# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# #TASK 0

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import StandardScaler

# %%
# %% Carregamento dos Dados (Ajuste Final para vetores Y)

# 1. Carregamento dos X (Mantemos o mais robusto)
solTestX = pd.read_csv('solTestX.txt', sep=r'\s+', engine='python')
solTrainX = pd.read_csv('solTrainX.txt', sep=r'\s+', engine='python')

# 2. Carregamento do Y como DUAS COLUNAS
# Lógica: O arquivo contém (Índice Tabulado | Valor Y). 
# sep=r'\s+' e engine='python' são necessários para lidar com a tabulação/espaço entre as duas colunas.
# header=None: essencial.
# names=['Index', 'Solubility']: Nomes explícitos para as duas colunas.
solTestY_raw = pd.read_csv('solTestY.txt', sep=r'\s+', engine='python', header=None, names=['Index', 'Solubility'], quoting=3, skiprows=1)
solTrainY_raw = pd.read_csv('solTrainY.txt', sep=r'\s+', engine='python', header=None, names=['Index', 'Solubility'], quoting=3, skiprows=1)


# 3. Limpeza, Conversão e Extração da Coluna Y
# O valor de solubilidade é a SEGUNDA coluna. Precisamos extrair e garantir que ela seja float.

# Extrai apenas a coluna de solubilidade do DataFrame temporário.
solTestY = solTestY_raw[['Solubility']].copy()
solTrainY = solTrainY_raw[['Solubility']].copy()

# O .astype(str).str.strip() é mantido por segurança, mas agora só está limpando a coluna correta.
solTestY['Solubility'] = solTestY['Solubility'].astype(str).str.strip().astype(float)
solTrainY['Solubility'] = solTrainY['Solubility'].astype(str).str.strip().astype(float)

print("Carregamento de todos os arquivos de dados concluído com sucesso.")

# %%
solTestX.shape, solTrainX.shape, solTestY.shape, solTrainY.shape

 # %%
 #, solTrainX.columns#, solTestY.head, solTrainY.head
solTrainY.columns = ['Solubility']

# %%
# print("--- solTrainX Head ---")
# print(solTrainX.head())
# print("\n--- solTrainX Shape ---")
# print(solTrainX.shape)

# %%
skew(solTrainY['Solubility'])

# %%
plt.figure(figsize=(8,10))
sns.histplot(solTrainY, kde=True)
plt.show()
solTrainY.min(), solTrainY.max()

# %%
continuous_var = [name for name in solTrainX.columns if "FP" not in name]
continuous_var
continuous_skew = solTrainX[continuous_var].apply(lambda x: skew(x.dropna())) 
top_skew = continuous_skew[continuous_skew > 1.0].sort_values(ascending=False).head(5)
top_skew

# %%
x_transformed = solTrainX.copy()

transformer_yj = PowerTransformer(method='yeo-johnson')
x_continuous_transformed = transformer_yj.fit_transform(solTrainX[continuous_var])

x_continuous_transformed_df = pd.DataFrame(x_continuous_transformed, columns=continuous_var, index=solTrainX.index)
x_transformed[continuous_var] = x_continuous_transformed_df

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_transformed)
solTrainX_final = pd.DataFrame(x_scaled, columns=solTrainX.columns, index=solTrainX.index)

# %%
plt.figure(figsize=(14,6))
plt.subplot(1,2,1)
sns.histplot(solTrainX[top_skew.index[0]], kde=True)
plt.title(f'Top Skew não transformado, skew: {continuous_skew[top_skew.index[0]]}')

plt.subplot(1,2,2)
sns.histplot(solTrainX_final[top_skew.index[0]], kde=True)
plt.title('Top Skew transformado')

# %%
#Pre-processamento do conjunto de teste 
x_test_transformed = solTestX.copy()
x_test_continuous_transformed = transformer_yj.transform(solTestX[continuous_var])

x_test_continuous_transformed_df = pd.DataFrame(x_test_continuous_transformed, columns=continuous_var, index=solTestX.index)
x_test_transformed[continuous_var] = x_test_continuous_transformed_df

x_test_scaled = scaler.transform(x_test_transformed)
solTestX_final = pd.DataFrame(x_test_scaled, columns=solTestX.columns, index=solTestX.index)

# %%
correlation_matrix = solTrainX_final.corr()
np.fill_diagonal(correlation_matrix.values, np.nan)
max_corr = correlation_matrix.abs().max().max()

# %%
predictor_result_correlation = solTrainX_final.corrwith(solTrainY['Solubility']).abs().sort_values(ascending=False)
print(f'preditores mais correlacionados com resultado: {predictor_result_correlation.head(5)}')

# %% [markdown]
# # TASK 01

# %% [markdown]
# # T1 a)

# %%
x_train = solTrainX_final.to_numpy() 
y_train = solTrainY['Solubility'].values.ravel() #
 
x_matrix_train_manual = np.insert(x_train, 0, 1, axis=1)
# CRÍTICO: Para a Equação Normal, Y precisa ser 2D. Convertemos temporariamente.
Y_train_2d = y_train.reshape(-1, 1)

xt_x = x_matrix_train_manual.T @ x_matrix_train_manual
xt_x_inv = np.linalg.inv(xt_x)
xt_x_inv_xt = xt_x_inv @ x_matrix_train_manual.T
bethas = xt_x_inv_xt @ Y_train_2d

x_matrix_train = x_train

# %%
x_test = solTestX_final.to_numpy()
y_test = solTestY['Solubility'].values.ravel()

x_matrix_test = np.insert(x_test, 0, 1, axis=1)

y_prediction = x_matrix_test @ bethas

# %%
from sklearn.metrics import mean_squared_error, r2_score
rmse_ols = np.sqrt(mean_squared_error(y_test, y_prediction))
r2_ols = r2_score(y_test, y_prediction)
rmse_ols, r2_ols

# %% [markdown]
# # T1 b)

# %%
from sklearn.linear_model import LinearRegression

ols_model = LinearRegression(fit_intercept=True)
ols_model.fit(x_matrix_train, y_train)
ols_model_prediction = ols_model.predict(x_test)


# %%
rmse_ols_model = np.sqrt(mean_squared_error(y_test, ols_model_prediction))
r2_ols_model = r2_score(y_test, ols_model_prediction)
rmse_ols_model, r2_ols_model

# %%
equality_percent = rmse_ols_model * 100 /rmse_ols
equality_percent2 = r2_ols * 100 / r2_ols_model
equality_percent, equality_percent2

# %%
from sklearn.model_selection import KFold

kf = KFold(n_splits=10, shuffle = True, random_state=42)

rmse_scores_cv = []
r2_scores_cv = []

for train_index, validation_index in kf.split(x_matrix_train):

    x_train_fold, x_validation_fold = x_matrix_train[train_index], x_matrix_train[validation_index]
    y_train_fold, y_validation_fold = y_train[train_index], y_train[validation_index]

    model_fold = LinearRegression(fit_intercept=True)
    model_fold.fit(x_train_fold, y_train_fold)
    model_fold_prediction = model_fold.predict(x_validation_fold)

    rmse_scores_cv.append(np.sqrt(mean_squared_error(y_validation_fold, model_fold_prediction)))
    r2_scores_cv.append(r2_score(y_validation_fold, model_fold_prediction))

rmse_scores_cv_mean = np.mean(rmse_scores_cv)
r2_scores_cv_mean = np.mean(r2_scores_cv)    

rmse_scores_cv_mean, r2_scores_cv_mean
    

# %%
from sklearn.model_selection import cross_val_score

rmse_cv_builtin = -cross_val_score(
    LinearRegression(fit_intercept=True),
    x_matrix_train,
    y_train,
    cv=kf,
    scoring='neg_mean_squared_error'
)
rmse_cv_builtin = np.sqrt(rmse_cv_builtin).mean()

r2_cv_builtin = cross_val_score(
    LinearRegression(fit_intercept=True),
    x_matrix_train,
    y_train,
    cv=kf,
    scoring='r2'
).mean()

rmse_cv_builtin, r2_cv_builtin

# %%
np.all(x_matrix_train[:, 0] == 1)

# %%
# Imports adicionais para TASK 02 e TASK 03

from sklearn.linear_model import Ridge, Lasso 
from sklearn.decomposition import PCA          # para PCR
from sklearn.cross_decomposition import PLSRegression  # para PLS


# %% [markdown]
# # TASK 02

# %% [markdown]
# # T2 a)

# %%
def add_intercept(X: np.ndarray) -> np.ndarray:
    """
    Adiciona coluna de 1's no início de X (intercepto explícito).
    """
    intercept = np.ones((X.shape[0], 1))
    return np.hstack([intercept, X])


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, lambda_reg: float) -> np.ndarray:
    """
    Ajusta um modelo Ridge usando fórmula fechada.
    
    β_ridge = (XᵀX + λI*)⁻¹ Xᵀy, onde I* tem zero na posição do intercepto.
    """
    # Garante formato de vetor coluna
    y = y.reshape(-1, 1)
    
    X_int = add_intercept(X)
    
    # XᵀX e Xᵀy
    XtX = X_int.T @ X_int
    Xty = X_int.T @ y
    
    # Cria matriz de penalização (0 para intercepto)
    n_params = XtX.shape[0]
    I = np.eye(n_params)
    I[0, 0] = 0.0  # não penalizar o intercepto
    
    # (XᵀX + λI)⁻¹ Xᵀy
    beta_ridge = np.linalg.inv(XtX + lambda_reg * I) @ Xty
    
    return beta_ridge  # vetor coluna


def predict_with_beta(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """
    Gera previsões usando β (incluindo intercepto como primeira posição).
    """
    X_int = add_intercept(X)
    y_pred = X_int @ beta
    return y_pred.ravel()



# ajustando com um λ qualquer 
lambda_example = 1.0
beta_ridge_example = fit_ridge_closed_form(x_matrix_train, y_train, lambda_example)
y_pred_test_ridge_example = predict_with_beta(x_test, beta_ridge_example)

rmse_ridge_example = np.sqrt(mean_squared_error(y_test, y_pred_test_ridge_example))
r2_ridge_example = r2_score(y_test, y_pred_test_ridge_example)
rmse_ridge_example, r2_ridge_example


# %% [markdown]
# # T2 b)

 # %%
 # - Definir uma grade de λ (em escala log).
# - Para cada λ:
#     * Fazer 10-fold CV no conjunto de treino.
#     * Guardar RMSE e R² médios nos folds.
# - Escolher λ com menor RMSE médio.
# - Comparar com curva de R².

# Definir grade de lambdas
lambda_grid = np.logspace(-3, 3, 15) 
kf_pen = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_cv_ridge_fs = []  # RMSE médio em cada λ
r2_cv_ridge_fs = []    # R² médio em cada λ

for lam in lambda_grid:
    fold_rmse = []
    fold_r2 = []
    
    for train_idx, val_idx in kf_pen.split(x_matrix_train):
        X_tr, X_val = x_matrix_train[train_idx], x_matrix_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # Ajusta Ridge "from scratch" no fold
        beta_fold = fit_ridge_closed_form(X_tr, y_tr, lambda_reg=lam)
        y_val_pred = predict_with_beta(X_val, beta_fold)
        
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        fold_r2.append(r2_score(y_val, y_val_pred))
    
    rmse_cv_ridge_fs.append(np.mean(fold_rmse))
    r2_cv_ridge_fs.append(np.mean(fold_r2))

rmse_cv_ridge_fs = np.array(rmse_cv_ridge_fs)
r2_cv_ridge_fs = np.array(r2_cv_ridge_fs)

rmse_cv_ridge_fs, r2_cv_ridge_fs


# %% [markdown]
# # T2 c)

# %%
# Perfil de validação cruzada (RMSE e R² vs λ)
# - Usamos log10(λ) no eixo x para visualização mais clara.

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(np.log10(lambda_grid), rmse_cv_ridge_fs, marker='o')
plt.xlabel('log10(lambda)')
plt.ylabel('RMSE (CV)')
plt.title('Perfil de RMSE (Ridge from scratch)')

plt.subplot(1,2,2)
plt.plot(np.log10(lambda_grid), r2_cv_ridge_fs, marker='o', color='orange')
plt.xlabel('log10(lambda)')
plt.ylabel('R² (CV)')
plt.title('Perfil de R² (Ridge from scratch)')

plt.tight_layout()
plt.show()


# %% [markdown]
# # T2 d)

# %%
# Seleção de λ ótimo e avaliação no conjunto de teste

# - Selecionamos λ* com RMSE médio mínimo.
# - Ajustamos o modelo Ridge usando todo o treino.
# - Avaliamos RMSE e R² no conjunto de teste.

# Seleção de lambda* (menor RMSE)
best_idx = np.argmin(rmse_cv_ridge_fs)
best_lambda_fs = lambda_grid[best_idx]
best_lambda_fs

# λ* em todo o treino
beta_ridge_best = fit_ridge_closed_form(x_matrix_train, y_train, best_lambda_fs)
y_test_pred_ridge_best = predict_with_beta(x_test, beta_ridge_best)

rmse_ridge_test_fs = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge_best))
r2_ridge_test_fs = r2_score(y_test, y_test_pred_ridge_best)

rmse_ridge_test_fs, r2_ridge_test_fs


# %%
# - Agora, usando Ridge do sklearn para comparação.
# - Podemos reaproveitar o mesmo `lambda_grid` (aqui chamado de alpha).

# %%
rmse_cv_ridge_skl = []
r2_cv_ridge_skl = []

for lam in lambda_grid:
    model_ridge = Ridge(alpha=lam, fit_intercept=True)  # Ridge do sklearn
    
    fold_rmse = []
    fold_r2 = []
    
    for train_idx, val_idx in kf_pen.split(x_matrix_train):
        X_tr, X_val = x_matrix_train[train_idx], x_matrix_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        model_ridge.fit(X_tr, y_tr)
        y_val_pred = model_ridge.predict(X_val)
        
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        fold_r2.append(r2_score(y_val, y_val_pred))
    
    rmse_cv_ridge_skl.append(np.mean(fold_rmse))
    r2_cv_ridge_skl.append(np.mean(fold_r2))

rmse_cv_ridge_skl = np.array(rmse_cv_ridge_skl)
r2_cv_ridge_skl = np.array(r2_cv_ridge_skl)

rmse_cv_ridge_skl, r2_cv_ridge_skl


# %%
plt.figure(figsize=(6,4))
plt.plot(np.log10(lambda_grid), rmse_cv_ridge_fs, marker='o', label='Ridge FS')
plt.plot(np.log10(lambda_grid), rmse_cv_ridge_skl, marker='s', label='Ridge sklearn')
plt.xlabel('log10(lambda)')
plt.ylabel('RMSE (CV)')
plt.legend()
plt.title('Comparação de RMSE (Ridge FS vs sklearn)')
plt.show()


# %%
# λ ótimo pela implementação sklearn
best_idx_skl = np.argmin(rmse_cv_ridge_skl)
best_lambda_skl = lambda_grid[best_idx_skl]
best_lambda_skl

# %%
ridge_best_skl = Ridge(alpha=best_lambda_skl, fit_intercept=True)
ridge_best_skl.fit(x_matrix_train, y_train)
y_test_pred_ridge_skl = ridge_best_skl.predict(x_test)

rmse_ridge_test_skl = np.sqrt(mean_squared_error(y_test, y_test_pred_ridge_skl))
r2_ridge_test_skl = r2_score(y_test, y_test_pred_ridge_skl)

rmse_ridge_test_skl, r2_ridge_test_skl


# %% [markdown]
# # TASK 03

# %% [markdown]
# # T3 a) PCR

# %%
# - Usar PCA para reduzir dimensionalidade de x_matrix_train.
# - Avaliar por k-fold CV (RMSE e R²).
# - Escolher k ótimo e avaliar no conjunto de teste.

max_components = 30  # pode ajustar depois

kf_pcr = KFold(n_splits=10, shuffle=True, random_state=42)

rmse_cv_pcr = []
r2_cv_pcr = []

n_components_grid = range(1, max_components + 1)

for n_comp in n_components_grid:
    fold_rmse = []
    fold_r2 = []
    
    for train_idx, val_idx in kf_pcr.split(x_matrix_train):
        X_tr, X_val = x_matrix_train[train_idx], x_matrix_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        # PCA ajustado só no fold de treino
        pca = PCA(n_components=n_comp)
        X_tr_pca = pca.fit_transform(X_tr)
        X_val_pca = pca.transform(X_val)
        
        # Regressão linear nos componentes
        lin_reg = LinearRegression(fit_intercept=True)
        lin_reg.fit(X_tr_pca, y_tr)
        y_val_pred = lin_reg.predict(X_val_pca)
        
        fold_rmse.append(np.sqrt(mean_squared_error(y_val, y_val_pred)))
        fold_r2.append(r2_score(y_val, y_val_pred))
    
    rmse_cv_pcr.append(np.mean(fold_rmse))
    r2_cv_pcr.append(np.mean(fold_r2))

rmse_cv_pcr = np.array(rmse_cv_pcr)
r2_cv_pcr = np.array(r2_cv_pcr)

rmse_cv_pcr, r2_cv_pcr


# %% [markdown]
# # T3 b)

# %%
# Perfil de CV para PCR (RMSE e R² vs nº de componentes)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(list(n_components_grid), rmse_cv_pcr, marker='o')
plt.xlabel('Número de componentes (PCR)')
plt.ylabel('RMSE (CV)')
plt.title('PCR - RMSE vs nº de componentes')

plt.subplot(1,2,2)
plt.plot(list(n_components_grid), r2_cv_pcr, marker='o', color='orange')
plt.xlabel('Número de componentes (PCR)')
plt.ylabel('R² (CV)')
plt.title('PCR - R² vs nº de componentes')

plt.tight_layout()
plt.show()


# %% [markdown]
# # T3 c)

# %%
# Seleção do nº ótimo de componentes (PCR) e avaliação no teste

best_idx_pcr = np.argmin(rmse_cv_pcr)
best_n_comp_pcr = list(n_components_grid)[best_idx_pcr]
best_n_comp_pcr

# %%
# Ajusta PCA + regressão linear em todo o treino com k* componentes
pca_final = PCA(n_components=best_n_comp_pcr)
X_train_pca_final = pca_final.fit_transform(x_matrix_train)
X_test_pca_final  = pca_final.transform(x_test)

lin_reg_final = LinearRegression(fit_intercept=True)
lin_reg_final.fit(X_train_pca_final, y_train)
y_test_pred_pcr = lin_reg_final.predict(X_test_pca_final)

rmse_pcr_test = np.sqrt(mean_squared_error(y_test, y_test_pred_pcr))
r2_pcr_test = r2_score(y_test, y_test_pred_pcr)

rmse_pcr_test, r2_pcr_test


# %% [markdown]
#
