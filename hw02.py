import pandas as pd           # Essencial para DataFrames (estruturas de dados tabulares).
import numpy as np            # Fundamental para operações matriciais e numéricas (base do OLS).
import seaborn as sns         # Usado para visualização estatística.
import matplotlib.pyplot as plt # Base para visualização de gráficos.

from scipy.stats import skew  # Calcula o coeficiente de assimetria.
from sklearn.preprocessing import PowerTransformer # Transforma a distribuição de dados (Yeo-Johnson).
from sklearn.preprocessing import StandardScaler   # Padroniza os dados (média 0, DP 1).


solTestX = pd.read_csv('solTestX.txt', sep=r'\s+', engine='python')
solTrainX = pd.read_csv('solTrainX.txt', sep=r'\s+', engine='python')
solTestY = pd.read_csv('solTestY.txt', sep=r'\s+', engine='python', header=None) # Y não tem cabeçalho.
solTrainY = pd.read_csv('solTrainY.txt', sep=r'\s+', engine='python', header=None) # Y não tem cabeçalho.

solTrainY.columns = ['Solubility'] 
print(f"Shape do Treinamento (X, Y): {solTrainX.shape}, {solTrainY.shape}")
print(f"Shape do Teste (X, Y): {solTestX.shape}, {solTestY.shape}")


print(f"\nSkewness (Assimetria) da Solubilidade (log10): {skew(solTrainY['Solubility']):.3f}")

plt.figure(figsize=(6, 4))
sns.histplot(solTrainY['Solubility'], kde=True)
plt.title('Distribuição da Solubilidade (log10)')
plt.show()



continuous_var = [name for name in solTrainX.columns if "FP" not in name]

continuous_skew = solTrainX[continuous_var].apply(lambda x: skew(x.dropna())) 

top_skew = continuous_skew[continuous_skew > 1.0].sort_values(ascending=False).head(5)
print(f"\nTop 5 Variáveis Mais Enviesadas:\n{top_skew}")



x_transformed = solTrainX.copy() 

transformer_yj = PowerTransformer(method='yeo-johnson') 

x_continuous_transformed = transformer_yj.fit_transform(solTrainX[continuous_var]) 

x_continuous_transformed_df = pd.DataFrame(x_continuous_transformed, columns=continuous_var, index=solTrainX.index) 
x_transformed[continuous_var] = x_continuous_transformed_df 


scaler = StandardScaler() 
x_scaled = scaler.fit_transform(x_transformed) 

solTrainX_final = pd.DataFrame(x_scaled, columns=solTrainX.columns, index=solTrainX.index)



x_test_transformed = solTestX.copy()

x_test_continuous_transformed = transformer_yj.transform(solTestX[continuous_var]) 

x_test_continuous_transformed_df = pd.DataFrame(x_test_continuous_transformed, columns=continuous_var, index=solTestX.index)
x_test_transformed[continuous_var] = x_test_continuous_transformed_df

x_test_scaled = scaler.transform(x_test_transformed) 

solTestX_final = pd.DataFrame(x_test_scaled, columns=solTestX.columns, index=solTestX.index)
print("\nPré-processamento concluído. solTrainX_final e solTestX_final criados.")



correlation_matrix = solTrainX_final.corr()

np.fill_diagonal(correlation_matrix.values, np.nan) 

max_corr = correlation_matrix.abs().max().max()
print(f"\nMáxima Correlação Absoluta entre Preditores (Multicolinearidade): {max_corr:.3f}")


predictor_result_correlation = solTrainX_final.corrwith(solTrainY['Solubility']).abs().sort_values(ascending=False)
print(f"Top 5 Preditores Mais Correlacionados com Y:\n{predictor_result_correlation.head(5)}")
