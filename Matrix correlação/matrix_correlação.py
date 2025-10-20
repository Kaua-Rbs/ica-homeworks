import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. CONTROLE A SUA MATRIZ AQUI ---



# Defina a lista de features que você quer do gráfico 
features_linhas = [
         
        'HF', 'LF_HF', 'HF_NU',  'RMSSD','SDRR' ,'SD1', 'SD2',
        
    
    
    #'SDRR_RMSSD_REL_RR','KURT_REL_RR','SKEW_REL_RR', 'VLF','VLF_PCT','LF','LF_PCT','LF_NU','HF','HF_PCT','HF_NU','TP','LF_HF','HF_LF', 'SD1','SD2','sampen','higuci'
  
    #'MEAN_RR','MEDIAN_RR','SDRR','RMSSD','SDSD','SDRR_RMSSD','HR','pNN25','pNN50','KURT','SKEW','MEAN_REL_RR','MEDIAN_REL_RR','SDRR_REL_RR','RMSSD_REL_RR','SDSD_REL_RR'
  ]

features_colunas = [
        'HF',      
        'LF_HF',          
        'HF_NU',        
        'RMSSD',     
        'SDRR' ,
        'SD1',
        'SD2',
         
    
    #'SDRR_RMSSD_REL_RR','KURT_REL_RR','SKEW_REL_RR', 'VLF','VLF_PCT','LF','LF_PCT','LF_NU','HF','HF_PCT','HF_NU','TP','LF_HF','HF_LF', 'SD1','SD2','sampen','higuci'
  ]


#cle 2. ATUALIZE OS CAMINHOS DOS SEUS ARQUIVOS CSV
caminho_arquivo_time = 'dados/Train Data/Train Data Zip/time_domain_features_train.csv'
caminho_arquivo_freq = 'dados/Train Data/Train Data Zip/frequency_domain_features_train.csv'
caminho_arquivo_nonlinear = 'dados/Train Data/Train Data Zip/heart_rate_non_linear_features_train.csv'


# --- 3. CARREGAMENTO E GERAÇÃO DO GRÁFICO (não precisa editar daqui para baixo) ---
try:
    # Carrega e une os arquivos de dados de forma simplificada
    df_time = pd.read_csv(caminho_arquivo_time)
    df_freq = pd.read_csv(caminho_arquivo_freq)
    df_nonlinear = pd.read_csv(caminho_arquivo_nonlinear)

    df_merged = pd.merge(df_time, df_freq, on='uuid')
    df_final = pd.merge(df_merged, df_nonlinear, on='uuid')

    # 1. Calcula a matriz de correlação COMPLETA para todas as features numéricas
    full_correlation_matrix = df_final.corr(numeric_only=True)

    # 2. Seleciona apenas o cruzamento entre as linhas e colunas que você definiu
   
    rectangular_corr_matrix = full_correlation_matrix.loc[features_linhas, features_colunas]

    # 3. Gera o gráfico (heatmap)
    plt.figure(figsize=(14, 8)) 
    sns.heatmap(
        rectangular_corr_matrix,
        annot=True,          #  Exibe os valores dentro das células
        cmap='coolwarm',     # Paleta de cores (vermelho=positivo, azul=negativo)
        fmt=".2f"            # Formata os números com duas casas decimais
    )
    plt.title('Matriz de Correlação: Time Domain x Frequency Domain', fontsize=16)
    plt.xticks(rotation=45, ha='right') # Melhora a legibilidade dos rótulos das colunas
    plt.yticks(rotation=0)
    plt.show()

except FileNotFoundError:
    print("ERRO: Um ou mais arquivos não foram encontrados. Verifique os caminhos definidos nas variáveis.")
except KeyError as e:
    # Caso não encontre a feature
    print(f"ERRO: A feature {e} definida em uma de suas listas não foi encontrada.")
    print("Por favor, verifique se todos os nomes de features estão escritos corretamente.")