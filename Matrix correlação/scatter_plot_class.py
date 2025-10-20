import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- BLOCO PRINCIPAL DO SCRIPT ---
if __name__ == "__main__":
    # ----------------------------------------------------------------------
    # 1. ESCOLHA AS FEATURES PARA A MATRIZ AQUI
    # ----------------------------------------------------------------------
    # Adicione ou remova os nomes das colunas que você quer na matriz.
    # A coluna 'condition' É OBRIGATÓRIA para que a diferenciação de classes funcione.
    features_para_matriz = [
        'HF',      
        'LF_HF',          
        'HF_NU',        
        'RMSSD',     
        'SDRR' ,
        'condition'   # Coluna para diferenciar as classes (OBRIGATÓRIA)
    ]

    # ----------------------------------------------------------------------
    # 2. DEFINIÇÃO DOS CAMINHOS (ajuste se necessário)
    # ----------------------------------------------------------------------
    caminho_time_domain = "dados/Train Data/Train Data Zip/time_domain_features_train.csv"
    caminho_freq_domain = "dados/Train Data/Train Data Zip/frequency_domain_features_train.csv"
    caminho_non_linear = "dados/Train Data/Train Data Zip/heart_rate_non_linear_features_train.csv"
    CHAVE_MERGE = 'uuid'

    try:
        # 3. Carregamento e combinação dos dados
        print("Carregando e combinando os arquivos de dados...")
        df_time = pd.read_csv(caminho_time_domain)
        df_freq = pd.read_csv(caminho_freq_domain)
        df_non_linear = pd.read_csv(caminho_non_linear)

        df_combinado = pd.merge(df_time, df_freq, on=CHAVE_MERGE, how='inner')
        df_combinado = pd.merge(df_combinado, df_non_linear, on=CHAVE_MERGE, how='inner')
        print(f"Dados combinados com sucesso. Total de {len(df_combinado)} registros.")

        # ----------------------------------------------------------------------
        # 4. GERAÇÃO DA MATRIZ DE GRÁFICOS (PAIRPLOT)
        # ----------------------------------------------------------------------
        print("\nGerando a matriz de gráficos (pairplot)... Isso pode levar alguns instantes.")

        # Filtra o DataFrame para conter apenas as colunas selecionadas, otimizando o processo
        df_plot = df_combinado[features_para_matriz]

        # Cria a matriz de gráficos
        pair_plot = sns.pairplot(
            df_plot,
            hue='condition',          # AQUI está a diferenciação por classes
            palette='viridis',        # Paleta de cores (pode ser 'plasma', 'inferno', etc.)
            diag_kind='kde',          # Mostra a densidade da distribuição na diagonal
            plot_kws={'alpha': 0.6, 's': 10} # Ajusta a transparência e tamanho dos pontos
        )
        
        # Adiciona um título geral à matriz
        pair_plot.fig.suptitle("Matriz de Dispersão Bivariada por Condição", y=1.02, fontsize=16)

        # Exibe o gráfico
        plt.show()

    except FileNotFoundError as e:
        print(f"Erro Crítico: Arquivo não encontrado. Verifique o caminho: {e}")
    except KeyError as e:
        print(f"Erro: A feature {e} não foi encontrada no DataFrame. Verifique se o nome está correto na lista 'features_para_matriz'.")
    except Exception as e:
        print(f"Ocorreu um erro geral durante o processo: {e}")