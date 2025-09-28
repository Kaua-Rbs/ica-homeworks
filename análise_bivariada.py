import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# A função de plotagem é simplificada para aceitar o DF JUNTADO
def plotar_dispersao_bivariada_final(df_plot, x_feature, y_feature):

    """
    Cria um gráfico de dispersão a partir de um DataFrame já pronto.
    """

    try:

        df_subset = df_plot[[x_feature, y_feature]].dropna().copy()

        # 1. Configuração do Gráfico
        plt.figure(figsize=(10, 7))
        # 2. Criação do Gráfico de Dispersão com Regressão (usando o DataFrame já juntado)
        sns.regplot(x=x_feature, y=y_feature, data=df_subset, 

                    scatter_kws={'alpha':1.0, 's':0.5}, # Otimizado para 40k pontos

                    line_kws={'color':'red', 'linewidth':2})

        # 3. Personalização 
        correlacao = df_subset[x_feature].corr(df_subset[y_feature])
        plt.title(f"Relação Bivariada entre {x_feature} e {y_feature}\n(Correlação: {correlacao:.2f})",
                  fontsize=16, fontweight='bold')
        plt.xlabel(x_feature, fontsize=12)
        plt.ylabel(y_feature, fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)
        
        # 4. Exibição
        plt.tight_layout()
        plt.show()


    except KeyError as e:
        print(f"Erro: Uma ou mais colunas não foram encontradas no DataFrame juntado: {e}")
    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        
if __name__ == "__main__":

    # ----------------------------------------------------------------------
    # 1. DEFINIÇÃO DOS CAMINHOS
    # ----------------------------------------------------------------------
    caminho_time_domain = "dados/Train Data/Train Data Zip/time_domain_features_train.csv"
    caminho_freq_domain = "dados/Train Data/Train Data Zip/frequency_domain_features_train.csv"
    caminho_non_linear = "dados/Train Data/Train Data Zip/heart_rate_non_linear_features_train.csv"
    # A CHAVE DEVE SER A COLUNA 'uuid' que identifica a medição em todos os arquivos
    CHAVE_MERGE = 'uuid' 
    try:

        df_time = pd.read_csv(caminho_time_domain, sep=",")
        df_freq = pd.read_csv(caminho_freq_domain, sep=",")
        df_non_linear = pd.read_csv(caminho_non_linear, sep=",")

        # Juntar df_time com df_freq

        df_combinado = pd.merge(df_time, df_freq, on=CHAVE_MERGE, how='inner')

        # Juntar o resultado com df_non_lineas
        df_combinado = pd.merge(df_combinado, df_non_linear, on=CHAVE_MERGE, how='inner')
        print(f"Dados combinados com sucesso. Total de {len(df_combinado)} registros.")

        # ----------------------------------------------------------------------
        # 3. PLOTAGEM USANDO FEATURES DE ARQUIVOS DIFERENTES
        # ----------------------------------------------------------------------    
        
        # Features para análise

        x_feature = 'MEAN_RR' 
        y_feature = 'VLF' # Supondo que VLF seja uma feature do domínio da frequência
        plotar_dispersao_bivariada_final(df_combinado, x_feature, y_feature)

        

        # Feature 2
        x_feature_2 = 'LF_HF' 
        y_feature_2 = 'ApEn' # Supondo que ApEn seja uma feature não-linear
        plotar_dispersao_bivariada_final(df_combinado, x_feature_2, y_feature_2)

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado: {e}")
    except Exception as e:
        print(f"Ocorreu um erro geral durante o merge ou plotagem: {e}")
