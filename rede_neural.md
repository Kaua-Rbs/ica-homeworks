# Documentação: Rede Neural para Predição de Heart Rate

## 📋 Sumário
1. [Arquitetura da Rede Neural](#arquitetura)
2. [Implementação Técnica](#implementação)
3. [Processo de Treinamento](#treinamento)
4. [Avaliação do Modelo](#avaliação)
5. [Exportação e Carregamento](#persistência)

---

## 🏗️ Arquitetura {#arquitetura}

### Estrutura Geral

A rede neural implementada é um **modelo de 2 camadas** (duas-camadas feed-forward):

```
Entrada (X: n×D)
    ↓
Camada Oculta (128 neurônios, Sigmoide)
    ↓
Camada de Saída (1 neurônio, Linear)
    ↓
Predição (ŷ: n×1)
```

**Onde:**
- **D** = número de features (variáveis preditoras) = 19 features após pré-processamento
- **n** = número de exemplos de treinamento
- **Camada Oculta**: 128 neurônios com ativação **Sigmoide** para capturar não-linearidades
- **Camada de Saída**: 1 neurônio com ativação **Linear** (apropriado para regressão)

### Dimensões dos Pesos

```
W1 (entrada → oculta):  (19, 128)
b1 (bias oculta):       (1, 128)

W2 (oculta → saída):    (128, 1)
b2 (bias saída):        (1, 1)

Total de parâmetros: 19×128 + 128 + 128×1 + 1 = 2,561 parâmetros
```

---

## 💻 Implementação Técnica {#implementação}

### Classe: `NeuralNetworkRegression`

#### 1. **Inicialização** (`__init__`)

```python
def __init__(self, input_size, hidden_size, output_size=1, weight_decay=0.0):
```

**Responsabilidades:**
- Inicializa os pesos W1 e W2 com distribuição uniforme $\mathcal{U}[-0.7, 0.7]$
- Inicializa os biases b1 e b2 com zeros
- Armazena o parâmetro de regularização L2 (`weight_decay`)

**Decisões de Design:**
- Intervalo de inicialização [-0.7, 0.7] é recomendado para dados padronizados
- Biases começam em zero (inicialização padrão)
- Weight decay permite controlar a magnitude dos pesos durante o treinamento

#### 2. **Forward Pass** (`forward`)

```python
def forward(self, X):
```

**Fluxo Computacional:**

1. **Camada Oculta:**
   $$z_1 = X \cdot W_1 + b_1$$
   $$a_1 = \sigma(z_1) = \frac{1}{1 + e^{-z_1}}$$

2. **Camada de Saída:**
   $$z_2 = a_1 \cdot W_2 + b_2$$
   $$\hat{y} = z_2 \quad \text{(sem ativação para regressão)}$$

**Detalhes Importantes:**
- Força conversão de X para `float64` para evitar erros numéricos (overflow em sigmoid)
- Armazena internamente `z1`, `a1`, `z2` para uso no backward pass
- Não aplica sigmoide na saída (regressão linear, não classificação)

#### 3. **Backward Pass** (`backward`)

Implementa o algoritmo de **backpropagation** com cálculo de gradientes:

**Etapa 1: Erro da Saída**
$$\delta_{out} = \hat{y} - y = \text{(Predição - Real)}$$

**Etapa 2: Retropropagação do Erro**
$$\delta_{hidden} = (\delta_{out} \cdot W_2^T) \odot \sigma'(a_1)$$

Onde $\sigma'(a_1) = a_1 \cdot (1 - a_1)$ é a derivada da sigmoide.

**Etapa 3: Cálculo dos Gradientes**
$$\frac{\partial L}{\partial W_2} = \frac{1}{m} a_1^T \cdot \delta_{out}$$
$$\frac{\partial L}{\partial b_2} = \frac{1}{m} \sum \delta_{out}$$
$$\frac{\partial L}{\partial W_1} = \frac{1}{m} X^T \cdot \delta_{hidden}$$
$$\frac{\partial L}{\partial b_1} = \frac{1}{m} \sum \delta_{hidden}$$

**Etapa 4: Atualização dos Pesos (Gradient Descent com L2 Regularização)**
$$W_2 := W_2 - \alpha \left(\frac{\partial L}{\partial W_2} + \lambda W_2\right)$$
$$W_1 := W_1 - \alpha \left(\frac{\partial L}{\partial W_1} + \lambda W_1\right)$$

Onde:
- $\alpha$ = taxa de aprendizado (learning_rate)
- $\lambda$ = weight_decay (parâmetro de regularização L2)

#### 4. **Função de Ativação Sigmoide**

```python
def _sigmoid(self, z):
    z = np.asarray(z, dtype=np.float64)
    return 1 / (1 + np.exp(-z))
```

**Propriedades:**
- Range: (0, 1)
- Função não-linear que introduz capacidade de modelar relações complexas
- Força conversão para float64 para estabilidade numérica (evita overflow)

#### 5. **Derivada da Sigmoide**

```python
def _sigmoid_derivative(self, a):
    return a * (1 - a)
```

Usa a propriedade: $\frac{d}{dz}\sigma(z) = \sigma(z) \cdot (1 - \sigma(z))$

---

## 🚀 Processo de Treinamento {#treinamento}

### Método: `train()`

```python
def train(self, X, y, epochs, learning_rate):
```

**Pseudocódigo:**
```
Para cada época (1 até epochs):
    1. Forward pass: calcular ŷ
    2. Calcular loss (MSE)
    3. Backward pass: calcular gradientes
    4. Atualizar pesos via gradient descent
```

**Parâmetros Utilizados:**

| Parâmetro | Valor | Justificativa |
|-----------|-------|---------------|
| Hidden Size | 128 | Melhor desempenho encontrado via Grid Search de hiperparâmetros |
| Learning Rate | 0.01 | Velocidade de convergência moderada |
| Weight Decay | 0.001 | Regularização L2 leve para reduzir overfitting |
| Epochs | 2000 | Suficiente para convergência |

**Loss Function (MSE):**
$$L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

**Monitoramento:**
- Imprime loss a cada 1000 épocas
- Retorna histórico completo de loss para visualização

---

## 📊 Avaliação do Modelo {#avaliação}

### Métricas Utilizadas

#### 1. **R² Score (Coeficiente de Determinação)**

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**Interpretação:**
- $R^2 = 1$: Predições perfeitas
- $R^2 = 0$: Modelo não melhor que predizer a média
- $R^2 < 0$: Modelo pior que a baseline

**Classificação de Performance:**
| Faixa | Qualidade |
|-------|-----------|
| R² > 0.9 | Excelente |
| 0.8 < R² ≤ 0.9 | Muito Bom |
| 0.7 < R² ≤ 0.8 | Bom |
| 0.5 < R² ≤ 0.7 | Aceitável |
| R² ≤ 0.5 | Fraco |

#### 2. **RMSE (Root Mean Squared Error)**

$$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

**Unidade:** bpm (batidas por minuto)

**Interpretação:** Erro médio em valores reais da variável alvo.

### Avaliação em Notebook

No arquivo `treinar_rede_neural.ipynb`, a avaliação segue este fluxo:

```python
# 1. Fazer predições
y_train_pred = model.forward(X_train)
y_test_pred = model.forward(X_test)

# 2. Calcular R² e RMSE
train_r2 = r2_score(YTrain, y_train_pred)
test_r2 = r2_score(YTest, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(YTrain, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(YTest, y_test_pred))

# 3. Detecção de Overfitting
overfitting = abs(train_r2 - test_r2)
```

### Visualizações

A avaliação inclui 4 gráficos:

1. **Scatter: Predito vs Real (Treino)**
   - Pontos próximos à diagonal = boas predições
   - Dispersão = incerteza do modelo

2. **Scatter: Predito vs Real (Teste)**
   - Valida generalização do modelo
   - Comparar com treino detecta overfitting

3. **Distribuição de Resíduos (Treino)**
   - Resíduo = valor real - predito
   - Distribuição centrada em 0 é desejável
   - Mostra if modelo tem viés sistemático

4. **Distribuição de Resíduos (Teste)**
   - Confirma que resíduos em teste também são normais
   - Assimetrias indicam problemas no modelo

---

## 🔍 Validação de Hiperparâmetros {#validacao}

A etapa de validação testa múltiplas configurações via **Grid Search**:

### Grid de Busca

```python
hidden_sizes = [32, 64, 128]
learning_rates = [0.001, 0.01]
weight_decays = [0.0, 0.001]
epochs_val = 300
```

**Total:** 3 × 2 × 2 = 12 combinações

### Estratégia de Validação

1. **Divisão de Dados:** Treino (80%) + Validação (20%)
2. **Para cada combinação:**
   - Treinar modelo com validação split
   - Avaliar R² e RMSE no conjunto de validação
3. **Seleção:** Configuração com maior R² na validação
4. **Retorno:** Melhores parâmetros para treinamento final

---

## 💾 Exportação e Carregamento {#persistência}

### Salvar Modelo

```python
model.save_model('modelos_treinados/modelo_hr_prediction.npz')
```

**Formato:** NPZ (NumPy compressed archive)

**Conteúdo Preservado:**
```
W1:           Pesos camada 1 (19, 64)
b1:           Biases camada 1 (1, 64)
W2:           Pesos camada 2 (64, 1)
b2:           Biases camada 2 (1, 1)
weight_decay: Hiperparâmetro de regularização
```

### Carregar Modelo

```python
modelo_carregado = NeuralNetworkRegression.load_model(
    'modelos_treinados/modelo_hr_prediction.npz'
)
```

**Processo:**
1. Lê arquivo NPZ
2. Extrai dimensões de W1 e W2 → reconstrói arquitetura (19 → 128 → 1)
3. Carrega todos os pesos treinados
4. Retorna instância pronta para predição

**Verificação de Integridade:**
```python
# Compara predições antes/depois
diff = np.abs(y_pred_original - y_pred_carregado).max()
# Deve ser próximo de 0
```

---

## 📈 Exemplo de Uso Completo

### Em novo notebook:

```python
from neural_network import NeuralNetworkRegression
import numpy as np

# 1. Carregar dados processados
X_novo = pd.read_csv('dados_processados/XTest.csv').values

# 2. Carregar modelo treinado
modelo = NeuralNetworkRegression.load_model(
    'modelos_treinados/modelo_hr_prediction.npz'
)

# 3. Fazer predições
predicoes = modelo.forward(X_novo)

# 4. Obter informações do modelo
info = modelo.get_model_info()
print(f"Arquitetura: {info['input_size']} → {info['hidden_size']} → {info['output_size']}")
print(f"Total de parâmetros: {info['total_params']}")
# Saída esperada: 19 → 128 → 1 (2561 parâmetros)
```

---

## 🔧 Decisões de Design e Justificativas

| Decisão | Justificativa |
|---------|---------------|
| 2 camadas (não mais) | Suficiente para dados tabulares; evita overfitting |
| Sigmoide na oculta | Não-linearidade; evita colapso para linear simples |
| Sem sigmoide na saída | Regressão: precisamos valores contínuos reais |
| Inicialização [-0.7, 0.7] | Padrão para dados padronizados; evita saturação inicial |
| Weight decay (L2) | Regularização: penaliza pesos grandes, reduz overfitting |
| Learning rate 0.01 | Balance: rápido o suficiente, estável o bastante |
| MSE como loss | Apropriado para regressão; diferenciável |

---

## ⚠️ Considerações e Limitações

1. **Dados Padronizados:** X deve ser padronizado (média=0, std=1) antes de treinar
2. **Escala de Y:** Não padronizamos Y; modelo aprende a escala direto
3. **Sem Dropout:** Não usamos dropout; confiamos em weight decay
4. **Batch Size:** Usamos batch completo (todas as amostras por época)
5. **Learning Rate Fixo:** Sem decay; poderia melhorar com scheduler
6. **Sem Validação Cross-Fold:** Usamos single train/test split

---

## 🎯 Próximos Passos Recomendados

1. **Tuning Avançado:** Testar epochs maiores (5000+) e learning rates menores
2. **Early Stopping:** Parar treinamento se val_loss não melhorar
3. **Batch Normalization:** Estabilizar training e acelerar convergência
4. **Dropout:** Reduzir overfitting ainda mais
5. **Arquiteturas Alternativas:** Testar 3+ camadas
6. **Ensemble:** Combinar múltiplos modelos

---

**Última atualização:** 8 de dezembro de 2025
**Arquivo relacionado:** `neural_network.py`, `treinar_rede_neural.ipynb`
