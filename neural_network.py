import numpy as np

class NeuralNetworkRegression:
    def __init__(self, input_size, hidden_size, output_size=1, weight_decay=0.0):
        """
        Inicialização da Rede Neural.
        Args:
            input_size: Número de preditores (D)
            hidden_size: Número de neurônios na camada oculta
            output_size: 1 (pois é regressão)
            weight_decay: Parâmetro lambda para regularização (L2) 
        """
        self.weight_decay = weight_decay
        
        # 1. Inicialização dos Pesos (Weights) e Viés (Bias)
        # O material sugere distribuição uniforme entre [-0.7, +0.7] para dados padronizados 
        self.W1 = np.random.uniform(-0.7, 0.7, (input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.uniform(-0.7, 0.7, (hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))

    def _sigmoid(self, z):
        # Função de ativação Sigmoide 
        z = np.asarray(z, dtype=np.float64)  # Garante tipo numérico
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        # Derivada da sigmoide: f'(z) = f(z) * (1 - f(z))
        return a * (1 - a)

    def forward(self, X):
        """
        Passo 3: Forward Computation 
        """
        # Garantir que X está em float64
        X = np.asarray(X, dtype=np.float64)
        
        # Camada Oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self._sigmoid(self.z1)  # Ativação não-linear
        
        # Camada de Saída (Linear para Regressão)
        # Nota: Não aplicamos sigmoide aqui pois queremos prever valores contínuos reais
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.output = self.z2 
        
        return self.output

    def backward(self, X, y, learning_rate):
        """
        Passo 4: Backward Computation (Backpropagation) 
        """
        # Garantir tipos numéricos
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        m = X.shape[0] # Número de exemplos
        
        # --- Cálculo dos Gradientes ---
        
        # Erro na saída (Predição - Real)
        # Derivada da função de custo MSE em relação à saída linear
        error_output = self.output - y 
        delta_output = error_output 
        
        # Erro na camada oculta (Retropropagação)
        # Multiplica o erro da saída pelos pesos W2 e pela derivada da sigmoide
        delta_hidden = np.dot(delta_output, self.W2.T) * self._sigmoid_derivative(self.a1)
        
        # Gradientes para W2 e b2
        dW2 = np.dot(self.a1.T, delta_output) / m
        db2 = np.sum(delta_output, axis=0, keepdims=True) / m
        
        # Gradientes para W1 e b1
        dW1 = np.dot(X.T, delta_hidden) / m
        db1 = np.sum(delta_hidden, axis=0, keepdims=True) / m
        
        # --- Atualização dos Pesos (Gradient Descent) ---
        # Inclui termo de Weight Decay (Regularização L2) 
        
        self.W2 -= learning_rate * (dW2 + self.weight_decay * self.W2)
        self.b2 -= learning_rate * db2
        
        self.W1 -= learning_rate * (dW1 + self.weight_decay * self.W1)
        self.b1 -= learning_rate * db1

    def train(self, X, y, epochs, learning_rate):
        loss_history = []
        
        for i in range(epochs):
            # 1. Forward
            y_pred = self.forward(X)
            
            # 2. Calcular Perda (MSE) para monitoramento
            loss = np.mean((y - y_pred) ** 2)
            loss_history.append(loss)
            
            # 3. Backward & Update
            self.backward(X, y, learning_rate)
            
            if i % 1000 == 0:
                print(f"Epoch {i}, Loss (MSE): {loss:.4f}")
                
        return loss_history

    def save_model(self, filepath):
        """
        Salva o modelo treinado em formato NPZ (NumPy).
        Preserva todos os pesos, biases e hiperparâmetros.
        
        Args:
            filepath: Caminho do arquivo (ex: 'modelo_treinado.npz')
        """
        np.savez(
            filepath,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            weight_decay=np.array(self.weight_decay)
        )
        print(f"✅ Modelo salvo com sucesso em: {filepath}")
        print(f"   Tamanho do arquivo: {np.savez.__doc__ or 'arquivo NPZ'}")

    @staticmethod
    def load_model(filepath):
        """
        Carrega um modelo previamente salvo.
        
        Args:
            filepath: Caminho do arquivo NPZ salvo
            
        Returns:
            model: Instância de NeuralNetworkRegression com pesos carregados
        """
        data = np.load(filepath)
        
        # Extrair dimensões dos pesos
        input_size = data['W1'].shape[0]
        hidden_size = data['W1'].shape[1]
        output_size = data['W2'].shape[1]
        weight_decay = float(data['weight_decay'])
        
        # Criar modelo com as mesmas dimensões
        model = NeuralNetworkRegression(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            weight_decay=weight_decay
        )
        
        # Carregar pesos treinados
        model.W1 = data['W1']
        model.b1 = data['b1']
        model.W2 = data['W2']
        model.b2 = data['b2']
        
        print(f"✅ Modelo carregado com sucesso de: {filepath}")
        print(f"   Arquitetura: {input_size} → {hidden_size} → {output_size}")
        print(f"   Weight Decay: {weight_decay}")
        
        return model

    def get_model_info(self):
        """
        Retorna informações sobre o modelo.
        """
        info = {
            'input_size': self.W1.shape[0],
            'hidden_size': self.W1.shape[1],
            'output_size': self.W2.shape[1],
            'weight_decay': self.weight_decay,
            'total_params': self.W1.size + self.b1.size + self.W2.size + self.b2.size
        }
        return info

# --- Funções Auxiliares de Pré-processamento ---

def standardize(data):
    """Padroniza os dados para média 0 e desvio padrão 1"""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Evitar divisão por zero
    std[std == 0] = 1 
    return (data - mean) / std, mean, std

def apply_standardization(data, mean, std):
    return (data - mean) / std

# --- Exemplo de Uso (Simulação) ---

if __name__ == "__main__":
    # 1. Criar dados dummy para teste (X com 5 preditores, Y contínuo)
    np.random.seed(42)
    X_raw = np.random.rand(100, 5) * 10 
    y_raw = 2 * X_raw[:, 0] - 1.5 * X_raw[:, 1] + 3 + np.random.normal(0, 1, 100)
    y_raw = y_raw.reshape(-1, 1) # Formato coluna

    # 2. Pré-processamento (Obrigatório segundo os slides)
    X_train_scaled, mean_X, std_X = standardize(X_raw)
    
    # 3. Configuração do Modelo
    # Hidden units: slides sugerem testar vários ou usar um número razoavelmente grande com regularização 
    model = NeuralNetworkRegression(
        input_size=5, 
        hidden_size=10, 
        weight_decay=0.01 
    )
    
    # 4. Treinamento
    print("Iniciando treinamento...")
    history = model.train(X_train_scaled, y_raw, epochs=5000, learning_rate=0.01)
    
    # 5. Predição
    predicoes = model.forward(X_train_scaled)
    print("\nExemplo de Predição:")
    print(f"Real: {y_raw[0][0]:.2f}, Previsto: {predicoes[0][0]:.2f}")