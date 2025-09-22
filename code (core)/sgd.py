"""
使用随机梯度下降 (SGD) 优化器训练三种不同的机器学习模型：
逻辑回归、支持向量机和多层感知机。
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import os

# --- 数据加载与预处理 ---
def load_and_preprocess_data(file_path, is_sparse=False):
    """加载并预处理数据"""
    if not os.path.exists(file_path):
        print(f"错误：找不到数据集 {file_path}")
        return None, None, None, None

    if is_sparse:
        data = pd.read_csv(file_path, header=None, low_memory=False)
        data = data.replace('?', np.nan)
        data = data.fillna(0)
        X = data.iloc[:, :-1].values.astype(float)
        y = data.iloc[:, -1].values
        y = np.where(y == 'ad.', 1, -1)
    else:
        data = pd.read_csv(file_path, header=None)
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        y = np.where(y == 1, 1, -1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, y_train, y_val

# --- 模型定义 ---

class LogisticRegressionModel:
    """逻辑回归模型"""
    def __init__(self, n_features, C=1e-5):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.C = C

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_params(self):
        return [self.weights, self.bias]

    def set_params(self, params):
        self.weights, self.bias = params

    def gradients(self, X, y):
        """计算梯度"""
        linear_model = np.dot(X, self.weights) + self.bias
        z = y * linear_model
        grad_factor = -y * (1 - self._sigmoid(z))
        
        dw = (1/len(y)) * np.dot(X.T, grad_factor) + 2 * self.C * self.weights
        db = (1/len(y)) * np.sum(grad_factor)
        return [dw, db]

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return np.sign(linear_model)

class SVMModel:
    """支持向量机模型"""
    def __init__(self, n_features, C=1e-3):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.C = C # 注意：这里的C与sklearn中的C相反，是正则化强度的直接乘数

    def get_params(self):
        return [self.weights, self.bias]

    def set_params(self, params):
        self.weights, self.bias = params

    def gradients(self, X, y):
        """计算次梯度"""
        linear_model = np.dot(X, self.weights) + self.bias
        condition = y * linear_model < 1
        
        # 将bool转为0/1，并调整形状用于乘法
        y_grad = np.where(condition, y, 0).reshape(-1, 1)

        dw = -np.mean(y_grad * X, axis=0) + 2 * self.C * self.weights
        db = -np.mean(y[condition]) if any(condition) else 0
        return [dw, db]

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return np.sign(linear_model)
        
class MLPModel:
    """多层感知机 (神经网络)"""
    def __init__(self, n_features, hidden_size=5, C=1e-7):
        self.w1 = np.random.randn(n_features, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros(1)
        self.C = C

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def get_params(self):
        return [self.w1, self.b1, self.w2, self.b2]

    def set_params(self, params):
        self.w1, self.b1, self.w2, self.b2 = params

    def gradients(self, X, y):
        """使用反向传播计算梯度"""
        n_samples = X.shape[0]
        y_mlp = np.where(y == 1, 1, 0).reshape(-1, 1)

        # 前向传播
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        y_hat = self._sigmoid(z2)

        # 反向传播
        # 损失函数为均方误差: 0.5 * (y_hat - y_mlp)^2
        # d_loss/d_y_hat = y_hat - y_mlp
        d_z2 = (y_hat - y_mlp) * y_hat * (1 - y_hat) # d_loss/d_z2
        
        dw2 = (1/n_samples) * np.dot(a1.T, d_z2) + 2 * self.C * self.w2
        db2 = (1/n_samples) * np.sum(d_z2, axis=0)

        d_a1 = np.dot(d_z2, self.w2.T)
        d_z1 = d_a1 * a1 * (1 - a1)
        
        dw1 = (1/n_samples) * np.dot(X.T, d_z1) + 2 * self.C * self.w1
        db1 = (1/n_samples) * np.sum(d_z1, axis=0)
        
        return [dw1, db1, dw2, db2]
        
    def predict(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        y_hat = self._sigmoid(z2)
        return np.where(y_hat > 0.5, 1, -1).flatten()

# --- 优化器定义 ---

class SGDOptimizer:
    def __init__(self, model, learning_rate=0.01, n_epochs=100, batch_size=32):
        self.model = model
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        n_samples = X.shape[0]
        for epoch in range(self.n_epochs):
            permutation = np.random.permutation(n_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]

                grads = self.model.gradients(X_batch, y_batch)
                params = self.model.get_params()
                
                # 学习率衰减
                lr = self.learning_rate / (1 + epoch * 0.1)

                updated_params = [p - lr * g for p, g in zip(params, grads)]
                self.model.set_params(updated_params)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs}')

# --- 主执行函数 ---

def main():
    # ******************************************************
    # *** 在这里选择要训练的模型               ***
    # *** 'lr' : Logistic Regression                   ***
    # *** 'svm': Support Vector Machine                ***
    # *** 'mlp': Multi-Layer Perceptron (Neural Network) ***
    # ******************************************************
    MODEL_TYPE = 'lr'

    model_map = {
        'lr': LogisticRegressionModel,
        'svm': SVMModel,
        'mlp': MLPModel
    }
    
    # --- Spambase (稠密) 数据集 ---
    print(f"--- [Spambase] 使用SGD训练 {MODEL_TYPE.upper()} ---")
    X_train, X_val, y_train, y_val = load_and_preprocess_data('spambase.data', is_sparse=False)
    
    if X_train is not None:
        n_features = X_train.shape[1]
        n_samples = len(X_train)
        
        # 根据模型类型设置不同超参数
        if MODEL_TYPE == 'svm':
            C = 1 / (1000 * n_samples)
            model = model_map[MODEL_TYPE](n_features, C=C)
        else:
            model = model_map[MODEL_TYPE](n_features)
            
        optimizer = SGDOptimizer(model, learning_rate=0.1)
        optimizer.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        print("\nSpambase 评估结果:")
        print(f"  Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
        print(f"  Recall: {recall_score(y_val, y_pred, zero_division=0):.4f}")

    # --- Ads (稀疏) 数据集 ---
    print(f"\n--- [Ads] 使用SGD训练 {MODEL_TYPE.upper()} ---")
    X_train, X_val, y_train, y_val = load_and_preprocess_data('ad.data', is_sparse=True)

    if X_train is not None:
        n_features = X_train.shape[1]
        n_samples = len(X_train)
        
        if MODEL_TYPE == 'svm':
            C = 1 / (1000 * n_samples)
            model = model_map[MODEL_TYPE](n_features, C=C)
        else:
            model = model_map[MODEL_TYPE](n_features)
            
        optimizer = SGDOptimizer(model, learning_rate=0.1)
        optimizer.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        print("\nAds 评估结果:")
        print(f"  Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
        print(f"  Recall: {recall_score(y_val, y_pred, zero_division=0):.4f}")

if __name__ == '__main__':
    main()
