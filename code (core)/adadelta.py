"""
使用 AdaDelta 优化器训练三种不同的机器学习模型：
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
    def __init__(self, n_features, C=1e-5):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.C = C
    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def get_params(self): return [self.weights, self.bias]
    def set_params(self, params): self.weights, self.bias = params
    def gradients(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        z = y * linear_model
        grad_factor = -y * (1 - self._sigmoid(z))
        dw = (1/len(y)) * np.dot(X.T, grad_factor) + 2 * self.C * self.weights
        db = (1/len(y)) * np.sum(grad_factor)
        return [dw, db]
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)

class SVMModel:
    def __init__(self, n_features, C=1e-3):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.C = C
    def get_params(self): return [self.weights, self.bias]
    def set_params(self, params): self.weights, self.bias = params
    def gradients(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        condition = y * linear_model < 1
        y_grad = np.where(condition, y, 0).reshape(-1, 1)
        dw = -np.mean(y_grad * X, axis=0) + 2 * self.C * self.weights
        db = -np.mean(y[condition]) if any(condition) else 0
        return [dw, db]
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
        
class MLPModel:
    def __init__(self, n_features, hidden_size=5, C=1e-7):
        self.w1 = np.random.randn(n_features, hidden_size) * 0.01
        self.b1 = np.zeros(hidden_size)
        self.w2 = np.random.randn(hidden_size, 1) * 0.01
        self.b2 = np.zeros(1)
        self.C = C
    def _sigmoid(self, z): return 1 / (1 + np.exp(-z))
    def get_params(self): return [self.w1, self.b1, self.w2, self.b2]
    def set_params(self, params): self.w1, self.b1, self.w2, self.b2 = params
    def gradients(self, X, y):
        n_samples = X.shape[0]
        y_mlp = np.where(y == 1, 1, 0).reshape(-1, 1)
        z1 = np.dot(X, self.w1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        y_hat = self._sigmoid(z2)
        d_z2 = (y_hat - y_mlp) * y_hat * (1 - y_hat)
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

class AdaDeltaOptimizer:
    def __init__(self, model, n_epochs=100, batch_size=32, decay_rate=0.95, epsilon=1e-6):
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        # 初始化累加器
        self.Eg = [np.zeros_like(p) for p in self.model.get_params()]
        self.Edx = [np.zeros_like(p) for p in self.model.get_params()]

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
                
                updated_params = []
                for j, (p, g) in enumerate(zip(params, grads)):
                    self.Eg[j] = self.decay_rate * self.Eg[j] + (1 - self.decay_rate) * g**2
                    
                    rms_dx = np.sqrt(self.Edx[j] + self.epsilon)
                    rms_g = np.sqrt(self.Eg[j] + self.epsilon)
                    delta_p = - (rms_dx / rms_g) * g
                    
                    self.Edx[j] = self.decay_rate * self.Edx[j] + (1 - self.decay_rate) * delta_p**2
                    
                    updated_params.append(p + delta_p)
                
                self.model.set_params(updated_params)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{self.n_epochs}')

# --- 主执行函数 ---
def main():
    MODEL_TYPE = 'lr'
    model_map = {'lr': LogisticRegressionModel, 'svm': SVMModel, 'mlp': MLPModel}
    print(f"--- [Spambase] 使用AdaDelta训练 {MODEL_TYPE.upper()} ---")
    X_train, X_val, y_train, y_val = load_and_preprocess_data('spambase.data', is_sparse=False)
    if X_train is not None:
        n_features, n_samples = X_train.shape[1], len(X_train)
        model = model_map[MODEL_TYPE](n_features, C=1/(1000*n_samples)) if MODEL_TYPE == 'svm' else model_map[MODEL_TYPE](n_features)
        optimizer = AdaDeltaOptimizer(model)
        optimizer.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print("\nSpambase 评估结果:")
        print(f"  Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
        print(f"  Recall: {recall_score(y_val, y_pred, zero_division=0):.4f}")

    print(f"\n--- [Ads] 使用AdaDelta训练 {MODEL_TYPE.upper()} ---")
    X_train, X_val, y_train, y_val = load_and_preprocess_data('ad.data', is_sparse=True)
    if X_train is not None:
        n_features, n_samples = X_train.shape[1], len(X_train)
        model = model_map[MODEL_TYPE](n_features, C=1/(1000*n_samples)) if MODEL_TYPE == 'svm' else model_map[MODEL_TYPE](n_features)
        optimizer = AdaDeltaOptimizer(model)
        optimizer.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        print("\nAds 评估结果:")
        print(f"  Accuracy: {accuracy_score(y_val, y_pred):.4f}")
        print(f"  Precision: {precision_score(y_val, y_pred, zero_division=0):.4f}")
        print(f"  Recall: {recall_score(y_val, y_pred, zero_division=0):.4f}")

if __name__ == '__main__':
    main()
