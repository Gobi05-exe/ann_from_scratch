import numpy as np
import pandas as pd

class Model:
    def __init__(self):
        self.layer_list = []
        
    def add(self, layer):
        self.layer_list.append(layer)
        
    def train(self, X_train, y_train):
        for i in range(len(self.layer_list)):
            if i == 0:
                self.layer_list[i].X = X_train
            else:
                self.layer_list[i].X = self.layer_list[i-1].a
            self.layer_list[i].forward_prop()
        
        print(self.layer_list[-1].a)
        print(self.layer_list[-1].a.shape)
        
    def predict(self, X):
        pass

class Layer:
    def __init__(self):
        self.X = np.array([])
        self.z = np.array([])
        self.a = np.array([])
    
    def forward_prop(self):
        pass
    
    def back_prop(self):
        pass
    
    def relu(self, a):
        return np.maximum(0, a)

class Dense(Layer):
    def __init__(self, units, activation):
        super().__init__()
        self.units = units
        self.activation = activation
        self.w = None
        self.b = None
        
    def initialize_weights(self, input_dim):
        self.w = np.random.randn(self.units, input_dim)
        self.b = np.random.randn(self.units, 1)
        
    def forward_prop(self):
        if self.w is None or self.b is None:
            input_dim = self.X.shape[1]
            self.initialize_weights(input_dim)
        
        self.z = np.dot(self.X, self.w.T) + self.b.T
        if self.activation == "relu":
            self.a = self.relu(self.z)
        else:
            self.a = self.z
        
        return self.w, self.b, self.a
    
    def back_prop(self):
        pass

dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

ann = Model()
ann.add(Dense(15, 'relu'))
ann.add(Dense(19, 'relu'))
ann.add(Dense(1, 'relu'))
ann.train(X, y)
