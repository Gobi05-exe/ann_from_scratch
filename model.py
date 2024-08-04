# Importing libraries
import numpy as np
import pandas as pd

# ANN Model class
class Model: 
    def __init__(self):
        self.layer_list = []  # Stores each layer
        
    def add(self, layer):  # Adding a layer to the model
        self.layer_list.append(layer)
    
    def train(self, X_train, y_train, epochs, learning_rate):  # Trains model
        for e in range(epochs):
            y_pred = np.array([])  # Predicted output array
    
            # Forward propagation
            for i in range(len(self.layer_list)): 
                if i == 0:
                    self.layer_list[i].X = X_train
                else:
                    self.layer_list[i].X = self.layer_list[i-1].z 
                y_pred = self.layer_list[i].forward_prop()
            
            if isinstance(self.layer_list[-1], Activation) and self.layer_list[-1].activation == "softmax":  # If multi-class classification
                temp = np.zeros_like(y_pred)
                temp[np.arange(y_pred.shape[0]), np.argmax(y_pred, axis=1)] = 1
                y_pred = temp

            op_grad = y_pred - y_train
            # Back propagation
            for i in range(len(self.layer_list)-1, -1, -1):
                op_grad, wt_grad, b_grad = self.layer_list[i].back_prop(op_grad)
                if isinstance(self.layer_list[i], Dense):
                    self.layer_list[i].update_params(wt_grad, b_grad, learning_rate)
            
            print("Epoch: {}/{}".format(e+1, epochs))
            print("Accuracy: {}%".format(self.get_accuracy(y_pred, y_train)))    
   
    def get_accuracy(self, y_pred, y_train):
        #if y_pred.shape == y_train.shape:
            #return np.mean(y_pred == y_train) * 100
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1)) * 100
    
    def predict(self, X):  # Takes input and returns a prediction
        for i in range(len(self.layer_list)):
            if i == 0:
                self.layer_list[i].X = X
            else:
                self.layer_list[i].X = self.layer_list[i-1].z 
            y_pred = self.layer_list[i].forward_prop()
        return y_pred

# Layer base class
class Layer:
    def __init__(self, units):
        self.units = units  # Number of neurons
        self.X = np.array([])  # Input array 
        self.z = np.array([])  # Output array
        self.ip_grad = np.array([])  # Input gradient
        self.wt_grad = np.array([])  # Weights gradient
        self.b_grad = np.array([])  # Biases gradient

# Dense layer class
class Dense(Layer):
    def __init__(self, units):
        super().__init__(units)
        self.w = None  # Weights
        self.b = None  # Biases
        
    def initialize_weights(self, input_dim):
        self.w = np.random.randn(self.units, input_dim)  # Initializing weights
        self.b = np.random.randn(self.units)  # Initializing biases
        
    def forward_prop(self):  # Forward propagation of dense layer
        if self.w is None or self.b is None:
            input_dim = self.X.shape[1]
            self.initialize_weights(input_dim)
        
        self.z = np.dot(self.X, self.w.T) + self.b
        return self.z
    
    def back_prop(self, op_grad):  # Back propagation of dense layer
        self.wt_grad = np.dot(op_grad.T, self.X)  # Weights gradient
        self.b_grad = np.sum(op_grad, axis=0)  # Biases gradient
        self.ip_grad = np.dot(op_grad, self.w)  # Input gradient
        return self.ip_grad, self.wt_grad, self.b_grad 
    
    def update_params(self, wt_grad, b_grad, learning_rate):  # Updates weights and biases
        self.w -= learning_rate * wt_grad
        self.b -= learning_rate * b_grad

# Activation Layer     
class Activation(Layer):
    def __init__(self, units, activation):
        super().__init__(units)
        self.activation = activation  # ReLU with MSE loss or softmax with cross-entropy loss
        
    def forward_prop(self):  # Forward propagation for activation layer
        if self.activation == "relu":
            self.z = np.maximum(self.X, 0)
            
        elif self.activation == "softmax":
            exp_X = np.exp(self.X - np.max(self.X, axis=1, keepdims=True))
            self.z = exp_X / np.sum(exp_X, axis=1, keepdims=True)
        return self.z
    
    def back_prop(self, op_grad):  # Back propagation of activation layer
        if self.activation == "relu":
            self.ip_grad = np.multiply(op_grad, (self.z > 0))
        elif self.activation == "softmax":
            self.ip_grad = op_grad  
        return self.ip_grad, None, None 


# Implementing the model 
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values.reshape(-1, 1)  # Reshaping target to (n_samples, 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


ann=Model()
ann.add(Dense(6))
ann.add(Activation(6,'relu'))
ann.add(Dense(6))
ann.add(Activation(6,'relu'))
ann.add(Dense(1))
ann.train(X_train,y_train,25,0.1)
