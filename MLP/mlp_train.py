#! python3
import numpy as np
from random import random

# Save the Activations and derivatives --> done
# Implement back propogation --> doneeeee 
# Implement Gradient Descent --> doneee
# Implement Train --> done
# Train our net with dummy data
# Make Predictions 
# Hurrayyy!!!!

class MLP:
    
    def __init__(self, n_inputs=3, n_hidden=[3,3], n_outputs=2):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        
        layers = [self.n_inputs] + self.n_hidden + [self.n_outputs]
        
        # Initiate random weights
        self.weights = []
        
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])    
            self.weights.append(w)    
            
        print('random weights', self.weights[:5])
        
        # Initiate activations
        self.activations = []
        for neurons in layers:
            a = np.zeros(neurons)
            self.activations.append(a)    
            
        # print('activations', self.activations)
        
        # Initiate derivatives
        self.derivatives = []
        for i in range(len(layers)-1):
            d = np.zeros((layers[i], layers[i+1]))
            self.derivatives.append(d)
            
        # print('derivatives', self.derivatives)
        
    def forward_propagate(self, inputs):
        
        activations = inputs
        self.activations[0] = inputs
        
        for i,w in enumerate(self.weights):
            # calculate the net inputs
            net_inputs = np.dot(self.activations[i], w)
            
            # calculate the activations
            activations = self._sigmoid(net_inputs)
            self.activations[i+1] = activations
        
        # print(activations)
        return activations
    
    def back_propagate(self, error, verbose=False):
        for i in reversed(range(len(self.derivatives))):
            activations = self.activations[i+1]
            delta = error * self._sigmoid_derivative(activations)
            delta_reshaped = delta.reshape(1, len(delta))
            current_activations = self.activations[i]
            current_activations_reshaped = current_activations.reshape(1, len(current_activations))
            derivative = np.matmul(delta_reshaped.T, current_activations_reshaped).T
            self.derivatives[i] = derivative
            
            error = np.dot(delta, self.weights[i].T)
            
            if verbose:
                print(f"Derivatives for W{i}: {self.derivatives[i]}")
            # print("---------")
            # print(activations)
            # print(delta_reshaped)
            # print('shape delta', delta_reshaped.shape)
            # print(current_activations_reshaped)
            # print('shape act', current_activations_reshaped.shape)
            # print(derivative)
            # print('derivative shape', derivative.shape)
        return error
    
    def gradient_descent(self, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] += self.derivatives[i] * learning_rate 
        
        # print('after gradient descent', self.weights[:5])
        
    def train(self, inputs, targets, epochs, learning_rate):
        
        for i in range(epochs):
            sum_error = 0 
            for inp, target in zip(inputs, targets):
                # Forward Propagation
                output = self.forward_propagate(inp)
                
                # Calculate Error
                error = target - output
                
                # Backward Propagation
                self.back_propagate(error)
                
                # Gradient Descent
                self.gradient_descent(learning_rate)
                
                
                
                sum_error += self._mse(target, output)
                
            # Report error
            print(f"Error: {sum_error/len(inputs)} at epoch {i+1}")
                
    def _mse(self, target, output):
        return np.average((target - output)**2)
    
    def _sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def _sigmoid(self, x): 
        return 1/(1 + np.exp(-x))
        

if __name__ == "__main__":
    
    # Create an MLP     
    mlp = MLP(2, [5], 1)
    
    # Create a dataset to train a network for the sum operation
    inputs = np.array([[random() / 2  for _ in range(2)] for _ in range(1000)])
    targets = np.array([[t[0] + t[1]] for t in inputs])
    
    # Train our mlp
    mlp.train(inputs, targets, epochs=10, learning_rate=0.5)
    
    
    # Predict
    inp = np.array([0.2, 0.3])
    target = np.array([0.9])
    out = mlp.forward_propagate(inp)
    print(f'The machine thinks {inp[0]} + {inp[1]} = {out}')