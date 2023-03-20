#! python3
import numpy as np

class MLP:
    
    def __init__(self, n_inputs=3, n_hidden=[3,5], n_outputs=2):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        
        layers = [self.n_inputs] + self.n_hidden + [self.n_outputs]
        
        # Initiate random weights
        self.weights = []
        
        for i in range(len(layers)-1):
            w = np.random.rand(layers[i], layers[i+1])    
            self.weights.append(w)        
        
    def forward_propagate(self, inputs):
        
        activations = inputs
        for w in self.weights:
            # calculate the net inputs
            net_inputs = np.dot(activations, w)
            
            # calculate the activations
            activations = self._sigmoid(net_inputs)
            
        return activations
    
    
    def _sigmoid(self, x): 
        return 1/(1 + np.exp(-x))
        

if __name__ == "__main__":
    
    # Create an MLP     
    mlp = MLP()

    # Create some random inputs
    inputs = np.random.rand(mlp.n_inputs)
    
    
    
    # Perform forward prop
    outputs = mlp.forward_propagate(inputs)
    
    # Print the results
    print(f"The network input is: {inputs}")
    print("The weights in the nn are:")
    for i in mlp.weights:
        print(i, end="\n\n")
        
    print(f"The network output is: {outputs}")
    