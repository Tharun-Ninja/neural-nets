#! python3
"""Implementing artificial neurons in python
"""
import math

# Euler's constant
E = math.e

# Activation function
def sigmoid(x):
    return 1/(1 + E**-x)

# Getting the sum of inputs with weights
def activate(inputs, weights):
    h = 0
    for inp, weight in zip(inputs, weights):
        h += inp * weight
    
    print("weight", h)
    # Perform activation
    h = sigmoid(h)
    return h


if __name__ == "__main__":
    
    inputs = [0.7,0.3,0.2]
    weights = [0.4,0.7,0.2]

    output = activate(inputs, weights)
    print(output)