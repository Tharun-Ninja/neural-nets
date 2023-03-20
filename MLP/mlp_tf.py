#! python3
import numpy as np
from random import random

from sklearn.model_selection import train_test_split
import tensorflow as tf

# Create a dataset to train a network for the sum operation
def generate_dataset(n_samples, test_size):
    x = np.array([[random() / 2  for _ in range(2)] for _ in range(n_samples)])
    y = np.array([[t[0] + t[1]] for t in x])

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
    
    return x_train, x_test, y_train, y_test

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = generate_dataset(5000, 0.3)
    
    # print("x_test", x_test)
    # print('y_test', y_test)
    
    # Build Model: 2 -> 5 -> 1
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(5, input_dim=2, activation='sigmoid'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile Model
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.2)
    model.compile(optimizer=optimizer, loss="MSE")
    
    # Train Model
    model.fit(x_train, y_train, epochs=50)
    
    # Evaluate Model
    print('Evaluate model')
    model.evaluate(x_test, y_test)
    
    # Make predictions
    data = np.array([[0.3, 0.6], [0.2, 0.4]])
    predictions = model.predict(data)
    
    print("Some predictions")
    for d, p in zip(data, predictions):
        print(f"{d} : {p}")