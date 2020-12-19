# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 07:51:19 2020

@author: Charbel
"""
import numpy as np

class Perceptron:
    
    def __init__(self, nb_input, nb_epoch, learning_rate):
        self.nb_input = nb_input
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.weights = [0,0,0]
        
    def predict(self, input_vals):
        if len(input_vals) == 3:
            return 1 if np.dot(self.weights, input_vals)>0.5 else 0
        else:
            mod_input = [1] + input_vals
            return 1 if np.dot(self.weights, mod_input)>0.5 else 0
    
    def train(self, x_train, Y_train):
        pred = [self.predict(x) for x in x_train]
        prev_error = np.sum(np.square(np.subtract(pred, Y_train)))/2
        conv_counter = 0
        for epoch in range(self.nb_epoch):
            for i in range(len(x_train)):
                x_train_local = [1] + x_train[i]
                t = self.predict(x_train_local)
                for j in range(len(x_train_local)):
                    self.weights[j] += (self.learning_rate * (Y_train[i] - t) * x_train_local[j])
            pred = [self.predict(x) for x in x_train]
            error = np.sum(np.square(np.subtract(pred, Y_train)))/2
            print(error)
            if(abs(prev_error-error)<1e-4):
                conv_counter += 1
            else:
                conv_counter = 0
                
            if(conv_counter >= 200):
                return epoch
            prev_error = error
        return epoch
               
    def getWeights(self):
        return self.weights
    