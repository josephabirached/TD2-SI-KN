# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 07:51:07 2020

@author: Charbel
"""

from perceptron import Perceptron
import csv

p = Perceptron(2,600,1e-3)
x_train = [[0,0],[0,1],[1,0],[1,1]]
Y_train = [0,0,0,1]
print(p.train(x_train,Y_train))

print(p.predict([0,0]))
print(p.predict([0,1]))
print(p.predict([1,0]))
print(p.predict([1,1]))

fields = ['w0', 'w1', 'w2']
weights = p.getWeights()

with open("data.csv" , 'w') as csvfile:
    csvwriter = csv.writer(csvfile)    
    csvwriter.writerows([fields,weights])  
    #csvwriter.writerow(weights) 

