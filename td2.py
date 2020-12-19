# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 11:42:45 2020

@author: Charbel
"""
import numpy as np

possible_entries = [[0, 0], [0, 1], [1, 0], [1, 1]]
possible_exits = [0, 0, 0, 1]
error_list = np.zeros(11*11).T.reshape(11,11)
for w1 in range(-5,6):
    for w2 in range(-5,6):
        w = np.array([w1,w2])
        pred = np.array([1 if np.dot(w, entry)>0.5 else 0 for entry in possible_entries])
        error_list[w1+5][w2+5] = np.sum(np.square(np.subtract(pred, possible_exits)))/2
        
        
  

import matplotlib.pyplot as plt

plt.imshow(error_list)
plt.show()     

