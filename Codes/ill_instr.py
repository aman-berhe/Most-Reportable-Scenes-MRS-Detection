# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:48:49 2020

@author: berhe
"""

import numpy as np
import pandas as pd
import pickle

def generateArr(arrSize):
    arr=np.arange(arrSize)
    arr=arr*arr
    
    return arr
def generaterand(arrSize):
    return np.random.random_sample(arrSize)

if __name__=='__main__':
    arrSize=10
    rangArr=generateArr(arrSize)
    randSamp=generaterand(arrSize)
    count=0
    for i in range(rangArr.shape[0]):
        if randSamp[i]<=rangArr[i]:
            count=count+1
    print("Done! {} counts found ".format(count))