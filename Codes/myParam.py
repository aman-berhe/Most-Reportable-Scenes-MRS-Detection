# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 17:32:46 2020

@author: berhe
"""

# Architectural constants.
FEAT = 'mfcc'  # Frames in input mel-spectrogram patch.
DIM = 2000  # Frequency bands in input mel-spectrogram patch.
EMBEDDING_SIZE = 128  # Size of embedding layer.
AUGMENT=0#0 is don't use augmented data a,d 1 is use augmented data
OUTPU_FILE=''
EPOCH=10 #number of epoches
MODEL=''# Choose a model LSTM or CNN
FEAT1="vggish"
FEAT2="trans"
FEAT3="tempo"
CONTEXT_SIZE=0