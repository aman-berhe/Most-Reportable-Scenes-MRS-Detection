# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:38:10 2020

@author: berhe
"""
from attention import AttentionWithContext,Attention,augmented_conv2d
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import LSTM
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials

import preprocessAudioFeat as paf
from sklearn.utils import shuffle
import numpy as np
import pickle
import json
import myParam# for setting dimentions and choosen features to use
import MRS_evaluation as me

import argparse


def data():
    x_train,y_train,x_test,y_test=paf.loadTrainingTest(myParam.FEAT)
    #print('Data Shapes',x_train.shape,x_test.shape)
    feat=myParam.FEAT
    if feat.lower()[0]=='v':
        x_train=paf.augment_reshapeVgg(x_train,myParam.DIM)
        x_test=paf.augment_reshapeVgg(x_test,myParam.DIM)
    else:
        x_train=paf.augment_reshape(x_train,myParam.DIM)
        x_test=paf.augment_reshape(x_test,myParam.DIM)
    
    if myParam.AUGMENT==0:
        y_train=np.array(y_train)
        y_test=np.array(y_test)
        print('Here: with out augmentation',x_train.shape,x_test.shape)
        #return  x_train,y_train,x_test,y_test
    else:
        mfccA,melspectogramA,tempoA,spec_centroidsA,vggFeatA=paf.loadAugmentedData()
        aug_labels=np.full((len(mfccA)), 1)
        
        mfcc_a=paf.augment_reshape(mfccA,myParam.DIM)
        mel_a=paf.augment_reshape(melspectogramA,myParam.DIM)
        vggFeat_a=paf.augment_reshapeVgg(vggFeatA,myParam.DIM)       
        if feat.lower()[0]=='v':
            conc_train=np.concatenate((x_train,vggFeat_a))
        elif feat=='mfcc':
            conc_train=np.concatenate((x_train,mfcc_a))
        else:
            conc_train=np.concatenate((x_train,mel_a))
        
        concat_label=np.concatenate((y_train,aug_labels))
        x_train, y_train = shuffle(conc_train, concat_label)
        print(x_train.shape,x_test.shape)
    return  x_train,x_test,y_train,y_test

def lstm_Model(x_train, x_test, y_train, y_test):

    input_shape = (x_train.shape[1], x_train.shape[2])
    print("Build LSTM RNN model ...")
    print(x_train.shape, x_test.shape)
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
    input_shape = (x_train.shape[1], x_train.shape[2])
    model = Sequential()

    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(drop_out))
    lstm_layers={{choice([2,3,4])}}
    if lstm_layers==2:
        model.add(LSTM(units=128, return_sequences=False))
        model.add(Dropout(drop_out))
    if lstm_layers==3:
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(drop_out))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dropout(drop_out))
    if lstm_layers==4:
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(drop_out))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(drop_out))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dropout(drop_out))
    
    
    if dense_layers==2:
        model.add(Dense(128, activation=activation_function))   
    if dense_layers==3:
        model.add(Dense(256, activation=activation_function))
        model.add(Dense(128, activation=activation_function))
    
    model.add(Dense(units=2, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    
    checkpointer = ModelCheckpoint(filepath='weights.best_LSTM_Optimization.hdf5',verbose=1, save_best_only=True)
    model.fit(x_train, y_train, batch_size=batch, epochs=10, validation_data=(x_test, y_test), callbacks=[checkpointer],verbose=2)
    
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    pred_test=model.predict_classes(x_test).tolist()
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/Results_LSTM_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed.txt",'a') as file:
        file.write(str(myParam.DIM)+'\t'+activation_function+'\t'+optimizer_func+'\t'+str(round(drop_out,3))+'\t'+str(dense_layers)+'\t'+str(score[1])+'\t'+str(rec)+'\t'+str(pre)+'\t'+str(f1))
        file.write('\n')
    with open("Results/Results_LSTM_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed_LaTex.txt",'a') as file:
        file.write(str(myParam.DIM)+' & '+activation_function+' & '+optimizer_func+' & '+str(round(drop_out,3))+' & '+str(dense_layers)+' & '+str(score[1])+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2)))
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':model}

def lstm_with_Attention(x_train, x_test, y_train, y_test):

    input_shape = (x_train.shape[1], x_train.shape[2])
    print("Build LSTM RNN model ...")
    print(x_train.shape, x_test.shape)
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    lstm_layers={{choice([2,3,4])}}
    
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(drop_out))
    """
    if lstm_layers==2:
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(drop_out))
    if lstm_layers==3:
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(drop_out))
        model.add(LSTM(units=32, return_sequences=False))
        model.add(Dropout(drop_out))
    if lstm_layers==4:
        model.add(LSTM(units=128, return_sequences=True))
        model.add(Dropout(drop_out))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dropout(drop_out))
        model.add(LSTM(units=32, return_sequences=True))
        model.add(Dropout(drop_out))
   """
    model.add(Attention())
    
    if dense_layers==2:
        model.add(Dense(128, activation=activation_function))   
    if dense_layers==3:
        model.add(Dense(256, activation=activation_function))
        model.add(Dense(128, activation=activation_function))
    
    model.add(Dense(units=2, activation="softmax"))
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    
    checkpointer = ModelCheckpoint(filepath='weights.best_LSTM_Optimization.hdf5',verbose=1, save_best_only=True)
    model.fit(x_train, y_train, batch_size=batch, epochs=10, validation_data=(x_test, y_test), callbacks=[checkpointer],verbose=2)
    
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    pred_test=model.predict_classes(x_test).tolist()
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/LSTM_Attention_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed.txt",'a') as file:
        file.write(str(myParam.DIM)+'\t'+activation_function+'\t'+optimizer_func+'\t'+str(round(drop_out,3))+'\t'+str(dense_layers)+'\t'+str(score[1])+'\t'+str(rec)+'\t'+str(pre)+'\t'+str(f1))
        file.write('\n')
    with open("Results/LSTM_Attention_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed_LaTex.txt",'a') as file:
        file.write(str(myParam.DIM)+' & '+activation_function+' & '+optimizer_func+' & '+str(round(drop_out,3))+' & '+str(dense_layers)+' & '+str(score[1])+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2)))
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':model}

if __name__=='__main__':
    """
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dimention',type=int,help='an integer for the same lenght dimentions')
    parser.add_argument('--feat',type=str,help='string, mfcc,mel,vggish are the values')
    parser.add_argument('--augment', type=int,help='True for augmenting MRS False other wise')
    parser.add_argument('--output', type=str,help='True for augmenting MRS False other wise')
    myParam.DIM=args.dimention
    myParam.FEAT=args.feat
    myParam.AUGMENT=args.augment
    myParam.OUTPU_FILE=args.output
    args = parser.parse_args()
    """
    features=['vggish','mel','mfcc']
    for feat in features:
        for t in [0]:
            if feat[0]=='v':
                myParam.DIM=200
            else:
                myParam.DIM=2000
            myParam.FEAT=feat
            myParam.AUGMENT=t
            myParam.OUTPU_FILE="dim_2000_"+str(t)
        print(type(myParam.DIM),type(myParam.FEAT),myParam.DIM,myParam.FEAT,myParam.AUGMENT)
        best_parameters,best_model=optim.minimize(model=lstm_with_Attention,
                                                   data=data,
                                                   algo=tpe.suggest,
                                                   max_evals=25,
                                                   trials=Trials())
        print(best_parameters)
        print(type(myParam.DIM),type(myParam.FEAT),myParam.DIM,myParam.FEAT,myParam.AUGMENT)
        with open("Results/Best_Parameters_LSTM_Attention_"+myParam.FEAT+"_"+myParam.OUTPU_FILE+"_Fixed.txt","a") as f:
            for att in best_parameters.keys():
                f.write(att+"\t"+str(best_parameters[att])+"\n")

    #with open("Best Parameters_LSTM_Pickle_"+myParam.FEAT,"wb") as fpickle:
     #   pickle.dump(fpickle,best_parameters)
        best_model.save("Results/Best_Parameters_LSTM_Attention_"+myParam.FEAT+"_"+myParam.OUTPU_FILE+"_Fixed.hdf5")