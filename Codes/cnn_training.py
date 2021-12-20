# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 09:36:02 2020

@author: berhe
"""
import warnings
warnings.filterwarnings('ignore')
from keras import models
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D,Conv1D,MaxPooling1D,GlobalAveragePooling1D,LSTM
from keras.optimizers import Adam,SGD,RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials
from attention import AttentionWithContext,Attention,augmented_conv2d

import preprocessAudioFeat as paf
from datetime import datetime
from sklearn.utils import shuffle
import numpy as np
import json

from sklearn.metrics import recall_score,precision_score,f1_score
import myParam
import argparse



def rec_pre_f1_MRS(y_test,pred_test):
    rec=recall_score(y_test,pred_test,average=None)
    pre=precision_score(y_test,pred_test,average=None)
    f1=f1_score(y_test,pred_test,average=None)
    return rec,pre,f1


results=[]
params={
        'units1': hp.choice('units1', [64,128,256,512,1024]),

        'dropout1': hp.uniform('dropout1', .25,.75),
        'dropout2': hp.uniform('dropout2',  .25,.75),

        'layers': hp.choice('layers',[1,2,3,4,5]),
        
        'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
        'activation': 'relu',
        'batch_size' : hp.choice('batch_size', [32,64,128,256]),
        'nb_epochs' :  10
        }
            
def data_aa():
    
    x_train,y_train,x_test,y_test=paf.loadTrainingTest(myParam.FEAT)
    feat=myParam.FEAT
    
    if feat.lower()[0]=='v':
        x_train=paf.augment_reshapeVgg(x_train,myParam.DIM)
        x_test=paf.augment_reshapeVgg(x_test,myParam.DIM)
    else:
        x_train=paf.augment_reshape(x_train,myParam.DIM)
        x_test=paf.augment_reshape(x_test,myParam.DIM)
        
    
    print("data loaded")
    print(x_test.shape,x_train.shape)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    return  x_train,y_train,x_test,y_test

def data():
    x_train,y_train,x_test,y_test=paf.loadDataFeaures(myParam.FEAT)
    #feat=myParam.FEAT
    """
    if feat.lower()[0]=='v':
        x_train=paf.augment_reshapeVgg(x_train,myParam.DIM)
        x_test=paf.augment_reshapeVgg(x_test,myParam.DIM)
    else:
        x_train=paf.augment_reshape(x_train,myParam.DIM)
        x_test=paf.augment_reshape(x_test,myParam.DIM)
    
    if myParam.AUGMENT==0:
        y_train=np.array(y_train)
        y_test=np.array(y_test)
        print(x_train.shape,x_test.shape)
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
    """
    return  x_train,x_test,y_train,y_test

def dataD():
    x_train, x_test, y_train, y_test=paf.loadDataFeaures(myParam.FEAT)
    print(x_train.shape,x_test.shape,y_test.shape,y_train.shape)
    return  x_train, x_test, y_train, y_test
def cnnModel(x_train,y_train,x_test,y_test):
    num_rows = x_train.shape[1]
    num_columns = x_train.shape[2]
    num_channels = 1

    x_train = x_train.reshape(x_train.shape[0], num_rows,num_columns, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_rows,num_columns, num_channels)

    num_labels = 2
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns,num_channels), activation=params['layers']))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(params['dropout1']))

    model.add(Conv2D(filters=32, kernel_size=2, activation=params['optimizer']))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(params['dropout1']))

    model.add(Conv2D(filters=32, kernel_size=2, activation=params['optimizer']))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(params['dropout1']))
    
    if params['layers']== 4:
        model.add(Conv2D(filters=16, kernel_size=1, activation=params['optimizer']))
        model.add(MaxPooling2D(pool_size=1))
        model.add(Dropout(params['dropout1']))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(num_labels, activation='softmax'))
    
    
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=params['optimizer'])
    num_epochs = 2
    num_batch_size = 32

    checkpointer = ModelCheckpoint(filepath='weights.best_CNN_Optimization.hdf5',
                                   verbose=1, save_best_only=True)
    start = datetime.now()
    model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=2)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    
    score = model.evaluate(x_test, y_test, verbose=0)

    pred_test=model.predict_classes(x_test).tolist()
    rec,prec,f1=rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/Results_CNN.text",'w') as file:
        file.write(params['dropout1']+'\t'+params['optimizer']+'\t'+params['layer']+'\t'+params['nb_epochs']+'\t'+str(score[1])+'\t'+str(rec)+'\t'+str(prec)+'\t'+str(f1))
        file.write('\n')
    
    return {'loss': -score[0], 'status': STATUS_OK,'model':model}

def cnn_with_Attention(x_train,x_test,y_train,y_test):
    num_labels = 2
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    cnn_layer={{choice([3,4,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=2, input_shape=(x_train.shape[1], x_train.shape[2]), activation=activation_function))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(drop_out))
    model.add(Attention())
    
    if dense_layers == 2:
        model.add(Dense(128, activation=activation_function))
    if dense_layers == 3:
        model.add(Dense(256, activation=activation_function))
        model.add(Dense(128, activation=activation_function))
    model.add(Dense(units=2, activation="softmax"))
    
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    model.add(Dense(num_labels, activation=activation_function))
    #num_epochs = 2
    #num_batch_size = 32

    checkpointer = ModelCheckpoint(filepath='weights.best_CNN_Optimization.hdf5',
                                   verbose=1, save_best_only=True)
    start = datetime.now()
    model.fit(x_train, y_train, batch_size=batch, epochs=10, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=2)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    
    score = model.evaluate(x_test, y_test, verbose=0)

    pred_test=model.predict_classes(x_test).tolist()
    rec=recall_score(y_test,pred_test,average=None)
    pre=precision_score(y_test,pred_test,average=None)
    f1=f1_score(y_test,pred_test,average=None)
    #rec,prec,f1=rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/CNN_Attention_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed.txt",'a') as file:
       file.write(str(myParam.DIM)+'\t'+activation_function+'\t'+optimizer_func+'\t'+str(round(drop_out,3))+'\t'+str(dense_layers)+'\t'+str(cnn_layer)+'\t'+str(score[1])+'\t'+str(rec)+'\t'+str(pre)+'\t'+str(f1))
       file.write('\n')
    with open("Results/CNN_Attention_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed_LaTex.txt",'a') as file:
        file.write(str(myParam.DIM)+' & '+activation_function+' & '+optimizer_func+' & '+str(round(drop_out,3))+' & '+str(dense_layers)+' & '+str(cnn_layer)+' & '+str(score[1])+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2)))
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':model}
    
def cnnModel2(x_train, x_test, y_train, y_test):
    num_rows = x_train.shape[1]
    num_columns = x_train.shape[2]
    num_channels = 1

    X_train = x_train.reshape(x_train.shape[0], num_rows,num_columns, num_channels)
    X_test = x_test.reshape(x_test.shape[0], num_rows,num_columns, num_channels)
    
    #num_labels = 2
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    cnn_layer={{choice([3,4,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns,num_channels), activation=activation_function))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(drop_out))

    model.add(Conv2D(filters=32, kernel_size=2, activation=activation_function))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(drop_out))

    model.add(Conv2D(filters=32, kernel_size=2, activation=activation_function))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(drop_out))
    
    if cnn_layer== 4:
        model.add(Conv2D(filters=16, kernel_size=1, activation=activation_function))
        model.add(MaxPooling2D(pool_size=1))
        model.add(Dropout(drop_out))
    if cnn_layer== 5:
        model.add(Conv2D(filters=32, kernel_size=1, activation=activation_function))
        model.add(MaxPooling2D(pool_size=1))
        model.add(Dropout(drop_out))
        model.add(Conv2D(filters=16, kernel_size=1, activation=activation_function))
        model.add(MaxPooling2D(pool_size=1))
        model.add(Dropout(drop_out))
    model.add(GlobalAveragePooling2D())
    
    model.add(Flatten())
    if dense_layers == 2:
        model.add(Dense(128, activation=activation_function))
    if dense_layers == 3:
        model.add(Dense(256, activation=activation_function))
        model.add(Dense(128, activation=activation_function))
    
    model.add(Dense(units=2, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    checkpointer = ModelCheckpoint(filepath='weights.best_CNN_Optimization.hdf5',
                                   verbose=1, save_best_only=True)
    start = datetime.now()
    print("Shape before training {} {} {}".format(x_train.shape, y_train.shape, y_test.shape, x_test.shape))
    model.fit(X_train, y_train, batch_size=batch, epochs=5, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=2)

    duration = datetime.now() - start
    print("Training completed in time: ", duration)
    
    score = model.evaluate(X_test, y_test, verbose=0)

    pred_test=model.predict_classes(X_test).tolist()
    rec=recall_score(y_test,pred_test,average=None)
    pre=precision_score(y_test,pred_test,average=None)
    f1=f1_score(y_test,pred_test,average=None)
    #rec,prec,f1=rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/CNN_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed.txt",'a') as file:
       file.write(str(myParam.DIM)+'\t'+activation_function+'\t'+optimizer_func+'\t'+str(round(drop_out,3))+'\t'+str(dense_layers)+'\t'+str(cnn_layer)+'\t'+str(score[1])+'\t'+str(rec)+'\t'+str(pre)+'\t'+str(f1))
       file.write('\n')
    with open("Results/CNN_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed_LaTex.txt",'a') as file:
        file.write(str(myParam.DIM)+' & '+activation_function+' & '+optimizer_func+' & '+str(round(drop_out,3))+' & '+str(dense_layers)+' & '+str(cnn_layer)+' & '+str(score[1])+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2)))
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':model}
 
if __name__ == '__main__': 
    #dim=int(input('enter dim '))
    #feat=input('enter feature ')
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
    music_feat=['tempo','Pitch_11_Freq_Intens']#'chromo']
    text_feat=['summary','trans']
    for feat in text_feat:
        #for t in [0,1]:
        if feat[0]=='v':
            myParam.DIM=200
        else:
            myParam.DIM=2000
        myParam.FEAT=feat
        myParam.AUGMENT=0
        myParam.OUTPU_FILE="dim_2000_"+str(0)
        print(type(myParam.DIM),type(myParam.FEAT),myParam.DIM,myParam.FEAT)
        best_parameters,best_model=optim.minimize(model=cnnModel2,
                                                   data=dataD,
                                                   algo=tpe.suggest,
                                                   max_evals=25,
                                                   trials=Trials())
        print(best_parameters)
        with open("Results/Best_Parameters_CNN_"+myParam.FEAT+myParam.OUTPU_FILE+".txt","a") as f:
            for att in best_parameters.keys():
                f.write(att+"\t"+str(best_parameters[att])+"\n")
        
        best_model.save("Results/Best_Parameters_CNN_"+myParam.FEAT+"_"+myParam.OUTPU_FILE+".hdf5")
