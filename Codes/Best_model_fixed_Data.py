#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:28:09 2021

@author: berhe
"""

#Deep learning models
import warnings
warnings.filterwarnings('ignore')
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input,TimeDistributed,InputLayer,Concatenate
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras import Model

from keras.callbacks import ModelCheckpoint
#utility python files

import preprocessAudioFeat as paf
import MRS_evaluation as me
import myParam

#basic python libraries
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

from numpy.random import seed
seed(1)#Reproducability

#Load Data 

"""
Load already splited dataset with out context
"""
def data(feat):
    x_train,y_train,x_test,y_test=paf.loadTrainingTest(feat)#myParam.FEAT)
    
    if feat.lower()[0]=='v':
        x_train=paf.augment_reshapeVgg(x_train,138)
        x_test=paf.augment_reshapeVgg(x_test,138)
    else:
        x_train=paf.augment_reshape(x_train,2000)
        x_test=paf.augment_reshape(x_test,2000)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    print('Data Shapes',x_train.shape,x_test.shape)
    return  x_train,x_test,y_train,y_test

def data1(feat):
    x_train, x_test, y_train, y_test=paf.loadDataFeaures(feat)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    print('Data Shapes',x_train.shape,x_test.shape)
    return  x_train,x_test,y_train,y_test

def dataAug(feat):
    x_train,y_train,x_test,y_test=paf.loadTrainingTest(feat)#myParam.FEAT)
    
    if feat.lower()[0]=='v':
        x_train=paf.augment_reshapeVgg(x_train,138)
        x_test=paf.augment_reshapeVgg(x_test,138)
    else:
        x_train=paf.augment_reshape(x_train,2000)
        x_test=paf.augment_reshape(x_test,2000)
    
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    mfccA,melspectogramA,tempoA,spec_centroidsA,vggFeatA=paf.loadAugmentedData()
    aug_labels=np.full((len(mfccA)), 1)
    aug_labels=np.array(aug_labels)
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    x_test=np.array(x_test)
    print(x_train.shape,y_train.shape)
    mfcc_a=paf.augment_reshape(mfccA,myParam.DIM)
    mel_a=paf.augment_reshape(melspectogramA,myParam.DIM)
    vggFeat_a=paf.augment_reshapeVgg(vggFeatA,138)       
    if feat.lower()[0]=='v':
        conc_train=np.concatenate((x_train,vggFeat_a))
    elif feat=='mfcc':
        mfcc_a=np.array(mfcc_a)
        conc_train=np.concatenate((x_train,mfcc_a))
    else:
        mel_a=np.array(mel_a)
        conc_train=np.concatenate((x_train,mel_a))
    
    concat_label=np.concatenate((y_train,aug_labels))
    y_train=np.array(y_train)
    x_train, y_train = shuffle(conc_train, concat_label)
    print(x_train.shape,x_test.shape)
    return  x_train,x_test,y_train,y_test

def data_context(feat):
    data_training_context=np.load("/vol/work3/berhe/MRS_Detection/Data/"+feat+"_context_data_"+str(myParam.CONTEXT_SIZE)+".npy",allow_pickle=True)
    #mfcc_training_context=np.load("Data/mel_context_data_5.npy")
    #mfcc_training_context=np.load("Data/vggish_context_data_5.npy")
    dataset_Df=pd.read_csv("Scene_Dataset_Normalized.csv")
    sceneLabels=dataset_Df.MRS.tolist()
    labels=[i if i==0 else 1 for i in sceneLabels]
    x_train, x_test, y_train, y_test=paf.split_Data(data_training_context,labels)
    print("shape",x_train.shape,x_test.shape)
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    return x_train, x_test, y_train, y_test

def model_lstm_classic_best(x_train,x_test,y_train,y_test):
    input_shape = (x_train.shape[1], x_train.shape[2])
    drop_out=0.31#{{uniform(0.0, 0.5)}} Fixed with best
    activation_function='linear'#{{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers= 2#{{choice([1,2,3])}}
    batch=32#{{choice([32,64])}}
    optimizer_func='adam'#{{choice(['rmsprop', 'adam', 'sgd'])}}
    lstm_layers=2#{{choice([2,3,4])}}
    
    model = Sequential()

    model.add(LSTM(units=256, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(drop_out))
    
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
    if dense_layers==3:
        model.add(Dense(256, activation=activation_function))
        model.add(Dense(128, activation=activation_function))
    if dense_layers==2:
        model.add(Dense(128, activation=activation_function))   
    
    model.add(Dense(units=2, activation="softmax"))

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer="rmsprop")
    
    checkpointer = ModelCheckpoint(filepath='best_LSTM_clasical_model_weights'+myParam.FEAT+'.hdf5',verbose=2, save_best_only=True)
    model.fit(x_train, y_train, batch_size=batch, epochs=15, validation_data=(x_test, y_test), callbacks=[checkpointer],verbose=1)
    
    score = model.evaluate(x_test, y_test, verbose=0)
    pred_test=model.predict_classes(x_test).tolist()
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/optimized_Models_Result/best_LSTM_clasical_"+myParam.FEAT+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT)+' & '+str(score[1])+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2)))
        file.write('\n')
        file.write('\hline')

def context_td_lstm(x_train,x_test,y_train,y_test):
    print(x_train.shape,y_train.shape,x_test.shape)
    inputshape=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))
    tm_lstm=TimeDistributed(LSTM(256, return_sequences=True))(inputshape)
    tm_lstm=TimeDistributed(Dropout(0.31))(tm_lstm)
 
    tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(0.31))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm)
    
    dense_input=Flatten()(tm_lstm)
    
    dense_input=Dense(128, activation='linear')(dense_input)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=inputshape,outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    merged_model.fit(x=x_train,y=y_train, validation_data=(x_test, y_test),epochs=10,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate(x_test, y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    
    pred_test=merged_model.predict(x_test)
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    
    with open("Results/optimized_Models_Result/best_TD_LSTM_"+myParam.FEAT+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    
def multi_modal_lstm_TOP(x_train,x_test,y_train,y_test):
    print(x_train.shape,y_train.shape,x_test.shape)
    inputshape=Input(shape=(x_train.shape[1], x_train.shape[2]))
    
    tm_lstm=LSTM(256, return_sequences=True)(inputshape)
    tm_lstm=Dropout(0.31)(tm_lstm)
    tm_lstm=LSTM(128, return_sequences=False)(tm_lstm)
    dense_input=Flatten()(tm_lstm)
    
    return dense_input,inputshape
    
def model_lstm_classic_multimodal(listFeat):
    learnedEmbeding=[]
    inputsShapes=[]
    x_trainings=[]
    x_tests=[]
    y_trainings=[]
    y_tests=[]
    for feats in listFeat:
        print(feats)
        x_train,x_test,y_train,y_test=data1(feats)
        print(x_train.shape,y_train.shape,x_test.shape)
        x_trainings.append(x_train)
        x_tests.append(x_test)
        y_trainings.append(y_train)
        y_tests.append(y_test)
        dense_inputs, inputshape=multi_modal_lstm_TOP(x_train,x_test,y_train,y_test)
        learnedEmbeding.append(dense_inputs)
        inputsShapes.append(inputshape)
    
    merged=Concatenate(axis=1)(learnedEmbeding)
    dense_input=Dense(128, activation='linear')(merged)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=inputsShapes,outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    merged_model.fit(x=x_trainings,y=y_trainings[0], validation_data=(x_tests, y_tests[0]),epochs=10,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate(x_tests, y_tests[0], verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    
    pred_test=merged_model.predict(x_tests)
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_tests[0],pred_test)
    
    featFile="_".join(listFeat)
    with open("Results/optimized_Models_Result/best_LSTM_clasical_"+featFile+"_LaTex.txt",'a') as file:
        file.write(str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
        
def multimodal_td_lstm_TOP(x_train,x_test,y_train,y_test):
    print(x_train.shape,y_train.shape,x_test.shape)
    
    inputshape=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))
    tm_lstm=TimeDistributed(LSTM(256, return_sequences=True))(inputshape)
    tm_lstm=TimeDistributed(Dropout(0.31))(tm_lstm)
 
    tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(0.31))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm)
    
    dense_input=Flatten()(tm_lstm)
    
    return dense_input,inputshape

def multimodal_td_lstm_best(listFeat):
    learnedEmbeding=[]
    inputsShapes=[]
    x_trainings=[]
    x_tests=[]
    y_trainings=[]
    y_tests=[]
    for feats in listFeat:
        print(feats)
        myParam.FEAT=feats
        x_train,x_test,y_train,y_test=data_context(feats)
        x_trainings.append(x_train)
        x_tests.append(x_test)
        y_trainings.append(y_train)
        y_tests.append(y_test)
        dense_inputs, inputshape=multimodal_td_lstm_TOP(x_train,x_test,y_train,y_test)
        learnedEmbeding.append(dense_inputs)
        inputsShapes.append(inputshape)
        
    merged=Concatenate(axis=1)(learnedEmbeding)
    dense_input=Dense(128, activation='linear')(merged)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=inputsShapes,outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    merged_model.fit(x=x_trainings,y=y_trainings[0], validation_data=(x_tests, y_tests[0]),epochs=10,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate(x_tests, y_tests[0], verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    
    pred_test=merged_model.predict(x_tests)
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_tests[0],pred_test)
    
    featFile="_".join(listFeat)
    with open("Results/optimized_Models_Result/best_TD_LSTM_"+featFile+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')


if __name__ == '__main__': 
    features=['mel','vggish','mffc']
    textFeat=['trans','summary']
    musicFeat=['tempogram','Pitch_11_Freq_Intens']
    allFeatures=features+textFeat+musicFeat
   
    #model_lstm_classic_multimodal(['vggish','trans','tempo'])
    #model_lstm_classic_multimodal(['mel','summary','tempo'])
    #model_lstm_classic_multimodal(['vggish','trans','Pitch_11_All'])
    #model_lstm_classic_multimodal(['vggish','summary','Pitch_11_All'])
    #model_lstm_classic_multimodal(['mel','tempo'])
    #model_lstm_classic_multimodal(['mel','Pitch_11_All'])
    
    """
    for feat in textFeat:
        myParam.FEAT=feat
        x_train,x_test,y_train,y_test=data1(feat)#data1("trans")
        model_lstm_classic_best(x_train,x_test,y_train,y_test)

    for feat in textFeat:
        myParam.FEAT=feat
        for c in [1,3,5,7]:
            myParam.CONTEXT_SIZE=c
            x_train,x_test,y_train,y_test=data_context()
            model_lstm_td_best(x_train, x_test, y_train, y_test)
    """
    for feat in textFeat:
        myParam.FEAT=feat
        for c in [1,3,5,7]:
            myParam.CONTEXT_SIZE=c
            x_train,y_train,x_test,y_test=data_context(feat)
            context_td_lstm(x_train,y_train,x_test,y_test)
    """
    for feat in features:
        for 
    myParam.FEAT=feat
    for c in [1,3,5,7]:
        myParam.CONTEXT_SIZE=c
        x_train,x_test,y_train,y_test=data_context()
        model_lstm_td_best(x_train, x_test, y_train, y_test)
    model_lstm_classic_multimodal(['mel','trans'])
    
    for c in [1,3,5,7]:
        myParam.CONTEXT_SIZE=c
        multimodal_td_lstm_best(['vggish','trans'])
        #multimodal_td_lstm_best(['mel','trans','Pitch_11_Freq_Intens'])
        #multimodal_td_lstm_best(['mel','summary','Pitch_11_Freq_Intens'])
        #multimodal_td_lstm_best(['mel','trans','tempogram'])
        #multimodal_td_lstm_best(['mel','summary','tempogram'])
    """
    #myParam.CONTEXT_SIZE=7
    #multimodal_td_lstm_best(['mel','summary','tempogram'])