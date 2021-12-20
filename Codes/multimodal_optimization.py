# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 09:20:14 2020

@author: berhe
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras import Model

from keras.layers import LSTM,Concatenate,Input,TimeDistributed,InputLayer
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.optimizers import Adam,SGD,RMSprop
from keras.callbacks import ModelCheckpoint
from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials


import preprocessAudioFeat as paf
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import MRS_evaluation as me

import myParam

from numpy.random import seed
seed(42)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

dir_files='/vol/work3/berhe/MRS_Detection/'

def data():
    x_train, x_test, y_train, y_test=paf.loadDataFeaures(myParam.FEAT1)
    x_train1, x_test1, y_train1, y_test1=paf.loadDataFeaures(myParam.FEAT2)
    return x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1

def data1():
    x_train, x_test, y_train, y_test=paf.loadDataFeaures(myParam.FEAT1)
    x_train1, x_test1, y_train1, y_test1=paf.loadDataFeaures(myParam.FEAT2)
    x_train2, x_test2, y_train2, y_test2=paf.loadDataFeaures(myParam.FEAT3)
    return x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1,x_train2, x_test2, y_train2, y_test2
    
def data_context():
    data_training_context=np.load("/vol/work3/berhe/MRS_Detection/Data/"+myParam.FEAT1+"_context_data_"+str(myParam.AUGMENT)+".npy",allow_pickle=True)
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
    

def multimodal_net(x_train, x_test, y_train, y_test):
    
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    lstm_layers={{choice([2,3,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
    
    model = Sequential()
    model.add(InputLayer(input_shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3])))
    #model.add(Reshape(target_shape=(2, 40, 1)))
    model.add(TimeDistributed(LSTM(256, return_sequences=True)))
    if lstm_layers==2:
        model.add(TimeDistributed(LSTM(128, return_sequences=True)))
        model.add(TimeDistributed(Dropout(drop_out)))
        model.add(TimeDistributed(LSTM(64, return_sequences=False)))
    if lstm_layers==3:
        model.add(TimeDistributed(LSTM(128, return_sequences=True)))
        model.add(TimeDistributed(Dropout(drop_out)))
        model.add(TimeDistributed(LSTM(64, return_sequences=True)))
        model.add(TimeDistributed(LSTM(64, return_sequences=False)))
    if lstm_layers==4:
        model.add(TimeDistributed(LSTM(128, return_sequences=True)))
        model.add(TimeDistributed(Dropout(drop_out)))
        model.add(TimeDistributed(LSTM(64, return_sequences=True)))
        model.add(TimeDistributed(Dropout(drop_out)))
        model.add(TimeDistributed(LSTM(32, return_sequences=True)))
        model.add(TimeDistributed(LSTM(32, return_sequences=False)))
    
    
    model.add(Flatten())
    if dense_layers==1:
        model.add(Dense(128, activation=activation_function))   
    if dense_layers==2:
        model.add(Dense(256, activation=activation_function))
        model.add(Dropout(drop_out))
        model.add(Dense(128, activation=activation_function))
    if dense_layers==3:
        model.add(Dense(256, activation=activation_function))
        model.add(Dropout(drop_out))
        model.add(Dense(128, activation=activation_function))
        model.add(Dropout(drop_out))
        model.add(Dense(64, activation=activation_function))
        
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    model.fit(x=x_train,y=y_train, validation_data=(x_test, y_test),epochs=10,batch_size=batch,callbacks=[es])
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
  
    pred_test=model.predict(x_test)
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/Context_Based_Optimization/Context_"+str(myParam.AUGMENT)+"_Optimization_"+myParam.FEAT1+'_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(lstm_layers)+' & '+str(dense_layers)+' & '+str(round(drop_out,2))+' & '+str(activation_function)+' & '+str(optimizer_func)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':model}
    
    
    
def multi_modal_lstm(x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1):
    #x_train, x_test, y_train, y_test=paf.loadDataFeaures(feat1)
    #x_train1, x_test1, y_train1, y_test1=paf.loadDataFeaures(feat2)
    
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    lstm_layers={{choice([2,3,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],1))
    tm_lstm=LSTM(256, return_sequences=True)(input1)
    tm_lstm=Dropout(drop_out)(tm_lstm)
    
    if lstm_layers==2:
        tm_lstm=LSTM(64, return_sequences=False)(tm_lstm)
        dense_input1=Flatten()(tm_lstm)
    if lstm_layers==3:
        tm_lstm=LSTM(128, return_sequences=True)(tm_lstm)
        tm_lstm=Dropout(drop_out)(tm_lstm)
        tm_lstm=LSTM(64, return_sequences=False)(tm_lstm)
        dense_input1=Flatten()(tm_lstm)
    if lstm_layers==4:
        tm_lstm=LSTM(128, return_sequences=True)(tm_lstm)
        tm_lstm=Dropout(drop_out)(tm_lstm)
        tm_lstm=LSTM(64, return_sequences=True)(tm_lstm)
        tm_lstm=Dropout(drop_out)(tm_lstm)
        tm_lstm=LSTM(32, return_sequences=False)(tm_lstm)
        dense_input1=Flatten()(tm_lstm)

    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],1))
    tm_lstm2=LSTM(256, return_sequences=True)(input2)
    tm_lstm2=Dropout(drop_out)(tm_lstm2)
    if lstm_layers==2:
        tm_lstm2=LSTM(64, return_sequences=False)(tm_lstm2)
        dense_input2=Flatten()(tm_lstm)
    if lstm_layers==3:
        tm_lstm2=LSTM(128, return_sequences=True)(tm_lstm2)
        tm_lstm2=Dropout(drop_out)(tm_lstm2)
        tm_lstm2=LSTM(64, return_sequences=False)(tm_lstm2)
        dense_input2=Flatten()(tm_lstm2)
    if lstm_layers==4:
        tm_lstm2=LSTM(128, return_sequences=True)(tm_lstm2)
        tm_lstm2=Dropout(drop_out)(tm_lstm2)
        tm_lstm2=LSTM(64, return_sequences=True)(tm_lstm2)
        tm_lstm2=Dropout(drop_out)(tm_lstm2)
        tm_lstm2=LSTM(32, return_sequences=False)(tm_lstm2)
        dense_input2=Flatten()(tm_lstm2)    

    
    merged=Concatenate(axis=1)([dense_input1,dense_input2])
    if dense_layers==1:
        dense_input=Dense(128, activation=activation_function)(merged)
    if dense_layers==2:
        dense_input=Dense(256, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(128, activation=activation_function)(merged)
        #dense_input=Dropout(drop_out)(dense_input)
    if dense_layers==3:
        dense_input=Dense(256, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(128, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(64, activation=activation_function)(merged)
    
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=[input1,input2],outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    merged_model.fit(x=[x_train,x_train1],y=y_train1, validation_data=([x_test,x_test1], y_test),epochs=10,batch_size=batch,callbacks=[es])
    
    score = merged_model.evaluate([x_test,x_test1], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    
    pred_test=merged_model.predict([x_test,x_test1])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/Multimodal_Optimization_Results/MultiModal_2_Optimization_"+myParam.FEAT1+"_"+myParam.FEAT2+'_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(lstm_layers)+' & '+str(dense_layers)+' & '+str(round(drop_out,2))+' & '+str(activation_function)+' & '+str(optimizer_func)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':merged_model}

def multimodal_td_cnn2(x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1):

    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    cnn_layer={{choice([2,3,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], 1)
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)
    
    X_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1],x_train1.shape[2], 1)
    X_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1],x_test1.shape[2], 1)
    
    print(x_train.shape,x_train1.shape,X_train.shape,X_train1.shape)
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],1))
    td_cnn=Conv2D(filters=16, kernel_size=2, activation=activation_function)(input1)
    td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
    td_cnn=Dropout(drop_out)(td_cnn)
    
    td_cnn=Conv2D(filters=128, kernel_size=2, activation=activation_function)(td_cnn)
    td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
    td_cnn=Dropout(drop_out)(td_cnn)
    
    if cnn_layer==3:
        td_cnn=Conv2D(filters=64, kernel_size=2, activation=activation_function)(td_cnn)
        td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
        td_cnn=Dropout(drop_out)(td_cnn)
    if cnn_layer==4:
        td_cnn=Conv2D(filters=64, kernel_size=2, activation=activation_function)(td_cnn)
        td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
        td_cnn=Dropout(drop_out)(td_cnn)
        
        td_cnn=Conv2D(filters=32, kernel_size=2, activation=activation_function)(td_cnn)
        td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
        td_cnn=Dropout(drop_out)(td_cnn)
        
    dense_input1=Flatten()(td_cnn)
    
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],1))
    td_cnn2=Conv2D(filters=16, kernel_size=2, activation=activation_function)(input2)
    td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
    td_cnn2=Dropout(drop_out)(td_cnn2)
    
    td_cnn2=Conv2D(filters=32, kernel_size=2, activation=activation_function)(td_cnn2)
    td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
    td_cnn2=Dropout(drop_out)(td_cnn2)
    
    if cnn_layer==3:
        td_cnn2=Conv2D(filters=16, kernel_size=2, activation=activation_function)(td_cnn2)
        td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
        td_cnn2=Dropout(drop_out)(td_cnn2)
    if cnn_layer==4:
        td_cnn2=Conv2D(filters=32, kernel_size=2, activation=activation_function)(td_cnn2)
        td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
        td_cnn2=Dropout(drop_out)(td_cnn2)
        
        td_cnn2=Conv2D(filters=16, kernel_size=2, activation=activation_function)(td_cnn2)
        td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
        td_cnn2=Dropout(drop_out)(td_cnn2)
    
        
    dense_input2=Flatten()(td_cnn2)
    
    merged=Concatenate(axis=1)([dense_input1,dense_input2])
    if dense_layers==1:
        dense_input=Dense(128, activation=activation_function)(merged)
    if dense_layers==2:
        dense_input=Dense(256, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(128, activation=activation_function)(merged)
    if dense_layers==3:
        dense_input=Dense(256, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(128, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(64, activation=activation_function)(merged)
    
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=[input1,input2],outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    merged_model.fit(x=[X_train,X_train1],y=y_train1, validation_data=( [X_test,X_test1], y_test), epochs=5, batch_size=batch,callbacks=[es])
    score = merged_model.evaluate([X_test,X_test1], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([X_test,X_test1])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    with open("Results/Multimodal_Optimization_Results/MultiModal_2_Optimization_No_Context_" +myParam.FEAT1 +"_"+myParam.FEAT2+'_CNN'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(cnn_layer)+' & '+str(batch)+' & '+str(dense_layers)+' & '+str(round(drop_out,2))+' & '+str(activation_function)+' & '+str(optimizer_func)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':merged_model}

def multimodal_tm_cnn3(x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1,x_train2, x_test2, y_train2, y_test2):
    print("Fetaures shape; ",x_train.shape,x_train2.shape)
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    cnn_layer={{choice([2,3,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
    x_train22=x_train2.reshape(x_train2.shape[0], x_train2.shape[2],x_train2.shape[1])
    x_test22=x_test2.reshape(x_test2.shape[0], x_test2.shape[2],x_test2.shape[1])
    
    X_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], 1)
    X_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)
    
    X_train1 = x_train1.reshape(x_train1.shape[0], x_train1.shape[1],x_train1.shape[2], 1)
    X_test1 = x_test1.reshape(x_test1.shape[0], x_test1.shape[1],x_test1.shape[2], 1)
    
    X_train2 = x_train22.reshape(x_train22.shape[0], x_train22.shape[1],x_train22.shape[2], 1)
    X_test2 = x_test22.reshape(x_test22.shape[0], x_test22.shape[1],x_test22.shape[2], 1)
    
    print("ALL SHAPES; ",x_train.shape,x_train1.shape,X_train.shape,X_train1.shape,X_train2.shape)
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],1))
    td_cnn=Conv2D(filters=16, kernel_size=2, activation=activation_function)(input1)
    td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
    td_cnn=Dropout(drop_out)(td_cnn)
    
    td_cnn=Conv2D(filters=128, kernel_size=2, activation=activation_function)(td_cnn)
    td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
    td_cnn=Dropout(drop_out)(td_cnn)
    
    if cnn_layer==3:
        td_cnn=Conv2D(filters=64, kernel_size=2, activation=activation_function)(td_cnn)
        td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
        td_cnn=Dropout(drop_out)(td_cnn)
    if cnn_layer==4:
        td_cnn=Conv2D(filters=64, kernel_size=2, activation=activation_function)(td_cnn)
        td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
        td_cnn=Dropout(drop_out)(td_cnn)
        
        td_cnn=Conv2D(filters=32, kernel_size=2, activation=activation_function)(td_cnn)
        td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
        td_cnn=Dropout(drop_out)(td_cnn)
        
    dense_input1=Flatten()(td_cnn)
    
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],1))
    td_cnn2=Conv2D(filters=16, kernel_size=2, activation=activation_function)(input2)
    td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
    td_cnn2=Dropout(drop_out)(td_cnn2)
    
    td_cnn2=Conv2D(filters=32, kernel_size=2, activation=activation_function)(td_cnn2)
    td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
    td_cnn2=Dropout(drop_out)(td_cnn2)
    
    if cnn_layer==3:
        td_cnn2=Conv2D(filters=16, kernel_size=2, activation=activation_function)(td_cnn2)
        td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
        td_cnn2=Dropout(drop_out)(td_cnn2)
    if cnn_layer==4:
        td_cnn2=Conv2D(filters=32, kernel_size=2, activation=activation_function)(td_cnn2)
        td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
        td_cnn2=Dropout(drop_out)(td_cnn2)
        
        td_cnn2=Conv2D(filters=16, kernel_size=2, activation=activation_function)(td_cnn2)
        td_cnn2=MaxPooling2D(pool_size=2,padding='same')(td_cnn2)
        td_cnn2=Dropout(drop_out)(td_cnn2)
    
        
    dense_input2=Flatten()(td_cnn2)
    
    input3=Input(shape=(x_train2.shape[1], x_train2.shape[2],1))
    td_cnn3=Conv2D(filters=5, kernel_size=2, activation=activation_function)(input3)
    td_cnn3=MaxPooling2D(pool_size=2,padding='same')(td_cnn3)
    td_cnn3=Dropout(drop_out)(td_cnn3)
    
    td_cnn3=Conv2D(filters=32, kernel_size=2, activation=activation_function)(td_cnn3)
    td_cnn3=MaxPooling2D(pool_size=2,padding='same')(td_cnn3)
    td_cnn3=Dropout(drop_out)(td_cnn3)
    
    if cnn_layer==3:
        td_cnn3=Conv2D(filters=5, kernel_size=2, activation=activation_function)(td_cnn3)
        td_cnn3=MaxPooling2D(pool_size=2,padding='same')(td_cnn3)
        td_cnn3=Dropout(drop_out)(td_cnn3)
    if cnn_layer==4:
        td_cnn3=Conv2D(filters=32, kernel_size=2, activation=activation_function)(td_cnn3)
        td_cnn3=MaxPooling2D(pool_size=2,padding='same')(td_cnn3)
        td_cnn3=Dropout(drop_out)(td_cnn3)
        
        td_cnn3=Conv2D(filters=16, kernel_size=2, activation=activation_function)(td_cnn3)
        td_cnn3=MaxPooling2D(pool_size=2,padding='same')(td_cnn3)
        td_cnn3=Dropout(drop_out)(td_cnn3)
    
        
    dense_input3=Flatten()(td_cnn3)
    
    merged=Concatenate(axis=-1)([dense_input1,dense_input2,dense_input3])
    if dense_layers==1:
        dense_input=Dense(128, activation=activation_function)(merged)
    if dense_layers==2:
        dense_input=Dense(256, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(128, activation=activation_function)(merged)
    if dense_layers==3:
        dense_input=Dense(256, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(128, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(64, activation=activation_function)(merged)
    
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=[input1,input2,input3],outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    merged_model.fit(x=[X_train,X_train1,X_train2],y=y_train1, validation_data=( [X_test,X_test1,X_test2], y_test), epochs=5, batch_size=32,callbacks=[es])
    score = merged_model.evaluate([X_test,X_test1,X_test2], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([X_test,X_test1,X_test2])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    with open("Results/Multimodal_Optimization_Results/MultiModal_3_Optimization_No_Context_" +myParam.FEAT1 +"_"+myParam.FEAT2+"_"+myParam.FEAT3+'_CNN'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(cnn_layer)+' & '+str(batch)+' & '+str(dense_layers)+' & '+str(round(drop_out,2))+' & '+str(activation_function)+' & '+str(optimizer_func)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':merged_model}
   

def multi_modal_lstm3(x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1,x_train2, x_test2, y_train2, y_test2):
   #x_train, x_test, y_train, y_test=paf.loadDataFeaures(feat1)
   #_train1, x_test1, y_train1, y_test1=paf.loadDataFeaures(feat2)
   #_train2, x_test2, y_train2, y_test2=paf.loadDataFeaures(feat3)
    
    
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    lstm_layers={{choice([2,3,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2]))
    tm_lstm=LSTM(256, return_sequences=True)(input1)
    tm_lstm=Dropout(drop_out)(tm_lstm)
    
    if lstm_layers==2:
        tm_lstm=LSTM(64, return_sequences=False)(tm_lstm)
        dense_input1=Flatten()(tm_lstm)
    if lstm_layers==3:
        tm_lstm=LSTM(128, return_sequences=True)(tm_lstm)
        tm_lstm=Dropout(drop_out)(tm_lstm)
        tm_lstm=LSTM(64, return_sequences=False)(tm_lstm)
        dense_input1=Flatten()(tm_lstm)
    if lstm_layers==4:
        tm_lstm=LSTM(128, return_sequences=True)(tm_lstm)
        tm_lstm=Dropout(drop_out)(tm_lstm)
        tm_lstm=LSTM(64, return_sequences=True)(tm_lstm)
        tm_lstm=Dropout(drop_out)(tm_lstm)
        tm_lstm=LSTM(32, return_sequences=False)(tm_lstm)
        dense_input1=Flatten()(tm_lstm)

    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2]))
    tm_lstm2=LSTM(256, return_sequences=True)(input2)
    tm_lstm2=Dropout(drop_out)(tm_lstm2)
    if lstm_layers==2:
        tm_lstm2=LSTM(64, return_sequences=False)(tm_lstm2)
        dense_input2=Flatten()(tm_lstm)
    if lstm_layers==3:
        tm_lstm2=LSTM(128, return_sequences=True)(tm_lstm2)
        tm_lstm2=Dropout(drop_out)(tm_lstm2)
        tm_lstm2=LSTM(64, return_sequences=False)(tm_lstm2)
        dense_input2=Flatten()(tm_lstm2)
    if lstm_layers==4:
        tm_lstm2=LSTM(128, return_sequences=True)(tm_lstm2)
        tm_lstm2=Dropout(drop_out)(tm_lstm2)
        tm_lstm2=LSTM(64, return_sequences=True)(tm_lstm2)
        tm_lstm2=Dropout(drop_out)(tm_lstm2)
        tm_lstm2=LSTM(32, return_sequences=False)(tm_lstm2)
        dense_input2=Flatten()(tm_lstm2)  
    
    input3=Input(shape=(x_train2.shape[1], x_train2.shape[2]))
    tm_lstm3=LSTM(256, return_sequences=True)(input3)
    tm_lstm3=Dropout(drop_out)(tm_lstm3)
    if lstm_layers==2:
         tm_lstm3=LSTM(64, return_sequences=False)( tm_lstm3)
         dense_input3=Flatten()( tm_lstm3)
    if lstm_layers==3:
        tm_lstm3=LSTM(128, return_sequences=True)(tm_lstm3)
        tm_lstm3=Dropout(drop_out)(tm_lstm3)
        tm_lstm3=LSTM(64, return_sequences=False)(tm_lstm3)
        dense_input3=Flatten()(tm_lstm3)
    if lstm_layers==4:
        tm_lstm3=LSTM(128, return_sequences=True)(tm_lstm3)
        tm_lstm3=Dropout(drop_out)(tm_lstm3)
        tm_lstm3=LSTM(64, return_sequences=True)(tm_lstm3)
        tm_lstm3=Dropout(drop_out)(tm_lstm3)
        tm_lstm3=LSTM(32, return_sequences=False)(tm_lstm3)
        dense_input3=Flatten()(tm_lstm3)

    
    merged=Concatenate(axis=1)([dense_input1,dense_input2,dense_input3])
    if dense_layers==1:
        dense_input=Dense(128, activation=activation_function)(merged)
    if dense_layers==2:
        dense_input=Dense(256, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(128, activation=activation_function)(merged)
        #dense_input=Dropout(drop_out)(dense_input)
    if dense_layers==3:
        dense_input=Dense(256, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(128, activation=activation_function)(merged)
        dense_input=Dropout(drop_out)(dense_input)
        dense_input=Dense(64, activation=activation_function)(merged)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=[input1,input2,input3],outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer=optimizer_func)
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    merged_model.fit(x=[x_train,x_train1,x_train2],y=y_train1, validation_data=( [x_test,x_test1,x_test2], y_test),epochs=10,batch_size=batch,callbacks=[es])
    
    score = merged_model.evaluate([x_test,x_test1,x_test2], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
  
    pred_test=merged_model.predict( [x_test,x_test1,x_test2])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/Multimodal_Optimization_Results/MultiModal_3_Optimization_"+myParam.FEAT1+"_"+myParam.FEAT2+"_"+myParam.FEAT3+'_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(lstm_layers)+' & '+str(dense_layers)+' & '+str(round(drop_out,2))+' & '+str(activation_function)+' & '+str(optimizer_func)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':merged_model}


        
if __name__=='__main__':
    
    audio_feat=['vggish','mel']
    music_feat=['tempo','Pitch_11_All']
    text_feat=['summary','trans']
    #pitch_feature=['pitch','magnitude']
    pitch_feat=["Pitch_11_Freq_Intens","Pitch_5_Freq"]

    allFeatures=audio_feat+music_feat+text_feat
    myParam.FEAT1='vggish'
    myParam.FEAT2='Pitch_11_All'
    #x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1=data()
    #multimodal_td_cnn2(x_train,y_train,x_test,y_test,x_train1, x_test1, y_train1, y_test1)
    """
    for af in ['mel']:
        for tf in text_feat:
            myParam.FEAT1=af
            myParam.FEAT2=tf
            print("features: {} {}".format(myParam.FEAT1, myParam.FEAT2))
            best_parameters,best_model=optim.minimize(model=multimodal_td_cnn2,
                                                       data=data,
                                                       algo=tpe.suggest,
                                                       max_evals=15,
                                                   trials=Trials())
    """
    myParam.FEAT1="mel"
    myParam.FEAT3="Pitch_11_All"
    myParam.FEAT2="trans"
    
    print("Features: {} {} {}".format(myParam.FEAT1, myParam.FEAT2,myParam.FEAT3))
    best_parameters,best_model=optim.minimize(model=multimodal_tm_cnn3,
                                               data=data1,
                                               algo=tpe.suggest,
                                               max_evals=5,
                                               trials=Trials())
    """
    for af in audio_feat:
        for tf in text_feat:
            for mf in music_feat:
                myParam.FEAT1=af
                myParam.FEAT2=tf
                myParam.FEAT3=mf
                print("Features: {} {} {}".format(myParam.FEAT1, myParam.FEAT2,myParam.FEAT3))
                best_parameters,best_model=optim.minimize(model=multimodal_tm_cnn3,
                                                           data=data1,
                                                           algo=tpe.suggest,
                                                           max_evals=15,
                                                           trials=Trials())
    """
    """
    for i in range(1,6):
        #for af in ['vggish']:
        for feat in pitch_feat:
            myParam.FEAT1=feat
            myParam.AUGMENT=i
            print("Features: {} {}".format(myParam.FEAT1, myParam.FEAT2))
            #try:
            best_parameters,best_model=optim.minimize(model=multimodal_net,
                                                           data=data_context,
                                                           algo=tpe.suggest,
                                                           max_evals=15,
                                                           trials=Trials())
        #except:
        #    print("Error on reading data: ")
      
"""