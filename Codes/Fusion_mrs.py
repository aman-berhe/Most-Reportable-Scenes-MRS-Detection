# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:58:43 2020

@author: berhe
"""
import warnings
warnings.filterwarnings('ignore')
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input,TimeDistributed,InputLayer,Concatenate
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras import Model

import numpy as np
import pandas as pd
import preprocessAudioFeat as paf
import MRS_evaluation as me
import myParam

from numpy.random import seed
seed(1)

dir_files='/vol/work3/berhe/MRS_Detection/'
def data(feat):
    data_training_context=np.load(dir_files+"Data/"+feat+"_context_data_"+str(myParam.AUGMENT)+".npy",allow_pickle=True)
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
    

def multimodal_net():
    
    x_train, x_test, y_train, y_test=data(myParam.FEAT)
    x_train1, x_test1, y_train1, y_test1=data('trans')
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))
    print(type(input1))
    tm_lstm=TimeDistributed(LSTM(256, return_sequences=True))(input1)
    print("ça Passe!!")
    tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(0.144))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(0.144))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm)
    #tm_lstm=TimeDistributed(Dropout(0.144))(tm_lstm)
    dense_input1=Flatten()(tm_lstm)
    
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],x_train1.shape[3]))
    tm_lstm2=TimeDistributed(LSTM(256, return_sequences=True))(input2)
    tm_lstm2=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm2)
    tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    tm_lstm2=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm2)
    tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    tm_lstm2=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm2)
    #tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)
    
    merged=Concatenate(axis=1)([dense_input1,dense_input2])
    
    
    dense_input=Dense(128, activation='linear')(merged)
    
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    
    merged_model=Model(inputs=[input1,input2],outputs=ouput_dense)
    
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
    
    merged_model.fit(x=[x_train,x_train1],y=y_train, validation_data=([x_test,x_test1], y_test),epochs=10,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate([x_test,x_test1], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([x_test,x_test1])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    #print(rec,pre,f1)
    
    with open(dir_files+"Results/Multimodal_results/MultiModal_"+myParam.OUTPU_FILE+'_'+myParam.FEAT+'_tm_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.AUGMENT)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')

def multimodal_net_3():
    
    x_train, x_test, y_train, y_test=data(myParam.FEAT)
    x_train1, x_test1, y_train1, y_test1=data('trans')
    x_train2, x_test2, y_train2, y_test2=data('tempogram')
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))
    print(type(input1))
    tm_lstm=TimeDistributed(LSTM(256, return_sequences=True))(input1)
    print("ça Passe!!")
    tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(0.144))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(0.144))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm)
    #tm_lstm=TimeDistributed(Dropout(0.144))(tm_lstm)
    dense_input1=Flatten()(tm_lstm)
    
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],x_train1.shape[3]))
    tm_lstm2=TimeDistributed(LSTM(256, return_sequences=True))(input2)
    tm_lstm2=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm2)
    tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    tm_lstm2=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm2)
    tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    tm_lstm2=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm2)
    #tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)
    
    input3=Input(shape=(x_train2.shape[1], x_train2.shape[2],x_train2.shape[3]))
    tm_lstm3=TimeDistributed(LSTM(256, return_sequences=True))(input3)
    tm_lstm3=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm3)
    tm_lstm3=TimeDistributed(Dropout(0.144))(tm_lstm3)
    tm_lstm3=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm3)
    tm_lstm3=TimeDistributed(Dropout(0.144))(tm_lstm3)
    tm_lstm3=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm3)
    #tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    dense_input3=Flatten()(tm_lstm3)
    
    merged=Concatenate(axis=1)([dense_input1,dense_input2,dense_input3])
    
    
    dense_input=Dense(128, activation='linear')(merged)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    
    merged_model=Model(inputs=[input1,input2,input3],outputs=ouput_dense)
    
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    
    merged_model.fit(x=[x_train,x_train1,x_train2],y=y_train, validation_data=([x_test,x_test1,x_test2], y_test),epochs=5,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate([x_test,x_test1,x_test2], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([x_test,x_test1,x_test2])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    #print(rec,pre,f1)
    
    with open(dir_files+"Results/Multimodal_results/MultiModal_3_Tempo_"+myParam.OUTPU_FILE+'_'+myParam.FEAT+'_tm_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.AUGMENT)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')

def multimodal_net_3_best(feat1,feat2,feat3):
    
    x_train, x_test, y_train, y_test=data(feat1)
    x_train1, x_test1, y_train1, y_test1=data(feat2)
    x_train2, x_test2, y_train2, y_test2=data(feat3)
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))
    print(type(input1))
    tm_lstm=TimeDistributed(LSTM(256, return_sequences=True))(input1)
    #tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(0.22))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm)
    
    dense_input1=Flatten()(tm_lstm)
    
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],x_train1.shape[3]))
    tm_lstm2=TimeDistributed(LSTM(256, return_sequences=True))(input2)
    #tm_lstm2=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm2)
    tm_lstm2=TimeDistributed(Dropout(0.22))(tm_lstm2)
    #tm_lstm2=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm2)
    #tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    tm_lstm2=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm2)
    #tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)
    
    input3=Input(shape=(x_train2.shape[1], x_train2.shape[2],x_train2.shape[3]))
    tm_lstm3=TimeDistributed(LSTM(256, return_sequences=True))(input3)
    tm_lstm3=TimeDistributed(Dropout(0.22))(tm_lstm3)
    #tm_lstm3=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm3)
    #tm_lstm3=TimeDistributed(Dropout(0.144))(tm_lstm3)
    #tm_lstm3=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm3)
    #tm_lstm3=TimeDistributed(Dropout(0.144))(tm_lstm3)
    tm_lstm3=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm3)
    #tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    dense_input3=Flatten()(tm_lstm3)
    
    merged=Concatenate(axis=1)([dense_input1,dense_input2,dense_input3])
    
    
    dense_input=Dense(128, activation='relu')(merged)
    
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    
    merged_model=Model(inputs=[input1,input2,input3],outputs=ouput_dense)
    
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
    
    merged_model.fit(x=[x_train,x_train1,x_train2],y=y_train, validation_data=([x_test,x_test1,x_test2], y_test),epochs=5,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate([x_test,x_test1,x_test2], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([x_test,x_test1,x_test2])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    #print(rec,pre,f1)
    
    with open(dir_files+"Results/Multimodal_Optimization_Results/MultiModal_Best_3_"+feat1+'_'+feat2+'_'+feat3+'_tm_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.AUGMENT)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')

def multimodal_net_2_best(feat1,feat2):
    
    x_train, x_test, y_train, y_test=data(feat1)
    x_train1, x_test1, y_train1, y_test1=data(feat2)
    
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))
    print(type(input1))
    tm_lstm=TimeDistributed(LSTM(256, return_sequences=True))(input1)
    #tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(0.22))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm)
    
    dense_input1=Flatten()(tm_lstm)
    
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],x_train1.shape[3]))
    tm_lstm2=TimeDistributed(LSTM(256, return_sequences=True))(input2)
    #tm_lstm2=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm2)
    tm_lstm2=TimeDistributed(Dropout(0.22))(tm_lstm2)
    #tm_lstm2=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm2)
    #tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    tm_lstm2=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm2)
    #tm_lstm2=TimeDistributed(Dropout(0.144))(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)
    
    
    merged=Concatenate(axis=1)([dense_input1,dense_input2])
    
    
    dense_input=Dense(128, activation='relu')(merged)
    
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    
    merged_model=Model(inputs=[input1,input2],outputs=ouput_dense)
    
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
    
    merged_model.fit(x=[x_train,x_train1],y=y_train, validation_data=([x_test,x_test1], y_test),epochs=10,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate([x_test,x_test1], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([x_test,x_test1])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    #print(rec,pre,f1)
    
    with open(dir_files+"Results/Multimodal_results/MultiModal_Best_2_"+feat1+"_"+feat2+'_tm_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.AUGMENT)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')

if __name__=='__main__':
    #for feat in ['mel',"vggish"]:#,'mel','mffc']:
    for t in [1,3,5,7]:
        #try:
        #myParam.FEAT=feat
        myParam.AUGMENT=t
        myParam.MODEL='tm_LSTM'
        myParam.OUTPU_FILE=''
        #multimodal_net_2_best(myParam.FEAT,'summary')
        multimodal_net_3_best("mel",'trans','Pitch_11_Freq_Intens')
