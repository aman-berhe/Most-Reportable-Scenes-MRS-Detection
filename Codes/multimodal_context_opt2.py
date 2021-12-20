import warnings
warnings.filterwarnings('ignore')
from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten,Input,TimeDistributed,InputLayer,Concatenate
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
from keras import Model

from hyperas import optim
from hyperas.distributions import choice, uniform
from hyperopt import fmin,tpe,hp,STATUS_OK,Trials

import preprocessAudioFeat as paf
import numpy as np
import MRS_evaluation as me
import pandas as pd

import myParam

from numpy.random import seed
seed(42)

dir_files='/vol/work3/berhe/MRS_Detection/'

def data():
    dir_files='/vol/work3/berhe/MRS_Detection/'
    data_training_context=np.load(dir_files+"Data/"+myParam.FEAT1+"_context_data_"+str(myParam.CONTEXT_SIZE)+".npy",allow_pickle=True)
    data_training_context2=np.load(dir_files+"Data/"+myParam.FEAT2+"_context_data_"+str(myParam.CONTEXT_SIZE)+".npy",allow_pickle=True)
    dataset_Df=pd.read_csv("Scene_Dataset_Normalized.csv")
    sceneLabels=dataset_Df.MRS.tolist()
    labels=[i if i==0 else 1 for i in sceneLabels]
    x_train, x_test, y_train, y_test=paf.split_Data(data_training_context,labels)
    x_train1, x_test1, y_train1, y_test1=paf.split_Data(data_training_context2,labels)
    print("shape",x_train.shape,x_test.shape,x_train1.shape,x_test1.shape)
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    x_train1=np.array(x_train1)
    x_test1=np.array(x_test1)
    y_train1=np.array(y_train1)
    y_test1=np.array(y_test1)
    return (x_train, y_train,x_test,y_test,x_train1,y_train1,x_test1,y_test1)

def data3():
    dir_files='/vol/work3/berhe/MRS_Detection/'
    data_training_context=np.load(dir_files+"Data/"+myParam.FEAT1+"_context_data_"+str(myParam.CONTEXT_SIZE)+".npy",allow_pickle=True)
    data_training_context2=np.load(dir_files+"Data/"+myParam.FEAT2+"_context_data_"+str(myParam.CONTEXT_SIZE)+".npy",allow_pickle=True)
    data_training_context3=np.load(dir_files+"Data/"+myParam.FEAT3+"_context_data_"+str(myParam.CONTEXT_SIZE)+".npy",allow_pickle=True)
    dataset_Df=pd.read_csv("Scene_Dataset_Normalized.csv")
    sceneLabels=dataset_Df.MRS.tolist()
    labels=[i if i==0 else 1 for i in sceneLabels]
    x_train, x_test, y_train, y_test=paf.split_Data(data_training_context,labels)
    x_train1, x_test1, y_train1, y_test1=paf.split_Data(data_training_context2,labels)
    x_train2, x_test2, y_train2, y_test2=paf.split_Data(data_training_context3,labels)
    print("shape",x_train.shape,x_test.shape,x_train1.shape,x_test1.shape,x_train2.shape,x_test2.shape)
    x_train=np.array(x_train)
    x_test=np.array(x_test)
    y_train=np.array(y_train)
    y_test=np.array(y_test)
    x_train1=np.array(x_train1)
    x_test1=np.array(x_test1)
    y_train1=np.array(y_train1)
    y_test1=np.array(y_test1)
    x_train2=np.array(x_train2)
    x_test2=np.array(x_test2)
    y_train2=np.array(y_train2)
    y_test2=np.array(y_test2)
    return (x_train, y_train,x_test,y_test,x_train1,y_train1,x_test1,y_test1,x_train2, x_test2, y_train2, y_test2)

def multimodal_optim(x_train, y_train,x_test,y_test,x_train1,y_train1,x_test1,y_test1):
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    lstm_layers={{choice([2,3,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3]))
    tm_lstm=TimeDistributed(LSTM(256, return_sequences=True))(input1)
    tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==2:
        tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==3:
        tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==4:
        tm_lstm=TimeDistributed(LSTM(256, return_sequences=True))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    tm_lstm=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm)
    dense_input1=Flatten()(tm_lstm)
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],x_train1.shape[3]))
    tm_lstm2=TimeDistributed(LSTM(256, return_sequences=True))(input2)
    tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==2:
        tm_lstm2=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==3:
        tm_lstm2=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==4:
        tm_lstm2=TimeDistributed(LSTM(256, return_sequences=True))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(LSTM(64, return_sequences=True))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    tm_lstm2=TimeDistributed(LSTM(64, return_sequences=False))(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)
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
    merged_model.fit(x=[x_train,x_train1],y=y_train1, validation_data=( [x_test,x_test1], y_test), epochs=5, batch_size=8,callbacks=[es])
    score = merged_model.evaluate([x_test,x_test1], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([x_test,x_test1])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    with open(dir_files+"Results/Multimodal_Optimization_Results/MultiModal_2_Optimization_Paper_" +myParam.FEAT1 +"_"+myParam.FEAT2+'_TM_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(lstm_layers)+' & '+str(batch)+' & '+str(dense_layers)+' & '+str(round(drop_out,2))+' & '+str(activation_function)+' & '+str(optimizer_func)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':merged_model}

def multimodal_optim_cnn(x_train, y_train,x_test,y_test,x_train1,y_train1,x_test1,y_test1):
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    lstm_layers={{choice([2,3,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3],1))
    #model.add(TimeDistributed(Conv2D(filters=16, kernel_size=2, activation=activation_function),input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3], 1)))    
    ##model.add(TimeDistributed( Conv2D(64, (3,3), padding='same', strides=(2,2), activation=activation_function)))    
    #model.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    #model.add(TimeDistributed(Dropout(drop_out)))
    tm_lstm=TimeDistributed(Conv2D(filters=16, kernel_size=2, activation=activation_function))(input1)
    tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==2:
        tm_lstm=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==3:
        tm_lstm=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==4:
        tm_lstm=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(Conv2D(32, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    
    dense_input1=Flatten()(tm_lstm)
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],x_train1.shape[3],1))
    tm_lstm2=TimeDistributed(Conv2D(filters=16, kernel_size=2, activation=activation_function))(input2)
    tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
    tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==2:
        tm_lstm2=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==3:
        tm_lstm2=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==4:
        tm_lstm2=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(Conv2D(32, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)
    
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
    merged_model.fit(x=[x_train,x_train1],y=y_train1, validation_data=( [x_test,x_test1], y_test), epochs=5, batch_size=32,callbacks=[es])
    score = merged_model.evaluate([x_test,x_test1], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([x_test,x_test1])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    with open(dir_files+"Results/Multimodal_Optimization_Results/MultiModal_2_Optimization_" +myParam.FEAT1 +"_"+myParam.FEAT2+'_TD_CNN'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(lstm_layers)+' & '+str(batch)+' & '+str(dense_layers)+' & '+str(round(drop_out,2))+' & '+str(activation_function)+' & '+str(optimizer_func)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':merged_model}

def multimodal_optim_cnn3(x_train, y_train,x_test,y_test,x_train1,y_train1,x_test1,y_test1,x_train2,y_train2,x_test2,y_test2):
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    lstm_layers={{choice([2,3,4])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    input1=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3],1))
    #model.add(TimeDistributed(Conv2D(filters=16, kernel_size=2, activation=activation_function),input_shape = (x_train.shape[1],x_train.shape[2],x_train.shape[3], 1)))    
    ##model.add(TimeDistributed( Conv2D(64, (3,3), padding='same', strides=(2,2), activation=activation_function)))    
    #model.add(TimeDistributed(MaxPooling2D(pool_size=2)))
    #model.add(TimeDistributed(Dropout(drop_out)))
    tm_lstm=TimeDistributed(Conv2D(filters=16, kernel_size=2, activation=activation_function))(input1)
    tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
    tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==2:
        tm_lstm=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==3:
        tm_lstm=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    if lstm_layers==4:
        tm_lstm=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
        tm_lstm=TimeDistributed(Conv2D(32, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm)
        tm_lstm=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm)
        tm_lstm=TimeDistributed(Dropout(drop_out))(tm_lstm)
    
    dense_input1=Flatten()(tm_lstm)
    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2],x_train1.shape[3],1))
    tm_lstm2=TimeDistributed(Conv2D(filters=16, kernel_size=2, activation=activation_function))(input2)
    tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
    tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==2:
        tm_lstm2=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==3:
        tm_lstm2=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    if lstm_layers==4:
        tm_lstm2=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
        tm_lstm2=TimeDistributed(Conv2D(32, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm2)
        tm_lstm2=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm2)
        tm_lstm2=TimeDistributed(Dropout(drop_out))(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)
    
    input3=Input(shape=(x_train2.shape[1], x_train2.shape[2],x_train2.shape[3],1))
    tm_lstm3=TimeDistributed(Conv2D(filters=16, kernel_size=2, activation=activation_function))(input3)
    tm_lstm3=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm3)
    tm_lstm3=TimeDistributed(Dropout(drop_out))(tm_lstm3)
    if lstm_layers==2:
        tm_lstm3=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm3)
        tm_lstm3=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm3)
        tm_lstm3=TimeDistributed(Dropout(drop_out))(tm_lstm3)
    if lstm_layers==3:
        tm_lstm3=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm3)
        tm_lstm3=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm3)
        tm_lstm3=TimeDistributed(Dropout(drop_out))(tm_lstm3)
        tm_lstm3=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm3)
        tm_lstm3=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm3)
        tm_lstm3=TimeDistributed(Dropout(drop_out))(tm_lstm3)
    if lstm_layers==4:
        tm_lstm3=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm3)
        tm_lstm3=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm3)
        tm_lstm3=TimeDistributed(Dropout(drop_out))(tm_lstm3)
        tm_lstm3=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm3)
        tm_lstm3=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm3)
        tm_lstm3=TimeDistributed(Dropout(drop_out))(tm_lstm3)
        tm_lstm3=TimeDistributed(Conv2D(32, (3,3),padding='same', strides=(2,2), activation=activation_function))(tm_lstm3)
        tm_lstm3=TimeDistributed(MaxPooling2D(pool_size=2))(tm_lstm3)
        tm_lstm3=TimeDistributed(Dropout(drop_out))(tm_lstm3)
    dense_input3=Flatten()(tm_lstm3)
    
    merged=Concatenate(axis=1)([dense_input1,dense_input2,dense_input3])
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
    merged_model.fit(x=[x_train,x_train1,x_train2],y=y_train1, validation_data=( [x_test,x_test1,x_test2], y_test), epochs=5, batch_size=4,callbacks=[es])
    score = merged_model.evaluate([x_test,x_test1,x_test2], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    pred_test=merged_model.predict([x_test,x_test1,x_test2])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    with open(dir_files+"Results/Multimodal_Optimization_Results/MultiModal_3_Optimization_" +myParam.FEAT1 +"_"+myParam.FEAT2+"_"+myParam.FEAT3+'_TD_CNN_3'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(lstm_layers)+' & '+str(batch)+' & '+str(dense_layers)+' & '+str(round(drop_out,2))+' & '+str(activation_function)+' & '+str(optimizer_func)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':merged_model}

if __name__=='__main__':
#    myParam.FEAT1="vggish"
#    myParam.FEAT2="trans"
#    myParam.CONTEXT_SIZE=1
#    x_train, x_test, y_train, y_test,x_train1, x_test1, y_train1, y_test1=data()
#    multimodal_optim(x_train, x_test, y_train, y_test,x_train1, x_test1, y_train1, y_test1)   
    audio_feat=['mel','vggish']
    music_feat=['tempogram','Pitch_11_Freq_Intens']
    text_feat=['trans','summary']
    #for af in audio_feat:
     #   for tf in text_feat:
      #      for mf in music_feat:
    myParam.FEAT1='vggish'
    myParam.FEAT2="Pitch_11_Freq_Intens"
    #myParam.FEAT2="tempogram"
    #myParam.CONTEXT_SIZE=5
    #x_train, x_test, y_train, y_test,x_train1, x_test1, y_train1, y_test1,x_train2, x_test2, y_train2, y_test2=data()
    #multimodal_optim_cnn3(x_train, x_test, y_train, y_test,x_train1, x_test1, y_train1, y_test1,x_train2, x_test2, y_train2, y_test2)
    
    print("Features: {} {}".format(myParam.FEAT1, myParam.FEAT2))
    #for cs in [5]:
    myParam.CONTEXT_SIZE=5
    best_parameters,best_model=optim.minimize(model=multimodal_optim,
                                           data=data,
                                           algo=tpe.suggest,
                                           max_evals=10,
                                           trials=Trials())