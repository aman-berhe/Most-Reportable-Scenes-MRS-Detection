# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 10:38:10 2020

@author: berhe
"""
from attention import AttentionWithContext,Attention,augmented_conv2d
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.callbacks import EarlyStopping
from keras import Model

from keras.layers import LSTM,Concatenate,Input
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
import myParam

from numpy.random import seed
seed(1)

dir_files='/vol/work3/berhe/MRS_Detection/'

def data():
    x_train,y_train,x_test,y_test=paf.loadTrainingTest(myParam.FEAT)#myParam.FEAT)
    #print('Data Shapes',x_train.shape,x_test.shape)
    feat=myParam.FEAT
    if feat.lower()[0]=='v':
        x_train=paf.augment_reshapeVgg(x_train,138)
        x_test=paf.augment_reshapeVgg(x_test,138)
    else:
        x_train=paf.augment_reshape(x_train,myParam.DIM)
        x_test=paf.augment_reshape(x_test,myParam.DIM)
    
    if myParam.AUGMENT==0:
        y_train=np.array(y_train)
        y_test=np.array(y_test)
        print('Here: with out augmentation',x_train.shape,x_test.shape,y_train.shape,y_test.shape)
        #return  x_train,y_train,x_test,y_test
    else:
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
        vggFeat_a=paf.augment_reshapeVgg(vggFeatA,myParam.DIM)       
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

def dataNew():
    x_train, x_test, y_train, y_test=paf.loadDataFeaures(myParam.FEAT1)
    
    return x_train,y_train,x_test,y_test
    


def multi_modal_lstm(feat1,feat2):
    x_train, x_test, y_train, y_test=paf.loadDataFeaures(feat1)
    x_train1, x_test1, y_train1, y_test1=paf.loadDataFeaures(feat2)
    
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2]))
    tm_lstm=LSTM(256, return_sequences=True)(input1)
    #tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm=Dropout(0.22)(tm_lstm)
    tm_lstm=LSTM(64, return_sequences=False)(tm_lstm)
    dense_input1=Flatten()(tm_lstm)
    

    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2]))
    tm_lstm2=LSTM(256, return_sequences=True)(input2)
    #tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm2=Dropout(0.22)(tm_lstm2)
    tm_lstm2=LSTM(64, return_sequences=False)(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)    

    
    merged=Concatenate(axis=1)([dense_input1,dense_input2])
    dense_input=Dense(128, activation='relu')(merged)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=[input1,input2],outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    merged_model.fit(x=[x_train,x_train1],y=y_train1, validation_data=([x_test,x_test1], y_test),epochs=10,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate([x_test,x_test1], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    
    pred_test=merged_model.predict([x_test,x_test1])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open(dir_files+"Results/Multimodal_Results_On_Previous_best _Models/MultiModal_Best_2_PATeI"+feat1+"_"+feat2+'_LSTM'+"_LaTex.txt",'a') as file:
        file.write(str(myParam.AUGMENT)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')

def multi_modal_lstm3(feat1,feat2,feat3):
    x_train, x_test, y_train, y_test=paf.loadDataFeaures(feat1)
    x_train1, x_test1, y_train1, y_test1=paf.loadDataFeaures(feat2)
    x_train2, x_test2, y_train2, y_test2=paf.loadDataFeaures(feat3)
    
    
    input1=Input(shape=(x_train.shape[1], x_train.shape[2]))
    tm_lstm=LSTM(256, return_sequences=True)(input1)
    #tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm=Dropout(0.22)(tm_lstm)
    tm_lstm=LSTM(64, return_sequences=False)(tm_lstm)
    dense_input1=Flatten()(tm_lstm)
    

    input2=Input(shape=(x_train1.shape[1], x_train1.shape[2]))
    tm_lstm2=LSTM(256, return_sequences=True)(input2)
    #tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm2=Dropout(0.22)(tm_lstm2)
    tm_lstm2=LSTM(64, return_sequences=False)(tm_lstm2)
    dense_input2=Flatten()(tm_lstm2)  
    
    input3=Input(shape=(x_train2.shape[1], x_train2.shape[2]))
    tm_lstm3=LSTM(256, return_sequences=True)(input3)
    #tm_lstm=TimeDistributed(LSTM(128, return_sequences=True))(tm_lstm)
    tm_lstm3=Dropout(0.22)(tm_lstm3)
    tm_lstm3=LSTM(64, return_sequences=False)(tm_lstm3)
    dense_input3=Flatten()(tm_lstm3) 

    
    merged=Concatenate(axis=1)([dense_input1,dense_input2,dense_input3])
    dense_input=Dense(128, activation='relu')(merged)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=[input1,input2,input3],outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    merged_model.fit(x=[x_train,x_train1,x_train2],y=y_train1, validation_data=([x_test,x_test1,x_test2], y_test),epochs=10,batch_size=32,callbacks=[es])
    
    score = merged_model.evaluate([x_test,x_test1,x_test2], y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    
    pred_test=merged_model.predict([x_test,x_test1,x_test2])
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open(dir_files+"Results/Multimodal_Results_On_Previous_best _Models/MultiModal_Best_3_PATI_"+feat1+"_"+feat2+"_"+feat3+"_LSTM_LaTex.txt",'a') as file:
        file.write(str(myParam.AUGMENT)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')



def lstm_Model(x_train, x_test, y_train, y_test):
    print(type(x_train))
    input_shape = (x_train.shape[1], x_train.shape[2])
    print(x_train.shape, x_test.shape)
    drop_out={{uniform(0.0, 0.5)}}
    activation_function={{choice(['sigmoid', 'relu', 'linear','relu'])}}
    dense_layers={{choice([1,2,3])}}
    batch={{choice([32,64])}}
    optimizer_func={{choice(['rmsprop', 'adam', 'sgd'])}}
    
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
    
    checkpointer = ModelCheckpoint(filepath='weights.best_LSTM_Optimization_'+myParam.FEAT1+'.hdf5',verbose=1, save_best_only=True)
    model.fit(x_train, y_train, batch_size=batch, epochs=10, validation_data=(x_test, y_test), callbacks=[checkpointer],verbose=2)
    
    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    pred_test=model.predict_classes(x_test).tolist()
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/LSTM_"+myParam.FEAT+"_Only"+"_Fixed.txt",'a') as file:
        file.write(str(myParam.DIM)+'\t'+activation_function+'\t'+optimizer_func+'\t'+str(round(drop_out,3))+'\t'+str(dense_layers)+'\t'+str(score[1])+'\t'+str(rec)+'\t'+str(pre)+'\t'+str(f1))
        file.write('\n')
    with open("Results/LSTM_"+myParam.FEAT+"_Only_Aug_"+"_Fixed_LaTex.txt",'a') as file:
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
    
    with open("Results/LSTM_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed.txt",'a') as file:
        file.write(str(myParam.DIM)+'\t'+activation_function+'\t'+optimizer_func+'\t'+str(round(drop_out,3))+'\t'+str(dense_layers)+'\t'+str(score[1])+'\t'+str(rec)+'\t'+str(pre)+'\t'+str(f1))
        file.write('\n')
    with open("Results/LSTM_"+myParam.FEAT+"_"+ myParam.OUTPU_FILE+"_Fixed_LaTex.txt",'a') as file:
        file.write(str(myParam.DIM)+' & '+activation_function+' & '+optimizer_func+' & '+str(round(drop_out,3))+' & '+str(dense_layers)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
    return {'loss': -score[0], 'status': STATUS_OK,'model':model}


if __name__=='__main__':
    
    myParam.FEAT="mel"
    myParam.AUGMENT=1
    myParam.DIM=2000
    print("Features: {}".format(myParam.FEAT))
    best_parameters,best_model=optim.minimize(model=lstm_Model,
                                                   data=data,
                                                   algo=tpe.suggest,
                                                   max_evals=15,
                                                   trials=Trials())
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
    
    features=['vggish','mel','mfcc']
    for feat in features:
        #for t in [0,1]:
        if feat[0]=='v':
            myParam.DIM=200
        else:
            myParam.DIM=2000
            myParam.FEAT=feat
        myParam.AUGMENT=1
        myParam.OUTPU_FILE="dim_2000_"+str(1)
        print(type(myParam.DIM),type(myParam.FEAT),myParam.DIM,myParam.FEAT,myParam.AUGMENT)
        best_parameters,best_model=optim.minimize(model=lstm_Model,
                                                   data=data,
                                                   algo=tpe.suggest,
                                               multi_modal_lstm3(feat1,feat2,feat3):    max_evals=25,
                                                   trials=Trials())
        print(best_parameters)
        print(type(myParam.DIM),type(myParam.FEAT),myParam.DIM,myParam.FEAT,myParam.AUGMENT)
        with open("Results/Best_Parameters_LSTM_"+myParam.FEAT+"_"+myParam.OUTPU_FILE+"_Fixed.txt","a") as f:
            for att in best_parameters.keys():
                f.write(att+"\t"+str(best_parameters[att])+"\n")

    #with open("Best Parameters_LSTM_Pickle_"+myParam.FEAT,"wb") as fpickle:
     #   pickle.dump(fpickle,best_parameters)
        best_model.save("Results/Best_Parameters_LSTM_"+myParam.FEAT+"_"+myParam.OUTPU_FILE+"_Fixed.hdf5")
feature_list=['vggish','mfcc','mel','summary','trans','tempo','chromo']
audio_feat=['vggish','mfcc','mel']
music_feat=['tempo','chromo']
text_feat=['summary','trans']
pitchFeatures=["Pitch_5_Freq","Pitch_11_All"]
for i in audio_feat:
    for j in text_feat:
        for k in music_feat:
            lt.multi_modal_lstm3(i,j,k)
"""