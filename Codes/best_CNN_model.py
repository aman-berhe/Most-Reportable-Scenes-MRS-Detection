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
seed(1)

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
    print('Data Shapes',x_train.shape,x_test.shape)
    return  x_train,x_test,y_train,y_test

def data_context():
    data_training_context=np.load("/vol/work3/berhe/MRS_Detection/Data/"+myParam.FEAT+"_context_data_"+str(myParam.CONTEXT_SIZE)+".npy",allow_pickle=True)
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

#Best values
#activation=relu, optimize=adam, dropout=0.178, cnn_layer=2, dense_layers=4

def model_cnn_classic_best(x_train,y_train,x_test,y_test):
    num_rows = x_train.shape[1]
    num_columns = x_train.shape[2]
    num_channels = 1

    x_train = x_train.reshape(x_train.shape[0], num_rows,num_columns, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_rows,num_columns, num_channels)

    
    model = Sequential()
    model.add(Conv2D(filters=16, kernel_size=2, input_shape=(num_rows, num_columns,num_channels), activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.18))

    model.add(Conv2D(filters=32, kernel_size=2, activation='linear'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.18))

    model.add(Conv2D(filters=32, kernel_size=2, activation='linear'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.18))
    
    model.add(Conv2D(filters=16, kernel_size=1, activation='linear'))
    model.add(MaxPooling2D(pool_size=1))
    model.add(Dropout(0.18))
    model.add(GlobalAveragePooling2D())
    
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))                

    model.add(Dense(2, activation='softmax'))
    
    
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='rmsprop')
    
    checkpointer = ModelCheckpoint(filepath='best_CNN_clasical_weights'+myParam.FEAT+'.hdf5',
                                   verbose=1, save_best_only=True)
    model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test), callbacks=[checkpointer], verbose=2)

    
    score = model.evaluate(x_test, y_test, verbose=0)
    pred_test=model.predict_classes(x_test).tolist()
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/optimized_Models_Result/best_CNN_clasical_"+myParam.FEAT+"_LaTex.txt",'a') as file:
        file.write(str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2)))
        file.write('\n')
        file.write('\hline')
        
def td_cnn_best(x_train,x_test,y_train,y_test):
    
    inputLayer=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3],1))
    td_cnn=TimeDistributed(Conv2D(filters=16, kernel_size=2, activation='linear'))(inputLayer)
    td_cnn=TimeDistributed(MaxPooling2D(pool_size=2))(td_cnn)
    td_cnn=TimeDistributed(Dropout(0.18))(td_cnn)
    
    
    td_cnn=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation='linear'))(td_cnn)
    td_cnn=TimeDistributed(MaxPooling2D(pool_size=2))(td_cnn)
    td_cnn=TimeDistributed(Dropout(0.18))(td_cnn)
    
    td_cnn=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation='linear'))(td_cnn)
    td_cnn=TimeDistributed(MaxPooling2D(pool_size=2))(td_cnn)
    td_cnn=TimeDistributed(Dropout(0.18))(td_cnn)
    
    td_cnn=TimeDistributed(Conv2D(32, (3,3),padding='same', strides=(2,2), activation='linear'))(td_cnn)
    td_cnn=TimeDistributed(MaxPooling2D(pool_size=2))(td_cnn)
    td_cnn=TimeDistributed(Dropout(0.18))(td_cnn)
    
    dense_input=Flatten()(td_cnn)
    
    dense_input=Dense(128, activation='linear')(dense_input)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    model=Model(inputs=inputLayer,outputs=ouput_dense)
    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    model.fit(x=x_train,y=y_train, validation_data=(x_test, y_test),epochs=10,batch_size=32,callbacks=[es])
    
    score = model.evaluate(x_test, y_test, verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    
    pred_test=model.predict(x_test)
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_test,pred_test)
    
    with open("Results/optimized_Models_Result/best_TD_CNN_"+myParam.FEAT+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
        
def multi_modal_cnn_TOP(x_train,x_test,y_train,y_test):
    inputLayer=Input(shape=(x_train.shape[1], x_train.shape[2],1))
    td_cnn=Conv2D(filters=16, kernel_size=2, activation='linear')(inputLayer)
    td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
    td_cnn=Dropout(0.18)(td_cnn)

    td_cnn=Conv2D(filters=128, kernel_size=2, activation='linear')(td_cnn)
    td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
    td_cnn=Dropout(0.18)(td_cnn)
    
    td_cnn=Conv2D(filters=64, kernel_size=2, activation='linear')(td_cnn)
    td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
    td_cnn=Dropout(0.18)(td_cnn)

    td_cnn=Conv2D(filters=32, kernel_size=2, activation='linear')(td_cnn)
    td_cnn=MaxPooling2D(pool_size=2,padding='same')(td_cnn)
    td_cnn=Dropout(0.18)(td_cnn)
    
    dense_input=Flatten()(td_cnn)

    return dense_input,inputLayer

def model_cnn_classic_best_multimodal(listFeat):
    learnedEmbeding=[]
    inputsShapes=[]
    x_trainings=[]
    x_tests=[]
    y_trainings=[]
    y_tests=[]
    for feat in listFeat:
        x_train,x_test,y_train,y_test=data1(feat)
        X_train = x_train.reshape(x_train.shape[0], x_train.shape[1],x_train.shape[2], 1)
        X_test = x_test.reshape(x_test.shape[0], x_test.shape[1],x_test.shape[2], 1)
        
        dense_input,inputLayer=multi_modal_cnn_TOP(x_train,x_test,y_train,y_test)
        learnedEmbeding.append(dense_input)
        inputsShapes.append(inputLayer)
        x_trainings.append(x_train)
        x_tests.append(x_test)
        y_trainings.append(y_train)
        y_tests.append(y_test)
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
    with open("Results/optimized_Models_Result/best_CNN_clasical_"+featFile+"_LaTex.txt",'a') as file:
        file.write(str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')
        
def multimodal_td_cnn_top(x_train,x_test,y_train,y_test):
    inputLayer=Input(shape=(x_train.shape[1], x_train.shape[2],x_train.shape[3],1))
    td_cnn=TimeDistributed(Conv2D(filters=16, kernel_size=2, activation='linear'))(inputLayer)
    td_cnn=TimeDistributed(MaxPooling2D(pool_size=2))(td_cnn)
    td_cnn=TimeDistributed(Dropout(0.18))(td_cnn)
    
    
    td_cnn=TimeDistributed(Conv2D(128, (3,3),padding='same', strides=(2,2), activation='linear'))(td_cnn)
    td_cnn=TimeDistributed(MaxPooling2D(pool_size=2))(td_cnn)
    td_cnn=TimeDistributed(Dropout(0.18))(td_cnn)
    
    td_cnn=TimeDistributed(Conv2D(64, (3,3),padding='same', strides=(2,2), activation='linear'))(td_cnn)
    td_cnn=TimeDistributed(MaxPooling2D(pool_size=2))(td_cnn)
    td_cnn=TimeDistributed(Dropout(0.18))(td_cnn)
    
    td_cnn=TimeDistributed(Conv2D(32, (3,3),padding='same', strides=(2,2), activation='linear'))(td_cnn)
    td_cnn=TimeDistributed(MaxPooling2D(pool_size=2))(td_cnn)
    td_cnn=TimeDistributed(Dropout(0.18))(td_cnn)
    
    dense_input=Flatten()(td_cnn)
    
    return dense_input,inputLayer

def multimodal_td_cnn_best(listFeat):
    learnedEmbeding=[]
    inputsShapes=[]
    x_trainings=[]
    x_tests=[]
    y_trainings=[]
    y_tests=[]
    for feats in listFeat:
        myParam.FEAT=feats
        print(feats)
        x_train,x_test,y_train,y_test=data_context()
        x_trainings.append(x_train)
        x_tests.append(x_test)
        y_trainings.append(y_train)
        y_tests.append(y_test)
        dense_inputs, inputLayer=multimodal_td_cnn_top(x_train,x_test,y_train,y_test)
        learnedEmbeding.append(dense_inputs)
        inputsShapes.append(inputLayer)
        
    merged=Concatenate(axis=1)(learnedEmbeding)
    dense_input=Dense(128, activation='linear')(merged)
    ouput_dense=Dense(2, activation='softmax')(dense_input)
    merged_model=Model(inputs=inputsShapes,outputs=ouput_dense)
    merged_model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    merged_model.fit(x=x_trainings,y=y_trainings[0], validation_data=(x_tests, y_tests[0]),epochs=10,batch_size=8,callbacks=[es])
    
    score = merged_model.evaluate(x_tests, y_tests[0], verbose=0)
    print("Lost : {} Accuracy {}".format(round(score[0],3),round(score[1],3)))
    
    pred_test=merged_model.predict(x_tests)
    pred_test=np.argmax(pred_test,axis=1)
    rec,pre,f1=me.rec_pre_f1_MRS(y_tests[0],pred_test)
    
    featFile="_".join(listFeat)
    with open("Results/optimized_Models_Result/best_TD_CNN_"+featFile+"_LaTex.txt",'a') as file:
        file.write(str(myParam.CONTEXT_SIZE)+' & '+str(round(score[1],2))+' & '+str(round(rec[1],2))+' & '+str(round(pre[1],2))+' & '+str(round(f1[1],2))+'\\\\')
        file.write('\n')
        file.write('\hline')
        file.write('\n')

if __name__ == '__main__': 
    features=['vggish','mel','mfcc']
    textFeat=['trans','summary']
    musicFeat=['tempogram','Pitch_11_Freq_Intens']
    allFeatures=features+textFeat+musicFeat
    """
    for feat in musicFeat:
        myParam.FEAT=feat
        x_train,x_test,y_train,y_test=data1(feat)
        model_cnn_classic_best(x_train,y_train,x_test,y_test)
    
    for feat in ['vggish','trans','summary','tempogram']:#features:
        myParam.FEAT=feat#'vggish'
        for c in [1,3,5,7]:
            myParam.CONTEXT_SIZE=c
            x_train,x_test,y_train,y_test=data_context()
            td_cnn_best(x_train,x_test,y_train,y_test)
    """
    
    #model_cnn_classic_best_multimodal(['mel','summary','tempogram'])
    #model_cnn_classic_best_multimodal(['mel','trans','tempogram'])
    #model_cnn_classic_best_multimodal(['vggish','summary','tempogram'])
    #model_cnn_classic_best_multimodal(['vggish','trans','tempogram'])
    #model_cnn_classic_best_multimodal(['vggish','tempogram'])
    
    """
    #for feat1 in features:
    feat1="mel"
    #for feat2 in textFeat:
        #for feat3 in musicFeat:
    feat2="summary"
    feat3="tempogram"
    #try:
    for c in [1,3,5,7]:
        myParam.CONTEXT_SIZE=c
        multimodal_td_cnn_best([feat1,feat2,feat3])
        #except:
         #   continue
    """
    myParam.CONTEXT_SIZE=7
    multimodal_td_cnn_best(["vggish","trans"])