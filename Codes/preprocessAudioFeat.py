#import librosa
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

def getAveragedData(audioData):
    averagedData=[]
    for i in audioData:
        averagedData.append(np.mean(i,axis=1))
    averagedData=np.array(averagedData)
    return averagedData

#Augmentation Just by taking last 10%,20% and 30% of an audio
def augmentMRS_data(pos_mrs,audioData,binaryLabels):
    additionData=[]
    additionalLabels=[]
    for i in range(len(pos_mrs)):
        perc=int((audioData[i].shape[1]*10)/100)
        for j in range(1,4):
            additionData.append(audioData[i][:,perc*j:audioData[i].shape[1]])
            additionalLabels.append(1)
    newData=audioData+additionData
    newlabels=binaryLabels+additionalLabels
    return newData,newlabels

def manipulateNoiseInjection(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data

def manipulateShiftingTime(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift    
    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data

def manipulateChangingPitch(data, sampling_rate, pitch_factor):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

def manipulateChangingSpeed(data, speed_factor):
    return librosa.effects.time_stretch(data, speed_factor)

def augmentMRS(posMRS_training):
    audioDir="/vol/work3/berhe/MRS_Detection/SceneAudio/"
    for i in posMRS_training:
        #print("Scene :{}".format(i+1))
        sys.stdout.write('Scene %s\r' % str(i+1))
        sys.stdout.flush()
        audioFile=audioDir+"Scene_"+str(i+1)+".wav"
        y, sr=librosa.load(audioFile)
        augmented_data=manipulateNoiseInjection(y,noise_factor=0.005)
        soundfile.write(audioDir+"AugmentedMRS/Noise_Scene_"+str(i+1)+".wav", augmented_data, sr)
        augmented_data=manipulateChangingSpeed(y,speed_factor=2)
        soundfile.write(audioDir+"AugmentedMRS/Speed_Scene_"+str(i+1)+".wav", augmented_data, sr)
        augmented_data=manipulateShiftingTime(y,sampling_rate=sr, shift_max=30, shift_direction="left")
        soundfile.write(audioDir+"AugmentedMRS/ShiftLeft_Scene_"+str(i+1)+".wav", augmented_data, sr)
        augmented_data=manipulateChangingPitch(y,sampling_rate=sr,pitch_factor=5)
        soundfile.write(audioDir+"AugmentedMRS/Pitch_Scene_"+str(i+1)+".wav", augmented_data, sr)
    print("finished")
    
#Reshaping Data to fit models


def sameLengthVgg(vggfeat):
    sceneLength=[i.shape[0] for i in vggfeat]
    minLength=min(sceneLength)
    reshapedData=[]
    for i in vggfeat:
        reshapedData.append(i[i.shape[1]-minLength:,:])
    reshapedData=np.array(reshapedData)
    return reshapedData

def generateSameLength(audioData,length=61):
    reshapedData=[]
    for i in audioData:
        reshapedData.append(i[:,i.shape[1]-length:])
    reshapedData=np.array(reshapedData)
    return reshapedData

def augment_reshape(audioData,length=2000):
    augmentZero=np.zeros(audioData[0].shape[0])
    d=np.zeros((audioData[0].shape[0],2000))
    reshapedData=[]
    for i in audioData:
        dataArray=i
        #print(dataArray.shape)
        try:
            if dataArray.shape[1] < length:
                for j in range(dataArray.shape[1]+1,length+1):
                    dataArray=np.concatenate((dataArray,augmentZero[:,None]),axis=1)
                reshapedData.append(dataArray)
            else:
                reshapedData.append(i[:,i.shape[1]-length:])
        except:
            reshapedData.append(d)
    reshapedData=np.array(reshapedData)
    return reshapedData

def loadAllFeat():
    with open ("MFCC_features_as_list","rb") as file:
        mfcc=pickle.load(file)
    with open ("MelSpectogram_features_as_list","rb") as file:
        melspectogram=pickle.load(file)
    with open ("Tempo_features_as_list","rb") as file:
        tempo=pickle.load(file)
    with open ("spectral_centroid_features_as_list","rb") as file:
        spec_centroids=pickle.load(file)
    with open ("/vol/work3/berhe/MRS_Detection/vggish/models/research/audioset/vggish/audioEmbedding_VGGish_new","rb") as file:
    #with open ("/vol/work3/berhe/MRS_Detection/vggish/models/research/audioset/vggish/audioEmbedding_ALL_Scenes","rb") as file:
        vggFeat=pickle.load(file)
        

    return mfcc,melspectogram,tempo,spec_centroids,vggFeat

def loadAugmentedData():
    with open ("MFCC_features_as_listAugmented","rb") as file:
        mfccA=pickle.load(file)
    with open ("MelSpectogram_features_as_listAugmented","rb") as file:
        melspectogramA=pickle.load(file)
    with open ("Tempo_features_as_listAugmented","rb") as file:
        tempoA=pickle.load(file)
    with open ("Spectogram_centroid_features_as_listAugmented","rb") as file:
        spec_centroidsA=pickle.load(file)
    with open ("/vol/work3/berhe/MRS_Detection/vggish/models/research/audioset/vggish/audioEmbedding_VGGish_new_Augmented","rb") as file:
        vggFeatA=pickle.load(file)

    return mfccA,melspectogramA,tempoA,spec_centroidsA,vggFeatA

def augment_reshapeVgg(audioData,length=138):
    augmentZero=np.zeros(audioData[0].shape[1])
    reshapedData=[]
    for i in audioData:
        dataArray=i
        if dataArray.shape[0] < length:
            for j in range(dataArray.shape[0]+1,length+1):
                dataArray=np.concatenate((dataArray,augmentZero[None,:]),axis=0)
            #print(dataArray.shape,dataArray[::-1].shape)
            reshapedData.append(dataArray)
        else:
            #print("here",i[i.shape[0]-avgLength:,:].shape)
            reshapedData.append(i[i.shape[0]-length:,:])
    reshapedData=np.array(reshapedData)
    return reshapedData


def split_Data(reshapedData,binaryLabels,testSize=0.25):
    X = reshapedData
    y = np.array(binaryLabels)

    # Encode the classification labels

    encoder = LabelEncoder()
    encoder.fit(y)
    yy = encoder.transform(y)
    #le = LabelEncoder()
    #yy = to_categorical(le.fit_transform(y))
    x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=testSize, random_state = 42)
    print(yy.shape,y_test.shape)

    return x_train, x_test, y_train, y_test
    
def load_musicFeat():
    with open ("Tempogram_features_as_list",'rb') as file:
        tempogramFeat=pickle.load(file)
    with open ("Chromagram_features_as_list",'rb') as file:
        chromagramFeat=pickle.load(file)
    tempogram=augment_reshape(tempogramFeat,2000)
    chromagram=augment_reshape(chromagramFeat,2000)
    
    return tempogram,chromagram
    
def loadTextEmbeding():
    with open ("Data/SceneSentenceEmbeddings_BERT_List",'rb') as file:
        textEmbed=pickle.load(file)
    with open ("Data/SceneSummarySentenceEmbeddings_BERT_List",'rb') as file:
        textSummaryEmbed=pickle.load(file)
    reshapedtextEmb=reshapeTextEmb(textEmbed,length=115)
    reshapedSummaryEmb=reshapeTextEmb(textSummaryEmbed,51)
    return reshapedtextEmb,reshapedSummaryEmb

def load_Pitch_frequency():
    with open("Pitch_11_All",'rb') as file:
        dataAll=pickle.load(file) 
    with open("Pitch_5_Frequency",'rb') as file:
        datafreq=pickle.load(file) 
    
    datafreq=augment_reshapeVgg(datafreq,2000)
    dataAll=augment_reshapeVgg(dataAll,2000)
    
    return datafreq, dataAll

def reshapeTextEmb(textEmbed,length=115):
    augmentZero=np.zeros(textEmbed[0][0].shape[0])
    reshapedData=[]
    for i in textEmbed:
        dataList=i
        if len(dataList) < length:
            while len(dataList) < length:
                dataList.append(augmentZero)
        else:
            dataList=dataList[len(dataList)-length:]
        reshapedData.append(np.array(dataList))
    reshapedData=np.array(reshapedData)
    return reshapedData

def loadTrainingTest(feature):
    if feature=="mfcc" or feature=="m":
        with open ("Data/TrainingData/mfcc_x_training","rb") as file:
            x_train=pickle.load(file)
        with open ("Data/TrainingData/mfcc_y_train","rb") as file:
            y_train=pickle.load(file)
        with open ("Data/TestData/mfcc_x_test","rb") as file:
            x_test=pickle.load(file)
        with open ("Data/TestData/mfcc_y_test","rb") as file:
            y_test=pickle.load(file)
        
        return x_train,y_train,x_test,y_test
    
    elif feature=="melspectogram" or feature=="mel":
        with open ("Data/TrainingData/mel_x_training","rb") as file:
            x_train=pickle.load(file)
        with open ("Data/TrainingData/mel_y_train","rb") as file:
            y_train=pickle.load(file)
        with open ("Data/TestData/mel_x_test","rb") as file:
            x_test=pickle.load(file)
        with open ("Data/TestData/mel_y_test","rb") as file:
            y_test=pickle.load(file)
        
        return x_train,y_train,x_test,y_test
    
    elif feature=="summary" or feature=="sum":
        with open ("Data/TrainingData/mel_x_training","rb") as file:
            x_train=pickle.load(file)
        with open ("Data/TrainingData/mel_y_train","rb") as file:
            y_train=pickle.load(file)
        with open ("Data/TestData/mel_x_test","rb") as file:
            x_test=pickle.load(file)
        with open ("Data/TestData/mel_y_test","rb") as file:
            y_test=pickle.load(file)
        return x_train,y_train,x_test,y_test
    
    elif feature=="tempo" or feature=="tempogram":
        with open ("Data/TrainingData/tempo_x_training","rb") as file:
            x_train=pickle.load(file)
        with open ("Data/TrainingData/tempo_y_train","rb") as file:
            y_train=pickle.load(file)
        with open ("Data/TestData/tempo_x_test","rb") as file:
            x_test=pickle.load(file)
        with open ("Data/TestData/tempo_y_test","rb") as file:
            y_test=pickle.load(file)
        return x_train,y_train,x_test,y_test
    
    elif feature=="vggish" or feature=="v":
        with open ("Data/TrainingData/vggish_q_x_training","rb") as file:
            x_train=pickle.load(file)
        with open ("Data/TrainingData/vggish_q_y_train","rb") as file:
            y_train=pickle.load(file)
        with open ("Data/TestData/vggish_q_x_test","rb") as file:
            x_test=pickle.load(file)
        with open ("Data/TestData/vggish_q_y_test","rb") as file:
            y_test=pickle.load(file)
        y_train=np.array(y_train)
        return x_train,y_train,x_test,y_test
    

def loadDataFeaures(feature):
    dataset_Df=pd.read_csv("Scene_Dataset_Normalized.csv")
    sceneLabels=dataset_Df.MRS.tolist()
    binaryLabels=[i if i==0 else 1 for i in sceneLabels]
    if feature=="mfcc" or feature[0:2]=="mf":
        with open ("MFCC_features_as_list","rb") as file:
            audioData=pickle.load(file)
        reshapedData=augment_reshape(audioData,length=2000)
        x_train, x_test, y_train, y_test=split_Data(reshapedData,binaryLabels,testSize=0.25)
        print('mfcc loaded')
        return x_train, x_test, y_train, y_test
    if feature=="mel" or feature=="melspectogram":
        with open ("MelSpectogram_features_as_list","rb") as file:
            audioData=pickle.load(file)
        reshapedData=augment_reshape(audioData,length=2000)
        x_train, x_test, y_train, y_test=split_Data(reshapedData,binaryLabels,testSize=0.25)
        print('mel loaded')
        return x_train, x_test, y_train, y_test
    if feature=='vggish' or feature=='v':
        with open ("/vol/work3/berhe/MRS_Detection/vggish/models/research/audioset/vggish/audioEmbedding_VGGish_new","rb") as file:
    #with open ("/vol/work3/berhe/MRS_Detection/vggish/models/research/audioset/vggish/audioEmbedding_ALL_Scenes","rb") as file:
            vggFeat=pickle.load(file)
        reshapedData=augment_reshapeVgg(vggFeat,length=138)
        x_train, x_test, y_train, y_test=split_Data(reshapedData,binaryLabels,testSize=0.25)
        print('Vggish loaded')
        return x_train, x_test, y_train, y_test
    if feature=="tempo" or feature=="tempogram":
        with open ("Tempogram_features_as_list",'rb') as file:
            tempogramFeat=pickle.load(file)
        print('tempo loaded')
        reshapedData=augment_reshape(tempogramFeat,2000)
        x_train, x_test, y_train, y_test=split_Data(reshapedData,binaryLabels,testSize=0.25)
        return x_train, x_test, y_train, y_test
    if feature=="chromo" or feature=="chromagram":
        with open ("Chromagram_features_as_list",'rb') as file:
            chromagramFeat=pickle.load(file)
        print('chromagram loaded')
        reshapedData=augment_reshape(chromagramFeat,length=2000)
        x_train, x_test, y_train, y_test=split_Data(reshapedData,binaryLabels,testSize=0.25)
        return x_train, x_test, y_train, y_test
        
    if feature=="trans" or feature=="trans_embeding":
        with open ("Data/SceneSentenceEmbeddings_BERT_List",'rb') as file:
            textEmbed=pickle.load(file)
        reshapedtextEmb=reshapeTextEmb(textEmbed,length=115)
        print('Trans loaded')
        x_train, x_test, y_train, y_test=split_Data(reshapedtextEmb,binaryLabels,testSize=0.25)
        return x_train, x_test, y_train, y_test
    
    if feature=='summary':
        with open ("Data/SceneSummarySentenceEmbeddings_BERT_List",'rb') as file:
            textSummaryEmbed=pickle.load(file)
        reshapedSummaryEmb=reshapeTextEmb(textSummaryEmbed,51)
        print('Summary loaded')
        x_train, x_test, y_train, y_test=split_Data(reshapedSummaryEmb,binaryLabels,testSize=0.25)
        return x_train, x_test, y_train, y_test
    
    if feature=='pitch':
        with open ("Pitch_Tracking_features_as_list",'rb') as file:
            pitchFeat=pickle.load(file)
        pitchFeat=augment_reshape(pitchFeat,2000)
        print('Pitch Loaded')
        x_train, x_test, y_train, y_test=split_Data(pitchFeat,binaryLabels,testSize=0.25)
        return x_train, x_test, y_train, y_test
    if feature=='magnitude':
        with open ("Magnitude_features_as_list",'rb') as file:
            magnitude=pickle.load(file)
        magnitude=augment_reshape(magnitude,2000)
        print('Magnitude Loaded')
        x_train, x_test, y_train, y_test=split_Data(magnitude,binaryLabels,testSize=0.25)
        return x_train, x_test, y_train, y_test
    if feature=="Pitch_5_Freq":
         with open("Pitch_5_Frequency",'rb') as file:
             pitchAll=pickle.load(file) 
         pitchAll=augment_reshapeVgg(pitchAll,2000)
         print('Pitch Loaded')
         x_train, x_test, y_train, y_test=split_Data(pitchAll,binaryLabels,testSize=0.25)
         return x_train, x_test, y_train, y_test
    if feature=="Pitch_11_All" or '11' in feature:
        with open("Pitch_11_All",'rb') as file:
            pitchfreq=pickle.load(file)
        pitchfreq=augment_reshapeVgg(pitchfreq,2000)
        print('Pitch Loaded')
        x_train, x_test, y_train, y_test=split_Data(pitchfreq,binaryLabels,testSize=0.25)
        return x_train, x_test, y_train, y_test
    else:
        print("no Features {}".format(feature))
