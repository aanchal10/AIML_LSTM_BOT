# -*- coding: utf-8 -*-
"""
Author ROMA JAIN
"""
import importlib
import warnings
import tensorflow as tf
import keras
from random import shuffle
from keras import backend as K
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation,SpatialDropout1D,Bidirectional
from keras.models import load_model
from keras.models import Model
from keras.layers import Dense, Input, Embedding
from keras.layers import Input, Dense, Dropout, Flatten,Activation,BatchNormalization
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import re
import sys
import pickle
import re,csv,os,json
import time
import codecs,argparse
from keras.preprocessing import text
from keras.optimizers import Adam
from utils import pickle_dump,remove_stopwords,ner_tagging,training

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
def set_keras_backend(backend):
    from keras import backend as K
    print(K.backend())

    if K.backend() != backend:
        os.environ['KERAS_BACKEND'] = backend
        importlib.reload(K)
        assert K.backend() == backend

set_keras_backend("tensorflow")

ap = argparse.ArgumentParser()
ap.add_argument("-flag", "--flag", required=True,help="predict or train")

def sensitivity(y_true, y_pred):
    '''sensitivity as one metric for checking accuracy of the model (unbalanced classes)'''
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def precision(y_true, y_pred):
    from keras import backend as K
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1score(y_true, y_pred):
    from keras import backend as K
    def recall(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))



def load_pretrained(EMEDDING_FILE,t,vocab_size):
    embeddings_index = dict()
    words_not_found=[]
    f = open(EMEDDING_FILE)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    vocab_words = len(embeddings_index)
    print('found %s word vectors' % vocab_words)
    embedding_matrix = np.zeros((vocab_size, 100))
    for word, i in t.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None  and len(embedding_vector) > 0:
            embedding_matrix[i] = embedding_vector
        else:
            pass
    return vocab_words,embedding_matrix,words_not_found
    

def get_XY(expressions,labels):
    lengt=[] 
    distint_words=[]
    for express in expressions:
        words = express.replace('\n','').split()
        leng = len(words)
        lengt.append(leng)
        for word in words:
            distint_words.append(word)
    
    maxlen = max(lengt)
    label_set=set(labels)
    
    tk = text.Tokenizer(oov_token='OOV')
    tk.fit_on_texts(expressions)
    
    distinct_words = set(distint_words)
    print("number of distinct.  ",len(distinct_words))
    X = tk.texts_to_sequences(expressions)
    label_encoder = LabelEncoder()
    integer_encode = label_encoder.fit_transform(labels)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encode.reshape(len(integer_encode), 1)
    Y = onehot_encoder.fit_transform(integer_encoded)
    return X,Y,maxlen,len(distinct_words),len(label_set),label_encoder,tk,integer_encode,distint_words,expressions,labels,onehot_encoder


def model_train(maxlen,X_train,y_train,embedding_matrix,embed,len_distinct,len_label,epochs,batch_size,weight,vector_size,X_test,Y_test):
    import tensorflow as tf
    from keras import backend as K
    set_keras_backend("tensorflow")
    model = Sequential()
    model.add(Embedding(len_distinct, 100, input_length=maxlen,weights=[embedding_matrix],trainable=True))
    model.add(SpatialDropout1D(0.1))
    model.add(Bidirectional(LSTM(64,return_sequences=False))) # was false#was 100 for latst data  for more_data its 200
    model.add(Dropout(0.1))#8 --> 0.2
    model.add(Dense(len_label, activation='softmax'))
    adam=keras.optimizers.Adam(lr=0.0002,beta_1=0.8, beta_2=0.99)#0.0001
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy',f1score,sensitivity,precision])
    print(model.summary())
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,shuffle=True,verbose=1,class_weight=weight,validation_data=(X_test, Y_test))
    return model,hist

def get_class_weights(y, smooth_factor):
    from collections import Counter 
    counter = Counter(y)
    if smooth_factor > 0:
        p = max(counter.values()) * smooth_factor
        for k in counter.keys():
            counter[k] += p
    majority = max(counter.values())
    return {cls: float(majority / count) for cls, count in counter.items()}



dir_path = os.path.dirname(os.path.realpath(__file__))
path_data = os.path.join(dir_path,'data')
path_models = os.path.join(dir_path,'models')
path_nlp_data = os.path.join(dir_path,'data/SRK_TRAINING_DATA.csv')
path_nlp_test_data = os.path.join(dir_path,'data/SRK_TEST_DATA.csv')

with open(path_nlp_data, 'r',encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    training_data = [(row[0],row[1]) for row in reader]
with open(path_nlp_test_data, 'r',encoding="utf-8-sig") as f:
    reader = csv.reader(f)
    test_data = [(row[0],row[1]) for row in reader]
def predict():
    maxlen = 31
    print("Loading model .....and predicting again")
    
    model = load_model(os.path.join(path_models,'LSTM_spatial_SRK.h5'),custom_objects={'sensitivity': sensitivity,'f1score':f1score,'precision':precision})
    with open(os.path.join(path_models,'lstm_tkSRK.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(os.path.join(path_models,'lstm_labelSRK.pickle'), 'rb') as handle:
        label_encoder = pickle.load(handle)
    count=0
    name_of_file_to_write= "validation_LSTM_SRK_GLOVE.csv"
    with open(name_of_file_to_write,"w",encoding="utf-8-sig") as csvfile:
        fieldnames = ['SENTENCE','ACTUAL INTENT','PREDICTED INTENT ON LSTM GLOVE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader() 
        srk_check=[]
        count=0
        for i in test_data:
            test_query = i[0]
            #test_query = remove_stopwords(i[0])
            if(len(test_query) > 0):
                seq = tokenizer.texts_to_sequences([test_query])
                X = sequence.pad_sequences(seq, maxlen=maxlen,padding='post')
                y_pred = model.predict(X)
                y_pred1 = np.argmax(y_pred,axis=1)
                classes = label_encoder.inverse_transform(y_pred1)[0]
                max_index= np.argmax(y_pred)
                if y_pred[0][max_index] < 0.2:
                    classes = 'Garbage'
                if classes == i[1]:
                    count+=1
                di={'SENTENCE' : test_query,'ACTUAL INTENT' : i[1],'PREDICTED INTENT ON LSTM GLOVE' :classes}
                writer.writerow(di)

    print("Accuracy on test data --> " ,(count/len(test_data))*100)
    
    print("DONE VALIDATING....")

def inference():
    import tensorflow as tf
    from keras.models import model_from_json
    import gensim
    model = load_model(os.path.join(path_models,'LSTM_spatial_SRK.h5'),custom_objects={'sensitivity': sensitivity,'f1score':f1score,'precision':precision})
    with open(os.path.join(path_models,'lstm_tkSRK.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open(os.path.join(path_models,'lstm_labelSRK.pickle'), 'rb') as handle:
        label_encoder = pickle.load(handle)
    print("input a sentence")
    test_query = input()
    print(test_query)
    seq = tokenizer.texts_to_sequences([test_query])
    X = sequence.pad_sequences(seq, maxlen=maxlen,padding='post')
    y_pred = model.predict(X)
    y_pred1 = np.argmax(y_pred,axis=1)
    classes = label_encoder.inverse_transform(y_pred1)[0]
    max_index= np.argmax(y_pred)
    print("predicted class is ",classes,'  ',"with prob ",y_pred[0][max_index])

   
def train():
    import tensorflow as tf

    EMEDDING_FILE = 'glove.6B.100d.txt'
    embed = 100
    vector_size=64#100#500
    epochs=17#35
    batch_size=16
    maxlen = 31

    X,y_train,maxlen,len_distinct,len_label,label_encoder,tk,labels,distint_words,expressions,label,onehot_encoder= get_XY([i[1] for i in training_data],[i[0] for i in training_data])
    len_distinct=3000#for added small talk!!!!#3045#2930#its 3600-more data
    len_vocab_words,embedding_matrix,words_not_found = load_pretrained(EMEDDING_FILE,tk,len_distinct)
    weight = get_class_weights(labels,0.1)
    X_train = sequence.pad_sequences(X, maxlen=maxlen,padding='post',truncating='post')
    X_test=X_train
    y_test=y_train
    from sklearn.utils import shuffle
    X_test, y_test = shuffle(X_test, y_test)
    with open(os.path.join(path_models,'lstm_tkSRK.pickle'), 'wb') as handle:
        pickle.dump(tk, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(path_models,'lstm_labelSRK.pickle'), 'wb') as handle:
        pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    model,history = model_train(maxlen,X_train,y_train,embedding_matrix,embed,len_distinct,len_label,epochs,batch_size,weight,vector_size,X_test,y_test)
    model.save('LSTM_spatial_SRK.h5')
    print(history.history['acc'])
    print("validating ---------->")
    print(history.history['val_acc'])
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    f1_score = history.history['f1score']
    val_f1score=history.history['val_f1score']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure(1)
    plt.show()
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.figure(2)
    plt.show()
    plt.plot(epochs, loss, 'b', label='Training f1score')
    plt.plot(epochs, val_loss, 'r', label='Validation f1score')
    plt.title('Training and validation f1score')
    plt.legend()
    plt.figure(3)
    plt.show()



if __name__ == '__main__':
    args = vars(ap.parse_args())
    if args['flag'] == 'train':
        train()
    if args['flag'] == 'test':
        predict()
    if args['flag'] == 'infer':
        inference()
    else:
        print(" wrong arugment !! use either 'train' or 'test' ")




