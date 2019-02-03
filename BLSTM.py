import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import *
from keras.utils.np_utils import to_categorical

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_validate   
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score

from scipy.sparse import csr_matrix
from gensim.models import FastText, KeyedVectors

### loading data ###

labels = []
texts = []
         
data = pd.read_csv("intents_new.csv",sep = ',', header = None, names = ['type', 'text'])
data['type'] = data['type'].map({'Как заблокировать сим-карту': 1, 'Верните деньги': 2,
                                 'Переведите на человека':3, 'Почему вы подключаете услуги без моего согласия?':4,
                                 'Не приходят смс':5,'Как поменять домашний регион?':6, 'Что за  тарифный план?':7,
                                 'Как заказать обратный звонок?':8,'Как узнать кодовое слово?':9, 'Позвоните мне!':10,
                                 'сим карта не работает':11, 'Тариф без интернета':12, 'Что со связью?':13 , 'Не отправляются смс':14,
                                 'Почему меня заблокировали?':15,'Как разблокировать сим-карту.':16,
                                 'Как сменить номер':17,'Статус заявки':18})
data.text=data.text.astype(str)
classes = np.array(data['type'])
texts = np.array(data['text'])

### tokenizer ###

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(texts)
phrases_emb = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
max_features = len(word_index)+1
lenths =[]
for sz in phrases_emb:
    lenths.append(len(sz))

max_len = max(lenths)
phrases_emb = sequence.pad_sequences(phrases_emb, maxlen=max_len)

### embeddings ###

modelFT = KeyedVectors.load('187//model.model')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

num_words = min(max_features, len(word_index)) + 1

embedding_dim = 300
embedding_matrix = np.zeros((num_words, embedding_dim))

for word, i in word_index.items():
    if len(word) == 1:
        word = '  '+word+'  '
    embedding_vector = modelFT.get_vector(word)
    embedding_matrix[i] = embedding_vector

#### Model coustruction ###

seed = 42
np.random.seed(seed)

F1_data = open('F1_data_file.txt', 'w')

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

for train, test in kfold.split(phrases_emb, classes): 
  # create model
    model = Sequential()
    model.add(Embedding(num_words,
                    embedding_dim,
                    input_length=max_len, trainable = False, weights = [embedding_matrix]))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))
    model.add(Bidirectional(CuDNNLSTM(32, return_sequences=True)))
    model.add(Dropout(0.25))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(units=18, activation='softmax'))
    model.summary()

    model.compile(loss = 'sparse_categorical_crossentropy', optimizer='adam' ,metrics = ['accuracy']) 
    

    batch_size = 32 #mini batch is preferable
    indices = np.arange(0, len(train), 1, dtype=np.int32)
    np.random.shuffle(indices)
    history = model.fit(phrases_emb[train], classes[train]-1, epochs=26, batch_size=batch_size, verbose=1, validation_split=0.1)
    
    #model F1 evaluate
    class_list = [i for i in range(18)]
    output = model.predict(phrases_emb[test])
    pred = [np.argmax(output[i]) + 1 for i in range(len(output))]

    f1_macro = f1_score(classes[test], pred, labels=class_list, average = 'macro' )
   # f1_micro = f1_score(classes[test], pred, labels=class_list, average = 'micro' )
    print('F1_macro = ',f1_macro)
    F1_data.write(str(f1_macro))
    F1_data.write("\n")
   
F1_data.close()

### Model saving ###

model_json = model.to_json()
json_file = open("BiLSTM_MaxPooling", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("BiLSTM_MaxPooling.h5")