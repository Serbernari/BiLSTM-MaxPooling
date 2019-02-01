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
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from scipy.sparse import csr_matrix
from gensim.models import FastText, KeyedVectors

### loading data ###

labels = []
texts = []
         
data = pd.read_csv("intents_new.csv",sep = ',', header = None, names = ['type', 'text'])
data['type'] = data['type'].map({'a': 1, 'b': 2,
                                 'c':3, 'd':4,
                                 'e':5,'f':6, 'g':7,
                                 'h':8,'o':9, 'p':10,
                                 'q':11, 'w':12, 'e':13 , 'r':14,
                                 't':15,'y':16,
                                 'n':17,'m':18})
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
# Test train split
x_train, x_test, y_train, y_test = train_test_split(phrases_emb, classes, test_size = 0.2, random_state = 42, stratify=classes)

### embeddings ###

modelFT = KeyedVectors.load('187//model.model')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

num_words = min(max_features, len(word_index)) + 1
print(num_words)

embedding_dim = 300
embedding_matrix = np.zeros((num_words, embedding_dim))
for word, i in word_index.items():
    if len(word) == 1:
        word = '  '+word+'  '
    embedding_vector = modelFT.get_vector(word)
    embedding_matrix[i] = embedding_vector

#### Model coustruction ###

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
indices = np.arange(0, len(y_train), 1, dtype=np.int32)
np.random.shuffle(indices)
history = model.fit(x_train[indices], y_train[indices]-1, epochs=26, batch_size=batch_size, verbose=1, validation_split=0.1)


### F1 metriс ###

class_list = [i for i in range(18)]
output = model.predict(x_test)
#print(output)
y_pred = [np.argmax(output[i]) + 1 for i in range(len(output))]
print('pred:',y_pred[:10])
print('real:',y_test[:10])
f1_macro = f1_score(y_test, y_pred, labels=class_list, average = 'macro' )
f1_micro = f1_score(y_test, y_pred, labels=class_list, average = 'micro' )
print('F1_macro = ',f1_macro)
print('F1_micro = ',f1_micro)
 

### Plot drawing ###

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

### Model saving ###

model_json = model.to_json()
json_file = open("BiLSTM_MaxPooling", "w")
json_file.write(model_json)
json_file.close()

model.save_weights("BiLSTM_MaxPooling.h5")
