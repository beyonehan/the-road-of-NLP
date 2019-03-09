
import pandas as pd
import numpy as np

import jieba

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint


train = pd.read_csv('comment-classification/train.csv')
test = pd.read_csv('comment-classification/test.csv')
vaild = pd.read_csv('comment-classification/validationset.csv')

# train = train[0:10000]

max_features = 20000
maxlen = 100

# list_sentences_train = train["content"].fillna("CVxTz").values

def cut(string): return list(jieba.cut(string))

def get_dummies(data,name): return pd.get_dummies(data[name], prefix=name)


def handle_dummies(data):
    columns_names = data.columns.values[2:]

    concat_list = [data]

    for name in columns_names:
        print(name)
        concat_list.append(get_dummies(data, name))

    return pd.concat(concat_list, axis=1)

train =  handle_dummies(train).iloc[:40000]

test  =  handle_dummies(test)

all_contents = train['content'].tolist()
all_contents = [' '.join(cut(s)) for s in all_contents]


y = train[train.columns[-80:]].values


list_sentences_test = test['content'].fillna("CVxTz").values

# tokenizer = text.Tokenizer(num_words=max_features)
# tokenizer.fit_on_texts(list(list_sentences_train))
# list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
# list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
# print (list_tokenized_train)
# X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
# X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(all_contents)
list_tokenized_train = tokenizer.texts_to_sequences(all_contents)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_model():
    embed_size = 128
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size)(inp)
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.05)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(80, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()
batch_size = 32
epochs = 2

file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

early = EarlyStopping(monitor="val_loss", mode="min", patience=20)


callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

model.load_weights(file_path)

y_test = model.predict(X_te)

