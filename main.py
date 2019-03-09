from data_handler import get_corpus, get_LSTM_data
from HMM import unsupervised_HMM
import numpy as np
from keras import *

def run_HMM(n_states, N_iters):
    corpus, detoken = get_corpus("data/shakespeare.txt", False)
    HMM = unsupervised_HMM(corpus, n_states, N_iters)

    for i in range(10):
        output, states = HMM.generate_emission(200)
        outstr = ""
        for token in output:
            outstr += (detoken[token] + " ")
        print(outstr)


def train_LSTM(X, y, v_size):

    cb = [callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')]

    model = Sequential()
    model.add(layers.LSTM(100, input_shape=(X.shape[1], X.shape[2])))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(v_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    print("train!")
    history = model.fit(X, y, epochs = 100, verbose = 2, callbacks = cb)
    print("done!")
    model.save('model.tmp')
    return model

def generate_seq(model, char_to_token, token_to_char, seed, n_chars):
    output = seed
    for _ in range(n_chars):
        tokenized = [char_to_token[char] for char in output]
        tokenized = preprocessing.sequence.pad_sequences([tokenized], maxlen=40, truncating='pre')
        tokenized = utils.to_categorical(tokenized, num_classes=len(char_to_token))
        predicted = model.predict_classes(tokenized, verbose=0)
        output += token_to_char[predicted[0]]
    return output

if __name__ == '__main__':
    X, y, v_size, char_to_token, token_to_char = get_LSTM_data("data/shakespeare.txt", True, 5)
    #model = train_LSTM(X, y, v_size)
    model = models.load_model('m1.model')
    print(generate_seq(model, char_to_token, token_to_char, "shall i compare thee to a summer's day?\n" , 1000))
