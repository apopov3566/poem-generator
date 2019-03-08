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

    cb = [callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=4, verbose=0, mode='auto')]

    model = Sequential()
    model.add(layers.LSTM(200, input_shape=(X.shape[1], X.shape[2])))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(v_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    print("train!")
    history = model.fit(X, y, epochs = 100, verbose = 2, callbacks = cb)
    print("done!")
    model.save('model.tmp')
    return model

def generate_seq(model, mapping, seq_length, seed_text, n_chars):
	in_text = seed_text
	for _ in range(n_chars):
		encoded = [mapping[char] for char in in_text]
		encoded = preprocessing.sequence.pad_sequences([encoded], maxlen=seq_length, truncating='pre')
		encoded = utils.to_categorical(encoded, num_classes=len(mapping))
		yhat = model.predict_classes(encoded, verbose=0)
		out_char = ''
		for char, index in mapping.items():
			if index == yhat:
				out_char = char
				break
		in_text += char
	return in_text

if __name__ == '__main__':
    X, y, v_size, char_to_token, token_to_char = get_LSTM_data("data/shakespeare.txt", True, 1)
    model = train_LSTM(X, y, v_size)
    #model = models.load_model('model.tmp')
    print(generate_seq(model, char_to_token, 40, "shall i compare thee to a summer's day?\n" , 1000))
