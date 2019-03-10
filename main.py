from data_handler import get_corpus, get_LSTM_data
from HMM import unsupervised_HMM
import numpy as np
from keras import *
import sys
import utils

def run_HMM(n_states, N_iters):
    corpus, detoken, _ = get_corpus("data/shakespeare.txt", False)
    HMM = unsupervised_HMM(corpus, n_states, N_iters)

    for i in range(10):
        output, states = HMM.generate_emission(200)
        outstr = ""
        for token in output:
            outstr += (detoken[token] + " ")
        print(outstr,"\n")


def run_HMM_rhyme(n_states, N_iters):
    # rhyme scheme:
    # abab cdcd efef gg
    

    corpus, detoken, reverse_dict = get_corpus("data/shakespeare.txt", False)
    rhyme_sets = utils.produce_rhyme_dictionary(corpus, detoken, reverse_dict)
    
    # this array corresponds to scheme abab cdcd efef gg
    
    rhyme_endings = utils.get_rhyme_based_on_scheme(rhyme_sets, [1,2,1,2,3,4,3,4,5,6,5,6,7,7],variety = True,variety_lb = 3)

    print("rhyming with: ",[detoken[i] for i in rhyme_endings])

    # TODO, implement backwards evolution in HMM
    HMM = unsupervised_HMM(corpus, n_states, N_iters)
    for i in rhyme_endings:
        output, states = HMM.generate_reverse_emmision(i, 20)
        outstr = ""
        for token in output:
            outstr += (detoken[token] + " ")
        print(outstr,"\n")






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
    
    LSTM = (len(sys.argv) >= 2 and '-LSTM' in sys.argv)
    HMM_simple = (len(sys.argv) >= 2 and '-HMMsimple' in sys.argv)
    HMM_rhyme = (len(sys.argv) >= 2 and '-HMM_rhyme' in sys.argv)
    if (len(sys.argv) >= 2 and '-h' in sys.argv or '-help' in sys.argv):
        print("python3 main.py -LSTM -HMMsimple -HMM_rhyme -help")
        sys.exit(0)

    if (LSTM):
        print("running LSTM")
        X, y, v_size, char_to_token, token_to_char = get_LSTM_data("data/shakespeare.txt", True, 5)
        #model = train_LSTM(X, y, v_size)
        model = models.load_model('m1.model')
        print(generate_seq(model, char_to_token, token_to_char, "shall i compare thee to a summer's day?\n" , 1000))

    if (HMM_simple):
        print("running HMM simple")
        run_HMM(10, 100)

    if (HMM_rhyme):
        print("running HMM rhyme")
        run_HMM_rhyme(10,100)
