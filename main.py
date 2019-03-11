from data_handler import get_corpus, get_LSTM_data, get_word_LSTM_data, get_corpus_syllable, infer_stress
from HMM import unsupervised_HMM
import numpy as np
import keras
from keras import *
import sys
import utils
from joblib import dump, load

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


def run_HMM_meter(n_states, N_iters):
    corpus, detoken, reverse_dict = get_corpus("data/shakespeare.txt", split_by_line = True)

    token_to_syllable = get_corpus_syllable("data/Syllable_dictionary.txt", reverse_dict, detoken)
    token_to_stress = infer_stress(token_to_syllable, corpus)

    for i,j in token_to_stress.items():
        print(i,j)

def run_HMM_haiku(n_states, N_iters):
    corpus, detoken, reverse_dict = get_corpus("data/shakespeare.txt", split_by_line = False)

    token_to_syllable = get_corpus_syllable("data/Syllable_dictionary.txt", reverse_dict, detoken)

    HMM = load('hmm_haiku.model')
    # HMM = unsupervised_HMM(corpus, n_states, N_iters)
    # dump(HMM, 'hmm_haiku.model')

    output, states = HMM.generate_emission(200)
    haiku = utils.syllabic_formatter(token_to_syllable, output)

    while(haiku is None):
        output, states = HMM.generate_emission(200)
        haiku = utils.syllabic_formatter(token_to_syllable, output)

    for line in haiku:
        outstr = ""
        for token in line:
            outstr += (detoken[token] + " ")
        print(outstr,"\n")


def train_LSTM(X, y, v_size, temp):

    cb = [callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')]

    model = Sequential()
    model.add(layers.LSTM(100, input_shape=(X.shape[1], X.shape[2])))
    model.add(layers.Dropout(0.2))
    model.add(layers.Lambda(lambda x: x/temp))
    model.add(layers.Dense(v_size, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
    print(model.summary())
    print("train!")
    history = model.fit(X, y, epochs = 100, verbose = 2, callbacks = cb)
    print("done!")
    model.save('model.tmp')
    return model


def train_word_embedded_LSTM(X, y, vocab_size, prev_words=22):
    ''' Trains a LSTM using word embeddings. '''

    # Setup the model.
    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size, output_dim=32, input_length=prev_words))
    model.add(layers.LSTM(150, dropout=0.3))
    model.add(layers.Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Train and save the model.
    cb = [callbacks.EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0, mode='auto')]
    history = model.fit(X, y, epochs=150, verbose=2, callbacks=cb)
    model.save('lstm_embedded_150_15_30.model')

    return model


def generate_seq(model, char_to_token, token_to_char, seed, n_lines=14):
    output = seed
    lines = 0

    while lines < n_lines:
        tokenized = [char_to_token[char] for char in output]
        tokenized = preprocessing.sequence.pad_sequences([tokenized], maxlen=40, truncating='pre')
        tokenized = keras.utils.to_categorical(tokenized, num_classes=len(char_to_token))
        predicted = model.predict_classes(tokenized, verbose=0)
        output += token_to_char[predicted[0]]

        # Check for a line end.
        if output[-1] == '\n':
            lines += 1

    return output


def generate_word_seq(model, word_to_token, token_to_word, seed, n_lines=13, prev_words=22):
    output = seed
    lines = 0

    # Convert the seed to a list of tokens.
    tokenized = [word_to_token[word] for word in seed.split(' ')]

    # Generate n_lines of text.
    while lines < n_lines:
        tokenized = preprocessing.sequence.pad_sequences([tokenized], maxlen=prev_words, truncating='pre')
        predicted = model.predict_classes(tokenized, verbose=0)

        # Add the predicted word.
        tokenized = np.append(tokenized, [predicted]);
        output += token_to_word[predicted[0]]

        # Check for a line end.
        if output[-1] == '\n':
            lines += 1
        else:
            output += ' '

    return output[:-1]

if __name__ == '__main__':

    LSTM = (len(sys.argv) >= 2 and '-LSTM' in sys.argv)
    LSTM_embed = (len(sys.argv) >= 2 and '-LSTM_embed' in sys.argv)
    HMM_simple = (len(sys.argv) >= 2 and '-HMM_simple' in sys.argv)
    HMM_rhyme = (len(sys.argv) >= 2 and '-HMM_rhyme' in sys.argv)
    HMM_meter = (len(sys.argv) >= 2 and '-HMM_meter' in sys.argv)
    HMM_haiku = (len(sys.argv) >= 2 and '-HMM_haiku' in sys.argv)

    if (len(sys.argv) >= 2 and '-h' in sys.argv or '-help' in sys.argv):
        print("python3 main.py -LSTM -LSTM_adv -HMM_simple -HMM_rhyme -HMM_meter -HMM_haiku -help")
        sys.exit(0)

    if (LSTM):
        print("running LSTM")
        X, y, v_size, char_to_token, token_to_char = get_LSTM_data("data/shakespeare.txt", True, 5)
        #model = train_LSTM(X, y, v_size)
        model1 = train_LSTM(X, y, v_size, 0.25)
        model1.save('lstm_t25.model')

        model2 = train_LSTM(X, y, v_size, 0.75)
        model2.save('lstm_t75.model')

        model3 = train_LSTM(X, y, v_size, 1.50)
        model3.save('lstm_t150.model')

        #model = models.load_model('m1.model')
        print(generate_seq(model, char_to_token, token_to_char, "shall i compare thee to a summer's day?\n"))

    if (LSTM_embed):
        print("running word embedded LSTM")
        prev_words = 15
        X, y, tokens, reverse_dict = get_word_LSTM_data("data/shakespeare.txt",
                                                        include_newlines=True,
                                                        prev_words=prev_words)
        vocab_size = len(tokens) + 1

        # model = train_word_embedded_LSTM(X, y, vocab_size, prev_words=prev_words)
        model = models.load_model('lstm_embedded_150_15_30.model')

        print(generate_word_seq(model, reverse_dict, tokens,
                                "shall i compare thee to a summers day \n",
                                prev_words=prev_words))

    if (HMM_simple):
        print("running HMM simple")
        run_HMM(10, 100)

    if (HMM_rhyme):
        print("running HMM rhyme")
        run_HMM_rhyme(10,100)

    if (HMM_meter):
        print("running HMM meter")
        run_HMM_meter(10,100)

    if (HMM_haiku):
        print("running HMM haiku")
        run_HMM_haiku(10,100)
