import string, re
import random
import numpy as np
from keras import *
from utils import *

def split_sequence(sequence, include_newlines, remove_punctuation = True):
    sequence = sequence.lower()
    if remove_punctuation:
        sequence = sequence.translate(str.maketrans('', '','\",.!?;:()'))

    if include_newlines:
        sequence = sequence.replace("\n", " \n ").split(" ")
    else:
        sequence = sequence.replace("\n", " ").split(" ")

    sequence = list(filter(("").__ne__, sequence))
    return sequence


def tokenize(corpus, tokenize_with_stress = False):
    tokenized = []
    reverse_dict = {}

    n = 0
    for sequence in corpus:
        for word in sequence:
            if word not in reverse_dict.keys():
                reverse_dict[word] = n
                n += 1

    for sequence in corpus:
        tokenized.append([])
        for word in sequence:
            tokenized[-1].append(reverse_dict[word])


    token_dict = {v: k for k, v in reverse_dict.items()}
    return tokenized, token_dict, reverse_dict


def get_corpus(location, split_by_line = False, include_newlines = False):
    f = open(location , 'r')
    if not split_by_line:
        poems = f.read().split("\n\n")

        for i in range(len(poems)):
            if i == 0:
                poems[i] = poems[i][poems[i].find("\n")+1:]
            else:
                poems[i] = poems[i][poems[i].find("\n")+1:]
                poems[i] = poems[i][poems[i].find("\n")+1:]
            poems[i] = split_sequence(poems[i], include_newlines)

        # print(poems[0])
        poems, tokens, reverse_dict = tokenize(poems)
        f.close()
        return poems, tokens, reverse_dict

    elif split_by_line:
        lines = f.read().split("\n")

        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        lines = list(filter(lambda x: not re.match(r'^ *[0-9]*$', x), lines))

        for i in range(len(lines)):
            lines[i] = split_sequence(lines[i], include_newlines)
            if include_newlines:
                lines[i].append("\n")

        #print(lines)
        lines, tokens, reverse_dict = tokenize(lines)

        return lines, tokens, reverse_dict

def get_LSTM_data(location, include_newlines = False, skipchars = 0):
    f = open(location , 'r')
    lines = f.read().split("\n")

    lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
    lines = list(filter(lambda x: not re.match(r'^ *[0-9]*$', x), lines))

    for i in range(len(lines)):
        lines[i] = split_sequence(lines[i], include_newlines, False)
        if include_newlines:
            lines[i].append("\n")

    texts = []
    for i in range(0, len(lines), 14):
        texts.append("")
        for line in lines[i:i+14]:
            for word in line:
                if word == "\n":
                    texts[int(i/14)] += "\n"
                else:
                    texts[int(i/14)] += (word + " ")

    full_text = ""
    for text in texts:
        full_text += text

    chars = sorted(list(set(full_text)))
    char_to_token = dict((c, i) for i, c in enumerate(chars))
    token_to_char = {v: k for k, v in char_to_token.items()}

    sequences = []
    labels = []

    for text in texts:
        for start in range(0, len(text) - 40, skipchars + 1):
            sequences.append(text[start:start+40])
            labels.append(text[start+40])

    tokenized_seq = []
    tokenized_labels = []
    for sequence in sequences:
        tokenized_seq.append([])
        for char in sequence:
            tokenized_seq[-1].append(char_to_token[char])
    for label in labels:
        tokenized_labels.append(char_to_token[label])

    v_size = len(char_to_token.keys())

    X = np.array([utils.to_categorical(label, num_classes=v_size) for label in tokenized_seq])
    y = utils.to_categorical(tokenized_labels, num_classes=v_size)
    return X, y, v_size, char_to_token, token_to_char


def get_corpus_syllable(stress_file, reverse_dict, detoken):
    ''' each word has two varieties of syllables. They can either take on 
    regular R amount of syllables, which is R = [a,b,c...], and then they can
    take on ending syllables of E = [a,b,c,d....]
    
    get_corpus_stress builds a token dicionary from token -> syllable sequence
    in the format of 
    {toke:dictionary}
    where dictionary ={R:[...]; E:[...]}

    '''
    import difflib

    f = open(stress_file , 'r')
    lines = f.read().split("\n")
    token_to_stress = dict()

    for i in lines:
        sequence = list(i.split(" "))
        word = sequence[0]
        worddict = dict()

        ending_char = "E"
        regular_char = "R"
        E = []
        R = []
        for i in sequence[1:]:
            if (ending_char in i):
                E.append(int(i.strip(ending_char)))
            elif (ending_char not in i):
                R.append(int(i))
            else:
                print("should not reach here")
                assert(False)

        worddict = {regular_char: R, ending_char:E}

        # an error that we have is not all words in the syllable dictionary 
        # is in the corpus, thus we will match the corpus to the closest
        # in the dictionary, then we will assign the syllable count to the
        # closest matching work in corpus
        try:
            word_token = reverse_dict[word]
        except KeyError as e:
            close_match = difflib.get_close_matches(word,reverse_dict.keys(), n = 1).pop()
            word_token = reverse_dict[close_match]
            print("could not find: ", word, ".using closest match: ", close_match)

        token_to_stress[word_token] = worddict


    detoken_keys = set(detoken.keys())
    token_to_stress_keys = set(token_to_stress.keys())
    print(detoken_keys - token_to_stress_keys)
    print(token_to_stress_keys - detoken_keys)
    assert(detoken_keys == token_to_stress_keys)













