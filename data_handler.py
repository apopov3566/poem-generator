import string, re
import random
import numpy as np
from keras import *

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
    return tokenized, token_dict


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

        #print(poems[0])
        poems, tokens = tokenize(poems)
        f.close()
        return poems, tokens

    elif split_by_line:
        lines = f.read().split("\n")

        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        lines = list(filter(lambda x: not re.match(r'^ *[0-9]*$', x), lines))

        for i in range(len(lines)):
            lines[i] = split_sequence(lines[i], include_newlines)
            if include_newlines:
                lines[i].append("\n")

        #print(lines)
        lines, tokens = tokenize(lines)

        return lines, tokens

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
