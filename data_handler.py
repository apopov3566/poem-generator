import string, re


def split_sequence(sequence):
    sequence = sequence.lower().translate(str.maketrans('', '','\",.!?;'))
    sequence = sequence.replace("\n", " \n ").split(" ")
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


def get_corpus(location, split_by_line = False):
    f = open(location , 'r')
    if not split_by_line:
        poems = f.read().split("\n\n")

        for i in range(len(poems)):
            poems[i] = poems[i][poems[i].find("\n")+1:]
            poems[i] = split_sequence(poems[i])

        print(poems[0])
        poems, tokens = tokenize(poems)
        return poems, tokens

    elif split_by_line:
        lines = f.read().split("\n")

        lines = list(filter(lambda x: not re.match(r'^\s*$', x), lines))
        lines = list(filter(lambda x: not re.match(r'^ *[0-9]*$', x), lines))

        for i in range(len(lines)):
            lines[i] = split_sequence(lines[i])
            lines[i].append("\n")

        print(lines)
        lines, tokens = tokenize(lines)
        return lines, tokens


#print(string.punctuation)
#print(get_corpus("data/shakespeare.txt", "line"))
