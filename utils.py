import pip
import numpy as np
import random

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


def recursive_flatten(lst):
    newlst = []
    if type(lst) is list:
        for i in lst:
            newlst.extend(recursive_flatten(i))
        return newlst
    else:
        return [lst]


def produce_rhyme_dictionary(corpus, detoken, reverse_dict):
    ''' takes in any dimension corpus
    and flattens it recursively, then
    generates a list of sets of rhyme classes
    '''
    # install rhyme packages
    try:
        import pronouncing
    except ImportError as e:
        utils.install("pronouncing")


    flat_corpus = recursive_flatten(corpus)
    vocab = set()
    for i in flat_corpus:
        vocab.add(detoken[i])
    distinct_word_count = len(vocab)

    rhyme_sets = []
    # keeps going while we haven't exhausted the set
    while(len(vocab) > 0):
        curword = vocab.pop()
        # produces all rhyming words
        cur_rhyme_set = set(pronouncing.rhymes(curword))
        
        cur_rhyme_set = vocab.intersection(cur_rhyme_set)

        cur_rhyme_set.add(curword)

        tokenized_rhymes = set([reverse_dict[i] for i in cur_rhyme_set])

        rhyme_sets.append(tokenized_rhymes)
        vocab -= cur_rhyme_set

    # ensure translation integrity
    dic_count = sum([len(s) for s in rhyme_sets])

    assert(dic_count == distinct_word_count)
    return rhyme_sets


def get_rhyme_based_on_scheme(rhyme_sets, scheme, variety = False, variety_lb = 0, randomize = True):
    ''' the scheme passed in should be in the format of
    [1,2,1,2,3,4,3,4,5,6,5,6,7,7]

    for the example of  
    abab cdcd efef gg
    
    variety demands that we pick rhyme sets with more than variety_lb words

    '''

    # randomly generate indicies that map rhyme scheme to sets
    
    scheme_to_set = dict()
    used = set()

    for i in scheme:
        set_idx = random.randint(0, len(rhyme_sets))

        if (variety):
            while(len(rhyme_sets[set_idx]) < variety_lb):
                set_idx = random.randint(0, len(rhyme_sets))

        while(set_idx in used):
            set_idx = random.randint(0, len(rhyme_sets))

        scheme_to_set[i] = set_idx
        used.add(set_idx)

    result = []

    dicsize = sum([len(s) for s in rhyme_sets])

    for i in scheme:
        cur_rhyme_set = rhyme_sets[scheme_to_set[i]]
        
        # since pop is deterministic, if we want random words from rhymeset
        # we must pop a random number of times and then push those back in
        random_count = 1
        if (randomize):
            random_count = random.randint(1, len(cur_rhyme_set))

        temp = [cur_rhyme_set.pop() for j in range(random_count)]
        result.append(temp[-1])

        for j in temp[::-1]:
            cur_rhyme_set.add(j)

        rhyme_sets[scheme_to_set[i]] = cur_rhyme_set

        # must maintain that dictionary is invariant before and after we pop
        
        assert(dicsize == sum([len(s) for s in rhyme_sets]))

    return result


def syllable_count(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


def recursive_evaluate_stress(stress_dict, stress_scheme, scheme_ptr, token_to_syllable, wordlst, wordlst_ptr):

    # for all possible syllables, advance pointer by that much and reach the
    # corresponding scheme into a string, then we add that string to the stress
    # dictionary
    
    def get_stress(scheme, scheme_ptr, syllable_len):
        # returns a list of stresses
        if scheme_ptr >= len(scheme):
            print("out of bounds ptr: ", scheme_ptr)
        return stress_scheme[scheme_ptr: scheme_ptr +  syllable_len]
    
    if (wordlst_ptr < len(wordlst)): # not done, so do
        word = wordlst[wordlst_ptr]
        possible_syllables = token_to_syllable[word]

        # at this point we determine whether this is the end of the list
        syllables = possible_syllables["R"]
        if (len(wordlst) - 1 == wordlst_ptr): # this is case where the word is at the end, then we use ending syllable count if available
            if ("E" in possible_syllables.keys()):
                syllables = possible_syllables["E"]

        for i in syllables:
            if word in stress_dict.keys():
                stress_pattern = get_stress(stress_scheme, scheme_ptr, i)
                if stress_pattern not in stress_dict[word]:
                    stress_dict[word].append(stress_pattern)
            else:
                stress_dict[word] = [get_stress(stress_scheme, scheme_ptr, i)]
            
            #recursive call by advancing the pointers
            recursive_evaluate_stress(stress_dict, stress_scheme, scheme_ptr + i, token_to_syllable, wordlst, wordlst_ptr + 1)












    


    


