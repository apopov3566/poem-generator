import os
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
from data_handler import get_corpus
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
)


def print_common_words(emissions, detoken, nwords):
    '''
    Print the nwords most common words given by the emissions array.

    emissions (1d array): The probability that word i will be emitted is emissions[i].
    detoken (dict: int -> string): Map the token of a word to the word itself.
    nwords: The number of words to print.

    returns: None
    '''

    # Find the tokens with the highest probability of occurring.
    tokens = list(np.argpartition(emissions, -nwords)[-nwords:])

    for token in tokens:
        print(detoken[token], end='\t')

    print()

    return


def main(model_file):
    ''' Run visualizations for the provided model. '''

    # Load the model.
    print('Visulaizations for the model ' + model_file)
    model = load(model_file)

    # Visualize the transition and observation matrices.
    visualize_sparsities(model, O_max_cols=10000, O_vmax=0.003)

    # Get the most common output words from each state.
    _, detoken, _ = get_corpus("data/shakespeare.txt", split_by_line=False)

    for state in range(model.L):
        print(f'state {state} -> ', end='')
        print_common_words(model.O[state], detoken, nwords=10)

    # Make  word cloud for each state.
    obs, obs_map = parse_observations(open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read())
    wordclouds = states_to_wordclouds(model, obs_map)

    # Make an animation.
    anim = animate_emission(model, obs_map, M=8)
    plt.show()


if __name__ == '__main__':
    main('hmm_haiku.model')
