import os
from joblib import load
import matplotlib.pyplot as plt
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
)


def main(model_file):
    ''' Run visualizations for the provided model. '''

    # Load the model.
    print('Visulaizations for the model ' + model_file)
    model = load(model_file)

    # Visualize the transition and observation matrices.
    visualize_sparsities(model, O_max_cols=10000)

    # Make  word cloud for each state.
    obs, obs_map = parse_observations(open(os.path.join(os.getcwd(), 'data/shakespeare.txt')).read())
    wordclouds = states_to_wordclouds(model, obs_map)

    # Make an animation.
    anim = animate_emission(model, obs_map, M=8)
    plt.show()


if __name__ == '__main__':
    main('hmm_haiku.model')
