import random, math
import numpy as np

class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0.
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.

            D:          Number of observations.

            A:          The transition matrix.

            O:          The observation matrix.

            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]


    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)      # Length of sequence.

        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.

        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        #probs[0][0] = 1

        for length in range(1,M+1):
            for this_state in range(self.L):
                max_prob = 0
                max_prob_seq = ''
                for prev_state in range(self.L):
                    config_prob = probs[length-1][prev_state] * self.A[prev_state][this_state] * self.O[this_state][x[length-1]]
                    if length == 1:
                        config_prob = self.A_start[this_state] * self.O[this_state][x[length-1]]
                    if config_prob > max_prob:
                        max_prob = config_prob
                        max_prob_seq = seqs[length-1][prev_state] + str(this_state)
                probs[length][this_state] = max_prob
                seqs[length][this_state] = max_prob_seq


        max_seq = seqs[M][probs[M].index(max(probs[M]))]
        return max_seq


    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)      # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        for this_state in range(self.L):
            alphas[1][this_state] = self.A_start[this_state] * self.O[this_state][x[0]]
        if normalize: #and length_sum > 0:
            for this_state in range(self.L):
                norm = sum(alphas[1])
                alphas[1][this_state] /= norm

        for length in range(2,M+1):
            for this_state in range(self.L):
                for prev_state in range(self.L):
                    alphas[length][this_state] += self.O[this_state][x[length-1]] * alphas[length - 1][prev_state] * self.A[prev_state][this_state]

            if normalize: #and length_sum > 0:
                norm = sum(alphas[length])
                for this_state in range(self.L):
                    alphas[length][this_state] /= norm
        return alphas


    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)      # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        betas[M] = [1] * self.L
        for length in range(M-1,0,-1):
            for this_state in range(self.L):
                for next_state in range(self.L):
                    betas[length][this_state] += betas[length + 1][next_state] * self.A[this_state][next_state] * self.O[next_state][x[length]]
            if normalize: #and length_sum > 0:
                norm = sum(betas[length])
                for this_state in range(self.L):
                    betas[length][this_state] /= norm
        return betas


    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.
        #print(self.A)
        self.A = [[0. for _ in range(self.L)] for _ in range(self.L)]

        counts = [0] * self.L
        for seq in range(len(Y)):
            for loc in range(len(Y[seq])-1):
                counts[Y[seq][loc]] += 1
            for loc in range(1,len(Y[seq])):
                self.A[Y[seq][loc-1]][Y[seq][loc]] += 1

        for this_state in range(self.L):
            for prev_state in range(self.L):
                self.A[prev_state][this_state] /= counts[prev_state]

        # Calculate each element of O using the M-step formulas.
        self.O = [[0. for _ in range(self.D)] for _ in range(self.L)]

        counts = [0] * self.L
        for seq in range(len(X)):
            for loc in range(len(X[seq])):
                counts[Y[seq][loc]] += 1
            for loc in range(len(X[seq])):
                self.O[Y[seq][loc]][X[seq][loc]] += 1

        for this_state in range(self.L):
            for x_val in range(self.D):
                self.O[this_state][x_val] /= counts[this_state]


    '''
    def get_Pyj(self, a, j, alphas, betas):
        Pyj = alphas[j][a] * betas[j][a]
        total = 0
        for this_state in range(self.L):
            total += alphas[j][this_state] * betas[j][this_state]
        return Pyj/total

    def get_Pyj2(self, a, b, j, x, alphas, betas):
        #print(a, b, j, len(alphas[0]))
        Pyj = alphas[j][a] * betas[j+1][b] * self.A[a][b] * self.O[b][x[j+1]]
        total = 0
        for this_state in range(self.L):
            for next_state in range(self.L):
                total += (alphas[j][this_state] * betas[j+1][next_state] * self.A[this_state][next_state] * self.O[next_state][x[j+1]])
        assert(Pyj/total < 1)
        return Pyj/total
    '''

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        print("progress: ")
        for iter in range(N_iters):
            
            if (int(iter * 100 / N_iters) % 10 == 0):
                print(int(iter * 100 / N_iters), "%", "complete")
            #print(self.A)

            #print("---")
            #print(self.A)
            #print("---")

            A_numerator = [[0. for _ in range(self.L)] for _ in range(self.L)]
            A_denominator = [0. for _ in range(self.L)]
            O_numerator = [[0. for _ in range(self.D)] for _ in range(self.L)]
            O_denominator = [0. for _ in range(self.L)]

            #Pyj2_sums = [[] for _ in range(len(X))]
            #Pyj_sums = [[] for _ in range(len(X))]

            for seq in range(len(X)):
                x = X[seq]
                #print(seq)
                alphas = self.forward(x, True)
                betas = self.backward(x, True)


                for j in range(1,len(x) + 1):

                    probs = [[0. for _ in range(self.L)] for _ in range(self.L)]
                    current = [0. for _ in range(self.L)]

                    for this_state in range(self.L):
                        current[this_state] += alphas[j][this_state] * betas[j][this_state]

                        if j < len(x):
                            for next_state in range(self.L):
                                probs[this_state][next_state] = alphas[j][this_state] * betas[j + 1][next_state] * self.A[this_state][next_state] * self.O[next_state][x[j]]

                    current_norm = sum(current)
                    prob_norm = 0.

                    if j < len(x):
                        for this_state in range(self.L):
                            prob_norm += sum(probs[this_state])

                    #print(current_norm)
                    #print(prob_norm)

                    for this_state in range(self.L):
                        current[this_state] /= current_norm

                        if j < len(x):
                            for next_state in range(self.L):
                                probs[this_state][next_state] /= prob_norm


                    for this_state in range(self.L):
                        for next_state in range(self.L):
                            if j < len(x):
                                A_numerator[this_state][next_state] += probs[this_state][next_state]
                        if j < len(x):
                            A_denominator[this_state] += current[this_state]

                        O_numerator[this_state][x[j - 1]] += current[this_state]
                        O_denominator[this_state] += current[this_state]

            for this_state in range(self.L):
                for next_state in range(self.L):
                    self.A[this_state][next_state] = A_numerator[this_state][next_state] / A_denominator[this_state]
                for x_val in range(self.D):
                    self.O[this_state][x_val] = O_numerator[this_state][x_val] / O_denominator[this_state]
        #print(self.O)


    def generate_reverse_emmision(self, ending, length):
        ''' ending is the point we start generating in reverse'''
        from numpy.random import choice

        def get_priors(state, matrix):
            state_given_is = [i[state] * 1.0 / self.L for i in matrix]
            norm_const = sum(state_given_is)
            return [j / norm_const for j in state_given_is]

        states = [choice([i for i in range(self.L)], p=get_priors(ending, self.O))]
        emission = [ending]
        # apply transitions
        for j in range(1,length):
            states.append(choice([i for i in range(self.L)], p=get_priors(states[-1], self.A)))
            emission.append(choice([i for i in range(self.D)], p=self.O[states[-1]]))

        return emission[::-1], states[::-1]

    def generate_reverse_emmision_syllable_stress(self, ending, syllable_length, token_to_syllable, token_to_stress):
        '''
        general idea of algorithm:
        we want to take any arbitrary ending, call generate reverse emission
        on windows of 2 words, meaning we generate 1 new word per call,
        then keep a running tab of syllable counts. If we match that syllable
        count, then we return, otherwise restart and discard current line
        
        here length is the number of syllables

        '''
        sequence = [ending]
        syllable_count = token_to_syllable[ending]["R"][0]
        # curstress = token_to_stress[ending][0][0] # take the first stressed syllable of the ending
        while(syllable_count != syllable_length):
            
            emission, state = self.generate_reverse_emmision(sequence[-1], 2)
            newterm = emission[0]
            #while(newterm not in token_to_stress.keys() or token_to_stress[newterm][0][-1] == curstress):
                # regenerate until stress is not equal
            #    emission, state = self.generate_reverse_emmision(sequence[-1], 2)
            #    newterm = emission[0]

            # curstress = token_to_stress[newterm][0][0]

            syllable_count += token_to_syllable[newterm]["R"][0]
            sequence.append(newterm)
            if (syllable_count == syllable_length):
                # we satisfy condition
                return sequence[::-1]
            elif(syllable_count > syllable_length):
                # its impossible to go back, then restart
                sequence = [ending]
                syllable_count = token_to_syllable[ending]["R"][0]
            #    curstress = token_to_stress[ending][0][0]

        
    def generate_emission(self, M):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random.

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        emission = []
        states = []

        state = 0

        for i in range(M):
            trans_prob = random.random()
            next_state = 0
            while trans_prob > 0:
                trans_prob -= self.A[state][next_state]
                next_state += 1
            state = next_state - 1
            states.append(state)

            emission_prob = random.random()
            next_emission = 0
            while emission_prob > 0:
                emission_prob -= self.O[state][next_emission]
                next_emission += 1
            emission.append(next_emission - 1)

        return emission, states




    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob


    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM

def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.

        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
