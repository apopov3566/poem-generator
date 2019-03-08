from data_handler import get_corpus
from HMM import unsupervised_HMM

def run_HMM(n_states, N_iters):
    corpus, detoken = get_corpus("data/shakespeare.txt", False)
    HMM = unsupervised_HMM(corpus, n_states, N_iters)

    for i in range(10):
        output, states = HMM.generate_emission(200)
        outstr = ""
        for token in output:
            outstr += (detoken[token] + " ")
        print(outstr)

if __name__ == '__main__':
    run_HMM(30, 20)
