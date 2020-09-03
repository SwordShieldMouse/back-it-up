import numpy as np 
import itertools

def make_big_list(algs, hyperparams):
    l = []
    for alg, param_names in algs:
        # print(list(itertools.product(*[HYPERPARAMS[param_name] for param_name in param_names])))
        for a in itertools.product(*[[(param, param_name) for param in hyperparams[param_name]] for param_name in param_names]):
            l.append((alg, *a))
    assert len(set(l)) == len(l) # guarantee uniqueness
    # calculate total number of experiments as a check
    n_total_experiments = get_n_total_experiments(algs, hyperparams)
    assert len(l) == n_total_experiments # guarantee we're running all the experiments
    return l

def get_n_total_experiments(algs, hyperparams):
    n_total_experiments = 0
    for alg, params_to_sweep in algs:
        n_combinations = np.prod([len(hyperparams[param]) for param in params_to_sweep])
        n_total_experiments += n_combinations
    return n_total_experiments


def get_instance(algs, hyperparams, ix):
    big_list = make_big_list(algs, hyperparams)
    alg = big_list[ix][0]
    agent_params = {key: value for value, key in big_list[ix][1:]}
    return alg, agent_params