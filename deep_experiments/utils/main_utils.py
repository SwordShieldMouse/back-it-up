from collections import OrderedDict


# Takes a string and returns and instance of an agent
# [env] is an instance of an environment
# [p] is a dictionary of agent parameters
def create_agent(agent_string, config):

    if agent_string == 'HydraReverseKL':
        from agents.HydraReverseKL import HydraReverseKL
        return HydraReverseKL(config)

    elif agent_string == 'HydraForwardKL':
        from deep_experiments.agents.HydraForwardKL import HydraForwardKL
        return HydraForwardKL(config)

    elif agent_string == 'ReverseKL':
        from agents.ReverseKL import ReverseKL
        return ReverseKL(config)

    elif agent_string == 'ForwardKL':
        from agents.ForwardKL import ForwardKL
        return ForwardKL(config)

    else:
        print("Don't know this agent")
        exit(0)


# takes a dictionary where each key maps to a list of different parameter choices for that key
# also takes an index where 0 < index < combinations(parameters)
# The algorithm does wrap back around if index > combinations(parameters), so setting index higher allows for multiple runs of same parameter settings
# Index is not necessarily in the order defined in the json file.
def get_sweep_parameters(parameters, index):
    out = OrderedDict()
    accum = 1
    for key in parameters:
        num = len(parameters[key])
        out[key] = parameters[key][int(index / accum) % num]
        accum *= num
    return (out, accum)






