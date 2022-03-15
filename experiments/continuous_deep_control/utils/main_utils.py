from collections import OrderedDict
import ast
import re

def tryeval(val):
    bytes_patt = re.compile("b'(?P<content>.+)'")
    m = bytes_patt.match(val)
    if m is not None:
        val = m.group("content")
    try:
        val = ast.literal_eval(val)
    except ValueError:
        pass
    return val

# Takes a string and returns and instance of an agent
# [env] is an instance of an environment
# [p] is a dictionary of agent parameters
def create_agent(agent_string, config):
    if agent_string == 'ReverseKL':
        from agents.ReverseKL import ReverseKL   
        return ReverseKL(config)

    elif agent_string == 'ForwardKL':
        from agents.ForwardKL import ForwardKL
        return ForwardKL(config)

    elif agent_string == 'GMMForwardKL':
        from agents.ForwardKL import ForwardKL_GMM
        return ForwardKL_GMM(config)

    elif agent_string == 'GMMReverseKL':
        from agents.ReverseKL import ReverseKL_GMM
        return ReverseKL_GMM(config)

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

# write to tf Summary
def write_summary(writer, increment, stuff_to_log, tag):
    raise NotImplementedError




