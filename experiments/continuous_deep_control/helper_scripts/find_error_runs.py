import re
import os

env = 'EasyContinuousMaze'

for agent in ['ForwardKL','ReverseKL']:
    print("##########{}##########".format(agent))
    base = '{}results'.format(env)
    patt = re.compile('{}_{}_setting_(?P<setting>\d+)_run_(?P<run>\d+)_agent_Params.txt'.format(env, agent ))

    files = [ f for f in os.listdir(base) if patt.match(f) is not None]

    sett = 5

    out_list = []
    for s in range(sett):
        for r in range(30):
            if '{}_{}_setting_{}_run_{}_agent_Params.txt'.format(env, agent, s, r) not in files:
                number = r*sett + s
                out_list.append(str(number))

    print(','.join(out_list))