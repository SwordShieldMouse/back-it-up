import numpy as np
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
import os
import datetime
import seaborn as sns


timeStamp = datetime.datetime.now().strftime('%m_%d_%H%M%S')
print("Start run at: {}".format(timeStamp))

saveDir = 'results/0_openGrid/{}'.format(timeStamp)


def plotHeatmap(V: np.array, config, pi_name, entropy, stepReward, saveDir):
    ax = sns.heatmap(V)

    ax.set_xticks(range(config["M"] + 1))
    ax.set_yticks(range(config["N"] + 1))
    ax.set_title(
        "Discrete Open Grid ({}x{}), policy: {} \n softQ temp: {}, stepReward: {}, termReward: {}".format(config["M"], config["N"], pi_name,
                                                                                                          entropy, stepReward, list(
                config["terminalStates"].values())))

    # plt.show()
    plt.savefig(
        '{}/v_openGrid_{}x{}_softQtemp_{}_stepReward_{}_terminalReward_{}.png'.format(saveDir, config["M"], config["N"], entropy, stepReward, list(config["terminalStates"].values())))
    plt.clf()

def main():
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    config = {
        "M": 10,
        "N": 10,
        "terminalStates": {
            (0, 9): 0,
            (9, 9): 0
        },
        "gamma": 0.99,
        "entropy_arr": [0.0, 0.01, 0.1, 0.5, 1.0],
        "stepReward_arr": [-5, -1, -0.1, -0.01]
    }

    # right, down, left, up

    pi_name, pi = getPolicy(config["M"], config["N"])

    for idx_e, e in enumerate(config["entropy_arr"]):
        for idx_r, sr in enumerate(config["stepReward_arr"]):

            subDir = saveDir + '/{}_stepReward_{}'.format(idx_r, sr)
            if not os.path.exists(subDir):
                os.makedirs(subDir)

            # terminalStates: dict, gamma: float, stepReward: float, thresh=1e-6):
            env = OpenGrid(config["M"], config["N"], config["terminalStates"], config["gamma"], sr)

            V = env.computeTrueVal(e, pi)
            plotHeatmap(V, config, pi_name, e, sr, subDir)


class OpenGrid:
    def __init__(self, ROW: int, COL: int, terminalStates: dict, gamma: float, stepReward: float, thresh=1e-6):

        self.M = ROW
        self.N = COL

        # dict containing coord as key and terminalReward as value
        self.terminalStates = terminalStates
        self.gamma = gamma

        self.stepReward = stepReward

        self.thresh = thresh

    def getNextState(self, r, c, a):

        if a == 0:  # right
            return (r, c+1) if c+1 < self.N else (r, c)
        elif a == 1:  # down
            return (r+1, c) if r+1 < self.M else (r, c)
        elif a == 2:  # left
            return (r, c-1) if c-1 > -1 else (r, c)
        elif a == 3:  # up
            return (r-1, c) if r-1 > -1 else (r, c)

    def isTerminalState(self, r, c):
        if (r, c) in self.terminalStates:
            return True
        else:
            return False

    def getReward(self, r, c):
        if self.isTerminalState(r, c):
            return self.terminalStates[(r,c)]
        else:
            return self.stepReward

    def computeTrueVal(self, entropy: float, policy: np.array):
        V = np.zeros((self.M, self.N))

        error = float('inf')
        iter = 0
        while error > self.thresh:

            error = 0
            # loop over each state
            for m in range(self.M):
                for n in range(self.N):

                    if self.isTerminalState(m, n):
                        V[m][n] = 0.0

                    else:
                        v_target = 0

                        for a in range(len(policy[m][n])):
                            m_p, n_p = self.getNextState(m, n, a)
                            r_p = self.getReward(m_p, n_p)

                            v_target += policy[m][n][a] * (r_p - entropy * np.log(policy[m][n][a]) + self.gamma * V[m_p][n_p])

                        error = max(error, abs(v_target - V[m][n]))
                        V[m][n] = v_target

            if iter % 100 == 0:
                print("iter {} error: {}".format(iter, error))
            iter += 1

        return V


# hard-coded for testing purposes
# Later will receive policies directly from agents
def getPolicy(M, N):

    policy_name = 'uniform'

    if policy_name == 'uniform':
        pi = 1/4 * np.ones((M, N, 4))
    else:
        raise ValueError("Invalid policy name")

    return policy_name, pi


if __name__ == '__main__':
    main()
