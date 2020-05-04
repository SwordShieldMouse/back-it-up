import numpy as np
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
import os
import datetime
import seaborn as sns

# from lib.pg_obj.src.envs.tabular_envs import GridWorld


timeStamp = datetime.datetime.now().strftime('%m_%d_%H%M%S')
print("Start run at: {}".format(timeStamp))

saveDir = 'results/0_openGrid/{}'.format(timeStamp)

def main():
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    config = {
        "M": 10,
        "N": 10,
        "terminalStates": {
            # (0, 9): 1,
            (9, 9): 1
        },
        "gamma": 0.99,
        "softQtemp_arr": [0.0, 0.1, 0.5, 1.0],
        "stepReward_arr": [-10, -1, -0.1, -0.01, 0]
    }

    pi_name, pi = getPolicy(config["M"], config["N"])

    # pi[0][0][0] = 0.997
    # pi[0][0][1] = 0.001
    # pi[0][0][2] = 0.001
    # pi[0][0][3] = 0.001
    #
    # pi[0][1][0] = 0.001
    # pi[0][1][1] = 0.997
    # pi[0][1][2] = 0.001
    # pi[0][1][3] = 0.001
    #
    # pi[2][0] = 0.001
    # pi[2][1] = 0.001
    # pi[2][2] = 0.997
    # pi[2][3] = 0.001

    for idx_e, e in enumerate(config["softQtemp_arr"]):
        for idx_r, sr in enumerate(config["stepReward_arr"]):

            subDir = saveDir + '/{}_stepReward_{}'.format(idx_r, sr)
            if not os.path.exists(subDir):
                os.makedirs(subDir)

            # terminalStates: dict, gamma: float, stepReward: float, thresh=1e-6):
            env = OpenGrid(config["M"], config["N"], config["terminalStates"], config["gamma"], sr)

            env.computeTrueVal(e, pi, subDir)


class OpenGrid:
    def __init__(self, ROW: int, COL: int, terminalStates: dict, gamma: float, stepReward: float, thresh=1e-5):

        self.M = ROW
        self.N = COL

        # dict containing coord as key and terminalReward as value
        self.terminalStates = terminalStates
        self.gamma = gamma

        self.stepReward = stepReward

        self.thresh = thresh

        self._starts = np.array([0, 0])
        self._state = None
        self._actions = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]])  # up, left, down, right
        self._action_eps = 0.0

        self.empty_Q = np.zeros((self.M * self.N, len(self._actions)))
        self.n_states = self.M * self.N

        self.n_actions = 4 * np.ones(100, dtype=np.int)

    def state_id(self, s):
        return s[0] * self.N + s[1]

    def reset(self):
        self._state = self._starts
        return self.state_id(self._state)

    def step(self, a):
        if self._action_eps != 0.0 and np.random.random() <= self._action_eps:
                a = np.random.randint(4)

        sp = self.getNextState(self._state, a)  # self._state + self._actions[a]
        r = self.getReward(sp)

        isTerm = True if self.isTerminalState(sp) else False

        self._state = sp

        return self.state_id(self._state), r, isTerm, None

    def getNextState(self, s, a):

        r = s[0]
        c = s[1]

        if a == 0:  # up
            return (r-1, c) if r-1 > -1 else (r, c)
        elif a == 1:  # left
            return (r, c-1) if c-1 > -1 else (r, c)
        elif a == 2:  # down
            return (r+1, c) if r+1 < self.M else (r, c)
        elif a == 3:  # right
            return (r, c+1) if c+1 < self.N else (r, c)

    def isTerminalState(self, s):
        if (s[0], s[1]) in self.terminalStates:
            return True
        else:
            return False

    def getReward(self, s):

        r = self.stepReward

        if self.isTerminalState(s):
            r += self.terminalStates[(s[0], s[1])]

        return r

    def computeTrueVal(self, entropy: float, learned_V_id, policy: np.array, subDir, epCount, stepCount):

        if not os.path.exists(subDir):
            os.makedirs(subDir)

        true_V = np.zeros((self.M, self.N))

        learned_V = np.zeros((self.M, self.N))

        # convert to learnred_V to grid layout

        for m in range(self.M):
            learned_V[m][:] = learned_V_id[self.N * m: self.N * m + self.N]

        error = float('inf')
        iter = 0
        while error > self.thresh:

            error = 0
            # loop over each state
            for m in range(self.M):
                for n in range(self.N):

                    s = [m,n]
                    s_id = self.state_id(s)
                    if self.isTerminalState(s):
                        true_V[m][n] = 0.0

                    else:
                        v_target = 0

                        for a in range(len(policy[s_id])):
                            m_p, n_p = self.getNextState(s, a)

                            s_p = [m_p, n_p]
                            r_p = self.getReward(s_p)

                            target = r_p - entropy * np.log(policy[s_id][a]) + self.gamma * true_V[m_p][n_p]
                            v_target += target * ( (1 - self._action_eps) * policy[s_id][a] + self._action_eps * 1/len(self._actions) )

                        error = max(error, abs(v_target - true_V[m][n]))
                        true_V[m][n] = v_target

            # if iter % 100 == 0:
            #     print("iter {} error: {}".format(iter, error))
            iter += 1

        # plot
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 11))

        self.plotHeatmap(true_V, ax1)
        self.plotHeatmap(learned_V, ax2)
        self.plotPolicy(policy, ax3)

        ax1.tick_params(labelsize=15)
        ax2.tick_params(labelsize=15)
        ax3.tick_params(labelsize=15)

        # ax1.set_aspect('equal')
        ax3.set_aspect('equal')

        ax1.set_title("true v", fontsize=15)
        ax2.set_title("learned v", fontsize=15)
        ax3.set_title("policy", fontsize=15)

        plt.suptitle(
            "Discrete Open Grid ({}x{}), Ep: {}, Step: {}\n softQ temp: {}, stepReward: {}, termReward: {}".format(
                self.M, self.N, epCount, stepCount, entropy, self.stepReward, list(self.terminalStates.values())), fontsize=15, y=0.95)

        # plt.show()
        plt.savefig('{}/steps_{}.png'.format(subDir, stepCount))

        plt.clf()
        plt.close()
        return true_V

    def plotHeatmap(self, V: np.array, ax):

        # TODO: set cbar length to match plot
        g1 = sns.heatmap(V, ax=ax, square=True)
        g1.set_xticks(range(self.M + 1))
        g1.set_yticks(range(self.N + 1))

    def plotPolicy(self, policy, ax):

        x = np.arange(0, self.N, 1.0)
        y = np.arange(0, self.M, 1.0)

        X, Y = np.meshgrid(x, y)

        for a in range(4):

            x_dir = np.ones((self.M, self.N)) if a == 3 or a == 1 else np.zeros((self.M, self.N))
            y_dir = np.ones((self.M, self.N)) if a == 2 or a == 0 else np.zeros((self.M, self.N))

            for m in range(self.M):
                for n in range(self.N):

                    s = [m,n]
                    s_id = self.state_id(s)
                    if a == 0 or a == 2:
                        y_dir[m][n] *= policy[s_id][a] * (-1) * (a - 1)
                    elif a == 1 or a == 3:
                        x_dir[m][n] *= policy[s_id][a] * (a - 2)

            ax.quiver(X, Y, x_dir, y_dir, units="inches", scale=2.5, scale_units="inches", width=0.025, headwidth=2,
                      headlength=2)

        ax.xaxis.set_ticks(x)
        ax.yaxis.set_ticks(y)

        ax.set_ylim(ax.get_ylim()[::-1])


# hard-coded for testing purposes
# Later will receive policies directly from agents
def getPolicy(M, N):

    policy_name = 'uniform'

    if policy_name == 'uniform':
        pi = 0.25 * np.ones((M * N, 4))
    else:
        raise ValueError("Invalid policy name")

    return policy_name, pi


if __name__ == '__main__':
    main()
