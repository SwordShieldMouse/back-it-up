import numpy as np 
# import cvxpy as cp
import time
import copy
import os
import sys
sys.path.append(os.getcwd())
import lib.pg_obj.src.utils.math_utils as utils

class BaseAgent():
    def __init__(self, env, env_params, agent_params, write_summaries=False, names_to_suppress=[], use_ep_length_r=False):
        self.env = env 
        self.env_params = env_params
        self.agent_params = agent_params 
        self.write_summaries = write_summaries 
        self.names_to_suppress = names_to_suppress
        self.name = self.generate_name()

        # RL stuff
        self.gamma = self.agent_params["gamma"]
        self.Q = copy.deepcopy(self.env.empty_Q)
        self.V = np.zeros(self.env.n_states)
        self.dual = copy.deepcopy(self.env.empty_Q) # the dual variable; conditioned on (s, a)
        self.policy = copy.deepcopy(self.env.empty_Q) # hold logits for policy
        self.policy_probs = None  # SS: hold probs for policy. Added for use in computing learned V
        self.lr = self.agent_params["lr"]
        if "alr" in self.agent_params:
            self.alr = self.agent_params["alr"]
        try:
            print("set default policy from env")
            self.policy = copy.deepcopy(self.env.empty_policy)
        except:
            print("no default policy from env")
        self.probs = None # for convex optim policy

        if "eps" in self.agent_params:
            self.eps_init = self.agent_params["eps"]
            self.eps_final = self.agent_params["eps"]

        if "eps_init" in self.agent_params:
            self.eps_init = self.agent_params["epsinit"]
        
        if "eps_final" in self.agent_params:
            self.eps_final = self.agent_params["epsfinal"]
        
        if "eps_zero_by" not in self.agent_params:
            self.eps_zero_by = self.env_params["max_frames"] // 10
        else:
            self.eps_zero_by = self.agent_params["epszeroby"]
        

        # for stats 
        self.use_ep_length_r = use_ep_length_r
        self.avg_rewards = []
        self.returns = []
        self.window = 10
        assert self.window <= self.env_params["max_frames"] // self.env_params["max_frames_per_ep"] # to make sure we have enough returns
        self.G_buffer = [np.nan] * self.window
        self.entropies = []
        self.td_errors = []
        self.all_probs = []
        self.plotting_data = []

    # For visualization purpose in Reverse/Forward KL
    # modifies self.V in-place using self.policy_probs and self.Q
    def compute_learnedV(self):

        assert (np.shape(self.policy_probs) == np.shape(self.Q))
        learnedV = np.array([np.sum(self.policy_probs[s] * (self.Q[s] - self.softQtemp * np.log(self.policy_probs[s])), axis=-1) for s in range(self.env.n_states)])

        assert (np.shape(learnedV) == (self.env.n_states,))

        # (n_states, )
        return learnedV

    def get_softmax_jacobian(self, p_s):
        # for given probabilities, return softmax jacobian
        # \nabla S_i = S_i {\delta_ij - S_j}_j
        # need to build matrix with columns \nabla S_i
        # columns indexed by i, rows indexed by j
        jac = np.multiply(p_s.reshape(1, -1), (np.eye(p_s.shape[0]) - p_s.reshape(-1, 1)))
        assert np.isnan(jac).sum() == 0
        return jac

    def soft_update_qv(self, s, a, r, sp, done, sac_update=1):
        probs = self.get_policy_probs()
        entropy = np.log(probs[s][a] + 1e-5)
        # record the entropy before updating
        self.entropies.append(np.dot(probs[s], -np.log(probs[s] + 1e-5)))
        assert np.isnan(entropy) == 0
        if sac_update == 1:
            self.V[s] += self.agent_params["lr"] * (self.Q[s][a] - self.softQtemp * entropy - self.V[s])
        else:
            self.V[s] += self.agent_params["lr"] * (r - self.softQtemp * entropy + self.gamma * (1. - done) * self.V[sp] - self.V[s])
        self.Q[s][a] += self.agent_params["lr"] * (r + self.gamma * (1. - done) * self.V[sp] - self.Q[s][a])

    def get_policy_probs(self):
        # get the policy probs, assuming softmax 
        probs = []
        for s in range(self.env.n_states):
            max_logit = np.max(self.policy[s])
            p = np.exp(self.policy[s] - max_logit) / np.sum(np.exp(self.policy[s] - max_logit))
            probs.append(p)
        assert np.isnan(probs).sum() == 0

        return np.array(probs)
    
    def get_action_probs(self, s):
        action_prefs = self.policy[s]
        max_logit = np.max(action_prefs)
        probs = np.exp(action_prefs - max_logit) / np.sum(np.exp(action_prefs - max_logit))
        assert np.isnan(probs).sum() == 0
        return probs

    def generate_name(self):
        # generate agent full name from agent_params and base_name 
        params = ["{}={}".format(name, self.agent_params[name]) for name in self.agent_params if name not in self.names_to_suppress]
        return "{}_{}".format(self.base_name, '_'.join(params))

    def act(self, s):
        pass 

    def policy_act(self, s, policy_s = None):

        # SS: Not necessary, we are computing all probs for all states anyway
        # if policy_s is None:
        #     policy_s = self.policy[s]
        # max_logit = np.max(policy_s)
        # self.probs = np.exp(policy_s - max_logit) / np.sum(np.exp(policy_s - max_logit))

        # SS: Adding new class variable, to use when computing learned V
        self.policy_probs = self.get_policy_probs()  # should be of shape (n_states, n_actions)
        self.all_probs.append(self.policy_probs)
        self.probs = self.policy_probs[s]  # SS: Not sure where this is used
        action = np.random.multinomial(1, self.policy_probs[s]).nonzero()[0][0]

        return action

    def eps_greedy_act(self, s):
        eps = self.get_eps()
        if np.random.random() < eps:
            action = np.random.randint(self.env.n_actions[s])
        else: 
            action = np.random.choice(np.where(self.Q[s] == self.Q[s].max())[0])
        return action


    def get_eps(self):
        eps = self.eps_init * (1. - self.frame / self.eps_zero_by) + (self.eps_final) * self.frame / self.eps_zero_by
        eps = max(min(eps, self.eps_init), self.eps_final)
        return eps

    def step(self, s, a, r, sp, done):
        pass

    def run(self):
        if self.write_summaries is True:
            tag = "{}_{}_{}".format(self.env.name, self.name, time.time())
            self.summary_writer = SummaryWriter(log_dir = "./summaries/{}".format(tag))
        self.frame = 0
        ep = 0
        while self.frame < self.env_params["max_frames"]:
            s = self.env.reset()
            G = 0
            done = False
            self.curr_frame_count = 0
            while done is not True:
                # self.env.render()
                a = self.act(s)
                sp, r, done, _ = self.env.step(a)
                # G += (self.gamma ** curr_frame_count) * r # if we want to use the discounted return as the eval metric
                G += r
                self.step(s, a, r, sp, done)
                self.curr_frame_count += 1
                if self.curr_frame_count >= self.env_params["max_frames_per_ep"]:
                    # NOTE: THIS NEEDS TO BE AFTER STEP SO THAT WE BOOTSTRAP CORRECTLY
                    done = True
                # clip action values if applicable 
                if "clip" in self.agent_params:
                    for s in range(len(self.Q)):
                        self.Q[s] = np.clip(self.Q[s], -self.agent_params["clip"], self.agent_params["clip"])
                s = sp
                self.frame += 1
                if self.write_summaries is True:
                    self.summary_writer.add_scalar("avg return", self.avg_rewards[-1], self.frame - 1)
            if self.use_ep_length_r is True:
                self.G_buffer[ep % self.window] = -self.curr_frame_count
            else:
                self.G_buffer[ep % self.window] = G
            # give episode lengths as return
            self.returns.append(G) 
            self.avg_rewards += [np.nanmean(self.G_buffer)] * self.curr_frame_count # don't start recording until 10 eps have been completed?
            self.plotting_data.append([np.nanmean(self.G_buffer), self.curr_frame_count])
            if self.write_summaries is True:
                self.summary_writer.add_scalar("return", G, ep)
            ep += 1
            print("ep = {} | frame = {} | G = {} | avg return = {} | ep length = {}".format(ep, self.frame - 1, G, self.avg_rewards[-1], self.curr_frame_count))
        # when done, ensure that number of avg returns matches number of frames
        self.avg_rewards = self.avg_rewards[:self.env_params["max_frames"]]
        self.plotting_data[-1][-1] -= np.sum(self.plotting_data, axis = 0)[-1] - self.env_params["max_frames"]
        assert np.sum(self.plotting_data, axis = 0)[-1] == self.env_params["max_frames"]

class QL(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "QL"
        super().__init__(*args, **kwargs)

    def act(self, s):
        return self.eps_greedy_act(s)

    def step(self, s, a, r, sp, done):
        self.Q[s][a] += self.agent_params["lr"] * (r + self.agent_params["gamma"] * (1. - done) * self.Q[sp].max() - self.Q[s][a])

class Random(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "Random"
        super().__init__(*args, **kwargs)
    
    def act(self, s):
        return np.random.randint(self.env.n_actions[s])

class MinMaxPolicy(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "MinMax"
        super().__init__(*args, **kwargs)

    def step(self, s, a, r, sp, done):
        gamma = self.agent_params["lr"]
        lr = self.agent_params["lr"]
        # obj is min_pi max_dual E[dual * td_error - 1/2 dual ** 2]
        # sample-based version is min_pi max_dual dual * expected_sarsa_error - 1/2 E_pi[dual ** 2]
        # could outright solve this for a transition?
        self.td_errors.append(np.power((r + gamma * (1. - done) * self.Q[sp].max() - self.Q[s][a]), 2))
        self.Q[s][a] += lr * (r + gamma * (1. - done) * self.Q[sp].max() - self.Q[s][a])
        
        max_logit = np.max(self.policy[sp])
        probs = np.exp(self.policy[sp] - max_logit) / np.sum(np.exp(self.policy[sp] - max_logit))
        # print(probs.shape, self.Q[sp].shape)
        delta = (r + gamma * (1. - done) * np.dot(probs, self.Q[sp]) - self.Q[s][a])
        # delta = r + gamma * sum_i S_i Q(sp, i) - Q(s, a)
        # \nabla delta = gamma * sum_i \nabla S_i Q(sp, i)
        # \nabla S_i = S_i {\delta_ij - S_j}_j
        # need to build matrix with columns \nabla S_i
        # columns indexed by i, rows indexed by j
        nablaS = np.multiply(probs.reshape(1, self.env.n_actions[sp]), (np.eye(self.env.n_actions[sp]) - probs.reshape(self.env.n_actions[sp], 1)))

        # stochastic obj is min_pi max_dual delta * dual - 1/2 dual ** 2
        self.dual[s][a] += lr * (delta - self.dual[s][a]) 
        self.policy[sp] -= lr * self.dual[s][a] * gamma * np.matmul(nablaS, self.Q[sp])

    def act(self, s):
        return self.policy_act(s)

class SGDPolicy(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "SGDPolicy"
        super().__init__(*args, **kwargs)
        # record policy probs 
        self.all_probs = []

    def step(self, s, a, r, sp, done):
        gamma = self.agent_params["gamma"]
        lr = self.agent_params["lr"]
        self.td_errors.append(np.power((r + gamma * (1. - done) * self.Q[sp].max() - self.Q[s][a]), 2))
        self.Q[s][a] += lr * (r + gamma * (1. - done) * self.Q[sp].max() - self.Q[s][a])
        
        # update policy wrt expected td error
        max_logit = np.max(self.policy[sp])
        probs = np.exp(self.policy[sp] - max_logit) / np.sum(np.exp(self.policy[sp] - max_logit))
        # print(probs.shape, self.Q[sp].shape)
        delta = (r + gamma * (1. - done) * np.dot(probs, self.Q[sp]) - self.Q[s][a])
        # delta = r + gamma * sum_i S_i Q(sp, i) - Q(s, a)
        # \nabla delta = 2 * delta * gamma * sum_i \nabla S_i Q(sp, i)
        # \nabla S_i = S_i {\delta_ij - S_j}_j
        # need to build matrix with columns \nabla S_i
        # columns indexed by i, rows indexed by j
        nablaS = np.multiply(probs.reshape(1, self.env.n_actions[sp]), (np.eye(self.env.n_actions[sp]) - probs.reshape(self.env.n_actions[sp], 1)))
        self.policy[sp] -= self.agent_params["alr"] * delta * gamma * np.matmul(nablaS, self.Q[sp])

    def act(self, s):
        return self.policy_act(s)

class ActorCritic(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "SGDPolicy"
        super().__init__(*args, **kwargs)
        # record policy probs 
        self.all_probs = []
        self.t = 0

    def step(self, s, a, r, sp, done):
        gamma = self.agent_params["gamma"]
        lr = self.agent_params["lr"]
        alr = self.agent_params["alr"]
        delta = (r + gamma * (1. - done) * self.V[sp] - self.V[s])
        self.td_errors.append(np.power(delta, 2))
        probs = self.get_action_probs(s)
        self.V[s] += lr * delta
        self.policy[s] += (self.gamma ** self.t) * alr * delta * self.get_softmax_jacobian(probs) / probs
        self.t += 1
        if done is True:
            self.t = 0

    def act(self, s):
        return self.policy_act(s)


class BoltzmannQ(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "BoltzmannQ"
        super().__init__(*args, **kwargs)
        self.softmaxtemp = self.agent_params["softmaxtemp"]

    def step(self, s, a, r, sp, done):
        self.Q[s][a] += self.agent_params["lr"] * (r + self.agent_params["gamma"] * (1. - done) * self.Q[sp].max() - self.Q[s][a])

    def act(self, s):
        max_logit = np.max(self.Q[s])
        probs = np.exp((self.Q[s] - max_logit) / self.softmaxtemp) / np.sum(np.exp((self.Q[s]
        - max_logit) / self.softmaxtemp))
        return self.policy_act(s, probs)

class ReverseKL(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "ReverseKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.softmaxtemp = self.agent_params["softmaxtemp"]
        self.noisyQ = self.agent_params["noisyQ"]
        self.learnQ = self.agent_params["learnQ"]
        self.allstates = self.agent_params["allstates"]
        self.trueQ_entropies = []
        self.sacupdate = self.agent_params["sacupdate"]
        self.integration = self.agent_params["integration"]

    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        # at each step, optimize the reverse KL D(pi || Q)
        # \nabla D(pi || Q) = - \sum_i Q(s, i) \nabla pi(i) - H(pi)

        # SS: self.policy_probs is already computed at agent.act()

        self.soft_update_qv(s, a, r, sp, done, self.sacupdate)

        if self.learnQ == 0:
            true_Q = self.env.true_Q(self.policy_probs, self.gamma, temp = self.softQtemp)
        else:
            true_Q = self.Q
        if self.noisyQ == 1:
            true_Q += np.random.normal(size = true_Q.shape)

        # update all s
        if self.allstates == 1:
            states_to_update = np.arange(self.env.n_states)
        else:
            states_to_update = [s]
        for s in states_to_update:
            softmax_jacobian = np.multiply(self.policy_probs[s].reshape(1, self.env.n_actions[s]), (np.eye(self.env.n_actions[s]) - self.policy_probs[s].reshape(self.env.n_actions[s], 1))) # of shape (params, actions)
            if self.integration == 1:
                # going along columns changes which pi(i) is looked at
                term1 = - np.matmul(softmax_jacobian, true_Q[s]).squeeze()
                # H(pi) = -\sum_i p_i log (p_i)
                # nabla H(pi) = -(\sum_i log(p_i) nabla p_i + \sum_i \nabla p_i)  
                term2 = np.matmul(softmax_jacobian, np.log(self.policy_probs[s])).squeeze() + np.sum(softmax_jacobian, axis = 1)
            else:
                term1 = - true_Q[s][a] * softmax_jacobian[:, a] / self.policy_probs[a]
                term2 = np.matmul(softmax_jacobian, np.log(self.policy_probs[s])).squeeze() + softmax_jacobian[:, a] / self.policy_probs[s][a]
            assert np.isnan(term1).sum() == 0
            assert np.isnan(term2).sum() == 0
            self.policy[s] -= self.agent_params["alr"] * (term1 + self.softmaxtemp * term2)
            

class ForwardKL(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "ForwardKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.softmaxtemp = self.agent_params["softmaxtemp"]
        self.noisyQ = self.agent_params["noisyQ"]
        self.learnQ = self.agent_params["learnQ"]
        self.allstates = self.agent_params["allstates"]
        self.trueQ_entropies = []
        self.sacupdate = self.agent_params["sacupdate"]
        self.integration = self.agent_params["integration"]


    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        # at each step, optimize forward KL D(Q || pi)
        # \nabla D(Q || pi) = - \sum_i exp(Q(i)) / Z * \nabla pi(i) / pi(i)

        # SS: self.policy_probs is already computed at agent.act()
        # probs = self.get_policy_probs()

        self.soft_update_qv(s, a, r, sp, done, self.sacupdate)
        if self.learnQ == 0:
            true_Q = self.env.true_Q(self.policy_probs, self.gamma, temp = self.softQtemp)
            # record true Q entropies at starting state 
        else:
            true_Q = self.Q
        if self.noisyQ == 1:
            true_Q += np.random.normal(size = true_Q.shape)
        if self.allstates == 1:
            states_to_update = np.arange(self.env.n_states)
        else:
            states_to_update = [s]
        for s in states_to_update:
            softmax_jacobian = np.multiply(self.policy_probs[s].reshape(1, self.env.n_actions[s]), (np.eye(self.env.n_actions[s]) - self.policy_probs[s].reshape(self.env.n_actions[s], 1)))
            max_Q = np.max(true_Q[s])
            boltzQ = np.exp((true_Q[s] - max_Q) / self.softmaxtemp) / np.sum(np.exp((true_Q[s] - max_Q) / self.softmaxtemp))
            if self.integration == 1:
                weighting = np.divide(boltzQ, self.policy_probs[s])
                # print(weighting.shape)
                assert np.isnan(weighting).sum() == 0
                self.policy[s] += self.agent_params["alr"] * np.matmul(softmax_jacobian, weighting).squeeze()
            else:
                # sample from the boltzman Q distribution
                new_a = np.random.multinomial(1, boltzQ).nonzero()[0][0]
                self.policy[s] += self.agent_params["alr"] * softmax_jacobian[:, new_a] / self.policy_probs[new_a]


class HardForwardKL(BaseAgent):
    """ Optimizes policy based on minimizing -log pi(max_action | s) """
    def __init__(self, *args, **kwargs):
        self.base_name = "HardForwardKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.noisyQ = self.agent_params["noisyQ"]
        self.learnQ = self.agent_params["learnQ"]
        self.allstates = self.agent_params["allstates"]
        self.trueQ_entropies = []
        self.sacupdate = self.agent_params["sacupdate"]

        

    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        probs = self.get_policy_probs()
        self.soft_update_qv(s, a, r, sp, done, self.sacupdate)
        if self.learnQ == 0:
            true_Q = self.env.true_Q(probs, self.gamma, temp = self.softQtemp)
            # record true Q entropies at starting state 
        else:
            true_Q = self.Q
        if self.noisyQ == 1:
            true_Q += np.random.normal(size = true_Q.shape)
        if self.allstates == 1:
            states_to_update = np.arange(self.env.n_states)
        else:
            states_to_update = [s]
        for s in states_to_update:
            softmax_jacobian = self.get_softmax_jacobian(probs[s]) # columns indexed by action

            max_action = utils.rand_argmax(true_Q[s])
            # print(weighting.shape)
            self.policy[s] += self.alr * softmax_jacobian[:, max_action] / probs[s][max_action]

class HardReverseKL(BaseAgent):
    """ Just Actor-Expert with SGD. The gradient is \sum_a \nabla pi(a | s) Q(s, a)"""
    def __init__(self, *args, **kwargs):
        self.base_name = "HardReverseKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.noisyQ = self.agent_params["noisyQ"]
        self.learnQ = self.agent_params["learnQ"]
        self.allstates = self.agent_params["allstates"]
        self.trueQ_entropies = []
        self.sacupdate = self.agent_params["sacupdate"]
        self.integration = self.agent_params["integration"]


    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        probs = self.get_policy_probs()
        self.soft_update_qv(s, a, r, sp, done, self.sacupdate)
        if self.learnQ == 0:
            true_Q = self.env.true_Q(probs, self.gamma, temp = self.softQtemp)
        else:
            true_Q = self.Q
        if self.noisyQ == 1:
            true_Q += np.random.normal(size = true_Q.shape)

        if self.allstates == 1:
            states_to_update = np.arange(self.env.n_states)
        else:
            states_to_update = [s]
        for s in states_to_update:
            softmax_jacobian = np.multiply(probs[s].reshape(1, self.env.n_actions[s]), (np.eye(self.env.n_actions[s]) - probs[s].reshape(self.env.n_actions[s], 1)))
            if self.integration == 1:
                # going along columns changes which pi(i) is looked at
                term1 = - np.matmul(softmax_jacobian, true_Q[s]).squeeze()
            else:
                term1 = - true_Q[s][a] * softmax_jacobian[:, a] / probs[a]
            assert np.isnan(term1).sum() == 0
            self.policy[s] -= self.agent_params["alr"] * term1 

class IntQL(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "IntQL"
        super().__init__(*args, **kwargs)
        self.r = copy.deepcopy(self.Q)
        self.Vp = copy.deepcopy(self.Q) # holds E_{s' \sim p(\cdot \mid s, a)}[V(s')] for each a

    def act(self, s):
        eps = self.get_eps()
        if np.random.random() < eps:
            action = np.random.randint(self.env.n_actions[s])
        else: 
            Qs = np.array(self.r[s]) + self.gamma * np.array(self.Vp[s])
            action = np.random.choice(np.where(Qs == Qs.max())[0])
        return action

    def step(self, s, a, r, sp, done):
        rs = np.array(self.r[s])
        Vps = np.array(self.Vp[s])
        rsp = np.array(self.r[sp])
        Vpsp = np.array(self.Vp[sp])
        Qp = rsp + self.gamma * Vpsp
        self.r[s][a] += self.agent_params["lr"] * (r - self.r[s][a])
        self.Vp[s][a] += self.agent_params["lr"] * (Qp.max() - self.Vp[s][a]) # this is what Q-learning amounts to with separate reward and successor value
        # self.Q[s][a] += self.agent_params["lr"] * (r + self.agent_params["gamma"] * (1. - done) * self.Q[sp].max() - self.Q[s][a])
