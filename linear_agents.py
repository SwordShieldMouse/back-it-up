import numpy as np 
import cvxpy as cp 
# from torch.utils.tensorboard import SummaryWriter
import time
import copy
import utils

cp.settings.ERROR = [cp.settings.USER_LIMIT]                                                                         
cp.settings.SOLUTION_PRESENT = [cp.settings.OPTIMAL, cp.settings.OPTIMAL_INACCURATE, cp.settings.SOLVER_ERROR]    

class BaseAgent():
    def __init__(self, env, env_params, agent_params, write_summaries=False, names_to_suppress=[]):
        self.env = env 
        # assert self.env.env_type == "linear"
        self.env_params = env_params
        self.agent_params = agent_params 
        self.write_summaries = write_summaries 
        self.names_to_suppress = names_to_suppress
        self.name = self.generate_name()

        # RL stuff
        self.gamma = self.agent_params["gamma"]
        self.Q = np.zeros([self.env.action_dim, self.env.obs_dim])
        self.V = np.zeros([self.env.obs_dim])
        self.dual = np.zeros([self.env.action_dim, self.env.obs_dim])
        self.policy = np.zeros([self.env.action_dim, self.env.obs_dim])
        # print(self.env.obs_dim)

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
            self.eps_init = self.agent_params["eps_init"]
        
        if "eps_final" in self.agent_params:
            self.eps_final = self.agent_params["eps_final"]
        
        if "eps_zero_by" not in self.agent_params:
            self.eps_zero_by = self.env_params["max_frames"] // 10
        else:
            self.eps_zero_by = self.agent_params["eps_zero_by"]
        

        # for stats 
        self.avg_rewards = []
        self.returns = []
        self.window = 10
        self.G_buffer = [np.nan] * self.window
        self.entropies = []
        self.td_errors = []
        self.all_probs = []
        self.plotting_data = []

    def get_softmax_jacobian(self, s):
        # given a state, return gradient of {pi(a | s)}_a with respect to all the weights of policy. This is of shape (policy shape, action_dim)
        # d pi(a | s) / dw_ij = pi(i | s) s_j (delta_ia - pi(a | s))
        probs = self.get_action_probs(s)
        delta = np.eye(self.env.action_dim)
        delta_minus_pi = delta - probs.reshape((1, -1))
        grad = (probs.reshape(-1, 1) * delta_minus_pi)[:, np.newaxis, :] * s.reshape((1, -1, 1))
        return grad

    def generate_name(self):
        # generate agent full name from agent_params and base_name 
        params = ["{}={}".format(name, self.agent_params[name]) for name in self.agent_params if name not in self.names_to_suppress]
        return "{}_{}".format(self.base_name, '_'.join(params))

    def act(self, s):
        pass 


    def get_action_probs(self, s):
        # print(self.policy.shape, s.shape)
        action_prefs = self.policy @ s
        max_logit = np.max(action_prefs)
        probs = np.exp(action_prefs - max_logit) / np.sum(np.exp(action_prefs - max_logit))
        return probs

    def policy_act(self, s, p = None):
        if p is None:
            probs = self.get_action_probs(s)
        else:
            probs = p
        # self.all_probs.append(self.probs)
        action = np.random.multinomial(1, probs).nonzero()[0][0]
        if self.write_summaries is True:
            self.summary_writer.add_scalar("entropy", -np.array(probs) @ np.log(probs), self.frame)
            self.summary_writer.add_scalar("action", action, self.frame)  
            # track the action gap 
            top2 = np.argsort(self.Q @ s)[-2:]
            self.summary_writer.add_scalar("action_gap", (self.Q @ s)[top2[1]] - (self.Q @ s)[top2[0]], self.frame)
        return action

    def eps_greedy_act(self, s):
        eps = self.get_eps()
        if np.random.random() < eps:
            action = np.random.randint(self.env.action_dim)
        else: 
            Qs = self.Q @ s
            action = np.random.choice(np.where(Qs == Qs.max())[0])
        if self.write_summaries is True:
            self.summary_writer.add_scalar("action", action, self.frame)  
            # track the action gap 
            top2 = np.argsort(self.Q[s, :])[-2:]
            self.summary_writer.add_scalar("action_gap", self.Q[s][top2[1]] - self.Q[s][top2[0]], self.frame)
        return action

    def soft_update_qv(self, s, a, r, sp, done, sac_update = 1):
        probs = self.get_action_probs(s)
        if sac_update == 1:
            self.V += self.agent_params["lr"] * ((self.Q @ s)[a] - self.softQtemp * np.log(probs[a]) - self.V @ s)* s
        else:
            self.V += self.agent_params["lr"] * (r - self.softQtemp * np.log(probs[a]) + self.gamma * (1. - done) * self.V @ sp - self.V @ s)* s
        self.Q[a, :] += self.agent_params["lr"] * (r + self.gamma * (1. - done) * self.V @ sp - (self.Q @ s)[a]) * s
  
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
            # print(s.shape)
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
                s = sp
                self.frame += 1
                if self.write_summaries is True:
                    self.summary_writer.add_scalar("avg return", self.avg_rewards[-1], self.frame - 1)
            self.G_buffer[ep % self.window] = G
            # give episode lengths as return
            # self.G_buffer[ep % self.window] = self.curr_frame_count
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
        self.Q[a, :] += self.agent_params["lr"] * (r + self.agent_params["gamma"] * (1. - done) * np.max(self.Q @ sp) - (self.Q @ s)[a]) * s

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
        self.td_errors.append(np.power((r + gamma * (1. - done) * np.max(self.Q @ sp) - (self.Q @ s)[a]), 2))
        self.Q[a, :] += self.agent_params["lr"] * (r + self.agent_params["gamma"] * (1. - done) * np.max(self.Q @ sp) - (self.Q @ s)[a]) * s        
        
        probs = self.get_action_probs(sp)

        delta = (r + gamma * (1. - done) * np.dot(probs, self.Q @ sp) - (self.Q @ s)[a])
        # delta = (r + gamma * sum_i pi(i | s) Q(sp)[i] - Q (s)[a]) 
        # \nabla delta = gamma * sum_i \nabla pi(i | s) Q(sp)[i]
        policy_grad = self.get_softmax_jacobian(sp)

        # stochastic obj is min_pi max_dual delta * dual(s)[a] - 1/2 dual(s)[a] ** 2
        self.dual[a, :] += lr * (delta * s - (self.dual @ s)[a] * s) 
        self.policy -= lr * (self.dual @ s)[a] * gamma * np.sum(policy_grad * self.Q[sp].reshape((1, 1, -1)), axis = -1)

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
        # self.td_errors.append(np.power((r + gamma * (1. - done) * self.Q[sp].max() - self.Q[s][a]), 2))
        self.Q[a, :] += self.agent_params["lr"] * (r + self.agent_params["gamma"] * (1. - done) * np.max(self.Q @ sp) - (self.Q @ s)[a]) * s
        
        # update policy wrt expected td error   
        next_action_prefs = self.policy @ sp
        max_logit = np.max(next_action_prefs)
        probs = np.exp(next_action_prefs - max_logit) / np.sum(np.exp(next_action_prefs - max_logit))
        # print(probs.shape, self.Q[sp].shape)
        delta = (r + gamma * (1. - done) * np.dot(probs, self.Q @ sp) - (self.Q @ s)[a])
        # delta = r + gamma * sum_i S_i Q(sp, i) - Q(s, a)
        # \nabla delta = 2 * delta * gamma * sum_i \nabla S_i Q(sp, i)
        # \nabla S_i = S_i {\delta_ij - S_j}_j
        # need to build matrix with columns \nabla S_i
        # columns indexed by i, rows indexed by j
        softmax_jacobian = self.get_softmax_jacobian(sp)
        self.policy -= lr * delta * gamma * np.sum(softmax_jacobian * (self.Q @ sp).reshape((1, 1, -1)), axis = -1)

    def act(self, s):
        return self.policy_act(s)

class BoltzmannQ(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "BoltzmannQ"
        super().__init__(*args, **kwargs)
        self.softmaxtemp = self.agent_params["softmaxtemp"]

    def step(self, s, a, r, sp, done):
        self.Q[a, :] += self.agent_params["lr"] * (r + self.agent_params["gamma"] * (1. - done) * np.max(self.Q @ sp) - (self.Q @ s)[a]) * s

    def act(self, s):
        action_prefs = self.Q @ s
        max_logit = np.max(action_prefs)
        probs = np.exp((action_prefs - max_logit) / self.softmaxtemp) / np.sum(np.exp((action_prefs
        - max_logit) / self.softmaxtemp))
        return self.policy_act(s, probs)

class ForwardKL(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "ForwardKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.softmaxtemp = self.agent_params["softmaxtemp"]
        self.trueQ_entropies = []
        self.integration = self.agent_params["integration"]
        self.sacupdate = self.agent_params["sacupdate"]
        

    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        # at each step, optimize forward KL D(Q || pi)
        # \nabla D(Q || pi) = - \sum_i exp(Q(i)) / Z * \nabla pi(i) / pi(i)
        probs = self.get_action_probs(s)
        self.soft_update_qv(s, a, r, sp, done, self.sacupdate)

        policy_grad = self.get_softmax_jacobian(s)

        Qs = self.Q @ s
        max_Q = np.max(Qs)
        boltzQ = np.exp((Qs - max_Q) / self.softmaxtemp) / np.sum(np.exp((Qs - max_Q) / self.softmaxtemp))
        # print(weighting.shape)
        if self.integration == 1:
            weighting = np.divide(boltzQ, probs)
            self.policy += self.agent_params["alr"] * np.sum(policy_grad * weighting.reshape(1, 1, -1), axis = -1)
        else:
            # sample from the boltzman Q distribution
            new_a = np.random.multinomial(1, boltzQ).nonzero()[0][0]
            # print(new_a)
            # print(policy_grad[:, :, new_a].shape, probs[new_a].shape)
            self.policy += self.agent_params["alr"] * policy_grad[:, :, new_a] / probs[new_a]

class ReverseKL(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "ReverseKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.softmaxtemp = self.agent_params["softmaxtemp"]
        self.trueQ_entropies = []
        self.integration = self.agent_params["integration"]
        self.sacupdate = self.agent_params["sacupdate"]



    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        # at each step, optimize the reverse KL D(pi || Q)
        # \nabla D(pi || Q) = - \sum_i Q(s, i) \nabla pi(i) - H(pi)
        probs = self.get_action_probs(s)
        self.soft_update_qv(s, a, r, sp, done, self.sacupdate)

        softmax_jacobian = self.get_softmax_jacobian(s)
        if self.integration == 1:
            # going along columns changes which pi(i) is looked at
            term1 = - np.sum(softmax_jacobian * (self.Q @ s).reshape(1, 1, -1), axis = -1)
            # H(pi) = -\sum_i p_i log (p_i)
            # nabla H(pi) = -(\sum_i log(p_i) nabla p_i + \sum_i \nabla p_i)  
            term2 = np.sum(softmax_jacobian * np.log(probs).reshape((1, 1, -1)), axis = -1) + np.sum(softmax_jacobian, axis = -1)
            self.policy -= self.agent_params["alr"] * (term1 + self.softmaxtemp * term2)
        else:
            term1 = - (self.Q @ s)[a] * softmax_jacobian[:, :, a] / probs[a]
            term2 = np.sum(softmax_jacobian * np.log(probs).reshape((1, 1, -1)), axis = -1)  + softmax_jacobian[:, :, a] / probs[a]
            self.policy -= self.agent_params["alr"] * (term1 + self.softmaxtemp * term2)
            

class HardForwardKL(BaseAgent):
    """ Optimizes policy based on minimizing -log pi(max_action | s) """
    def __init__(self, *args, **kwargs):
        self.base_name = "HardForwardKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.trueQ_entropies = []
        self.sacupdate = self.agent_params["sacupdate"]

        
    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        probs = self.get_action_probs(s)
        self.soft_update_qv(s, a, r, sp, done, self.sacupdate)

        softmax_jacobian = self.get_softmax_jacobian(s)

        max_action = utils.rand_argmax(self.Q @ s)

        self.policy += self.agent_params["alr"] * softmax_jacobian[:, :, max_action] / probs[max_action]


class HardReverseKL(BaseAgent):
    """ Just Actor-Expert with SGD. The gradient is \sum_a \nabla pi(a | s) Q(s, a)"""
    def __init__(self, *args, **kwargs):
        self.base_name = "HardReverseKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.trueQ_entropies = []
        self.integration = self.agent_params["integration"]
        self.sacupdate = self.agent_params["sacupdate"]



    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        probs = self.get_action_probs(s)
        self.soft_update_qv(s, a, r, sp, done, self.sacupdate)


        softmax_jacobian = self.get_softmax_jacobian(s)
        if self.integration == 1:
            # going along columns changes which pi(i) is looked at
            term1 = - np.sum(softmax_jacobian * (self.Q @ s).reshape(1, 1, -1), axis = -1)
            # H(pi) = -\sum_i p_i log (p_i)
            # nabla H(pi) = -(\sum_i log(p_i) nabla p_i + \sum_i \nabla p_i)  
            self.policy -= self.agent_params["alr"] * term1
        else:
            term1 = - (self.Q @ s)[a] * softmax_jacobian[:, :, a] / probs[a]
            self.policy -= self.agent_params["alr"] * term1 
            
