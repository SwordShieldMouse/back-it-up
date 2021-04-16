import os 
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical
import time
import numpy as np
import copy

import src.utils.replay as replay
import src.utils.math_utils as utils
import src.representations.minatar_nets as minatar_nets
import src.representations.deep_discrete_nets as deep_discrete_nets

torch.set_default_dtype(torch.float)
torch.set_printoptions(precision = 8)
dtype = torch.float
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

OPTIMS = {"adam": torch.optim.Adam, "rmsprop": torch.optim.RMSprop, "sgd": torch.optim.SGD}

class BaseAgent():
    def __init__(self, env, env_params, agent_params, write_summaries=False, names_to_suppress=["replay", "seed"], use_ep_length_r = False):
        self.env = env 
        self.env_params = env_params
        self.agent_params = agent_params 
        self.write_summaries = write_summaries
        self.names_to_suppress = names_to_suppress
        self.name = self.generate_name()

        # RL stuff
        optimizer = OPTIMS[self.agent_params["optim"]]
        if self.agent_params["fa"] == "minatar":
            in_channels = self.env.in_channels
            num_actions = self.env.num_actions
        n_hidden = self.agent_params["hidden"]
        self.gamma = self.agent_params["gamma"]
        if self.agent_params["fa"] == "deep":
            self.VQPolicy = deep_discrete_nets.VQPolicy(self.env.obs_dim, self.env.action_dim, n_hidden)
            self.target_net = copy.deepcopy(self.VQPolicy)
            self.target_net.requires_grad = False
        elif self.agent_params["fa"] == "minatar":
            self.VQPolicy = minatar_nets.MinAtarVQPolicy(num_actions, n_hidden, in_channels).to(device)
        self.optim = optimizer(self.VQPolicy.parameters(), lr = self.agent_params["lr"])

        # self.use_exp_replay = self.agent_params["replay"]
        self.use_exp_replay = True
        self.replay = replay.Replay(32, int(1e5), self.env.reset().shape)
        self.use_target = False # default false but this is changed by the individual algs on a case-by-case basis

        # for stats 
        self.use_ep_length_r = use_ep_length_r
        self.avg_rewards = []
        self.returns = []
        self.window = 20
        self.G_buffer = [np.nan] * self.window
        self.entropies = []
        self.td_errors = []
        self.t = 0

    def generate_name(self):
        # generate agent full name from agent_params and base_name 
        params = ["{}={}".format(name, self.agent_params[name]) for name in self.agent_params if name not in self.names_to_suppress]
        return "{}_{}".format(self.base_name, '_'.join(params))

    def act(self, s):
        pass 

    def soft_update_qv(self, s, a, r, sp, done, sac_update = 1):
        probs = self.policy(s)
        if sac_update == 1:
            v_loss = torch.pow(self.Q(s)[a].detach() - self.softQtemp * torch.log(probs[a] + 1e-5).detach() - self.V(s), 2)
        else:
            v_loss = torch.pow(r - self.softQtemp * torch.log(probs[a] + 1e-5).detach() + self.gamma * (1. - done) * self.V(sp).detach() - self.V(s), 2)
        q_loss = torch.pow(r + self.gamma * (1. - done) * self.V(sp).detach() - self.Q(s)[a], 2)
        return q_loss, v_loss
    
    def batch_soft_update_qv(self, QS, VS, piS, A, R, VSp, D, sac_update):
        # print(A.shape)
        A.unsqueeze_(-1)
        gather_t = 0
        t0 = time.time()
        QSA = torch.gather(QS, -1, A).squeeze()
        with torch.no_grad():
            piSA = torch.gather(piS, -1, A).squeeze()
        gather_t += time.time() - t0
        VSp.squeeze_()
        VS.squeeze_()
        calc_t = 0
        t0 = time.time()
        if sac_update == 1:
            v_loss = torch.pow(QSA.detach() - self.softQtemp * torch.log(piSA + 1e-5) - VS, 2) 
            q_loss = torch.pow(R + self.gamma * (1. - D) * VSp.detach() - QSA, 2)
        else:
            with torch.no_grad():
                entropy = - self.softQtemp * torch.log(piSA + 1e-5)
                target = R + self.gamma * (1. - D) * VSp.detach()
            v_loss = torch.pow(target + entropy - VS, 2)
            q_loss = torch.pow(target - QSA, 2)
        calc_t += time.time() - t0
        # print(calc_t / (calc_t + gather_t), gather_t / (calc_t + gather_t))
        assert q_loss.numel() == QSA.numel(), (q_loss.shape, QSA.shape)
        assert v_loss.numel() == VS.numel(), (v_loss.shape, VS.shape)
        return q_loss, v_loss

    def policy_act(self, s):
        with torch.no_grad():
            probs = self.VQPolicy(s)[-1]
            m = Categorical(probs = probs)
            action = m.sample()
        return action.item()

    def sample_buffer(self):
        S, A, R, Sp, D = self.replay.sample()
        
        S = S.to(device)
        A = A.to(device)
        R = R.to(device)
        Sp = Sp.to(device)
        D = D.to(device)

        return S, A, R, Sp, D

    def step(self, s, a, r, sp, done):
        pass

    def run(self):
        # initialize optimizer here to allow easy overriding of default function approximator 
        self.optim = OPTIMS[self.agent_params["optim"]](self.VQPolicy.parameters(), lr = self.agent_params["lr"])
        self.frame = 0
        ep = 0
        t_start = time.time()
        env_time = 0
        step_time = 0
        print_freq = 100
        while self.frame < self.env_params["max_frames"]:
            s = torch.tensor(self.env.reset(), dtype = dtype, device = device)
            G = 0
            done = False
            self.curr_frame_count = 0
            while done is not True:
                
                a = self.act(s)

                t0 = time.time()
                sp, r, done, _ = self.env.step(a)
                env_time += time.time() - t0
                
                G += r
                sp = torch.tensor(sp, dtype = dtype, device = device)
                
                t0 = time.time()
                self.step(s, a, r, sp, done)
                step_time += time.time() - t0

                self.curr_frame_count += 1
                if self.curr_frame_count >= self.env_params["max_frames_per_ep"]:
                    # NOTE: THIS NEEDS TO BE AFTER STEP SO THAT WE BOOTSTRAP CORRECTLY
                    done = True
                s = sp
                self.frame += 1
                if self.use_target is True:
                    if (self.frame % self.sync_period == 0):
                        # sync target net
                        self.target_net.load_state_dict(self.VQPolicy.state_dict())     
            if self.use_ep_length_r is True:
                self.G_buffer[ep % self.window] = -self.curr_frame_count
            else:
                self.G_buffer[ep % self.window] = G
            self.returns.append((G, self.curr_frame_count))
            self.avg_rewards += [np.nanmean(self.G_buffer)] * self.curr_frame_count # don't start recording until 10 eps have been completed?
            ep += 1
            if ep % print_freq == 0:
                fps = round(self.frame / (time.time() - t_start), 2)
                step_time_frac = round(step_time / (step_time + env_time), 6)
                step_fps = round(self.frame / step_time, 3)
                env_time_frac = round(env_time / (step_time + env_time), 6)
                env_fps = round(self.frame / env_time, 3)
                print("ep = {} | frame = {} | G = {} | avg G = {} | ep length = {} | fps = {} | step time = {}, {} | env time = {}, {}".format(ep, self.frame - 1, G, self.avg_rewards[-1], self.curr_frame_count, fps, step_time_frac, step_fps, env_time_frac, env_fps))
        # when done, ensure that number of avg returns matches number of frames
        self.avg_rewards = self.avg_rewards[:self.env_params["max_frames"]]



class ForwardKL(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "ForwardKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.sacupdate = self.agent_params["sacupdate"]
        # self.softmaxtemp = self.agent_params["softmaxtemp"]
        # self.trueQ_entropies = []
        self.softmax_t = 0
        self.qv_t = 0
        

    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        # at each step, optimize forward KL D(Q || pi)
        # \nabla D(Q || pi) = - \sum_i exp(Q(i)) / Z * \nabla pi(i) / pi(i)
        # if self.agent_params["fa"] == "minatar":
        if self.use_exp_replay is True:
            # do exp replay for minatar
            self.replay.push(s, a, r, sp, done)
            S, A, R, Sp, D = self.sample_buffer()

            VS, QS, piS = self.VQPolicy(S)
            VSp, QSp, piSp = self.VQPolicy(Sp)

            t0 = time.time()
            with torch.no_grad():
                BQ = torch.softmax(QS / self.softQtemp, dim = -1)
            assert BQ.shape == QS.shape
            self.softmax_t += time.time() - t0

            t0 = time.time()
            q_loss, v_loss = self.batch_soft_update_qv(QS, VS, piS, A, R, VSp, D, self.sacupdate)
            self.qv_t += time.time() - t0

            inner = BQ * torch.log(piS + 1e-5)
            assert inner.shape == BQ.shape
            policy_loss = -torch.sum(inner, dim = -1)
        else:
            probs = self.policy(s)
            q_loss, v_loss = self.soft_update_qv(s, a, r, sp, done, self.sacupdate)
            Qs = self.Q(s).detach()
            max_Q = np.max(Qs.cpu().numpy())
            boltzQ = torch.exp((Qs - max_Q) / self.softQtemp) / torch.sum(torch.exp((Qs - max_Q) / self.softQtemp))
            # weighting = boltzQ / probs
            policy_loss = - torch.sum(boltzQ * torch.log(probs + 1e-5))

        # print(self.softmax_t / (self.qv_t + self.softmax_t), self.qv_t / (self.qv_t + self.softmax_t))

        loss = (q_loss + v_loss + policy_loss).mean()
        assert torch.isnan(loss) == 0, (q_loss, v_loss, policy_loss)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()



class ReverseKL(BaseAgent):
    def __init__(self, *args, **kwargs):
        self.base_name = "ReverseKL"
        super().__init__(*args, **kwargs)
        self.softQtemp = self.agent_params["softQtemp"]
        self.sacupdate = self.agent_params["sacupdate"]
        # self.softmaxtemp = self.agent_params["softmaxtemp"]
        # self.trueQ_entropies = []

    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        # at each step, optimize the reverse KL D(pi || Q)
        # \nabla D(pi || Q) = - \sum_i Q(s, i) \nabla pi(i) - H(pi)
        if self.use_exp_replay is True:
            self.replay.push(s, a, r, sp, done)
            S, A, R, Sp, D = self.sample_buffer()

            VS, QS, piS = self.VQPolicy(S)
            VSp, QSp, piSp = self.VQPolicy(Sp)
            q_loss, v_loss = self.batch_soft_update_qv(QS, VS, piS, A, R, VSp, D, self.sacupdate)
            with torch.no_grad():
                BQ = torch.softmax(QS / self.softQtemp, dim = 1)
            # print(BQ)
            policy_loss = torch.mean(torch.sum(piS * torch.log(piS / (BQ + 1e-5) + 1e-5), dim = 1))
            q_loss = q_loss.mean()
            v_loss = v_loss.mean()
        else:
            probs = self.policy(s)
            q_loss, v_loss = self.soft_update_qv(s, a, r, sp, done, self.sacupdate)

            policy_loss = -torch.sum(self.Q(s).detach() * probs) - self.softQtemp * torch.sum(-torch.log(probs + 1e-5) * probs)
            # print(policy_loss.shape, self.env.action_dim)

        assert torch.isnan(q_loss + v_loss + policy_loss) == 0, "nan loss"


        self.optim.zero_grad()
        (q_loss + v_loss + policy_loss).backward()
        self.optim.step()


class HardForwardKL(BaseAgent):
    """ Optimizes policy based on minimizing -log pi(max_action | s) """
    def __init__(self, *args, **kwargs):
        self.base_name = "HardForwardKL"
        super().__init__(*args, **kwargs)
        # self.softQtemp = self.agent_params["softQtemp"]
        self.softQtemp = 0
        self.sacupdate = self.agent_params["sacupdate"]
        # self.trueQ_entropies = []
        
    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        # if self.agent_params["fa"] == "minatar":
        if self.use_exp_replay is True:
            # do exp replay for minatar
            self.replay.push(s, a, r, sp, done)
            S, A, R, Sp, D = self.sample_buffer()
            VS, QS, piS = self.VQPolicy(S)
            VSp, QSp, piSp = self.VQPolicy(Sp)
            max_actions = torch.max(QS, dim = -1)[1]
            assert max_actions.numel() == S.shape[0], (max_actions.numel(), S.shape[0])
            piSmax = torch.gather(piS, -1, max_actions.unsqueeze(-1))
            assert piSmax.numel() == S.shape[0]

            q_loss, v_loss = self.batch_soft_update_qv(QS, VS, piS, A, R, VSp, D, self.sacupdate)
            policy_loss = -torch.mean(torch.log(piSmax + 1e-5))
            q_loss = q_loss.mean()
            v_loss = v_loss.mean()
        else:
            probs = self.policy(s)
            q_loss, v_loss = self.soft_update_qv(s, a, r, sp, done, self.sacupdate)

            max_action = utils.rand_argmax(self.Q(s).detach().cpu().numpy())
            policy_loss = - torch.log(probs[max_action] + 1e-5)
            self.V_optim.zero_grad()
            self.policy_optim.zero_grad()
            self.Q_optim.zero_grad()
        assert torch.isnan(q_loss + v_loss + policy_loss) == 0, "nan loss"
        
        self.optim.zero_grad()
        (q_loss + v_loss + policy_loss).backward()
        self.optim.step()


class HardReverseKL(BaseAgent):
    """ Just Actor-Expert with SGD. The gradient is \sum_a \nabla pi(a | s) Q(s, a)"""
    def __init__(self, *args, **kwargs):
        self.base_name = "HardReverseKL"
        super().__init__(*args, **kwargs)
        # self.softQtemp = self.agent_params["softQtemp"]
        self.softQtemp = 0
        self.sacupdate = self.agent_params["sacupdate"]
        # self.trueQ_entropies = []

    def act(self, s):
        return self.policy_act(s)

    def step(self, s, a, r, sp, done):
        # if self.agent_params["fa"] == "minatar":
        if self.use_exp_replay is True:
            # do exp replay for minatar
            self.replay.push(s, a, r, sp, done)
            S, A, R, Sp, D = self.sample_buffer()
            VS, QS, piS = self.VQPolicy(S)
            VSp, QSp, piSp = self.VQPolicy(Sp)
            q_loss, v_loss = self.batch_soft_update_qv(QS, VS, piS, A, R, VSp, D, self.sacupdate)
            all_actions = QS.detach() * piS
            assert all_actions.shape == QS.shape
            policy_loss = -torch.mean(torch.sum(all_actions, dim = -1), dim = 0)
            q_loss = q_loss.mean()
            v_loss = v_loss.mean()
        else:
            probs = self.policy(s)
            q_loss, v_loss = self.soft_update_qv(s, a, r, sp, done, self.sacupdate)
            
            policy_loss = -torch.sum(self.Q(s).detach() * probs)
            
        assert torch.isnan(q_loss + v_loss + policy_loss) == 0, "nan loss"
        self.optim.zero_grad()
        (q_loss + v_loss + policy_loss).backward()
        self.optim.step()