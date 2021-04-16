import numpy as np
from datetime import datetime
import time
import os
import pickle
import torch
# from guppy import hpy

# h = hpy()
        
class Experiment(object):
    def __init__(self, agent, train_environment, test_environment, seed, writer, write_log, write_plot, resume_params):
        self.agent = agent
        self.train_environment = train_environment
        self.train_environment.set_random_seed(seed)

        # for eval purpose
        self.test_environment = test_environment  # copy.deepcopy(environment) # this didn't work for Box2D env
        self.test_environment.set_random_seed(seed)

        self.train_rewards_per_episode = []
        self.train_cum_steps = []
        self.eval_mean_rewards_per_episode = []
        self.eval_std_rewards_per_episode = []

        self.total_step_count = 0
        self.writer = writer

        # boolean to log result for tensorboard
        self.write_log = write_log
        self.write_plot = write_plot

        self.cum_train_time = 0.0
        self.cum_eval_time = 0.0

        # save/resume params
        self.resume_training = resume_params['resume_training']
        self.save_data_bdir = resume_params['save_data_bdir']
        self.save_data_interval = resume_params['save_data_interval']
        self.save_data_fname = resume_params['save_data_fname']
        # save params ContinuousMaze
        self.steps_per_netsave = resume_params['steps_per_netsave']
        self.no_netsave = resume_params['no_netsave']
        self.netsave_data_bdir = resume_params['netsave_data_bdir']

    @profile
    def run(self):

        self.episode_count = 0

        # For total time
        start_run = datetime.now()
        print("Start run at: " + str(start_run)+'\n')

        # evaluate once at beginning
        self.cum_eval_time += self.eval()

        if self.resume_training:
            self.link_variables_and_names()
            self.load_data()
        
        self.last_time_saved = time.time()
        while self.total_step_count < self.train_environment.TOTAL_STEPS_LIMIT:
            if (self.episode_count + 1) % self.save_data_interval == 0 and self.resume_training:
                self.save_data()
                self.last_time_saved = time.time()
            elif self.resume_training:
                time_since_save = time.time() - self.last_time_saved
                if time_since_save >= 600:
                    self.save_data()
                    self.last_time_saved = time.time()
            # runs a single episode and returns the accumulated reward for that episode
            train_start_time = time.time()
            episode_reward, num_steps, force_terminated, eval_session_time = self.run_episode_train(is_train=True)
            train_end_time = time.time()

            train_ep_time = train_end_time - train_start_time - eval_session_time

            self.cum_train_time += train_ep_time
            print("Train:: ep: " + str(self.episode_count) + ", r: " + str(episode_reward) + ", n_steps: " + str(num_steps) + ", elapsed: " + time.strftime("%H:%M:%S", time.gmtime(train_ep_time)))

            if not force_terminated: 
                self.train_rewards_per_episode.append(episode_reward)
                self.train_cum_steps.append(self.total_step_count)
        
            self.episode_count += 1

        self.train_environment.close()  # clear environment memory

        end_run = datetime.now()
        print("End run at: " + str(end_run)+'\n')
        print("Total Time taken: "+str(end_run - start_run) + '\n')
        print("Training Time: " + time.strftime("%H:%M:%S", time.gmtime(self.cum_train_time)))
        print("Evaluation Time: " + time.strftime("%H:%M:%S", time.gmtime(self.cum_eval_time)))

        return self.train_rewards_per_episode, self.eval_mean_rewards_per_episode, self.eval_std_rewards_per_episode, self.train_cum_steps

    # Runs a single episode (TRAIN)
    def run_episode_train(self, is_train):

        eval_session_time = 0.0

        obs = self.train_environment.reset()
        self.agent.reset()  # Need to be careful in Agent not to reset the weight

        episode_reward = 0.
        done = False
        Aold = self.agent.start(obs, is_train)

        episode_step_count = 0

        while not (done or episode_step_count == self.train_environment.EPISODE_STEPS_LIMIT or self.total_step_count == self.train_environment.TOTAL_STEPS_LIMIT):
            if self.train_environment.name == "ContinuousMaze" and self.total_step_count % self.steps_per_netsave == 0 and self.no_netsave is False:
                netsave_dir = os.path.join(self.netsave_data_bdir,os.path.splitext(self.save_data_fname)[0], '{}'.format(self.total_step_count))
                if not os.path.isdir(netsave_dir):
                    os.makedirs(netsave_dir, exist_ok=True)
                self.save_nets_custom_path(netsave_dir)
            episode_step_count += 1
            self.total_step_count += 1

            obs_n, reward, done, info = self.train_environment.step(Aold)
            episode_reward += reward

            # if the episode was externally terminated by episode step limit, don't do update
            # (except ContinuousBandits, where the episode is only 1 step)
            if self.train_environment.name.startswith('ContinuousBandits'):
                is_truncated = False
            else:
                if done and episode_step_count == self.train_environment.EPISODE_STEPS_LIMIT:
                    is_truncated = True
                else:
                    is_truncated = False

            self.agent.update(obs, obs_n, float(reward), Aold, done, is_truncated)

            if not done:
                Aold = self.agent.step(obs_n, is_train)

            obs = obs_n

            if self.total_step_count % self.train_environment.eval_interval == 0:
                eval_session_time += self.eval()

        # check if this episode is finished because of Total Training Step Limit
        if not (done or episode_step_count == self.train_environment.EPISODE_STEPS_LIMIT):
            force_terminated = True
        else:
            force_terminated = False
        return episode_reward, episode_step_count, force_terminated, eval_session_time

    def link_variables_and_names(self):
        #Diverse counters
        self.sr_diverse_names = ['cum_eval_time', 'cum_train_time', 'total_step_count', 'episode_count','train_cum_steps', 'train_rewards_per_episode']
        self.sr_diverse_vars = [None] * len(self.sr_diverse_names)

        #Networks
        self.sr_nets_names = ['pi_net', 'q_net', 'v_net']
        self.sr_nets_vars = [self.agent.network_manager.network.pi_net, self.agent.network_manager.network.q_net, self.agent.network_manager.network.v_net]

        if self.agent.network_manager.use_target:
            self.sr_nets_names.append('target_v_net')
            self.sr_nets_vars.append(self.agent.network_manager.network.target_v_net)

        #Optimizers
        self.sr_optimizers_names = ['pi_optimizer', 'q_optimizer', 'v_optimizer']
        self.sr_optimizers_vars = [self.agent.network_manager.network.pi_optimizer, self.agent.network_manager.network.q_optimizer, self.agent.network_manager.network.v_optimizer]

        #Replay buffer
        self.sr_buffer_names = ['replay_buffer']
        self.sr_buffer_vars = [self.agent.replay_buffer]

        #Join all variables
        self.sr_all_names = self.sr_diverse_names + self.sr_nets_names + self.sr_optimizers_names + self.sr_buffer_names
        self.sr_all_vars = self.sr_diverse_vars + self.sr_nets_vars + self.sr_optimizers_vars + self.sr_buffer_vars

    def save_data(self):

        sr_all_vars_state_dicts = [getattr(self, n) for n in self.sr_diverse_names] + [a.state_dict() for a in self.sr_nets_vars] + [a.state_dict() for a in self.sr_optimizers_vars] + [pickle.dumps(a) for a in self.sr_buffer_vars]

        out_dict = dict(zip(self.sr_all_names, sr_all_vars_state_dicts))

        out_temp_fname = os.path.join(self.save_data_bdir, 'temp_' + self.save_data_fname)
        out_fname = os.path.join(self.save_data_bdir, self.save_data_fname)

        torch.save(out_dict, out_temp_fname)
        os.rename(out_temp_fname, out_fname)

    def save_nets_custom_path(self, cpath):

        sr_nets_names = ['pi_net', 'q_net', 'v_net']
        sr_nets_vars = [self.agent.network_manager.network.pi_net, self.agent.network_manager.network.q_net, self.agent.network_manager.network.v_net]

        sr_all_vars_state_dicts = [a.state_dict() for a in sr_nets_vars]

        out_dict = dict(zip(sr_nets_names, sr_all_vars_state_dicts))

        out_temp_fname = os.path.join(cpath, 'temp_' + self.save_data_fname)
        out_fname = os.path.join(cpath, self.save_data_fname)

        torch.save(out_dict, out_temp_fname)
        os.rename(out_temp_fname, out_fname)        

    def load_data(self):
        in_fname = os.path.join(self.save_data_bdir, self.save_data_fname)
        if os.path.isfile(in_fname):
            checkpoint = torch.load(in_fname)
            for name, var in zip(self.sr_all_names, self.sr_all_vars):
                if name in self.sr_optimizers_names or name in self.sr_nets_names:
                    var.load_state_dict(checkpoint[name])
                elif name in self.sr_buffer_names:
                    tmp = pickle.loads(checkpoint[name])
                    for ii in dir(var):
                        if ii.startswith('__'):
                            continue
                        if hasattr(var, ii):
                            setattr(var, ii, getattr(tmp, ii) )
                elif name in self.sr_diverse_names:
                    setattr(self, name, checkpoint[name])
                else:
                    raise NotImplementedError

    def eval(self):
        temp_rewards_per_episode = []

        eval_session_time = 0.0

        for i in range(self.test_environment.eval_episodes):
            eval_start_time = time.time()
            episode_reward, num_steps = self.run_episode_eval(self.test_environment, is_train=False)
            eval_end_time = time.time()
            temp_rewards_per_episode.append(episode_reward)

            eval_elapsed_time = eval_end_time - eval_start_time

            eval_session_time += eval_elapsed_time
            print("=== EVAL :: ep: " + str(i) + ", r: " + str(episode_reward) + ", n_steps: " + str(num_steps) + ", elapsed: " + time.strftime("%H:%M:%S", time.gmtime(eval_elapsed_time)))

        mean = np.mean(temp_rewards_per_episode)

        self.eval_mean_rewards_per_episode.append(mean)
        self.eval_std_rewards_per_episode.append(np.std(temp_rewards_per_episode))

        self.cum_eval_time += eval_session_time

        return eval_session_time

    # Runs a single episode (EVAL)
    def run_episode_eval(self, test_env, is_train):
        obs = test_env.reset()
        self.agent.reset()

        episode_reward = 0.
        done = False
        Aold = self.agent.start(obs, is_train)

        episode_step_count = 0
        while not (done or episode_step_count == test_env.EPISODE_STEPS_LIMIT):
            if self.train_environment.name == "ContinuousMaze" and episode_step_count == 5000:
                break                
            
            obs_n, reward, done, info = test_env.step(Aold)

            episode_reward += reward  
            if not done:          
                Aold = self.agent.step(obs_n, is_train)

            obs = obs_n
            episode_step_count += 1

        return episode_reward, episode_step_count
