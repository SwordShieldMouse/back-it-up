import numpy as np
from datetime import datetime
import time

        
class Experiment(object):
    def __init__(self, agent, train_environment, test_environment, seed, writer, write_log, write_plot):
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

    def run(self):

        episode_count = 0

        # For total time
        start_run = datetime.now()
        print("Start run at: " + str(start_run)+'\n')

        # evaluate once at beginning
        self.cum_eval_time += self.eval()
        
        while self.total_step_count < self.train_environment.TOTAL_STEPS_LIMIT:
            # runs a single episode and returns the accumulated reward for that episode
            train_start_time = time.time()
            episode_reward, num_steps, force_terminated, eval_session_time = self.run_episode_train(is_train=True)
            train_end_time = time.time()

            train_ep_time = train_end_time - train_start_time - eval_session_time

            self.cum_train_time += train_ep_time
            print("Train:: ep: " + str(episode_count) + ", r: " + str(episode_reward) + ", n_steps: " + str(num_steps) + ", elapsed: " + time.strftime("%H:%M:%S", time.gmtime(train_ep_time)))

            if not force_terminated: 
                self.train_rewards_per_episode.append(episode_reward)
                self.train_cum_steps.append(self.total_step_count)
        
            episode_count += 1

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
            
            obs_n, reward, done, info = test_env.step(Aold)

            episode_reward += reward  
            if not done:          
                Aold = self.agent.step(obs_n, is_train)

            obs = obs_n
            episode_step_count += 1

        return episode_reward, episode_step_count
