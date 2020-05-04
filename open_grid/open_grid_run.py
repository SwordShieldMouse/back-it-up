import sys
import numpy as np
from lib.pg_obj.src.agents.tabular_agents import ForwardKL, ReverseKL, HardForwardKL, HardReverseKL
import open_grid.open_grid_utils as open_grid_utils
import datetime
import os
import subprocess
# ========= Taken from tabular_run.py

SEEDS = [609, 8166, 1286, 3403, 398, 404, 2757, 5536, 3535, 5768, 6034, 5703, 1885, 6052, 6434, 3026, 4009, 4212, 2829, 7483, 2267, 2861, 1444, 4950, 1845, 4048, 2521, 9204, 5936, 4626]

LRS = [0.3, 0.6, 0.9]
ALRS = LRS
SOFTMAXTEMPS = [0.01, 0.1, 0.5, 1]
SOFTQTEMPS = [0, 0.1, 0.5, 1]

GAMMAS = [0.99]
INTEGRATION = [1]
SAC_UPDATE = [0]
FA = ["tabular"] # just for plotting purposes

HYPERPARAMS = {"sacupdate": SAC_UPDATE, "integration": INTEGRATION, "gamma": GAMMAS, "alr": ALRS, "lr": LRS, "softmaxtemp": SOFTMAXTEMPS, "softQtemp": SOFTQTEMPS, "fa": FA}
ALGS = [(HardForwardKL, ["alr","lr", "softQtemp", "gamma", "sacupdate", "fa"]),
        (HardReverseKL, ["sacupdate", "integration", "gamma", "alr", "lr", "softQtemp", "fa"]),
        (ReverseKL, ["sacupdate", "integration", "gamma", "alr", "lr", "softQtemp", "softmaxtemp", "fa"]),
        (ForwardKL, ["sacupdate", "integration", "alr", "lr", "softQtemp", "softmaxtemp", "gamma", "fa"])]

# =========

# OpenGrid Config

M, N = 10, 10
terminalStates = {
            (9, 9): 1
            # (9, 9): 0
        }
stepRewards = [-10, -0.01]

# -10, Reverse
reverse_agent_params = { "alr": 0.3,
                "lr": 0.6,
                "softQtemp": 0.5,
                "softmaxtemp": 1.0,

                "gamma": 0.99,
                "integration": 1,
                "sacupdate": 0,
                "noisyQ": 0,
                "learnQ": 1,
                "allstates": 0}

# -10, Forward
forward_agent_params = { "alr": 0.9,
                "lr": 0.6,
                "softQtemp": 0.5,
                "softmaxtemp": 1.0,

                "gamma": 0.99,
                "integration": 1,
                "sacupdate": 0,
                "noisyQ": 0,
                "learnQ": 1,
                "allstates": 0}


def main():
    timeStamp = datetime.datetime.now().strftime('%m_%d_%H%M%S')
    print("Start run at: {}".format(timeStamp))

    saveDir = 'results/0_openGrid/{}'.format(timeStamp)

    index = int(sys.argv[1])
    max_frames = int(sys.argv[2])
    max_frames_per_ep = int(sys.argv[3])



    if not os.path.exists(saveDir):
        os.makedirs(saveDir)

    env_params = {"max_frames": max_frames, "max_frames_per_ep": max_frames_per_ep}

    # Create env and agent
    env = open_grid_utils.OpenGrid(M, N, terminalStates, 0.99, stepRewards[0])

    if sys.argv[4] == 'forward':
        agent_params = forward_agent_params
        agent = ForwardKL(env=env, agent_params=agent_params, env_params=env_params, use_ep_length_r=True)
    elif sys.argv[4] == 'reverse':
        agent_params = reverse_agent_params
        agent = ReverseKL(env=env, agent_params=agent_params, env_params=env_params, use_ep_length_r=True)
    else:
        raise ValueError("Invalid agent")

    # extracted from agent.run : Run method should be outside the agent
    frame = 0
    ep = 0
    while frame < env_params["max_frames"]:
        s = env.reset()

        G = 0
        done = False
        curr_frame_count = 0
        while done is not True:
            # self.env.render()
            a = agent.act(s)
            env.computeTrueVal(agent_params["softQtemp"], agent.compute_learnedV(), agent.all_probs[-1], saveDir+'/figures', ep, frame)
            sp, r, done, _ = env.step(a)

            # G += (self.gamma ** curr_frame_count) * r # if we want to use the discounted return as the eval metric
            G += r

            agent.step(s, a, r, sp, done)
            curr_frame_count += 1
            if curr_frame_count >= env_params["max_frames_per_ep"]:
                # NOTE: THIS NEEDS TO BE AFTER STEP SO THAT WE BOOTSTRAP CORRECTLY
                done = True
            # clip action values if applicable
            if "clip" in agent_params:
                for s in range(len(agent.Q)):
                    agent.Q[s] = np.clip(agent.Q[s], -agent.agent_params["clip"], agent.agent_params["clip"])
            s = sp
            frame += 1
        if agent.use_ep_length_r is True:
            agent.G_buffer[ep % agent.window] = -curr_frame_count
        else:
            agent.G_buffer[ep % agent.window] = G
        # give episode lengths as return
        agent.returns.append(G)
        agent.avg_rewards += [np.nanmean(
            agent.G_buffer)] * curr_frame_count  # don't start recording until 10 eps have been completed?
        agent.plotting_data.append([np.nanmean(agent.G_buffer), curr_frame_count])

        ep += 1
        print("ep = {} | frame = {} | G = {} | avg return = {} | ep length = {}".format(ep, frame - 1, G,
                                                                                        agent.avg_rewards[-1],
                                                                                        curr_frame_count))
    # when done, ensure that number of avg returns matches number of frames
    agent.avg_rewards = agent.avg_rewards[:env_params["max_frames"]]
    agent.plotting_data[-1][-1] -= np.sum(agent.plotting_data, axis=0)[-1] - env_params["max_frames"]
    assert np.sum(agent.plotting_data, axis=0)[-1] == env_params["max_frames"]

    subprocess.run(
        ["ffmpeg", "-framerate", "24", "-i", "{}/figures/steps_%01d.png".format(saveDir), "{}.mp4".format(saveDir)])
if __name__ == '__main__':
    main()
