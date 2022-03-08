import os

for i in range(125 * 5):
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/fkl.json --index {}' >> commands.txt".format(i))
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/rkl.json --index {}' >> commands.txt".format(i))

    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/MultimodalContinuousWorld.json --agent_json jsonfiles/agent/world/fkl.json --index {}' >> commands.txt".format(i))
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/MultimodalContinuousWorld.json --agent_json jsonfiles/agent/world/rkl.json --index {}' >> commands.txt".format(i))

# for i in range(375 * 5):
#     os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/fkl_gmm.json --index {}' >> commands.txt".format(i))
#     os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/rkl_gmm.json --index {}' >> commands.txt".format(i))
#
#     os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/MultimodalContinuousWorld.json --agent_json jsonfiles/agent/world/fkl_gmm.json --index {}' >> commands.txt".format(i))
#     os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/MultimodalContinuousWorld.json --agent_json jsonfiles/agent/world/rkl_gmm.json --index {}' >> commands.txt".format(i))