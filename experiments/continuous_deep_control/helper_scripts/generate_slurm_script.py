import os

for i in range(125 * 5):
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/fkl.json --index {}' >> cedar.txt".format(i))
    os.system("echo 'pipenv run python nonlinear_run.py --env_json jsonfiles/environment/EasyContinuousWorld.json --agent_json jsonfiles/agent/world/rkl.json --index {}' >> cedar.txt".format(i))