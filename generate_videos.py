from multiprocessing import Process, Semaphore
import os

with open("experiments/continuous_deep_control/slurm_scripts/video_runs/video_runs.txt", "r") as f:
    ffile = f.read().splitlines() 

def f(sem, command, device, idx):
    sem.acquire()
    os.system('export CUDA_VISIBLE_DEVICES={}; '.format(device) + command + " > {}.txt".format(idx))
    sem.release()

sem = [Semaphore(2), Semaphore(2)]

p_count = 0
p_list = []

for idx, l in enumerate(ffile):
    device = idx % 2
    s_sem = sem[idx%2]
    i_args = {
        "sem": s_sem,
        "command": l,
        "device": device,
        "idx": idx
    }
    p = Process(target=f, kwargs=i_args)
    p.start()
    p_list.append( p )
    p_count += 1

    if len(p_list) >= 10:
        for pp in p_list:
            pp.join()
        p_list = []    