import gym
import env
import PPO_model
import PPO_model_dag_train
import torch
import time
import os
import copy

def get_validate_env(env_paras):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    #file_path = "/home/zss/data/00_FJSP/DAFJS_valid/{0}j_{1}m/".format(env_paras["num_jobs"], env_paras["num_mas"])
    file_path = "/home/zss/data/00_FJSP/DAFJS_valid/{0}j_{1}m/".format(env_paras["num_jobs"], env_paras["num_mas"])
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path+valid_data_files[i]
    print("valid_data_files=",valid_data_files)
    env = gym.make('fjsp-v2', case=valid_data_files, env_paras=env_paras, data_source='file')
    return env

def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = PPO_model_dag_train.Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    gantt_result = env.validate_gantt()[0]
    # if not gantt_result:
    #     print("Scheduling Error！！！！！！")
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch
