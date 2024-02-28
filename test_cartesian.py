import copy
import json
import os
import random
import time as time

import gym
import pandas as pd
import torch
import numpy as np

import pynvml
import PPO_model_cartesian


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # PyTorch initialization
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu)
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type=='cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # Load config and init objects
    with open("./config_cartesian.json", 'r') as load_f:
        load_dict = json.load(load_f)
    env_paras = load_dict["env_paras"]  # "env_paras": {"num_jobs": 10,"num_mas": 5,"batch_size": 20, "ope_feat_dim" : 6,
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]
    test_paras = load_dict["test_paras"] # "num_ins": 100,"rules": ["DRL"],"sample": false,"num_sample": 100,"data_path": "1005"
    env_paras["device"] = device
    model_paras["device"] = device
    env_test_paras = copy.deepcopy(env_paras)
    num_ins = test_paras["num_ins"]
    num_ins=1
    if test_paras["sample"]:
        env_test_paras["batch_size"] = test_paras["num_sample"]  ##100
    else:
        env_test_paras["batch_size"] = 1
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    #data_path = "./data_test/{0}/".format(test_paras["data_path"]) ##1005
    read_dafjs="./data_test/read_DAFJS"
    with open(read_dafjs) as file_object1:
        line1 = file_object1.readlines()
    print("line1=", line1)
    # print("line[1].strip().split()", line1[0].strip().split())


    # data_path="./data_test/dafjs/"
    data_path = "/home/zss/data/00_FJSP/DAFJS_train/{0}j_{1}m/".format(env_paras["num_jobs"], env_paras["num_mas"])

    test_files = os.listdir(data_path)
    test_files.sort(key=lambda x: str(x))
    test_files = test_files[:num_ins]
    print("test_files=",test_files)
    # test_files=['DAFJS01']
    # test_files = ['dafjs10']
    mod_files = os.listdir('./model/')[:]

    memories = PPO_model_cartesian.Memory()
    model = PPO_model_cartesian.PPO(model_paras, train_paras)
    rules = test_paras["rules"] ## DRL
    envs = []  # Store multiple environments

    # Detect and add models to "rules"
    if "DRL" in rules:
        for root, ds, fs in os.walk('./model/'):
            for f in fs:
                if f.endswith('.pt'):
                    rules.append(f)
    if len(rules) != 1:
        if "DRL" in rules:
            rules.remove("DRL")

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save_cartesian/test_{0}'.format(str_time)
    os.makedirs(save_path)
    writer = pd.ExcelWriter(
        '{0}/makespan_{1}.xlsx'.format(save_path, str_time))  # Makespan data storage path
    writer_time = pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time))  # time data storage path
    file_name = [test_files[i] for i in range(num_ins)]
    data_file = pd.DataFrame(file_name, columns=["file_name"])
    print("data_file=",data_file)
    data_file.to_excel(writer, sheet_name='Sheet1', index=False)
    writer.save()
    writer.close()
    data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)
    writer_time.save()
    writer_time.close()
    print("data_file.to_excel=",data_file)

    # Rule-by-rule (model-by-model) testing
    start = time.time()
    for i_rules in range(len(rules)):
        rule = rules[i_rules]
        # Load trained model
        if rule.endswith('.pt'):
            if device.type == 'cuda':
                model_CKPT = torch.load('./model/' + mod_files[i_rules])
                # model_CKPT = torch.load('/home/zss/data/00_FJSP/0118_no_print/save_dag_train'
                #                         '/train_20240120_234638/save_best_4_5_60.pt')
                # model_CKPT = torch.load(
                #     '/home/zss/data/00_FJSP/0118_no_print/save_dag_train/train_20240119_105216/save_best_4_5_190.pt')
                # model_CKPT =torch.load('/home/zss/data/00_FJSP/0118_no_print/save/train_20240108_162538/save_best_10_5_870.pt')
                # model_CKPT = torch.load('/home/zss/data/00_FJSP/0124_no_print/save_dag_train/train_20240125_160828/save_best_4_5_960.pt')
                model_CKPT = torch.load('/home/zss/data/00_FJSP/0124_no_print/save_dag_train/train_20240126_192133/save_best_4_5_610.pt')
            else:
                model_CKPT = torch.load('./model/' + mod_files[i_rules], map_location='cpu')
            print('\nloading checkpoint:', mod_files[i_rules])
            model.policy.load_state_dict(model_CKPT)
            model.policy_old.load_state_dict(model_CKPT)
        print('rule:', rule)

        # Schedule instance by instance
        step_time_last = time.time()
        makespans = []
        times = []
        for i_ins in range(num_ins):
            test_file = data_path + test_files[i_ins]
            print("test_file=",test_file)
            with open(test_file) as file_object:
                line = file_object.readlines()
                print("line=",line)
                # line_split =line1[i_ins].strip().split()
                line_split = line1[0].strip().split()
                # print("line_split",line_split)
                # ins_num_jobs = int(line_split[0])
                # ins_num_mas = int(line_split[1])
                ins_num_jobs = env_paras["num_jobs"]
                ins_num_mas = env_paras["num_mas"]
                print("ins_num_jobs=",ins_num_jobs)
                print("ins_num_mas=", ins_num_mas)

            env_test_paras["num_jobs"] = ins_num_jobs  ## update according to the actual dataset
            env_test_paras["num_mas"] = ins_num_mas

            # Environment object already exists
            if len(envs) == num_ins:
                env = envs[i_ins]
                print("len(envs) == num_ins")
            # Create environment object
            else:
                # Clear the existing environment
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                if meminfo.used / meminfo.total > 0.7:  ## prevent memory too large
                    envs.clear()
                # DRL-S, each env contains multiple (=num_sample) copies of one instance
                if test_paras["sample"]:
                    env = gym.make('fjsp-v3', case=[test_file] * test_paras["num_sample"],
                                   env_paras=env_test_paras, data_source='file')
                # DRL-G, each env contains one instance
                else:
                    print("test_file=",[test_file])
                    print("env_test_paras=",env_test_paras)
                    env = gym.make('fjsp-v3', case=[test_file], env_paras=env_test_paras, data_source='file')
                    ## if 'case' then generatre on the fly
                    ## has been registrated in env_init
                envs.append(copy.deepcopy(env))
                print("len(envs)=",len(envs))
                print("Create env[{0}]".format(i_ins))

            # Schedule an instance/environment
            # DRL-S
            if test_paras["sample"]:
                makespan, time_re = schedule(env, model, memories, flag_sample=test_paras["sample"])
                makespans.append(torch.min(makespan))
                times.append(time_re)
            # DRL-G
            else:
                time_s = []
                makespan_s = []  # In fact, the results obtained by DRL-G do not change
                # for j in range(test_paras["num_average"]):  ## 10 as average
                for j in range(1):  ## 10 as average
                    makespan, time_re = schedule(env, model, memories)
                    makespan_s.append(makespan)
                    time_s.append(time_re)
                    env.reset()
                    print("env.reset()")
                print("makespan_s",makespan_s)
                makespans.append(torch.mean(torch.tensor(makespan_s)))
                times.append(torch.mean(torch.tensor(time_s)))
                print("makespans=",makespans)
                average_makespan = torch.mean(torch.tensor(makespans))
                print("average_makespan=", average_makespan)
            print("finish env {0}".format(i_ins))
        print("rule_spend_time: ", time.time() - step_time_last)

        # Save makespan and time data to files
        writer = pd.ExcelWriter(
            '{0}/makespan_{1}.xlsx'.format(save_path, str_time))  # Makespan data storage path
        writer_time = pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time))  # time data storage path
        data = pd.DataFrame(torch.tensor(makespans).t().tolist(), columns=[rule])
        data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=i_rules + 1)
        writer._save()
        writer.close()
        data = pd.DataFrame(torch.tensor(times).t().tolist(), columns=[rule])
        data.to_excel(writer_time, sheet_name='Sheet1', index=False, startcol=i_rules + 1)
        writer_time._save()
        writer_time.close()

        for env in envs:
            # print("env in envs: env.reset()")
            env.reset()  ##reset 100 env for different rule

    print("total_spend_time: ", time.time() - start)

def schedule(env, model, memories, flag_sample=False):
    # Get state and completion signal
    state = env.state
    dones = env.done_batch
    done = False  # Unfinished at the beginning
    last_time = time.time()
    i = 0
    while ~done:
        i += 1
        with torch.no_grad():
            actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
        # print("action=",actions)
        state, rewards, dones = env.step(actions)  # environment transit
        done = dones.all()
    spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
    # print("spend_time: ", spend_time)
    env.render()

    # Verify the solution
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    return copy.deepcopy(env.makespan_batch), spend_time


if __name__ == '__main__':
    main()