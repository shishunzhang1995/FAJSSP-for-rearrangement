import sys
import gym
import torch

from dataclasses import dataclass
from env_dag_train.load_data import load_fjs, nums_detec,get_index,start_time,start_time_estimate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import copy
from utils.my_utils import read_json, write_json


@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch:  torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ope_procing_batch: torch.Tensor = None
    mask_ope_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    num_opes:torch.Tensor = None
    former_batch:torch.Tensor = None
    later_batch:torch.Tensor=None
    nums_ope_batch:torch.Tensor=None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_ope_procing_batch, mask_ope_finish_batch, mask_ma_procing_batch, ope_step_batch, time,nums_ope_batch):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_ope_procing_batch = mask_ope_procing_batch
        self.mask_ope_finish_batch = mask_ope_finish_batch
        # self.mask_job_procing_batch = mask_job_procing_batch
        # self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time

def convert_feat_job_2_ope(feat_job_batch, opes_appertain_batch):
    '''
    Convert job features into operation features (such as dimension)
    '''
    return feat_job_batch.gather(1, opes_appertain_batch)

class FJSPEnv(gym.Env):
    '''
    FJSP environment
    '''
    def __init__(self, case, env_paras, data_source='case'):
        # print("enter env init")
        '''
        :param case: The instance generator or the addresses of the instances
        :param env_paras: A dictionary of parameters for the environment
        :param data_source: Indicates that the instances came from a generator or files: for train on the fly, for test on file
        '''

        # load paras
        # static
        self.show_mode = env_paras["show_mode"]  # Result display mode (deprecated in the final experiment)

        self.batch_size = env_paras["batch_size"]  # for train: 20 Number of parallel instances during training; for test 1
        # self.batch_size=2
        print("self.batch_size=",self.batch_size)
        self.num_jobs = env_paras["num_jobs"]  # Number of jobs 10
        self.num_mas = env_paras["num_mas"]  # Number of machines 5
        self.paras = env_paras  # Parameters
        self.device = env_paras["device"]  # Computing device for PyTorch

        # load instance
        num_data = 8  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)] #[[], [], [], [], [], [], [], []]
        # complete_opes=[[] for _ in range(self.batch_size)]
        # print("complete_opes=",complete_opes)
        self.former=[]
        self.later=[]
        self.nums_ope=[]  # [[7, 6, 5, 10], [7, 13, 5, 6], [5, 10, 6, 6], [9, 4, 7, 6], [7, 5, 8, 4]
        self.every_ope_number=[] ## [28,25,....,37,...]
        self.num_opes = 0
        self.num_prior = 0
        lines = []
        if data_source=='case':  # Generate instances through generators ## train on the fly ?
            for i in range(self.batch_size):
                lines.append(case.get_case(i)[0])  # Generate an instance and save it
                num_jobs = env_paras["num_jobs"]
                num_mas = env_paras["num_mas"]
                num_opes = int(lines[0].strip().split()[0])
                num_priors=int(lines[0].strip().split()[1])
                self.every_ope_number.append(num_opes)
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)
                self.num_prior=max(self.num_prior,num_priors)
        else:  # Load instances from files
            # print("case=", case)
            # print("len(case)=",len(case))
            for i in range(self.batch_size):
                with open(case[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)
                num_jobs = env_paras["num_jobs"]
                num_mas = env_paras["num_mas"]
                num_opes = int(line[0].strip().split()[0])
                num_priors = int(line[0].strip().split()[1])
                self.every_ope_number.append(num_opes)
                self.num_opes = max(self.num_opes, num_opes)  ## for test only 1
                self.num_prior = max(self.num_prior, num_priors)
                # Records the maximum number of operations (total) in the parallel 20 instances
        # load feats
        print("self.every_ope_number=",self.every_ope_number)
        # print("self.num_opes=",self.num_opes)  ## 50+
        # print("self.batch_size=",self.batch_size)
        for i in range(self.batch_size):
            # print("lines[i]=",lines[i])
            load_data = load_fjs(lines[i], num_mas, self.num_opes,self.num_prior)
            ### 8 kind of features for an instance
            for j in range(num_data):
                tensors[j].append(load_data[j])
            self.former.append(load_data[8])
            self.later.append(load_data[9])
            self.nums_ope.append(load_data[10])
        # print("self.nums_ope=",self.nums_ope) ## [[7, 6, 5, 10], [7, 13, 5, 6], [5, 10, 6, 6], [9, 4, 7, 6], [7, 5, 8, 4]

        # dynamic feats
        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # print("tensors[0]=",tensors[0])
        # print("self.proc_times_batch=",self.proc_times_batch)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # print("self.ope_ma_adj_batch=", self.ope_ma_adj_batch)
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()
        # print("self.cal_cumul_adj_batch=", self.cal_cumul_adj_batch)
        self.complete_opes= [[] for _ in range(self.batch_size)]
        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1  ##no use later
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances ## for train [0,1,...19]; for test [0]]
        # print("self.batch_idxes=",self.batch_idxes)
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        # print("self.time=",self.time)
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations of each batch
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        '''
        features, dynamic
            ope:
                Status
                Number of neighboring machines
                Processing time
                Number of unscheduled operations in the job
                Job completion time
                Start time
            ma:
                Number of neighboring operations
                Available time
                Utilization
        '''
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, self.paras["ope_feat_dim"], self.num_opes))  ##6
        feat_mas_batch = torch.zeros(size=(self.batch_size, self.paras["ma_feat_dim"], num_mas))   ##3

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        ## the number of machine can be choosed [4., 3., 3., 2., 3.
        print("feat_opes_batch[:, 0, :]=", feat_opes_batch[:, 0, :])
        print("feat_opes_batch[:, 1, :]=",feat_opes_batch[:, 1, :]) #ope_ma_adj_batch
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        ##[[15.3333, 10.6667, 16.2500,  6.0000, 14.4000  average operation time
        print("feat_opes_batch[:, 2, :]=", feat_opes_batch[:, 2, :])
        ##feat_opes_batch[:, 2, :]= tensor([[86.5000, 71.0000, 23.6667,  3.0000, 98.3333, 43.5000, 52.2500,  1.0000,
        feat_opes_batch[:, 3, :] = convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        ##[9., 9., 9., 9., 9., 9., 9., 9., 9., 5., 5., 5., 5., 5., 5., 5., 5., 5.,
         ##5., 7., 7., 7., 7., 7., 7., 7., Number of unscheduled operations in the job
        print("feat_opes_batch[:, 3, :]=", feat_opes_batch[:, 3, :])
        for i in range(self.batch_size):
            # for ii in range(self.num_opes):
            for ii in range(self.every_ope_number[i]):
                feat_opes_batch[:, 5, :][i][ii] = start_time(ii, self.former[i], self.later[i], feat_opes_batch[:, 2, :][i])
        print("feat_opes_batch[:, 5, :]=",feat_opes_batch[:, 5, :])
        ##tensor([[  0.0000,  86.5000, 157.5000, 181.1667,   0.0000,  98.3333, 141.8333,
         ##194.0833, 195.0833
        end_time_batch1 = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]
        end_time_batch = [[] for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            for ii in range(len(self.num_ope_biases_batch[i])):
                end_time_batch[i].append(max(end_time_batch1[i][int(self.num_ope_biases_batch[i][ii])
                                                                :int(self.end_ope_biases_batch[i][ii]) + 1]))
        end_time_batch = torch.tensor(end_time_batch)

        # end_time_batch = (feat_opes_batch[:, 5, :] +
        #                   feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        print("feat_opes_batch[:, 4, :]=",feat_opes_batch[:, 4, :])


        ## the least end time of every job  (operation time is averaged)

        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        print("feat_mas_batch[:, 0, :]=",feat_mas_batch[:, 0, :])

        # print("feat_mas_batch[:, 0, :]",feat_mas_batch[:, 0, :])
        ##tensor([[14., 17., 15., 18., 18.]]) how many operation per machine can do

        self.feat_opes_batch = feat_opes_batch  ##operation feature  6
        self.feat_mas_batch = feat_mas_batch    ##machine feature  3

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)

        # shape: (batch_size, num_opes), True for opes in process
        self.mask_ope_procing_batch = torch.full(size=(self.batch_size, self.num_opes), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_ope_finish_batch = torch.full(size=(self.batch_size, self.num_opes), dtype=torch.bool, fill_value=False)

        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)
        # print("self.mask_job_procing_batch=",self.mask_job_procing_batch)
        # print("self.mask_job_finish_batch=", self.mask_job_finish_batch)
        # print("self.mask_ope_procing_batch=", self.mask_ope_procing_batch)
        # print("self.mask_ope_finish_batch=", self.mask_ope_finish_batch)

        # print("self.mask_ma_procing_batch=", self.mask_ma_procing_batch)
        '''
        Partial Schedule (state) of jobs/operations, dynamic  in order for gantt() and render (no use)
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4)) ## (b,52,4)

        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]  #least Start 1x52

        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :] #least end 1x52
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        # print("self.makespan_batch=",self.makespan_batch)
        #self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)
        self.done_batch = self.mask_ope_finish_batch.all(dim=1)  # shape: (batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_ope_procing_batch=self.mask_ope_procing_batch,
                              mask_ope_finish_batch=self.mask_ope_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes,
                              num_opes=self.num_opes,  ## 37 max
                              former_batch=torch.Tensor(self.former),
                              later_batch=torch.Tensor(self.later),
                              nums_ope_batch=torch.Tensor(self.nums_ope))

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)

    def step(self, actions):
        '''
        Environment transition function
        '''
        opes = actions[0, :]  ## from 0-50+ total number
        opes_list=opes.tolist()
        # for i in range(self.batch_size):
        #     self.complete_opes[i].append(opes_list[i])
        #
        # print("self.complete_opes_batch=",self.complete_opes)

        mas = actions[1, :]   ## 0-4   ### wrong machine?
        jobs = actions[2, :]  ## 0-9
        print("opes,mas,jobs",opes,mas,jobs)
        self.N += 1  #self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations

        # Removed unselected O-M arcs of the scheduled operations
        # remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj = torch.zeros(size=(len(self.batch_idxes), self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1  ## the chosen machine 3
        # print("remain_ope_ma_adj=",remain_ope_ma_adj) ##tensor([[0, 1, 0, 0, 0]])  choose the 2nd machine
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]  ##the chosen opes 3rd job,1st opes
        # the 3rd job 1st operation: [1, 0, 1, 0, 0], ->  [0, 0, 1, 0, 0], only the 3rd machine be chosen
        # print("self.ope_ma_adj_batch=", self.ope_ma_adj_batch)
        self.proc_times_batch *= self.ope_ma_adj_batch
        # print("self.proc_times_batch=", self.proc_times_batch)
        # the 3rd job 1st operation: [18, 0, 17, 0, 0], ->  [0, 0, 17, 0, 0], only the 3rd machine be chosen

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        # print("proc_times=",proc_times)  ##17
        # print("self.feat_opes_batch[self.batch_idxes, :3, opes]=", self.feat_opes_batch[self.batch_idxes, :3, opes])
        # print("self.feat_opes_batch[self.batch_idxes, :3]=", self.feat_opes_batch[self.batch_idxes, :3])
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack((torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                                                                        torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                                                                        proc_times), dim=1)
        ## update the feature of the chosen opes
        # print("self.feat_opes_batch[self.batch_idxes, :3, opes]=",self.feat_opes_batch[self.batch_idxes, :3, opes])

        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1, opes - 1)

        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i]+1] -= 1

        # Update 'Start time' and 'Job completion time'
        # print("self.feat_opes_batch[self.batch_idxes, 5, opes]=",self.feat_opes_batch[self.batch_idxes, 5, opes])
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        ## the start time of ope is now, self.time (because in the next_time, the time will be updated as the time
        # that at least one ope-M can be scheduled)

        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes

        # print("un_scheduled", un_scheduled)
        # print("mean_proc_time", mean_proc_time) ## the opes will return not average but the only machine process time
        # print("start_times of scheduled opes", start_times)

        # estimate_times=torch.zeros(size=(self.batch_size, self.num_opes))
        estimate_times = torch.zeros(size=(len(self.batch_idxes), self.num_opes))
        # for i in range(self.batch_size):
        for i in range(len(self.batch_idxes)):
            for ii in range(self.num_opes):
                estimate_times[i][ii] = start_time_estimate(ii,is_scheduled[i], start_times[i],self.former[i],
                                                            self.later[i], mean_proc_time[i])
        # print("estimate_times=",estimate_times)
        estimate_times=estimate_times*un_scheduled

        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        # print("self.feat_opes_batch[self.batch_idxes, 5, :]=",self.feat_opes_batch[self.batch_idxes, 5, :])
        end_time_batch = [[] for _ in range(len(self.batch_idxes))]
        estimate_end_time_opes_batch = self.feat_opes_batch[self.batch_idxes, 5, :] \
                                       + self.feat_opes_batch[self.batch_idxes, 2, :]
        for i in range(len(self.batch_idxes)):
            for ii in range(len(self.num_ope_biases_batch[i])):
                ss = torch.max(estimate_end_time_opes_batch[i][int(self.num_ope_biases_batch[i][ii])
                                                               :int(self.end_ope_biases_batch[i][ii]) + 1])
                end_time_batch[i].append(ss)
        end_time_batch = torch.tensor(end_time_batch)
        # end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
        #                   self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        #
        # print("self.time=",self.time)

        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[self.batch_idxes,:])


        # Update partial schedule (state)
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas), dim=1)
        # print("self.schedules_batch[self.batch_idxes, opes, :2]=",self.schedules_batch[self.batch_idxes, opes, :2])
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))  ## idle 1->0 eligible -> uneligible
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times  ## accumulate operation time ?
        self.machines_batch[self.batch_idxes, mas, 3] = opes.float()  ##which job -> ope is the machine doing

        print("self.machines_batch[self.batch_idxes, mas, 2]=",self.machines_batch[self.batch_idxes, mas, 2])
        print("self.time=",self.time)
        # print("self.machines_batch[self.batch_idxes, mas]=",self.machines_batch[self.batch_idxes, mas])

        # Update feature vectors of machines
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)

        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        # print("cur_time",cur_time)
        # print("utiliz",utiliz)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        ##tensor([[ 0,  5, 11, 17, 21, 25, 30, 36, 42, 48]]) -> tensor([[ 0,  5, 12, 17, 21, 25, 30, 36, 42, 48]])
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ope_procing_batch[self.batch_idxes, opes] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch==self.end_ope_biases_batch+1,
                                                 True, self.mask_job_finish_batch)
        # self.mask_ope_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
        #                                          True, self.mask_job_finish_batch)
        # print("self.mask_ope_procing_batch=",self.mask_ope_procing_batch)
        # print("self.mask_ope_finish_batch=", self.mask_ope_finish_batch)
        # print("self.mask_ma_procing_batch=", self.mask_ma_procing_batch)

        # self.done_batch = self.mask_job_finish_batch.all(dim=1)
        # for i in range(self.batch_size):
        for i in range(len(self.batch_idxes)):
            if is_scheduled[i][:self.every_ope_number[i]].all():
                self.done_batch[i]=True
        # self.done_batch = is_scheduled.all(dim=1)
        # print("self.done_batch=",self.done_batch)
        self.done = self.done_batch.all()
        # if self.done:
        #     # print("Goaaaaaal!!!")

        # max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        # max=torch.max(estimate_end_time_opes_batch,dim=1)[0]
        # print("max=",max)
        # ## the makespan= latest job end time
        # self.reward_batch = self.makespan_batch - max  # the last time makespan (makespan) - new time makespan (max)
        # self.makespan_batch = max  ##update the last time makespan as new

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()

        while ~((~((flag_trans_2_next_time==0) & (~self.done_batch))).all()):
            # print("go to next time!")
            # print("self.time=",self.time)
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()  ## to next time, then judge again if there's no eligible,
            # and if need to go to next time.
        #     print("flag_trans_2_next_time=",flag_trans_2_next_time)
        # print("self.time=",self.time)

        # for i in range(self.batch_size):
        for i in range(len(self.batch_idxes)):
            if is_scheduled[i][:self.every_ope_number[i]].all():
                self.done_batch[i] = True
        # print("self.done_batch=", self.done_batch)
        self.done = self.done_batch.all()
        if self.done:
            print("Goaaaaaal!!!")
        # for i in range(self.batch_size):
        for i in range(len(self.batch_idxes)):
            if self.done_batch[i]:
                # print("Done this instance!!!")
                self.feat_opes_batch[i, 5, opes] = self.time[i]
                # print("self.feat_opes_batch[i, 5, opes]=",
                #       self.feat_opes_batch[i, 5, opes])

                is_scheduled = self.feat_opes_batch[i, 0, :]
                mean_proc_time = self.feat_opes_batch[i, 2, :]
                start_times = self.feat_opes_batch[i, 5, :] * is_scheduled  # real start time of scheduled opes
                un_scheduled = 1 - is_scheduled  # unscheduled opes

                estimate_times = torch.zeros(self.num_opes)
                for ii in range(self.num_opes):
                    estimate_times[ii] = start_time_estimate(ii, is_scheduled, start_times, self.former,
                                                                self.later, mean_proc_time)

                estimate_times = estimate_times * un_scheduled

                self.feat_opes_batch[i, 5, :] = start_times + estimate_times
                # print("self.feat_opes_batch[i, 5, :]=",self.feat_opes_batch[i, 5, :])
                # print("self.feat_opes_batch[i, 2, :]=", self.feat_opes_batch[i, 2, :])
                # print("end_time_batch=",end_time_batch)
                # end_time_batch[i] = self.feat_opes_batch[i, 5, :] + self.feat_opes_batch[i, 2, :]
                # self.feat_opes_batch[i, 4, :] = convert_feat_job_2_ope(end_time_batch,
                #                                                                       self.opes_appertain_batch[
                #                                                                       i, :])
        end_time_batch = [[] for _ in range(len(self.batch_idxes))]
        estimate_end_time_opes_batch = self.feat_opes_batch[self.batch_idxes, 5, :] \
                                       + self.feat_opes_batch[self.batch_idxes, 2, :]
        for i in range(len(self.batch_idxes)):
            for ii in range(len(self.num_ope_biases_batch[i])):
                ss = torch.max(estimate_end_time_opes_batch[i][int(self.num_ope_biases_batch[i][ii])
                                                               :int(self.end_ope_biases_batch[i][ii]) + 1])
                end_time_batch[i].append(ss)
        end_time_batch = torch.tensor(end_time_batch)
        self.feat_opes_batch[self.batch_idxes, 4, :] = convert_feat_job_2_ope(end_time_batch,
                                                                              self.opes_appertain_batch[
                                                                              self.batch_idxes, :])
        max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        ## the makespan= latest job end time
        self.reward_batch = self.makespan_batch - max  # the last time makespan (makespan) - new time makespan (max)
        self.makespan_batch = max  ##update the last time makespan as new


        # Update the vector for uncompleted instances
        mask_finish = (self.N+1) <= self.nums_opes
        # print("self.N=",self.N)
        # print("self.nums_opes=",self.nums_opes)
        # print("mask_finish=",mask_finish)
        # print("self.batch_idxes=",self.batch_idxes)
        # if ~(mask_finish.all()):
        #     self.batch_idxes = torch.arange(self.batch_size)[mask_finish]
        # print("self.batch_idxes=", self.batch_idxes)

        # Update state of the environment
        print("is_scheduled", is_scheduled)
        print("self.feat_opes_batch[self.batch_idxes, 0, :]=", self.feat_opes_batch[self.batch_idxes, 0, :])
        print("self.feat_opes_batch[self.batch_idxes, 1, :]=", self.feat_opes_batch[self.batch_idxes, 1, :])
        print("self.feat_opes_batch[self.batch_idxes, 2, :]=", self.feat_opes_batch[self.batch_idxes, 2, :])
        print("self.feat_opes_batch[self.batch_idxes, 3, :]=", self.feat_opes_batch[self.batch_idxes, 3, :])
        print("self.feat_opes_batch[self.batch_idxes, 4, :]=", self.feat_opes_batch[self.batch_idxes, 4, :])
        print("self.feat_opes_batch[self.batch_idxes, 5, :]=", self.feat_opes_batch[self.batch_idxes, 5, :])
        print("self.feat_mas_batch[self.batch_idxes, 0, :]=", self.feat_mas_batch[self.batch_idxes, 0, :])
        print("self.feat_mas_batch[self.batch_idxes, 1, :]=", self.feat_mas_batch[self.batch_idxes, 1, :])
        print("self.feat_mas_batch[self.batch_idxes, 2, :]=", self.feat_mas_batch[self.batch_idxes, 2, :])
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
            self.ope_ma_adj_batch, self.mask_ope_procing_batch, self.mask_ope_finish_batch, self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time,self.nums_ope_batch)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        # print("if no eligible check")
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(self.proc_times_batch)
        ope_eligible=torch.full(size=(self.batch_size, self.num_opes), dtype=torch.bool, fill_value=False)
        for i in range(self.batch_size):
            for ii in range(self.num_opes):
                ii_in_later_index = get_index(self.later[i], ii)
                if len(ii_in_later_index)==0:
                    if not (self.mask_ope_finish_batch[i][ii] or self.mask_ope_procing_batch[i][ii]):
                       ope_eligible[i][ii] = True
                else:
                    all_former_done=[]
                    for former_ii in range(len(ii_in_later_index)):
                        all_former_done.append(self.mask_ope_finish_batch[i][self.former[i][ii_in_later_index[former_ii]]])
                    all_former_done=torch.tensor(all_former_done)
                    if all_former_done.all():
                        ope_eligible[i][ii] = True
                    if self.mask_ope_finish_batch[i][ii] or self.mask_ope_procing_batch[i][ii]:
                       ope_eligible[i][ii] = False
        # print("ope_eligible=",ope_eligible)
        ope_eligible=ope_eligible.unsqueeze(-1)
        # print("ope_eligible.size()=",ope_eligible.size())
        # print("self.proc_times_batch.size()=",self.proc_times_batch.size())
        # print("ope_eligible.t()=", ope_eligible.t())
        ope_eligible = ope_eligible.expand_as(self.proc_times_batch)
        # print("ope_eligible.size()=", ope_eligible.size())
        #ma_eligible= tensor([[[ True, False,  True,  True,  True],
         #[ True, False,  True,  True,  True],
         #[ True, False,  True,  True,  True],
         #[ True, False,  True,  True,  True]]])

        ## there will be no concept of job_eligible
        # print("ma_eligible=",ma_eligible) ## the machine in procing can not be used
        # # print("job_eligible=",job_eligible)
        # print("ma_eligible & ope_eligible=",ma_eligible & ope_eligible)

        # flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
        #                                    dim=[1, 2])
        # flag_trans_2_next_time = torch.sum(torch.where(ma_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
        #     dim=[1, 2])
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & ope_eligible, self.proc_times_batch.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        # print("flag_trans_2_next_time=", flag_trans_2_next_time)
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        ## self.time[self.batch_idxes] + proc_times
        # print("available_time of machines, a=",a)
        # remain available_time greater than current time
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        # print("remain available_time greater than current time, b=", b)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # print("the minimum value of available_time, c=",c)
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # print("the machines that completed, d",d)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        # print("The time for each batch to transit to, e=",e)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        # print("self.machines_batch=",self.machines_batch)
        # print("aa=",aa)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)
        # print("self.machines_batch=",self.machines_batch)

        utiliz = self.machines_batch[:, :, 2]
        # print("utiliz=",utiliz)
        cur_time = self.time[:, None].expand_as(utiliz)
        # print("cur_time=", cur_time)
        utiliz = torch.minimum(utiliz, cur_time)
        # print("utiliz=", utiliz)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        # print("utiliz=", utiliz)

        self.feat_mas_batch[:, 2, :] = utiliz

        # jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        # jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        # job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        # print("jobs=",jobs)
        # print("jobs_index=", jobs_index)
        # print("job_idxes=", job_idxes)

        opes = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0)
        opes_index = np.argwhere(opes.cpu() >= 0).to(self.device)
        ope_idxes = opes[opes_index[0], opes_index[1]].long()
        batch_idxes = opes_index[0]
        # print("opes=", opes)
        # print("opes_index=", opes_index)
        # print("ope_idxes=", ope_idxes)
        # print("batch_idxes=", batch_idxes)

        # self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ope_procing_batch[batch_idxes,ope_idxes]=False

        self.mask_ma_procing_batch[d] = False
        # self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
        #                                          True, self.mask_job_finish_batch)
        self.mask_ope_finish_batch[batch_idxes, ope_idxes] = True

        # print("self.mask_ma_procing_batch=", self.mask_ma_procing_batch)
        # print("self.mask_ope_procing_batch=",self.mask_ope_procing_batch)
        # print("self.mask_ope_finish_batch=", self.mask_ope_finish_batch)

    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ope_procing_batch = torch.full(size=(self.batch_size, self.num_opes), dtype=torch.bool,
                                                 fill_value=False)
        self.mask_ope_finish_batch = torch.full(size=(self.batch_size, self.num_opes), dtype=torch.bool,
                                                fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_ope_finish_batch.all(dim=1)
        self.complete_opes = [[] for _ in range(self.batch_size)]
        return self.state

    def render(self, mode='human'):
        '''
        Deprecated in the final experiment
        '''
        if self.show_mode == 'draw':
            num_jobs = self.num_jobs
            num_mas = self.num_mas
            print(sys.argv[0])
            color = read_json("./utils/color_config")["gantt_color"]
            if len(color) < num_jobs:
                num_append_color = num_jobs - len(color)
                color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
                          range(num_append_color)]
            write_json({"gantt_color": color}, "./utils/color_config")
            labels = [''] * num_jobs
            for j in range(num_jobs):
                labels[j] = "job {0}".format(j)
            for batch_id in range(self.batch_size):
                schedules = self.schedules_batch[batch_id].to('cpu')
                fig, ax = plt.subplots()
                patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(num_jobs)]
                ax.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=14)
                fig = ax.figure
                fig.set_size_inches(16, 8)
                y_ticks = []
                y_ticks_loc = []
                for i in range(num_mas):
                    y_ticks.append('Machine {0}'.format(num_mas-1-i))
                    y_ticks_loc.insert(0, i)
                ax.set_yticks(y_ticks_loc, y_ticks, fontsize=16)
                # for i in range(num_mas):
                #     y_ticks.append('Machine {0}'.format(i))
                #     y_ticks_loc.insert(0, i)

                ax.set_xlabel('Time')
                ax.set_ylabel('Machine')
                ax.set_title('Job Shop Scheduling Gantt Chart')
                ax.grid(True)
                for i in range(num_mas-1,-1,-1):
                    y_ticks.append('Machine {0}'.format(i))
                    y_ticks_loc.insert(0, i)
                labels = [''] * num_jobs
                for j in range(num_jobs):
                    labels[j] = "job {0}".format(j)
                # patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
                # axes.cla()
                # axes.set_title(u'FJSP Schedule',fontsize=18)
                # axes.grid(linestyle='-.', color='gray', alpha=0.2)
                # axes.set_xlabel('Time',fontsize=16)
                # axes.set_ylabel('Machine',fontsize=16)
                # axes.set_yticks(y_ticks_loc, y_ticks,fontsize=16)
                # axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
                # axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
                #
                # axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=16)

                for i in range(int(self.nums_opes[batch_id])):
                    id_ope = i
                    print("id_ope=",id_ope)
                    idx_job, idx_ope = self.get_idx(id_ope, batch_id)
                    print("idx_job, idx_ope=",idx_job, idx_ope)
                    id_machine = schedules[id_ope][1]
                    print("id_machine",id_machine)
                    id_machine=int(id_machine)
                    operation_start=int(schedules[id_ope][2])
                    operation_duration=int(schedules[id_ope][3])-int(schedules[id_ope][2])

                    # axes.barh(id_machine,
                    #          0.2,
                    #          left=schedules[id_ope][2],
                    #          color='#b2b2b2',
                    #          height=0.5)
                    ax.broken_barh(
                        [(operation_start, operation_duration)],
                        (id_machine - 0.4, 0.8),
                        facecolors=color[idx_job],
                        edgecolor='black'
                    )
                    # ax.barh(id_machine,
                    #          schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
                    #          left=schedules[id_ope][2]+0.2,
                    #          color=color[idx_job],
                    #          height=0.5)
                    # ax.text(0.5*(schedules[id_ope][2]+schedules[id_ope][3]),schedules[id_ope][1],f'{id_ope}',ha='center',va='bottom',fontsize=14)
                    middle_of_operation = operation_start + operation_duration / 2
                    ax.text(
                        middle_of_operation,
                        id_machine,
                        id_ope,
                        ha='center',
                        va='center',
                        fontsize=16
                    )
                plt.show()
        return

    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass
