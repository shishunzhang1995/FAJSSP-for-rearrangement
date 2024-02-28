import torch
import numpy as np


def get_index(list, a):
    return [i for i in range(len(list)) if list[i] == a]

def start_time(i,former,later,average_ope_time):   ## i is the number of ope
    ## start_time(i)=max(start_time(former1)+operation_time(fomer1),former2,...,formerN))
    # print("i=",i)
    i_in_later_index = get_index(later, i)  ##[], [1,3,5]
    # print("i_in_later_index=",i_in_later_index)
    if len(i_in_later_index) != 0:
        former_end_time = 0
        for ii in i_in_later_index:
            if former_end_time<start_time(former[ii],former,later,average_ope_time)+ average_ope_time[former[ii]]:
               former_end_time=start_time(former[ii],former,later,average_ope_time)+ average_ope_time[former[ii]]
        start_time_i = former_end_time
        # start_time_i= former[i_in_later_index]
    else:
        start_time_i = 0
    return start_time_i

def start_time_estimate(i,is_scheduled,start_times,former,later,average_ope_time):   ## i is the number of ope
    ## start_time(i)=max(start_time(former1)+operation_time(fomer1),former2,...,formerN))
    # print("i=",i)
    i_in_later_index = get_index(later, i)  ##[], [1,3,5]
    # print("i_in_later_index=",i_in_later_index)
    if len(i_in_later_index) != 0:
        former_end_time = []
        for ii in range(len(i_in_later_index)):
            ## judge if the former of one ope is scheduled
            if is_scheduled[former[i_in_later_index[ii]]]:
                former_end_time.append(start_times[former[i_in_later_index[ii]]]
                                       +average_ope_time[former[i_in_later_index[ii]]])
            else:
                former_end_time.append(start_time_estimate(former[i_in_later_index[ii]],is_scheduled,start_times,
                                                           former,later,average_ope_time)
                                       + average_ope_time[former[i_in_later_index[ii]]])
        start_time_i = max(former_end_time)
        # start_time_i= former[i_in_later_index]
    else:
        start_time_i = start_times[i]
    return start_time_i


def matrix_cal_1(origin_i,i,former,later,matrix_cal_cumul):
    i_index = get_index(former, i)
    for ii in range(len(i_index)):
        i_later = later[i_index[ii]]
        matrix_cal_cumul[origin_i][i_later] = 1
        matrix_cal_cumul[i][i_later] = 1
        matrix_cal_1(origin_i,i_later,former,later,matrix_cal_cumul)

def load_fjs(lines, num_mas, num_opes,num_priors):
    '''
    Load the local FJSP instance.
    '''
    flag = 0
    matrix_proc_time = torch.zeros(size=(num_opes, num_mas)) ## [37,5]  [28,5]
    # print("matrix_proc_time.size()=",matrix_proc_time.size())
    matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False) ##[26,26]
    # print("matrix_pre_proc.size()=", matrix_pre_proc.size())
    matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int() ##[26,26]
    # print("matrix_cal_cumul.size()=", matrix_cal_cumul.size())
    nums_ope = []  # A list of the number of operations for each job
    opes_appertain = np.array([])
    num_ope_biases = []  # The id of the first operation of each job
    # print("lines=",lines)
    operation_total_number=int(lines[0].strip().split()[0])
    percedence_number=int(lines[0].strip().split()[1])
    dynamic_jobs=[]
    all_jobs=[]
    for i in range(1,1+percedence_number):
        # print("lines[i]=",lines[i])
        former=int(lines[i].strip().split()[0])
        later=int(lines[i].strip().split()[1])
        matrix_pre_proc[former][later]=True
        if i==1:
            dynamic_jobs=[former,later]
            # print("dynamic_jobs=",dynamic_jobs)
        else:
            if former in dynamic_jobs:
              dynamic_jobs.append(later)
            else:
                if later in dynamic_jobs:
                    dynamic_jobs.append(former)
            dynamic_jobs = list(range(min(dynamic_jobs), max(dynamic_jobs) + 1))
            # print("dynamic_jobs=", dynamic_jobs)
            if former not in dynamic_jobs and later not in dynamic_jobs:
                all_jobs.append(dynamic_jobs)
                dynamic_jobs=[former,later]
        if i==percedence_number:
            all_jobs.append(dynamic_jobs)
    # print("all_jobs=",all_jobs)
    # nums_ope=[]

    for i in range(len(all_jobs)):
        opes_appertain = np.concatenate((opes_appertain, np.ones(len(all_jobs[i]))*i))
        nums_ope.append(len(all_jobs[i]))
        num_ope_biases.append(all_jobs[i][0])

    # print("nums_ope=",nums_ope)
    # print("opes_appertain=",opes_appertain)
    for i in range(1+percedence_number,1+percedence_number+operation_total_number):
        operation_time=lines[i].strip().split()
        # print("operation_time=",operation_time)
        # total_machine=int(operation_time[0])
        machine_id=[]
        ope_time=[]
        for index in range(len(operation_time)):
            if index % 2 ==0 and index!=0:
                ope_time.append(int(operation_time[index]))
            elif index % 2 ==1:
                machine_id.append(int(operation_time[index]))
        for ii in range(len(machine_id)):
            matrix_proc_time[i-percedence_number-1][machine_id[ii]] =ope_time[ii]

    # print("matrix_proc_time=",matrix_proc_time)
    matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
    # print("matrix_ope_ma_adj=",matrix_ope_ma_adj)
    # Fill zero if the operations are insufficient (for parallel computation)
    opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes-opes_appertain.size)))
    # print("opes_appertain=",opes_appertain)
    # print("matrix_pre_proc=",matrix_pre_proc)
    # print("matrix_pre_proc.t()=", matrix_pre_proc.t())
    # print("num_ope_biases=", num_ope_biases)  ## the num_ope_biases indicate the first opreation in a job,
    # but not actually the first number in all operations.
    # for i in range(num_opes):
    #     for j in range(num_opes):
    #         if matrix_pre_proc[i][j] ==True:
    #             matrix_cal_cumul[i][j]=1
    #             matrix
    former=[]
    later=[]
    for i in range(1,1+percedence_number):
        # print("lines[i]=", lines[i])
        former.append(int(lines[i].strip().split()[0]))
        later.append(int(lines[i].strip().split()[1]))
    # print("former=",former)
    # print("later=",later)

    for i in range(num_opes):
        matrix_cal_1(i,i,former,later,matrix_cal_cumul)
    # Fill zero if the operations are insufficient (for parallel computation)
    # print("num_priors=",num_priors)
    # print("len(former)=",len(former))
    former = np.concatenate((former, -1*np.ones(num_priors - len(former))))
    later = np.concatenate((later, 100*np.ones(num_priors - len(later))))
    former=np.array(former).astype(dtype=int).tolist()
    later=np.array(later).astype(dtype=int).tolist()
    # print("former=",former)


    # print("matrix_cal_cumul[0]=",matrix_cal_cumul[0])
    # print("matrix_cal_cumul[4]=", matrix_cal_cumul[4])

    return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
           torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
           torch.tensor(nums_ope).int(), matrix_cal_cumul,former,later,nums_ope

def nums_detec(lines):
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i]!="\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes

def nums_detec_dag(lines):
    '''
    Count the number of jobs, machines and operations
    '''
    num_opes = 0
    for i in range(1, len(lines)):
        num_opes += int(lines[i].strip().split()[0]) if lines[i]!="\n" else 0
    line_split = lines[0].strip().split()
    num_jobs = int(line_split[0])
    num_mas = int(line_split[1])
    return num_jobs, num_mas, num_opes

def edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
    '''
    Detect information of a job
    '''
    line_split = line.split()
    flag = 0
    flag_time = 0
    flag_new_ope = 1
    idx_ope = -1
    num_ope = 0  # Store the number of operations of this job
    num_option = np.array([])  # Store the number of processable machines for each operation of this job
    mac = 0
    for i in line_split:
        x = int(i)
        # The first number indicates the number of operations of this job
        if flag == 0:
            num_ope = x
            flag += 1
        # new operation detected
        elif flag == flag_new_ope:
            idx_ope += 1
            flag_new_ope += x * 2 + 1
            num_option = np.append(num_option, x)
            if idx_ope != num_ope-1:
                matrix_pre_proc[idx_ope+num_ope_bias][idx_ope+num_ope_bias+1] = True
            if idx_ope != 0:
                vector = torch.zeros(matrix_cal_cumul.size(0))
                vector[idx_ope+num_ope_bias-1] = 1
                matrix_cal_cumul[:, idx_ope+num_ope_bias] = matrix_cal_cumul[:, idx_ope+num_ope_bias-1]+vector
            flag += 1
        # not proc_time (machine)
        elif flag_time == 0:
            mac = x-1
            flag += 1
            flag_time = 1
        # proc_time
        else:
            matrix_proc_time[idx_ope+num_ope_bias][mac] = x
            flag += 1
            flag_time = 0
    return num_ope