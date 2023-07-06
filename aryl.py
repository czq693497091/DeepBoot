# 假设没有排队作业
from collections import OrderedDict
import collections
import copy
import math
import numpy as np
import logging
# from tokenize import group

LOG = logging.getLogger('Aryl')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# log_file = './logs/simulator.log'


ch = logging.StreamHandler()  
ch.setFormatter(formatter)
LOG.addHandler(ch)

class Aryl:
    def __init__(self):
        self.infer_schedule = True
        self.aryl = True
    
    def sort_jobs(self,jobs):
        for job in jobs:
            jobs[job].run_time = self.predict_remain_time(jobs[job],1)
            
        jobs = OrderedDict(sorted(
            jobs.items(), key=lambda kv: (kv[1].run_time)))

        return jobs
    
    def max_value_dp(self,ws,vs,m):        
        n = len(ws) - 1 
        dp = np.zeros(shape=(n+1,m+1))
        for i in range(1,n+1): # 任务
            for j in range(m,-1,-1):
                for k in range(len(ws[i])): 
                    if j >= ws[i][k]:
                        dp[i][j] = max(dp[i][j],dp[i-1][j],dp[i-1][j-ws[i][k]]+vs[i][k])
        
        j = m
        ways = np.zeros(n+1,dtype=int) 
        for i in range(n,0,-1):
            for k in range(len(ws[i])):
                if j >= ws[i][k] and dp[i][j] == dp[i-1][j-ws[i][k]] + vs[i][k]:
                    ways[i] = ws[i][k]
                    j -= ws[i][k]
                    break
                
        return ways[1:]

    
    def allocate_elastic(self,prev_allocations,jobs,free_gpus):
        jobs = self.sort_jobs(jobs) 
        num_gpus = sum(free_gpus.values()) 
        groups = []
        for job,info in jobs.items():
            g = [] # New Group
            # groups.append(g)
            if info.max_replicas == 1:
                g.append((1,info.run_time))
            
            else:
                for w in range(1,info.max_replicas - info.min_replicas+1):
                    weight = w + info.min_replicas # 每个作业重量
                    # t_max = predict_remain_time(info,1) # 最慢的情况下就是只拿1个GPU
                    
                    value = info.run_time * w / (w + info.min_replicas + 1) # 作业价值
                    g.append((weight,value))
            
            groups.append(g)
        
        ws = [[],]
        vs = [[],]
        for g in groups:
            temp_w = [w for w,v in g]
            temp_v = [v for w,v in g]
            ws.append(temp_w)
            vs.append(temp_v)
        
        # LOG.info("groups: %s",groups)
        # LOG.info("ws: %s",ws)
        # LOG.info("vs: %s",vs)
        ways = self.max_value_dp(ws,vs,num_gpus)
        
        num_replicas = {}
        for i, job in enumerate(jobs):
            num_replicas[job] = ways[i]
        
        
        temp_alloc = copy.deepcopy(prev_allocations)
        alloc = self.replicas2allocation(
            jobs = jobs,
            allocations = temp_alloc,
            num_replicas = num_replicas,
            available_gpus = free_gpus
        )

        return alloc 
    
    def select_node(self, num_replica, free_gpus):
        ORIGIN_SELECT = False 
        
        node_idx, count = free_gpus.most_common(1)[0]
        if ORIGIN_SELECT: 
            
            return node_idx, count
        
        if num_replica > count: 
            return node_idx, count
        
        else:
            f = {k:v for k,v in dict(free_gpus).items() if v >= num_replica}
            # print(f)
            nodes, cnts = list(f.keys()),list(f.values())
            node_id = np.argmin(cnts)
            node = nodes[node_id] 
            return node, cnts[node_id]
    
    def replicas2allocation(self, jobs, allocations, num_replicas, available_gpus):
        job_keys = sorted(jobs, key=lambda k: num_replicas[k])
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}
        free_gpus = collections.Counter(available_gpus) - collections.Counter(sum(allocations.values(), []))
        
        for key in job_keys:
            if num_replicas[key] > 0 and not allocations.get(key):
                # Allocate resources.
                allocations[key] = []
       
                while len(allocations[key]) < num_replicas[key]:
                    gpu_need = num_replicas[key] - len(allocations[key])
                    node_idx, count = self.select_node(gpu_need, free_gpus)
                    num = min(count, gpu_need)
                    allocations[key].extend([node_idx] * num)
                    free_gpus[node_idx] -= num
        
        return allocations
    
    def get_free_gpus(self, total_gpus,allocations):
        return collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
    

    def infer_pod_status_trans(self,infer_pod_status):
        _infer_pod_status = dict()
        for _,pods in infer_pod_status.items():
            for name, info in pods.items():
                _infer_pod_status[name] = info
        
        self.infer_pod_status = _infer_pod_status
        
    def optimize(self, jobs, nodes, prev_allocations, node_template, clock, infer_pod_status):
        
        LOG.info("real Aryl optimize")
        sleep_pods = set()
        infer_pods = set()
        self.infer_pod_status_trans(infer_pod_status)
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        
        nodes = OrderedDict(  # Sort preemptible nodes last.
            sorted(nodes.items(), key=lambda kv: (kv[1].preemptible, kv[0])))
        
        jobs = OrderedDict(sorted(
            jobs.items(), key=lambda kv: (kv[1].creation_timestamp)))
        
        self.node_id_dict = dict(zip(nodes.keys(),range(len(nodes))))

        for name,info in self.infer_pod_status.items():
            if info['status'] == 'SLEEP':
                sleep_pods.add(name)
            else: # PROTECT or RUNNING
                infer_pods.add(name)

        train_jobs = {}
        infer_jobs = {}
        sleep_jobs = {}
        if self.infer_schedule:                       
            train_jobs = {k: v for k, v in jobs.items() if not v.inference}
            infer_jobs = {k: v for k, v in jobs.items() if v.inference and k not in sleep_pods}
            sleep_jobs = {k: v for k, v in jobs.items() if k in sleep_pods and k in prev_allocations}
        
        else:
            train_jobs = jobs
        
        if len(train_jobs) == 0:
            return prev_allocations, len(nodes)

        LOG.info("prev allocation: %s",prev_allocations)
        
        infer_nodes = set()
        infer_alloc = {}
        prev_train_alloc = {}
        for job,alloc in prev_allocations.items():
            if job not in infer_jobs:
                if 'infer' not in job:
                    prev_train_alloc[job] = alloc
                continue
            else:
                infer_alloc[job] = alloc
            
            for node_id in set(alloc):
                if self.node_id_dict[node_id] >= len(nodes) // 2: 
                    infer_nodes.add(node_id)
        
        
        LOG.info(infer_nodes) 
        allocations = {}
        allocations.update(infer_alloc) 
        
        free_gpus = self.get_free_gpus(total_gpus,infer_alloc)

        train_alloc = self.allocate_elastic(prev_train_alloc,train_jobs,free_gpus)
        remain_gpus = self.get_free_gpus(free_gpus,train_alloc)
        allocations.update(train_alloc)
        
        
        sleep_alloc = {}
        for job in sleep_jobs:
            alloc = prev_allocations[job]
            if remain_gpus[alloc[0]] > 0:
                sleep_alloc[job] = alloc
                remain_gpus[alloc[0]] -= 1
        
        allocations.update(sleep_alloc)
        return allocations, len(nodes)
    
    def predict_step_time(self, job, num_replicas):
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        local_bsz = math.ceil(job.target_batch_size / num_replicas - 1e-8)
        accum_steps = math.ceil(local_bsz / job.application.max_local_bsz - 1e-8) - 1
        if num_replicas == 1:
            accum_steps = max(1, accum_steps)
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = num_replicas * (accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(job.application.max_batch_size / count))
        #throughput = job.speedup_fn._goodput_fn.throughput(len(placement), num_replicas, atomic_bsz, accum_steps)
        #return atomic_bsz * count / throughput
        step_time, sync_time = job.application.get_throughput(placement, atomic_bsz)
        return step_time + (step_time - sync_time) * accum_steps

    def predict_remain_time(self, job,num_replicas):
        if num_replicas == 0:
            return 1e8 # 
        
        completion_epoch = job.application.get_completion_epoch(
                        job.target_batch_size)
                
        if completion_epoch <= job.epoch:
            job.remaining = 1
            
        else:
            job.remaining = (job.application.get_iteration(job.target_batch_size, completion_epoch) -
                                job.application.get_iteration(job.target_batch_size, job.epoch))
        
        return self.predict_step_time(job,num_replicas) * job.remaining

        