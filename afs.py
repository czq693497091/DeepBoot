import collections
import math
import copy
import logging
from collections import OrderedDict
import numpy as np

LOG = logging.getLogger('afs')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


ch = logging.StreamHandler()  
ch.setFormatter(formatter)
LOG.addHandler(ch)



class AFS(object):
    def __init__(self):
        self.infer_schedule = True 
        self.infer_pod_status = None
        self.interval = 60
        
        self.exist_train_jobs = dict()
        self.tmp_train_jobs = dict() 
        
        self.aryl = True 
    
    # def get_throughput_ratio(self,job)
    def top_priority(self,jobs,num_replicas):
        if len(jobs) == 0:
            return None
        job_names = list(jobs.keys())
        
        j0 = job_names[0]
        for job in job_names[1:]:
            ja = j0
            jb = job
            if num_replicas[ja] == 0 and num_replicas[jb] == 0:
                if self.predict_remain_time(jobs[ja],1) < self.predict_remain_time(jobs[jb],1):
                    j0 = ja
                else:
                    j0 = jb
            else:
                if num_replicas[ja] >= num_replicas[jb]:
                    ja, jb = jb, ja # swap
                
                if self.get_afs_weight(jobs[jb],num_replicas[jb]) > self.get_afs_weight(jobs[ja],num_replicas[ja]):
                    j0 = jb
                else:
                    j0 = ja      
        return j0

    def afs_L(self,jobs,free_gpus,prev_allocations):
        num_free_gpus = sum(free_gpus.values())

        job_names = list(jobs.keys())
        num_replicas = {job:0 for job in job_names}

        while num_free_gpus > 0:
            j0 = self.top_priority(jobs,num_replicas)
            num_replicas[j0] += 1
            num_free_gpus -= 1

        temp_alloc = copy.deepcopy(prev_allocations)
        alloc = self.replicas2allocation(
            jobs = jobs,
            allocations = temp_alloc,
            num_replicas = num_replicas,
            available_gpus = free_gpus
        )
        

        return alloc

    def infer_pod_status_trans(self,infer_pod_status):
        _infer_pod_status = dict()
        for _,pods in infer_pod_status.items():
            for name, info in pods.items():
                _infer_pod_status[name] = info
        
        self.infer_pod_status = _infer_pod_status
        
    def get_free_gpus(self, total_gpus,allocations):
        return collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
    
    def select_node(self, num_replica, free_gpus):

        ORIGIN_SELECT = False # False就是新的分配方案
        
        node_idx, count = free_gpus.most_common(1)[0]
        if ORIGIN_SELECT: # 自带的分配方式
            
            return node_idx, count
        
        if num_replica > count: # 一个节点无法承受这个任务
            return node_idx, count
        
        else:
            f = {k:v for k,v in dict(free_gpus).items() if v >= num_replica}
            nodes, cnts = list(f.keys()),list(f.values())
            node_id = np.argmin(cnts)
            node = nodes[node_id] # 选出满足要求且空闲最少的节点
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

    def optimize(self, jobs, nodes, prev_allocations, node_template, clock, infer_pod_status):
    
        sleep_pods = set()
        infer_pods = set()
        prev_train_alloc = {}
        
        self.infer_pod_status_trans(infer_pod_status)
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        
        nodes = OrderedDict(  # Sort preemptible nodes last.
            sorted(nodes.items(), key=lambda kv: (kv[1].preemptible, kv[0])))
        
        # 根据提交时间对作业进行排名
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
        
                
        allocations = {}
        allocations.update(infer_alloc) 

        free_gpus = self.get_free_gpus(total_gpus,infer_alloc)
        
        train_alloc = self.afs_L(train_jobs,free_gpus,prev_train_alloc)

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

    def predict_remain_time(self,job,num_replicas):
        completion_epoch = job.application.get_completion_epoch(
                    job.target_batch_size)
        if completion_epoch <= job.epoch:
            job.remaining = 1
        else:
            job.remaining = (job.application.get_iteration(job.target_batch_size, completion_epoch) -
                                job.application.get_iteration(job.target_batch_size, job.epoch))

        step_time = self.predict_step_time(job,num_replicas)
        return job.remaining * step_time


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

    def get_throughout(self,job,num_replicas):
        if num_replicas == 0:
            return 0
        
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
        
        iter_time = step_time + (step_time - sync_time) * accum_steps
        return atomic_bsz * count / iter_time
    
    def get_afs_weight(self,job,num_replicas):
        
        return (self.get_throughout(job,num_replicas + 1) 
                - self.get_throughout(job,num_replicas)) / self.get_throughout(job,num_replicas+1)