import collections
import math
import copy
import logging

# from sympy import intervals

LOG = logging.getLogger('optimus')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()  
ch.setFormatter(formatter)
LOG.addHandler(ch)

class Optimus(object):
    def __init__(self):
        self.infer_schedule = True 
        self.infer_pod_status = None
        self.interval = 60
        
        self.exist_train_jobs = dict()
        self.tmp_train_jobs = dict() 
        
        self.optimus = True 
        
    def infer_pod_status_trans(self,infer_pod_status):
        _infer_pod_status = dict()
        for _,pods in infer_pod_status.items():
            for name, info in pods.items():
                _infer_pod_status[name] = info
        
        self.infer_pod_status = _infer_pod_status
        
    def get_free_gpus(self, total_gpus,allocations):
        return collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
    
    def optimize(self, jobs, nodes, prev_allocations, node_template, clock, infer_pod_status):
    
        LOG.info("Optimus optimize")
        sleep_pods = set()
        infer_pods = set()
        self.infer_pod_status_trans(infer_pod_status)
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        
        for name,info in self.infer_pod_status.items():
            if info['status'] == 'SLEEP':
                sleep_pods.add(name)
            else: # PROTECT or RUNNING
                infer_pods.add(name)

        # print("jobs:",jobs)
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
        for job,alloc in prev_allocations.items():
            if job not in infer_jobs:
                continue
            
            for node_id in set(alloc):
                if node_id >= len(nodes) // 2: 
                    infer_nodes.add(node_id)
        
        
        LOG.info(infer_nodes)
                    
        allocations = {k: v for k, v in prev_allocations.items() if k in jobs and k not in sleep_pods}

       
        for job in train_jobs.values():
            completion_epoch = job.application.get_completion_epoch(
                    job.target_batch_size)
            
            if completion_epoch <= job.epoch:
                job.remaining = 1
                
            else:
                job.remaining = (job.application.get_iteration(job.target_batch_size, completion_epoch) -
                                 job.application.get_iteration(job.target_batch_size, job.epoch))
        min_replicas = {}
        
        for key, job in train_jobs.items():
            min_replicas[key] = 1  # math.ceil(job.target_batch_size / job.application.max_local_bsz)
        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        
        
        
        num_replicas = {}
        gain = {} 

        if self.infer_schedule:
            for key, job in infer_jobs.items():
                if num_gpus > 0:
                    num_replicas[key] = 1
                    num_gpus -= 1        
                else: 
                    break
            
        for key, job in sorted(train_jobs.items(), key=lambda item: min_replicas[item[0]]):
            if min_replicas[key] > num_gpus:
                num_replicas[key] = 0
                gain[key] = 0
                continue
            num_replicas[key] = min_replicas[key]
            num_gpus -= min_replicas[key]
            if num_replicas[key] + 1 > job.max_replicas or num_gpus < 1:
                gain[key] = 0
            else:
                gain[key] = (self.predict_step_time(job, num_replicas[key]) -
                            self.predict_step_time(job, num_replicas[key] + 1)) * job.remaining

        while num_gpus > 0 and max(gain.values()) > 0:
            key = max(gain, key=lambda k: gain[k]) # 找到收益最大的那个job
            job = train_jobs[key]
            num_replicas[key] += 1
            
            if num_replicas[key] + 1 > job.max_replicas or num_gpus < 1:
                gain[key] = 0
            else:
                gain[key] = (self.predict_step_time(job, num_replicas[key]) -
                            self.predict_step_time(job, num_replicas[key] + 1)) * job.remaining
            num_gpus -= 1
            
        allocations = {k: v for k, v in allocations.items() if k in num_replicas and len(v) == num_replicas[k]}
        job_keys = sorted(train_jobs, key=lambda k: num_replicas[k])
        free_gpus = self.get_free_gpus(total_gpus,allocations)
           
        if self.infer_schedule:
            for job in num_replicas:
                if job in infer_jobs and job not in allocations:
                    node_idx, count = free_gpus.most_common(1)[0]
                    allocations[job] = [node_idx]
                    free_gpus[node_idx] -= 1
                    


        free_gpus = self.get_free_gpus(total_gpus,allocations)

        for key in job_keys:
            if num_replicas[key] > 0 and not allocations.get(key):
                # Allocate resources.
                allocations[key] = []
                while len(allocations[key]) < num_replicas[key]:
                    node_idx, count = free_gpus.most_common(1)[0]
                    num = min(count, num_replicas[key] - len(allocations[key]))
                    allocations[key].extend([node_idx] * num)
                    free_gpus[node_idx] -= num
        
        if self.optimus:
            for job in train_jobs:
                if job not in allocations:
                    continue
                
                alloc = allocations[job]
                new_alloc = []
                for node_id in alloc:
                    if node_id not in infer_nodes:
                        new_alloc.append(node_id)
                    
                    else:
                        free_gpus[node_id] += 1
                
                allocations[job] = new_alloc
                        
        if self.infer_schedule:
            for key, job in sleep_jobs.items():
                node_idx = prev_allocations[key][0]
                if free_gpus[node_idx] > 0:
                    allocations[key] = [node_idx]
                    free_gpus[node_idx] -= 1

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
        step_time, sync_time = job.application.get_throughput(placement, atomic_bsz)
        return step_time + (step_time - sync_time) * accum_steps
