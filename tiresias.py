import collections
import copy
import numpy as np
import logging

LOG = logging.getLogger('tiresias')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()  
ch.setFormatter(formatter)
LOG.addHandler(ch)


class TiresiasPolicy(object):
    def __init__(self, time_fn):
        self._time_fn = time_fn
        self._queue_threshold = 3600 * 16
        self._solve_starvation = 0
        self._queue_0 = [] 
        self._queue_1 = []
        self._status = {}
        self._last_check_time = collections.Counter()
        self._total_executed_time = collections.Counter()
        self._executed_time = collections.Counter()
        self._last_pending_time = collections.Counter()
        self._pending_time = collections.Counter()
        
        self.infer_schedule = True
        self.infer_pod_status = None

    def get_free_gpus(self, total_gpus,allocations):
        return collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
    
    def select_node(self, num_replica, free_gpus):
        ORIGIN_SELECT = True
        
        node_idx, count = free_gpus.most_common(1)[0]
        if ORIGIN_SELECT: 
            
            return node_idx, count
        
        if num_replica > count: 
            return node_idx, count
        
        else:
            f = {k:v for k,v in dict(free_gpus).items() if v >= num_replica}
            print(f)
            nodes, cnts = list(f.keys()),list(f.values())
            node_id = np.argmin(cnts)
            node = nodes[node_id]
            return node, cnts[node_id]
        

    def infer_pod_status_trans(self,infer_pod_status):
        _infer_pod_status = dict()
        for _,pods in infer_pod_status.items():
            for name, info in pods.items():
                _infer_pod_status[name] = info

        self.infer_pod_status = _infer_pod_status
        
   
    
    
    def optimize(self, jobs, nodes, prev_allocations, node_template, clock, infer_pod_status):
        event_time = int(self._time_fn())

        sleep_pods = set()
        infer_pods = set()
        self.infer_pod_status_trans(infer_pod_status)
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        
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
        
        self._queue_0 = [key for key in self._queue_0 if key in train_jobs]
        self._queue_1 = [key for key in self._queue_1 if key in train_jobs]        
        self._status = {key: val for key, val in self._status.items() if key in train_jobs}
        
        allocations = {key: val for key, val in prev_allocations.items() if key in jobs and key not in sleep_pods}
        # Add new jobs to pending.
        
        for key, job in train_jobs.items():
            if key not in self._status:
                self._status[key] = 'PENDING'
                self._queue_0.append(key)
                
        # Update queues.
        for key, job in train_jobs.items():
            assert self._status[key] in ('RUNNING', 'PENDING')
            if self._status[key] == 'RUNNING':  # Job is running.
                tmp = int(event_time - self._last_check_time[key])
                self._total_executed_time[key] = int(self._total_executed_time[key] + tmp)
                
                self._executed_time[key] = int(self._executed_time[key] + tmp) # decide job priority queue
                self._last_check_time[key] = event_time
                # check demotion
                
                gputime = int(self._executed_time[key] * job.max_replicas)
                
                if key in self._queue_0 and gputime >= self._queue_threshold:
                    self._queue_0.pop(self._queue_0.index(key))
                    self._queue_1.append(key)
                    print("job {} demote to Q1".format(key))
                    
            elif self._status[key] == 'PENDING':
                tmp = int(event_time - self._last_check_time[key]) 
                self._last_check_time[key] = event_time
                
                self._pending_time[key] = int(self._pending_time[key] + tmp) #this is the total pending_time
                
                if self._executed_time[key] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    self._last_pending_time[key] = int(self._last_pending_time[key] + tmp) #this is the total pending_time
                    
                # Q0 job no need to push_back, and must be a runned 
                if self._solve_starvation > 0 and key not in self._queue_0 and \
                        self._total_executed_time[key] > 0 and self._executed_time[key] > 0:
                    if self._last_pending_time[key] >= int(self._executed_time[key] * self._solve_starvation):
                        self._executed_time[key] = 0
                        self._last_pending_time[key] = 0
                        self._queue_0.append(key)
                        self._queue_1.pop(self._queue_1.index(key))
                        
        # Update statuses.
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        num_gpus = sum(total_gpus.values()) - len(infer_jobs)
        
        LOG.info("num gpus: %s",num_gpus)
        for queue in (self._queue_0, self._queue_1):
            for idx in queue:
                if train_jobs[idx].max_replicas <= num_gpus:
                    self._status[idx] = 'RUNNING'
                    num_gpus -= train_jobs[idx].max_replicas
                
                # GPU不够分配直接挂起
                else:
                    self._status[idx] = 'PENDING'
                    allocations.pop(idx, None)
                    
        # Update allocations.
        free_gpus = self.get_free_gpus(total_gpus,allocations)

        for queue in (self._queue_0, self._queue_1):
            for idx in queue:
                if self._status[idx] == 'RUNNING' and not allocations.get(idx):
                    # Allocate resources.
                   
                    allocations[idx] = []     
                    while len(allocations[idx]) < train_jobs[idx].max_replicas:
                        gpu_need = train_jobs[idx].max_replicas - len(allocations[idx])
                        node_idx, count = self.select_node(gpu_need, free_gpus)
                        num = min(count, gpu_need)
                        allocations[idx].extend([node_idx] * num)
                        free_gpus[node_idx] -= num
                        
        if self.infer_schedule:
            for key, job in sleep_jobs.items():
                node_idx = prev_allocations[key][0]
                if free_gpus[node_idx] > 0:
                    allocations[key] = [node_idx]
                    free_gpus[node_idx] -= 1

        # Objective values, allocations, active nodes.
        return allocations, len(nodes)
