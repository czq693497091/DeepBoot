import numpy as np
from collections import OrderedDict
import logging
import collections
import copy


LOG = logging.getLogger('infer_scheduler')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


ch = logging.StreamHandler()  
ch.setFormatter(formatter)
LOG.addHandler(ch)

def array2onehot(array,num_classes=None):
    if not num_classes:
        num_classes = np.max(array) + 1
    return np.array(np.eye(num_classes)[array],dtype=np.int)

class InferScheduler(object):
    def __init__(self):
        self.infer_schedule = True
        self.aryl = True
        self.random = False
        
    
    def _allocations_to_state(self, allocations, jobs, nodes):
        jobs_index = {key: idx for idx, key in enumerate(jobs)}
        nodes_index = {key: idx for idx, key in enumerate(nodes)}
        state = np.zeros((len(jobs), len(nodes)), dtype=np.int)
        for job_key, alloc in allocations.items():
            for node_key in (key for key in alloc if key in nodes_index):
                state[jobs_index[job_key], nodes_index[node_key]] += 1
        return state

    
    def _state_to_allocations(self, state, jobs, nodes):
        allocations = {}
        for job_idx, job_key in enumerate(jobs):
            for node_idx, node_key in enumerate(nodes):
                count = state[job_idx, node_idx]
                allocations.setdefault(job_key, []).extend([node_key] * count)
        return allocations
    

    def _get_job_speedups(self, states):
        speedup = []
        num_nodes = np.count_nonzero(states, axis=2)
        num_replicas = np.array(np.sum(states, axis=2),dtype=np.int)
        
        job_list = []
        for job in self.job_vals:
            if not job.inference:
                job_list.append(job)
        
        for idx, job in enumerate(job_list):
  
            speedup.append(job.speedup_fn(
                num_nodes[:, idx], num_replicas[:, idx]))
        
        # LOG.info("speedup: %s",speedup)
        return np.stack(speedup, axis=1).astype(np.float)

    def take_available_gpu(self):


        num_gpu_in_each_node = np.array([x.resources['nvidia.com/gpu'] for x in self.node_vals])
        self.gpu_num = num_gpu_in_each_node[0]
        empty_gpu = num_gpu_in_each_node - self.node_training_gpus - self.node_infer_gpus - self.node_sleep_gpus
        return empty_gpu

    def _aryl_schedule_empty_or_sleep_gpu(self,gpus,is_sleep=False):
        
        half_node = len(self._nodes) // 2
        idx = half_node
        if is_sleep:
            sleep_flag = True
        
        while self.remain_infer_tasks > 0:
    
            while idx < len(self._nodes) and gpus[idx] <= 0:
                idx += 1
                if is_sleep: 
                    sleep_flag = True
            
            if idx == len(self._nodes):
                break
            
            node_id = idx
            
            if is_sleep and sleep_flag:
                sleep_task_id = np.where(self.sleep_state[:,node_id] == 1)[0][0] 
                self.sleep_state[sleep_task_id][node_id] = 0
                
            gpus[node_id] -= 1
            self.remain_infer_tasks -= 1
            
            self.update_state[self.curr_infer_job_id][node_id] = 1
            self.curr_infer_job_id += 1

       
            # training_state: train_jobs x num_nodes
            num_train_gpus = np.sum(self.training_state[:,node_id])
            self.training_state[:,node_id] = 0
            gpus[node_id] += num_train_gpus
            
            # 如果有从训练任务里抢到训练GPU 那么下一轮就不是占用sleep任务
            if num_train_gpus > 0 and is_sleep:
                sleep_flag = False
                
    def _schedule_empty_or_sleep_gpu(self,gpus,is_sleep=False):
        node_priority = np.argsort(self.node_training_gpus) # 优先排序训练任务最少的GPU
        idx = 0 # node_id
        while self.remain_infer_tasks > 0:

            while idx < len(self._nodes) and gpus[node_priority[idx]] == 0:
                idx += 1
            
            if idx == len(self._nodes):
                break
            
            node_id = node_priority[idx]
            
            if is_sleep:
                # 选出第一个这种sleep任务
                sleep_task_id = np.where(self.sleep_state[:,node_id] == 1)[0][0] 
                self.sleep_state[sleep_task_id][node_id] = 0
                
            gpus[node_id] -= 1
            self.remain_infer_tasks -= 1
 
            self.update_state[self.curr_infer_job_id][node_id] = 1
            self.curr_infer_job_id += 1

    def schedule_available_gpu(self):
        '''
        这里先把空/sleep的GPU用完
        在每次用掉一个GPU, 也要指定对应哪个推理任务
        用掉哪个sleep的GPU需要指定一下
        '''
        empty_gpu = self.take_available_gpu()        

        # LOG.info("empty gpu: %s",empty_gpu)
        
        if self.aryl:
            self._aryl_schedule_empty_or_sleep_gpu(empty_gpu,False)
            if self.remain_infer_tasks == 0:
                return
            self._aryl_schedule_empty_or_sleep_gpu(self.node_sleep_gpus,True)
            
        else:
            self._schedule_empty_or_sleep_gpu(empty_gpu,False)
            if self.remain_infer_tasks == 0:
                return
            self._schedule_empty_or_sleep_gpu(self.node_sleep_gpus,True)
    
    def schedule_training_gpu(self):
     
        self.num_training_gpus = np.sum(self.node_training_gpus)
        if self.num_training_gpus == 0:
            return
        
        self._schedule_training_gpu()
        # self._schedule_training_gpu_next_round()
    
    def _aryl_schedule_training_gpu(self):
        gpus = np.sum(self.training_state,axis=0) # 每个节点训练任务占用GPU的个数
        half_node = len(self._nodes) // 2
        
        node_priority = np.argsort(gpus) # 训练任务最少的节点进行抢占
        
        node_priority = [index for index in node_priority if index >= half_node]        

        idx = 0
        # LOG.info("node priority: %s",node_priority)
        while self.remain_infer_tasks > 0:
            
            while idx < half_node and gpus[node_priority[idx]] <= 0:
                idx += 1
            
            if idx == half_node:
                break

            node_id = node_priority[idx]
            gpus[node_id] -= 1
            self.remain_infer_tasks -= 1
            
            self.update_state[self.curr_infer_job_id][node_id] = 1
            self.curr_infer_job_id += 1
            
            num_train_gpus = np.sum(self.training_state[:,node_id])
            self.training_state[:,node_id] = 0
            
            if num_train_gpus != 0:

                gpus[node_id] = max(num_train_gpus - 1, 0)

    def _schedule_training_gpu(self):
        while self.remain_infer_tasks > 0 and self.num_training_gpus > 0:
            node_delta_onehot = array2onehot(np.arange(self.num_training_jobs*self.num_nodes))
       
            # 这步求出所有推理空间可以放置的组合空间
            self.infer_state_space = node_delta_onehot.reshape(self.num_training_jobs * self.num_nodes,self.num_training_jobs,self.num_nodes)
            diff_space = self.training_state - self.infer_state_space
            available_idx = ~np.any(diff_space == -1,axis=(1,2))
            available_idx = np.array(np.where(available_idx == True)[0])
            available_space = diff_space[available_idx]
            self.available_infer_state_space = self.infer_state_space[available_idx]
            self.curr_speedup = self._get_job_speedups(available_space)
            best_idx = np.argmax(np.sum(self.curr_speedup,axis=1))
            best_case = self.available_infer_state_space[best_idx]
            self.best_cases.append(best_case)

            node_id = np.where(best_case == 1)[1][0]

            self.update_state[self.curr_infer_job_id][node_id] = 1
            self.training_state -= best_case
            self.curr_infer_job_id += 1
            self.remain_infer_tasks -= 1
            self.num_training_gpus -= 1
            

    def _schedule_training_gpu_next_round(self):
        
        # LOG.info("remain infer task: %s",self.remain_infer_tasks)
        while self.remain_infer_tasks > 0 and self.num_training_gpus > 0:
        # for i in range(min(self.remain_infer_tasks,self.num_training_gpus)):
            self.curr_training_state = self.curr_training_state - self.best_cases[-1]
            # LOG.info("curr training state: %s %s",self.curr_training_state.shape,self.curr_training_state)
            update_training_job_idx = np.where(self.best_cases[-1])[0][0]
            update_training_job_curr_state = self.curr_training_state[update_training_job_idx]

            save_case_idx = np.array(np.where(np.any(self.available_infer_state_space[:,update_training_job_idx,:] == 1,axis=1)==False))[0]

            self.curr_speedup = self.curr_speedup[save_case_idx]
            self.available_infer_state_space[save_case_idx]

            node_delta_onehot = array2onehot(np.arange(self.num_nodes))
            update_infer_state_space = node_delta_onehot

            # num_cases x num_nodes, num_cases = num_nodes
            diff_space = update_training_job_curr_state - update_infer_state_space

            update_available_idx = ~np.any(diff_space == -1,axis=1)
            
            # LOG.info("update available idx: %s",update_available_idx)
            update_available_idx = np.array(np.where(update_available_idx == True)[0])
            update_available_space = diff_space[update_available_idx]
            update_available_infer_state_space = update_infer_state_space[update_available_idx] # 当前该job可行的infer state

            state_space_template = np.zeros(shape=(len(update_available_idx),self.num_training_jobs,self.num_nodes),dtype=np.int)

            state_space_template[:,update_training_job_idx,:] = update_available_infer_state_space
            update_available_infer_state_space = state_space_template
            
            # LOG.info("update available space: %s %s",update_available_space.shape,update_available_space)
            
            update_job_speedup = self._get_job_speedups(update_available_space[:,np.newaxis,:])
            self.curr_speedup = np.concatenate((self.curr_speedup,update_job_speedup))

            self.available_infer_state_space = np.concatenate((self.available_infer_state_space,update_available_infer_state_space))
            available_best_idx = np.argmax(self.curr_speedup)
            best_case = self.available_infer_state_space[available_best_idx]
            self.best_cases.append(best_case)
            
            node_id = np.where(best_case == 1)[1][0]

            self.update_state[self.curr_infer_job_id][node_id] = 1
            self.training_state -= best_case
            self.curr_infer_job_id += 1
            self.remain_infer_tasks -= 1
            self.num_training_gpus -= 1
    
    def pollux_config_init(self,jobs,nodes,infer_pod_status):
        # LOG.info("pollux config init")
        # LOG.info("InferScheduler infer pod status: %s",infer_pod_status)
        sleep_pods = set()
        for app, pods in infer_pod_status.items():
            for pod, info in pods.items():
                if info.get('status','') == 'SLEEP':
                    sleep_pods.add(pod)
        # LOG.info("sleep pods: %s",sleep_pods)
        # LOG.info("base state: %s",self.base_state)
        self.inference = np.array([x.inference for x in jobs])
        self.sleep = np.array([job.name in sleep_pods for job in jobs])
        
        self.training = ~self.inference 
        self.inference[self.sleep == True] = False 

        self.training_state = self.base_state[self.training]
        self.infer_state = self.base_state[self.inference]
        self.sleep_state = self.base_state[self.sleep]

        rtypes = sorted(set.union(*[set(job.resources) for job in jobs]))
        # Build array of job resources: <num_jobs> x <num_rtypes>. Each entry
        # [j, r] is the amount of resource r requested by a replica of job j.
        self._job_resources = np.zeros((len(jobs), len(rtypes)), np.int64)
        for j, job in enumerate(jobs):
            for r, rtype in enumerate(rtypes):
                self._job_resources[j, r] = job.resources.get(rtype, 0)
        # Build array of node resources: <num_nodes> x <num_rtypes>. Each
        # entry [n, r] is the amount of resource r available on node n.
        self._node_resources = np.zeros((len(nodes), len(rtypes)), np.int64)
        for n, node in enumerate(nodes):
            for r, rtype in enumerate(rtypes):
                self._node_resources[n, r] = node.resources.get(rtype, 0)
        # Calculate dominant per-replica resource shares for each job.
        shares = self._job_resources / np.sum(self._node_resources, axis=0)
        self._dominant_share = np.amax(shares, axis=1)
        # Change base goodput to fair-share goodput.

        fair_replicas = np.ceil(1.0 / self._dominant_share / len(jobs))
       
        fair_nodes = np.ceil(len(nodes) * self._dominant_share)
        for job, num_nodes, num_replicas in zip(jobs, fair_nodes, fair_replicas):
            if not hasattr(job.speedup_fn, "_goodput_fn"):
                job.speedup_fn = lambda n, r: r / num_replicas
                continue
            job.speedup_fn._base_goodput = job.speedup_fn._goodput_fn.optimize(
                num_nodes=num_nodes, num_replicas=num_replicas,
                max_batch_size=job.speedup_fn._max_batch_size,
                atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                accumulation=job.speedup_fn._accumulation)[0]
            
        # Upper bound each job: <replicas on node 0> <replicas on node 1> ...
        self._max_replicas = np.zeros(self.base_state.shape, dtype=np.int)
        for j, job in enumerate(jobs):
            for n, node in enumerate(nodes):
                self._max_replicas[j, n] = min(
                    node.resources[rtype] // job.resources[rtype]
                    for rtype in rtypes if job.resources.get(rtype, 0) > 0)
    
    
    def infer_pod_status_trans(self,infer_pod_status):
        _infer_pod_status = dict()
        for _,pods in infer_pod_status.items():
            for name, info in pods.items():
                _infer_pod_status[name] = info
        
        self.infer_pod_status = _infer_pod_status
    
    def get_free_gpus(self, total_gpus,allocations):
        return collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
    
    def update_alloc(self,prev_alloc,free_gpus):
        new_alloc = {}
        # free_gpus = copy.deepcopy(remain_gpus)
        for job, alloc in prev_alloc.items():
            new_alloc[job] = []
            for node in alloc:
                if node in free_gpus and free_gpus[node] > 0:
                    new_alloc[job].append(node)
                    free_gpus[node] -= 1
        
        return new_alloc

    def random_allocate(self,jobs,nodes,prev_allocations,infer_pod_status):
        self.infer_pod_status_trans(infer_pod_status)

        # LOG.info("random allocate")
        # LOG.info("jobs: %s",jobs)
        # LOG.info("random prev allocation: %s",prev_allocations)
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        
        
        for job in jobs:
            if job not in prev_allocations:
                prev_allocations[job] = []
                
            
        
        # LOG.info("update prev allocation: %s",prev_allocations)
        # LOG.info("infer pod status: %s",infer_pod_status)
        sleep_pods = set()
        infer_pods = set()

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
        
        infer_nodes = set()
        infer_alloc = {}
        prev_train_alloc = {}
        prev_sleep_alloc = {}
        infer_job_not_alloc = set()
        for job,alloc in prev_allocations.items():
            if job in train_jobs:
                prev_train_alloc[job] = alloc
            elif job in sleep_jobs:
                prev_sleep_alloc[job] = alloc
                
            if job not in infer_jobs:
                continue
            
            if len(alloc) > 0:
                infer_alloc[job] = alloc
            else:
                infer_job_not_alloc.add(job)
            
            for node_id in set(alloc):
                if self.node_id_dict[node_id] >= len(nodes) // 2: 
                    infer_nodes.add(node_id)

        allocations = {}
        allocations.update(infer_alloc) 
        
        free_gpus = self.get_free_gpus(total_gpus,allocations)
        
        available_gpu_list = []
        
        for node, num_gpu in free_gpus.items():
            available_gpu_list.extend([node] * num_gpu)
        
        
        for i, job in enumerate(infer_job_not_alloc):
            if i >= len(available_gpu_list):
                allocations[job] = []
            else:
                node = available_gpu_list[i]
                allocations[job] = [node]
                free_gpus[node] -= 1
        
                   
        new_train_alloc = self.update_alloc(prev_train_alloc,free_gpus)
        allocations.update(new_train_alloc)
        # free_gpus = self.get_free_gpus(total_gpus,allocations)
        sleep_alloc = self.update_alloc(prev_sleep_alloc,free_gpus)
        allocations.update(sleep_alloc)

        LOG.info("update alloc: %s",allocations)
        return allocations, len(nodes)
        
    def ATSI_infer_schedule(self, jobs, nodes, base_allocations, node_template, infer_pod_status):
        '''
        results = self.infer_scheduler.optimize(job_infos,node_infos,
            self.allocations,node_infos[0],self.infer_pod_status)
        '''
        LOG.info("InferScheduler optimize")
        LOG.info("base alloc: %s",base_allocations)
        LOG.info("infer pod status: %s",infer_pod_status)
        def ispinned(key, job):
            return not job.preemptible and base_allocations.get(key, []) != []
        
        jobs = OrderedDict(sorted(jobs.items(),
                                  key=lambda kv: (not ispinned(kv[0], kv[1]),
                                                  kv[1].attained_service,
                                                  kv[1].creation_timestamp)))

        nodes = OrderedDict(  # Sort preemptible nodes last.
            sorted(nodes.items(), key=lambda kv: (kv[1].preemptible, kv[0])))

        self._jobs = jobs
        self._nodes = nodes
        self.job_vals = list(jobs.values())
        self.node_vals = list(nodes.values())
        
        self._base_alloc = base_allocations
        self.base_state = self._allocations_to_state(base_allocations,jobs,nodes)
        
        self.pollux_config_init(
            jobs=list(jobs.values()),
            nodes=list(nodes.values()),
            infer_pod_status=infer_pod_status
        )
        
        update_infer_ids = np.where(np.all(self.infer_state == 0,axis=-1))[0]
        self.update_state = self.infer_state[update_infer_ids]
        
        self.node_training_gpus = np.sum(self.training_state,axis=0) 
        self.node_infer_gpus = np.sum(self.infer_state,axis=0) 
        self.node_sleep_gpus = np.sum(self.sleep_state,axis=0)

        self.best_cases = [] 
        
        self.num_training_jobs, self.num_nodes = self.training_state.shape 
        self.curr_training_state = self.training_state 
        self.remain_infer_tasks = len(self.update_state)
        
        self.curr_infer_job_id = 0

        if self.remain_infer_tasks > 0: 
            self.schedule_available_gpu()
        
        if self.remain_infer_tasks > 0: 
            if self.aryl:
                LOG.info("aryl")
                LOG.info("remain infer tasks: %s",self.remain_infer_tasks)
                self._aryl_schedule_training_gpu()
            else:
                self.schedule_training_gpu()
        
        self.infer_state[update_infer_ids] = self.update_state
        self.base_state[self.training] = self.training_state
        self.base_state[self.inference] = self.infer_state
        self.base_state[self.sleep] = self.sleep_state
        
        alloc = self._state_to_allocations(self.base_state,jobs,nodes)
        # LOG.info("infer scheduler alloc: %s",alloc)
        return alloc, None
                
    def optimize(self, jobs, nodes, base_allocations, node_template, infer_pod_status):
        if self.random:
            return self.random_allocate(jobs,nodes,base_allocations,infer_pod_status)

        else:
            return self.ATSI_infer_schedule(jobs, nodes, base_allocations, node_template, infer_pod_status)
