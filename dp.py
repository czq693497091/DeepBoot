# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import logging
import numpy as np
from collections import OrderedDict
import collections


LOG = logging.getLogger('simulator')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DeepBoot(object): # Use DP to calculate
    def __init__(self):
        self._prev_states = None
        self._prev_jobs = None
        self._prev_nodes = None
        self.jobs = None
        self.nodes = None
        self.total_gpus = None
        self.sched_train = True
        self.infer_schedule = True 
    
    def select_node(self, num_replica, free_gpus):
        '''
        num_replica: gpus needed by current tasks
        free_gpus: free gpus in each node
        '''
        ORIGIN_SELECT = False # False就是新的分配方案
        
        node_idx, count = free_gpus.most_common(1)[0]
        if ORIGIN_SELECT: 
            return node_idx, count
        
        if num_replica > count: 
            return node_idx, count
    
        else:
            f = {k:v for k,v in dict(free_gpus).items() if v >= num_replica}
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

    def _get_speedup(self, job, num_replicas): 
        # LOG.info("total gpus: %s",self.total_gpus)
        gpus_each_node = list(self.total_gpus.values())[0]
        num_nodes = num_replicas // gpus_each_node
        if num_replicas % gpus_each_node != 0:
            num_nodes += 1
            
        return job.speedup_fn(num_nodes, num_replicas)
        
    def max_value_dp(self,ws,vs,m):
        # group knapsack
        
        n = len(ws) - 1 
        dp = np.zeros(shape=(n+1,m+1))
        for i in range(1,n+1): # 任务
            for j in range(m,-1,-1):
                dp[i][j] = dp[i-1][j]
                for k in range(len(ws[i])): 
                    if j >= ws[i][k]:
                        dp[i][j] = max(dp[i][j],dp[i-1][j-ws[i][k]]+vs[i][k])
        
        j = m
        ways = np.zeros(n+1,dtype=int) 
        for i in range(n,0,-1):
            for k in range(len(ws[i])):
                if j >= ws[i][k] and dp[i][j] == dp[i-1][j-ws[i][k]] + vs[i][k]:
                    ways[i] = ws[i][k] # limitation of task k
                    j -= ws[i][k]
                    break
                
        return ways[1:]
      
    def allocate_elastic(self,prev_allocations, jobs,free_gpus):
        num_gpus = sum(free_gpus.values())      
        ws = [[]]
        vs = [[]]
        for job, info in jobs.items():
            temp_w = []
            temp_v = []
            
            num_restarts = info.num_restarts
            age = info.age
            delay = 10
            factor = max(age - num_restarts * delay, 0.0) / (age + delay)
            
            for w in range(1,info.max_replicas + 1):
                temp_w.append(w)
                speedup = self._get_speedup(info,w)
                if job not in prev_allocations or w != len(prev_allocations[job]):
                    speedup *= factor
                    
                temp_v.append(speedup)
        
            ws.append(temp_w)
            vs.append(temp_v)
               
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
         
    def infer_pod_status_trans(self,infer_pod_status):
        _infer_pod_status = dict()
        for _,pods in infer_pod_status.items():
            for name, info in pods.items():
                _infer_pod_status[name] = info
        
        self.infer_pod_status = _infer_pod_status

    def get_free_gpus(self, total_gpus,allocations):
        return collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
    
    def optimize(self, jobs, nodes, base_allocations, node_template, clock = None, infer_pod_status = None):
        def ispinned(key, job):
            return not job.preemptible and base_allocations.get(key, []) != []
        
        sleep_pods = set()
        infer_pods = set()
        self.total_gpus = total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        self.infer_pod_status_trans(infer_pod_status)
        self.node_id_dict = dict(zip(nodes.keys(),range(len(nodes))))
        
        prev_allocations = base_allocations
        
        for name,info in self.infer_pod_status.items():
            if info['status'] == 'SLEEP':
                sleep_pods.add(name)
            else: # PROTECT or RUNNING
                infer_pods.add(name)
                
        self.jobs = jobs = OrderedDict(sorted(jobs.items(),
                                  key=lambda kv: (not ispinned(kv[0], kv[1]),
                                                  kv[1].attained_service,
                                                  kv[1].creation_timestamp)))
        
        self.nodes = nodes = OrderedDict(  # Sort preemptible nodes last.
            sorted(nodes.items(), key=lambda kv: (kv[1].preemptible, kv[0])))
        
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

        self._jobs = train_jobs
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
                # [n_node // 2, n_node) is the inference node
                if self.node_id_dict[node_id] >= len(nodes) // 2: 
                    infer_nodes.add(node_id)
        
        
        rtypes = sorted(set.union(*[set(job.resources) for job in self._jobs.values()]))
        # Build array of job resources: <num_jobs> x <num_rtypes>. Each entry
        # [j, r] is the amount of resource r requested by a replica of job j.
        self._job_resources = np.zeros((len(self._jobs), len(rtypes)), np.int64)
        for j, job in enumerate(self._jobs.values()):
            for r, rtype in enumerate(rtypes):
                self._job_resources[j, r] = job.resources.get(rtype, 0)

        # Build array of node resources: <num_nodes> x <num_rtypes>. Each
        # entry [n, r] is the amount of resource r available on node n.
        self._node_resources = np.zeros((len(nodes), len(rtypes)), np.int64) 
        for n, node in enumerate(nodes.values()):
            for r, rtype in enumerate(rtypes):
                self._node_resources[n, r] = node.resources.get(rtype, 0)
        

        for job_name, alloc in infer_alloc.items():
            for r, rtype in enumerate(rtypes):
                # LOG.info("alloc: %s",alloc)
                if len(alloc) == 0:
                    continue
                node_id = self.node_id_dict[alloc[0]]
                self._node_resources[node_id] -= infer_jobs[job_name].resources.get(rtype, 0)
            
        # Calculate dominant per-replica resource shares for each job.
        shares = self._job_resources / np.sum(self._node_resources, axis=0)
        self._dominant_share = np.amax(shares, axis=1)
           
        # Change base goodput to fair-share goodput.
        fair_replicas = np.ceil(1.0 / self._dominant_share / len(self._jobs))
        fair_nodes = np.ceil(len(nodes) * self._dominant_share)
        
        # 主要是这里要更新speedup
        for job, num_nodes, num_replicas in zip(self._jobs.values(), fair_nodes, fair_replicas):
            # LOG.info("num nodes: %s, num replicas: %s",num_nodes, num_replicas)
            if not hasattr(job.speedup_fn, "_goodput_fn"):
                job.speedup_fn = lambda n, r: r / num_replicas
                continue 
            
            
            job.speedup_fn._base_goodput = job.speedup_fn._goodput_fn.optimize(
                num_nodes=num_nodes, num_replicas=max(num_replicas,num_nodes),
                max_batch_size=job.speedup_fn._max_batch_size,
                atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                accumulation=job.speedup_fn._accumulation)[0]
    
                
        allocations = {}
        allocations.update(infer_alloc) 
        
        free_gpus = self.get_free_gpus(total_gpus,infer_alloc)

        train_alloc = self.allocate_elastic(prev_train_alloc, self._jobs, free_gpus)
        remain_gpus = self.get_free_gpus(free_gpus,train_alloc)
        allocations.update(train_alloc) 

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

                    

