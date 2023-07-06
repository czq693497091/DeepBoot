import argparse
import collections
import copy
import glob
import json
import math
import multiprocessing
import os

import numpy as np
import pandas

from applications import APPLICATIONS, APPLICATIONS_DELAY, FIRST_DELAY, NEXT_DELAY
from goodput import GoodputFunction, fit_perf_params
from speedup import SpeedupFunction
from utils import JobInfo, NodeInfo
from pollux import PolluxPolicy
from afs import AFS
from optimus import Optimus
from aryl import Aryl
from dp import DeepBoot

from tiresias import TiresiasPolicy
import logging
import os
import time
from infer_scheduler import InferScheduler

LOG = logging.getLogger('simulator')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# log_file = './logs/simulator.log'
# if os.path.exists(log_file):
#     os.remove(log_file)

# fh = logging.FileHandler(log_file)
# fh.setFormatter(formatter)
# LOG.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(formatter)
LOG.addHandler(ch)

# By default use ATS-I to schedule the inference task, otherwise use pollux's NSGA II
INFER_SCHEDULER = True  

# By default True, which means if use NSGA II to schedule inference task, we make sure training allocation updates
# only when interval (60s)
REPAIR_TRAIN = True  

# Inference schedule in Aryl
ARYL = True

# ATS-I to manage the lifecycle of inference task
AISS = True 

# AFE to reduce the allocation update cost
AFE = False  

PROTECT_TIMES = 1  
RANDOM_ALLOCATE = 0

t_protect_max = 120
t_bonus = 15
t0 = 30
NUM_NODE = -1
schedule_cost_dict = {}  


def calculate_protect_time(info):
    t_protect_max = PROTECT_TIMES * 120
    t_bonus = PROTECT_TIMES * 15
    t0 = PROTECT_TIMES * 30
    return int(min(t0 + info.get('cache_times', 0) * t_bonus, t_protect_max))


def app_trans(app_name):
    app_name_split = app_name.split('-')
    # LOG.info("app name split: %s")
    if len(app_name_split) == 2 or 'infer' not in app_name:
        return app_name
    else:
        return "-".join(app_name_split[:-1])


class Job(object):
    pretrain = {}

    def __init__(self, name, application, submission_time,
                 target_num_replicas=None, target_batch_size=None, duration=None):
        self.name = name
        self.application = application
        self.submission_time = submission_time
        self.target_num_replicas = target_num_replicas
        self.target_batch_size = target_batch_size
        self.completion_time = None
        self.current_time = 0
        self.rescale_time = 0
        self.placement = ()
        self.atomic_bsz = 0
        self.accum_steps = 0
        self.profile = {}
        self.perf_params = None
        self.grad_params = None
        self.best_metric = None
        self.progress = 0.0
        self.epoch = 0
        self.attained_service = 0
        self.num_restarts = None
        self.inference = False
        self.total_delay = 0  
        self.total_delay_with_placement = 0  
        self.duration = duration
        self.start_execute_time = None
        self.evaluate_finish_time = None
        self.run_time = 0

        self.current_rescale_time = 0
        self.protect_time = 30

        # (timestamp, prev_alloc, curr_alloc)
        self.placement_update_history = []
        self.app = self.application.name
        if 'infer' in self.app:
            self.target_batch_size = 1
            self.target_num_replicas = 1
            self.inference = True
            self.atomic_bsz = 1
            self.pod_name = None 
            self.status = "START"
            self.use_cache = False

    @property
    def max_profiled_replicas(self):
        return max((k[1] for k in self.profile), default=0)

    def get_goodput_fn(self):
        app = self.application
        return GoodputFunction(self.perf_params, self.grad_params, app.init_batch_size)

    def get_speedup_fn(self):
        if self.perf_params is None:
            return lambda n, r: r

        app = self.application
        return SpeedupFunction(self.get_goodput_fn(), app.max_batch_size,
                               (app.min_local_bsz, app.max_local_bsz),
                               accumulation=True)

    def update_local_bsz(self, placement):
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size
        if batch_size is None and self.perf_params is None:
            batch_size = max(app.init_batch_size,
                             app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn()
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, app.max_batch_size,
                (app.min_local_bsz, app.max_local_bsz), accumulation=True)
        else:
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(
                local_bsz / app.max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.init_batch_size:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(
                local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))

    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var):
        self.grad_params = (grad_sqr, grad_var)
        if (num_nodes, num_replicas, local_bsz) in self.profile:
            return
        self.profile[num_nodes, num_replicas, local_bsz] = step_time, sync_time
        num_nodes = np.array([key[0] for key in self.profile])
        num_replicas = np.array([key[1] for key in self.profile])
        local_bsz = np.array([key[2] for key in self.profile])
        step_time = np.array([val[0] for val in self.profile.values()])
        sync_time = np.array([val[1] for val in self.profile.values()])
        compute_time = step_time - sync_time
        self.perf_params = fit_perf_params(
            num_nodes, num_replicas, local_bsz, compute_time, step_time)

    def step(self, seconds, interference=0.0, cluster=None):
        
        infer_pod_status = cluster.infer_pod_status
        if not self.placement:
            # No resources are allocated to this job.
            if not self.inference or self.status == 'FINISH' or self.submission_time > cluster.clock:
                self.current_time += seconds
                return

            # 推理任务找里面适配的pod
            elif self.inference and self.status == 'START' and len(infer_pod_status.get(self.app, {})) > 0:
                LOG.info("%s seeks for infer pod", self.name)
                LOG.info("infer pod status: %s", infer_pod_status)

                # 缓存次数高的pod优先执行
                infer_pod_status[self.app] = dict(sorted(infer_pod_status[self.app].items(),
                                                         key=lambda x: x[1].get('cache_times', 0), reverse=True))
                infer_pod_app = infer_pod_status[self.app]

                for pod_name, info in infer_pod_app.items():
                    '''
                    满足该条件则可以走推理缓存,不需要启动时间
                    '''
                    if info['status'] == 'SLEEP' or info['status'] == 'PROTECT' and AISS:
                        LOG.info("find the infer pod %s", pod_name)

                        self.status = 'RUNNING'
                        self.pod_name = pod_name
                        self.rescale_time = 0 
                        self.use_cache = True  
                        self.start_execute_time = cluster.clock
                        self.evaluate_finish_time = self.start_execute_time + self.duration
                        self.completion_time = self.evaluate_finish_time
                        self.current_time = cluster.clock
                        self.total_delay = self.start_execute_time - self.submission_time
                        cluster.submit_time[self.evaluate_finish_time] = 1

                        if info['status'] == 'SLEEP':
                            info['cache_times'] = 0

                        else:
                            info['cache_times'] = info.get(
                                'cache_times', 0) + 1  # 缓存命中保护时间 + 1

                        info['curr_job'] = self.name
                        info['live_time'] = calculate_protect_time(info)
                        info['status'] = 'RUNNING'
                        info['completion_time'] = self.completion_time

                        return
    
                LOG.info("No PROTECT or SLEEP pod")
                self.current_time += seconds
                return

            else:
                self.current_time += seconds
                return

        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        self.total_delay += delay
        self.total_delay_with_placement += delay * sum(self.placement)
        seconds -= delay

        while seconds > 0 and self.completion_time is None:
            assert self.epoch < self.application.max_epochs

            if not self.inference:
                # Calculate current job configurations.
                placement = tuple(filter(None, self.placement))
                num_nodes, num_replicas = len(placement), sum(placement)
                local_bsz = self.atomic_bsz
                batch_size = num_replicas * \
                    self.atomic_bsz * (self.accum_steps + 1)
                scale = batch_size / self.application.init_batch_size
                # Calculate true (simulated) throughput.
                step_time, sync_time = \
                    self.application.get_throughput(placement, self.atomic_bsz)
                accum_time = step_time - sync_time
                # Calculate true (simulated) efficiency.
                grad_sqr, grad_var = \
                    self.application.get_grad_stats(batch_size, self.epoch)
                gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)
                # Update the estimated throughput/efficiency parameters.
                self.update_params(num_nodes, num_replicas, self.atomic_bsz,
                                   step_time, sync_time, grad_sqr, grad_var)
                # Calculate true (simulated) goodput.
                total_time = step_time + accum_time * self.accum_steps
                goodput = gain / total_time * (1.0 - interference)
                # Update current epoch and progress.
                next_progress = self.application.get_progress(self.epoch + 1)
                if self.progress + goodput * seconds < next_progress:
                    # Used up the entire time interval without finishing an epoch.
                    self.progress += goodput * seconds
                    self.current_time += seconds
                    self.attained_service += seconds * sum(self.placement)
                    self.run_time += seconds
                    seconds = 0
                else:
                    # Crossed an epoch boundary before finishing the time interval.
                    self.epoch += 1
                    delta = round(
                        float((next_progress - self.progress) / goodput))
                    assert delta <= seconds
                    completion_epoch = \
                        self.application.get_completion_epoch(batch_size)
                    if self.epoch > completion_epoch:
                        self.completion_time = self.current_time + delta
                    self.progress = next_progress
                    self.best_metric = \
                        self.application.get_best_metric(
                            batch_size, self.epoch)
                    self.current_time += delta
                    self.attained_service += delta * sum(self.placement)
                    self.run_time += delta
                    seconds -= delta
                # Re-scale batch size between epochs.
                self.update_local_bsz(self.placement)

            else:  # Inference task
                LOG.info("else infer: %s %s", self.name, self.pod_name)
                info = infer_pod_status[self.app][self.pod_name]
                self.completion_time = self.current_time + self.duration
                self.current_time += seconds

                flag = False
                if cluster.clock >= self.completion_time:

                    info['status'] = 'PROTECT'
                    info['curr_job'] = None
                    flag = True

                else:
                    info['status'] = 'RUNNING'
                    info['curr_job'] = self.name

                info['live_time'] = calculate_protect_time(info)
                info['completion_time'] = self.completion_time

                if not AISS and flag:
                    infer_pod_status[self.app].pop(self.pod_name)

                return
        self.current_time += seconds  # Add any remaining time.

    def calculate_rescale_time(self, origin_placement, current_placement):
        app_name = self.application.name

        first_delay = FIRST_DELAY[app_name]
        next_delay = NEXT_DELAY[app_name]
        if len(origin_placement) == 0:  # First start, do not avoid the reduce cost
            return first_delay

        elif len(current_placement) == 0:
            return 0

        # 这里实际上是因为折算了
        if sum(origin_placement) < sum(current_placement):  # 扩容, 因为新重启的容器还要经历一次完整的重启时间 客观起见去掉
            return next_delay + (first_delay - next_delay) * (sum(current_placement) - sum(origin_placement)) // sum(current_placement)

        elif sum(origin_placement) > sum(current_placement):
            return next_delay

        else:  # 同扩同删的情况
            return first_delay

    def calculate_real_rescale_time(self, origin_placement, current_placement):
        app_name = self.application.name

        first_delay = FIRST_DELAY[app_name]
        next_delay = NEXT_DELAY[app_name]
        if len(origin_placement) == 0:  
            return first_delay
        elif len(current_placement) == 0:
            return 0

        if sum(origin_placement) != sum(current_placement): 
            return next_delay

        else:
            return first_delay

    def reallocate(self, placement):
        if not self.inference:
            self.placement_update_history.append(
                (self.current_time, self.placement, tuple(placement))
            )

        if placement:
            LOG.info("origin placement: %s, curr placement: %s",
                     self.placement, placement)
            origin_placement = self.placement
            self.placement = tuple(placement)
            if not self.inference:
                self.update_local_bsz(self.placement)
                if AFE:
                    self.rescale_time = self.calculate_rescale_time(
                        origin_placement, placement)
                else:
                    delay_dict = APPLICATIONS_DELAY

                    # Start re-scale countdown. 这里要根据任务类型改
                    self.rescale_time = delay_dict[self.application.name]
                # self.rescale_time = 0 # 理论上限
                self.current_rescale_time = self.rescale_time

                if self.num_restarts is None:
                    self.num_restarts = 0
                else:
                    self.num_restarts += 1

            elif len(placement) > 0 and self.start_execute_time is None:
                app_name = app_trans(self.app)
                self.rescale_time = APPLICATIONS_DELAY[app_name]

                self.start_execute_time = self.current_time
                self.evaluate_finish_time = self.start_execute_time + \
                    self.duration + self.rescale_time

        else:  # De-allocate all resources.
            self.placement = ()
            self.atomic_bsz = 0


class Cluster(object):
    def __init__(self, workload, policy, min_nodes, num_gpus=4,
                 max_nodes=None, interference=0.0,
                 low_util=None, high_util=None):
        # assert 1 <= num_gpus <= 4
        self.workload = workload
        self.policy = policy
        self.min_nodes = self.num_nodes = min_nodes
        self.num_gpus = num_gpus
        self.max_nodes = min_nodes if max_nodes is None else max_nodes
        self.interference = interference
        self.low_util = low_util
        self.high_util = high_util
        self.current_time = 0
        self.clock = 0

        self.infer_scheduler = InferScheduler()
        self.infer_scheduler.aryl = ARYL

        self.infer_scheduler.random = RANDOM_ALLOCATE

        self.infer_pod_status = dict()
        self.protect_time = 30

        self.optimize_history = []  # time, base_state.shape, cost

        total_gpus = self.num_gpus * self.num_nodes
        self.gpu_util_dict = {"clock": [], "real_gpu_use": [
        ], "real_running_gpu_use": [], "gpu_use": []}
        
        self.metric_dict = {
            "clock": [],
            "sum_goodput": [],
            "avg_goodput": [],
            "sum_speedup": [],
            "avg_speedup": []
        }
        
        LOG.info("simulator total gpu: %s", total_gpus)

        if isinstance(policy, PolluxPolicy) or isinstance(policy, DeepBoot):
            self.jobs = [Job(row.name, APPLICATIONS[app_trans(row.application)], row.time, duration=row.duration)
                         for row in workload.itertuples()]
            for job in self.jobs:
                if job.application.name == "ncf":
                    job.target_batch_size = 32768
        elif isinstance(policy, TiresiasPolicy):
            self.jobs = [Job(row.name, APPLICATIONS[app_trans(row.application)], row.time,
                             target_num_replicas=row.num_replicas,
                             target_batch_size=row.batch_size, duration=row.duration)
                         for row in workload.itertuples()]
        elif isinstance(policy, Optimus):
            self.jobs = [Job(row.name, APPLICATIONS[app_trans(row.application)], row.time,
                             target_batch_size=row.batch_size, duration=row.duration)
                         for row in workload.itertuples()]

        elif isinstance(policy, AFS) or isinstance(policy, Aryl):
            self.jobs = [Job(row.name, APPLICATIONS[app_trans(row.application)], row.time, duration=row.duration,
                             target_num_replicas=row.num_replicas,
                             target_batch_size=row.batch_size)
                         for row in workload.itertuples()]
            for job in self.jobs:
                if job.application.name == "ncf":
                    job.target_batch_size = 32768

        self.job_dict = dict()

        for i, row in enumerate(workload.itertuples()):
            self.jobs[i].app = row.application

        for job in self.jobs:
            self.job_dict[job.name] = job

        self.allocations = {}
        self.logs = []
        self.utility = []
        self.current_log = []

        # new add #
        self.finish_job_set = set()
        self.submit_time = {}
        for job in self.jobs:
            self.submit_time[job.submission_time] = job.duration

    def aryl_remove(self, allocation):
        # LOG.info("alloc: %s",allocation)
        infer_nodes = set()
        for job, alloc in allocation.items():
            if 'infer' in job and len(alloc) > 0:
                infer_nodes.add(alloc[0])

        new_alloc = {job: [] for job in allocation}
        for job in new_alloc:
            if 'infer' in job:
                new_alloc[job] = allocation[job]
                continue

            for node in allocation[job]:
                if node not in infer_nodes:
                    new_alloc[job].append(node)

        return new_alloc

    def update_infer_pod_status(self):

        flag = False
        remove_pods = []
        for application, pods in self.infer_pod_status.items():
            for pod, info in pods.items():
                status = info['status']
                if status == 'RUNNING':

                    if self.clock >= info['completion_time']: 
                        flag = True

                        if AISS:
                            info['status'] = 'PROTECT'
                            info['live_time'] = calculate_protect_time(info)
                            info['curr_job'] = None
                            info['completion_time'] = np.inf

                        else:
                            remove_pods.append((application, pod))
                else:
                    if info['live_time'] > 0:
                        info['live_time'] -= 1

                    else:
                        info['status'] = 'SLEEP'
                        info['cache_times'] = 0

        if not AISS and len(remove_pods) > 0:
            # 直接删掉状态
            LOG.info("remove pods: %s", remove_pods)
            for application, pod in remove_pods:
                self.infer_pod_status[application].pop(pod)
        if flag:
            LOG.info("infer pod status update: %s", self.infer_pod_status)

    def _allocations_to_state(self, allocations, jobs, nodes):
        jobs_index = {key: idx for idx, key in enumerate(jobs)}
        nodes_index = {key: idx for idx, key in enumerate(nodes)}
        state = np.zeros((len(jobs), len(nodes)), dtype=np.int)
        for job_key, alloc in allocations.items():
            for node_key in (key for key in alloc if key in nodes_index):
                state[jobs_index[job_key], nodes_index[node_key]] += 1
        return state

    def step(self, seconds=60, interval=60):
        '''
        seconds: time spend from previous schedule
        interval: interval for training tasks
        '''

        # self.update_infer_pod_status()
        interfere_nodes = set(idx for idx in range(self.num_nodes)
                              if sum(len(set(val)) > 1 and idx in val
                                     for key, val in self.allocations.items()) > 1)

        for job in self.jobs:
            job.clock = self.clock
            if job.completion_time and job.completion_time <= self.clock:
                job.status = 'FINISH'
            alloc_set = set(self.allocations.get(job.name, []))
            interference = 0.0
            if len(alloc_set) > 1 and any(idx in interfere_nodes for idx in alloc_set):
                interference = self.interference

            job.step(seconds, interference=interference, cluster=self)

            if job.completion_time and job.name not in self.finish_job_set:
                # finish_job_list.append(job)
                self.finish_job_set.add(job.name)
                LOG.info("finish job set %s", self.finish_job_set)

        self.current_time += seconds
        LOG.info("cluster current time: %s", self.current_time)
        assert all(job.current_time == self.current_time for job in self.jobs)
        job_infos = self.get_job_infos()
        if job_infos:
            if self.max_nodes > self.min_nodes:
                # Autoscale cluster if needed.
                self.utility.append(self.get_utility(
                    self.num_nodes, job_infos, self.allocations))
                if len(self.utility) > 15:
                    self.utility.pop(0)
                    utility = sum(self.utility) / len(self.utility)
                    if (self.num_nodes > self.min_nodes and utility < self.low_util) or \
                            (self.num_nodes < self.max_nodes and utility > self.high_util):
                        self.autoscale(job_infos)
                        self.utility.clear()
                    LOG.info("Utility: %s", utility)
                LOG.info("Nodes: %s", self.num_nodes)
            # Optimize allocations.
            node_infos = self.get_node_infos()
            self.allocations = {
                k: v for k, v in self.allocations.items() if k in job_infos}

            t1 = time.time()

            LOG.info("infer scheduler: %s", INFER_SCHEDULER)
            if INFER_SCHEDULER:  
                if self.clock % interval == 0:
                    results = self.policy.optimize(
                        job_infos,
                        node_infos,
                        self.allocations,
                        node_infos[0],
                        self.clock,
                        self.infer_pod_status,
                    )

                else:
                    results = self.infer_scheduler.optimize(
                        job_infos,
                        node_infos,
                        self.allocations,
                        node_infos[0],
                        self.infer_pod_status,
                    )

            else:
                results = self.policy.optimize(
                    job_infos,
                    node_infos,
                    self.allocations,
                    node_infos[0],
                    self.clock,
                    self.infer_pod_status
                )

            t2 = time.time()

            optimize_time = round(t2 - t1, 3)

            num_jobs = len(job_infos)
            num_nodes = len(node_infos)

            if num_jobs not in schedule_cost_dict:
                schedule_cost_dict[num_jobs] = []

            schedule_cost_dict[num_jobs].append(
                {'cost': optimize_time, 'clock': self.clock, 'nodes': num_nodes}
            )

            allocations, _ = results

            if ARYL:  
                allocations = self.aryl_remove(allocations)

            LOG.info("allocations: %s", allocations)
            LOG.info("optimize time: %s", optimize_time)
            LOG.info("schedule_cost_dict: %s", {
                     'cost': optimize_time, 'clock': self.clock, 'nodes': num_nodes})
            # LOG.info("infer pod status: %s",self.infer_pod_status)
            for job, alloc in allocations.items():
                job_application = self.job_dict[job].app
                if job_application in self.infer_pod_status and job in self.infer_pod_status[job_application]:
                    pod_info = self.infer_pod_status[job_application][job]
                    if pod_info['status'] == 'SLEEP' and len(alloc) == 0:
                        LOG.info("pop %s", job)
                        self.infer_pod_status[job_application].pop(job)

            states = self._allocations_to_state(
                allocations, job_infos, node_infos)

            self.optimize_history.append(
                (self.clock, states.shape, optimize_time)
            )

            used_gpus = collections.Counter(sum(allocations.values(), []))
            assert all(val <= node_infos[key].resources["nvidia.com/gpu"]
                       for key, val in used_gpus.items())
            for job in self.jobs:
                if allocations.get(job.name) != self.allocations.get(job.name):
                    alloc = allocations.get(job.name, [])
                    job.alloc = alloc
                    placement = []
                    for i in range(len(alloc)):
                        if i == 0 or alloc[i] != alloc[i - 1]:
                            placement.append(1)
                        else:
                            placement[-1] += 1
                    job.reallocate(placement)
                    if job.evaluate_finish_time and job.evaluate_finish_time not in self.submit_time:
                        self.submit_time[job.evaluate_finish_time] = 1

            self.allocations = allocations
            self.init_new_pod_status()

        self.current_log = {
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "optimize_history": self.optimize_history,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "allocation": self.allocations.get(job.name, []),
                    "placement": job.placement,
                    "batch_size": job.atomic_bsz * (job.accum_steps + 1) * sum(job.placement),
                    "accum_steps": job.accum_steps,
                    "submission_time": job.submission_time,
                    "completion_time": job.completion_time,
                    "grad_params": job.grad_params,
                    "rescale_time":job.rescale_time,
                    "run_time": job.run_time,
                    "start_execute_time": job.start_execute_time,
                    "evaluate_finish_time": job.evaluate_finish_time,
                    "delay_time": job.total_delay,
                    "placement_update_history": job.placement_update_history
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        }
        # self.logs.append()

    def init_new_pod_status(self):
        for job in self.jobs:
            # job = self.job_dict[name]
            if not job.inference or job.name in self.finish_job_set or job.name not in self.allocations or len(self.allocations[job.name]) == 0:
                continue

            if job.app not in self.infer_pod_status:
                self.infer_pod_status[job.app] = dict()

            if job.name not in self.infer_pod_status[job.app]:
                self.infer_pod_status[job.app][job.name] = {
                    'curr_job': job.name,
                    'status': 'RUNNING',
                    'live_time': job.protect_time,
                    'completion_time': np.inf
                }

                job.pod_name = job.name

    def autoscale(self, job_infos):
        target_utility = (self.low_util + self.high_util) / 2
        min_nodes = self.min_nodes
        max_nodes = self.max_nodes
        num_nodes = self.num_nodes
        while min_nodes + 1 < max_nodes:
            utility = self.get_utility(num_nodes, job_infos)
            if utility < target_utility:
                max_nodes = num_nodes
            elif utility > target_utility:
                min_nodes = num_nodes
            else:
                break
            num_nodes = (min_nodes + max_nodes) // 2
        min_util = self.get_utility(min_nodes, job_infos)
        max_util = self.get_utility(max_nodes, job_infos)
        if abs(target_utility - min_util) < abs(target_utility - max_util):
            self.num_nodes = min_nodes
        else:
            self.num_nodes = max_nodes

    def get_utility(self, num_nodes, job_infos, allocations=None):
        node_infos = self.get_node_infos(num_nodes)
        if allocations is None:
            # policy = copy.deepcopy(self.policy)
            results = self.policy.optimize(job_infos, node_infos,
                                           self.allocations)
            allocations = results[0][1]
        sum_speedup = 0.0
        for key, alloc in allocations.items():
            if key in job_infos:
                speedup_fn = job_infos[key].speedup_fn
                speedup = speedup_fn(len(set(alloc)), len(alloc))
                sum_speedup += speedup
        return sum_speedup / (num_nodes * self.num_gpus)

    def is_valid_job(self, job):

        # 训练任务不变
        if self.current_time >= job.submission_time and job.completion_time is None:
            return True

        if not job.inference:  # 训练任务不满足上述条件的直接返回False
            return False

        application = job.app
        cond1 = not job.use_cache  # 使用推力缓存的不计入

        # 如果这个job是作为推理缓存，那么需要计入job info
        cond2 = False

        # 如果最终状态为END
        if application in self.infer_pod_status:
            if job.name in self.infer_pod_status[application]:
                pod = self.infer_pod_status[application][job.name]
                # LOG.info("podinfo: %s, cond2",pod)
                cond2 = pod['status'] != 'END'  # RUNNING, PROTECT, SLEEP都是True

        return cond1 and cond2

    def get_job_infos(self):

        job_infos = {}
        for job in self.jobs:
            if self.is_valid_job(job):
                if isinstance(self.policy, TiresiasPolicy):
                    job_infos[job.name] = self.get_tiresias_job_info(job)
                elif isinstance(self.policy, Optimus):
                    
                    job_infos[job.name] = self.get_optimus_job_info(job)

                elif isinstance(self.policy, AFS) or isinstance(self.policy, Aryl):
                    job_infos[job.name] = self.get_afs_job_info(job)

                else:
                    job_infos[job.name] = self.get_pollux_job_info(job)

                # 判断是否为推理任务,在排序时优先级最高
                job_infos[job.name].inference = job.inference
                job_infos[job.name].duration = job.duration

        return job_infos

    def get_pollux_job_info(self, job):
        job_info = JobInfo(
            job=job,
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            run_time=job.run_time,
            min_replicas=0,
            max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz),

            # max_replicas=min(64,  # simulator can't handle more.
            #                  job.application.max_batch_size // job.application.min_local_bsz),
            preemptible=True,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_optimus_job_info(self, job):
        job_info = JobInfo(
            job=job,
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            run_time=job.run_time,
            min_replicas=0,
            # max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
            #                 job.target_batch_size // job.application.min_local_bsz),
            max_replicas=(job.target_batch_size //
                          job.application.min_local_bsz),
            preemptible=True,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.epoch = job.epoch
        job_info.application = job.application
        job_info.target_batch_size = job.target_batch_size
        return job_info

    def get_afs_job_info(self, job):
        job_info = JobInfo(
            job=job,
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            run_time=job.run_time,
            min_replicas=0,
            max_replicas=job.target_num_replicas,
            # max_replicas=(job.target_batch_size // job.application.min_local_bsz),
            preemptible=True,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.epoch = job.epoch
        job_info.application = job.application
        job_info.target_batch_size = job.target_batch_size
        return job_info

    def get_tiresias_job_info(self, job):
        return JobInfo(
            job=job,
            resources={"nvidia.com/gpu": 1},
            speedup_fn=None,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            run_time=job.run_time,
            min_replicas=0,
            max_replicas=job.target_num_replicas,
            preemptible=True,
        )

    def get_node_infos(self, num_nodes=None):
        return {
            idx: NodeInfo({"nvidia.com/gpu": self.num_gpus}, preemptible=False)
            for idx in range(num_nodes or self.num_nodes)
        }

    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    def output_logs(self, path):
        LOG.info("output_logs")
        if os.path.isdir(path):
            path = os.path.join(path, 'jobinfo.log')
        with open(path, "w") as f:
            # record = self.logs[-1]
            record = self.current_log
            json.dump(record, f)
            f.write("\n")

    def output_gpu_util_info(self, path):
        with open(path, "w") as f:
            record = self.gpu_util_dict
            # for record in self.logs:
            json.dump(record, f)
            
    def output_metric_info(self,path):
        with open(path, "w") as f:
            record = self.metric_dict
            json.dump(record, f)

    def get_jcts(self):
        return {
            val["name"]: val["completion_time"] - val["submission_time"]
            # for val in self.logs[-1]["submitted_jobs"]
            for val in self.current_log["submitted_jobs"]
            if val["completion_time"] is not None
        }

    def calculate_goodput_and_speedup(self):
        speedups = []
        goodputs = []
        job_infos = self.get_job_infos()
        for job in self.jobs:
            if job.inference or job.name not in self.allocations:
                continue
            # job.name
            if job.submission_time <= self.current_time and job.completion_time is None:
                
                if job.grad_params is None or job.perf_params is None:
                    continue
                job_info = job_infos[job.name]
                
                job_alloc = self.allocations[job.name]
                
                num_replicas = len(job_alloc)
                num_nodes = len(set(job_alloc))

                goodput = job_info.speedup_fn._base_goodput
                
                goodputs.append(goodput)
                
                if not hasattr(job_info.speedup_fn, "_goodput_fn"):
                    speedup_fn = lambda n, r: r / num_replicas
                else:
                    # print("has speedup_fn")
                    speedup_fn = job_info.speedup_fn
                
                speedup = speedup_fn(num_nodes,num_replicas)
                speedups.append(speedup)

                # print("goodput:",goodput)
                # print("speedup:",speedup)

        sum_goodput = 0
        avg_goodput = 0
        sum_speedup = 0
        avg_speedup = 0
        
        if len(goodputs) != 0:
            sum_goodput = round(np.sum(goodputs),2)
            avg_goodput = round(np.average(goodputs),2)
        
        if len(speedups) != 0:
            sum_speedup = round(np.sum(speedups),2)
            avg_speedup = round(np.average(speedups),2)
        
        return sum_goodput, avg_goodput, sum_speedup, avg_speedup

    def calculate_real_gpu_usage(self):
        real_used_gpu = 0
        used_gpu = 0
        real_running_gpu = 0

        for app, pods in self.infer_pod_status.items():
            for name, info in pods.items():
                if info['status'] == 'RUNNING':
                    real_running_gpu += 1

        for job in self.jobs:
            if job.submission_time <= self.current_time and job.completion_time is None:
                # if not job.inference:

                if job.current_rescale_time == 0:
                    real_used_gpu += sum(job.placement)

                    if not job.inference:
                        real_running_gpu += sum(job.placement)
                used_gpu += sum(job.placement)

            job.current_rescale_time = max(job.current_rescale_time - 1, 0)

        return real_used_gpu, real_running_gpu, used_gpu

    def event_update(self):
        is_training_interval = self.clock % args.interval == 0  # 训练任务固定调度周期

        if is_training_interval:
            LOG.info("case1 schedule interval")
            return True

        # 有任务完成
        is_job_finish = self.clock in self.submit_time and self.submit_time[self.clock] > 0

        if is_job_finish:
            LOG.info("case2 job finish")
            return True

        suspend_infer_jobs = []
        suspend_job_app = set()
        for job in self.jobs:
            if job.inference and job.completion_time is None and job.submission_time <= self.clock and len(job.placement) == 0:
                suspend_infer_jobs.append(job)
                suspend_job_app.add(job.app)

        if len(suspend_job_app) == 0:
            return False

        for app, pods in self.infer_pod_status.items():
            for name, info in pods.items():
                status = info['status']
                if status == 'SLEEP':
                    return True

                elif status == 'PROTECT' and app in suspend_job_app:
                    return True

        return False


def simulate(args):
    workload = pandas.read_csv(args.workload)
    if args.policy == "tiresias":
        policy = TiresiasPolicy(lambda: simulator.current_time)
    elif args.policy == "optimus":
        policy = Optimus()
    elif args.policy == 'afs':
        policy = AFS()
    elif args.policy == 'aryl': 
        policy = Aryl()

    elif args.policy == 'dp':
        policy = DeepBoot()

    else:
        policy = PolluxPolicy()
        policy.sched_train = INFER_SCHEDULER
        policy.repair_train = REPAIR_TRAIN

    policy.interval = args.interval
    policy.infer_priority = args.infer_priority

    simulator = Cluster(workload, policy, args.min_nodes, num_gpus=args.num_gpus,
                        max_nodes=args.max_nodes, interference=args.interference,
                        low_util=args.low_util, high_util=args.high_util)

    simulator.protect_time = args.protect_time

    previous_clock = 0

    while not simulator.all_complete():

        simulator.clock += 1
        real_gpu_util, real_running_gpu_util, gpu_util = simulator.calculate_real_gpu_usage()

        # If calculate the goodput and speedup in the whle process, use follow code
        # sum_goodput, avg_goodput, sum_speedup, avg_speedup = simulator.calculate_goodput_and_speedup()

        # simulator.metric_dict['clock'].append(simulator.clock)
        # simulator.metric_dict['sum_goodput'].append(sum_goodput)
        # simulator.metric_dict['avg_goodput'].append(avg_goodput)
        # simulator.metric_dict['sum_speedup'].append(sum_speedup)
        # simulator.metric_dict['avg_speedup'].append(avg_speedup)
        
        simulator.gpu_util_dict['clock'].append(simulator.clock)
        simulator.gpu_util_dict['real_gpu_use'].append(real_gpu_util)
        simulator.gpu_util_dict['real_running_gpu_use'].append(
            real_running_gpu_util)
        simulator.gpu_util_dict['gpu_use'].append(gpu_util)
        
        
        

        simulator.update_infer_pod_status()

        if not simulator.event_update():
            continue

        LOG.info("+++ Clock: %s, real_gpu_use: %s, gpu_use: %s +++",
                 simulator.clock, real_gpu_util, gpu_util)

        interval = simulator.clock - previous_clock
        LOG.info("previous: %s", previous_clock)
        LOG.info("clock: %s", simulator.clock)
        LOG.info("interval: %s", interval)

        simulator.step(interval, args.interval)
        LOG.info("infer pod status: %s", simulator.infer_pod_status)

        infer_pods = set()
        for app, pods in simulator.infer_pod_status.items():
            infer_pods = infer_pods.union(set(pods.keys()))

        # infer_pods.union
        LOG.info("infer pods: %s", infer_pods)
        LOG.info("---------------- SIMULATOR TIME: {} ----------------"
                 .format(simulator.current_time))
        LOG.info("Active jobs:")
        # for val in simulator.logs[-1]["submitted_jobs"]:
        for val in simulator.current_log['submitted_jobs']:
            if val["submission_time"] <= simulator.current_time and (val["completion_time"] is None or val['name'] in infer_pods):
                LOG.info("    {}:\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[placement {}] \t[rescale time] {} \t[start execute time {}] \t[evaluation finish time {}] \t[total delay time {}] \t[completion time {}]".format(
                    val["name"], val["epoch"], val["num_restarts"], val["batch_size"], val["placement"], val["rescale_time"], val["start_execute_time"], val["evaluate_finish_time"], val["delay_time"], val['completion_time']))
        used_gpus = sum(map(len, simulator.allocations.values()))
        LOG.info("GPU utilization: {}".format(used_gpus))
        LOG.info("Completed jobs:")
        jct_dict = simulator.get_jcts()
        LOG.info(jct_dict)

        LOG.info("Average JCT: %s", sum(jct_dict.values()) /
                 len(jct_dict) if jct_dict else 0)

        previous_clock = simulator.clock 

    if args.output:
        simulator.output_logs(args.output)
        simulator.output_gpu_util_info(args.gpu_output)
        simulator.output_metric_info(args.metric_output)

    result_jcts = simulator.get_jcts()
    return simulator.logs, result_jcts, simulator.gpu_util_dict, simulator.metric_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str,
                        default="workloads/short_workloads.csv")
    parser.add_argument("--policy", type=str, default="dp",
                        choices=["tiresias", "optimus", "pollux", "afs", "aryl", 'dp'])
    parser.add_argument("--min-nodes", type=int, default=16,
                        help="min number of nodes in the cluster")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="max number of nodes for cluster autoscaling")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--infer_priority", type=int, default=1,
                        help="infer job has higher priority than training job")
    parser.add_argument("--protect_time", type=int, default=30,
                        help="protect time for inference replicas")
    parser.add_argument("--interference", type=float, default=0.0,
                        help="job slowdown due to interference")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="number of GPUs per node")

    parser.add_argument("--ARYL", type=int, default=0,
                        help="whether aryl schedule")  

    parser.add_argument("--AISS", type=int, default=1,
                        help="whether aiss lifesycle")

    parser.add_argument("--AFE", type=int, default=0,
                        help="whether AFE optimize elastic")

    parser.add_argument("--INFER_SCHEDULER", type=int, default=1,
                        help="1 means using Pollux to schedule Inference tassks")

    parser.add_argument("--REPAIR_TRAIN", type=int, default=1,
                        help="1 means training task can't expand unless in interval")

    parser.add_argument("--protect_times", type=float, default=1.0,
                        help="1 means using Pollux to schedule Inference tasks")

    parser.add_argument("--random_allocate", type=int,
                        default=0, help="inference task random allocate")

    parser.add_argument("--log_file", type=int, default=0,
                        help="log out")

    parser.add_argument("--low-util", type=float,
                        help="low utility threshold")
    parser.add_argument("--high-util", type=float,
                        help="high utility threshold")
    parser.add_argument("--output", type=str,
                        help="path to output logs")
    parser.add_argument("--gpu_output", type=str,
                        help="path to output gpu usage info")
    parser.add_argument("--metric_output", type=str,
                        help="path to output metric info")

    args = parser.parse_args()

    AISS = args.AISS
    AFE = args.AFE
    INFER_SCHEDULER = args.INFER_SCHEDULER
    REPAIR_TRAIN = args.REPAIR_TRAIN
    RANDOM_ALLOCATE = args.random_allocate
    ARYL = args.ARYL
    PROTECT_TIMES = args.protect_times
    NUM_NODE = args.min_nodes

    LOG.info("REPAIR_TRAIN: %s", REPAIR_TRAIN)
    LOG.info("protect times: %s", PROTECT_TIMES)
    LOG.info("random allocate: %s", args.random_allocate)

    # exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    files = os.listdir(args.output)
    for file in files:
        os.remove(os.path.join(args.output, file))

    if args.log_file:
        log_file = args.output + '/simulator.log'
        if os.path.exists(log_file):
            os.remove(log_file)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        LOG.addHandler(fh)

    LOG.info("output: %s", args.output)

    # exit()
    if os.path.isdir(args.workload):
        if not os.path.exists(args.output):
            os.mkdir(args.output)
        assert args.output is not None and os.path.isdir(args.output)
        args_list = []
        for workload in glob.glob(args.workload + "/*.csv"):
            name = os.path.basename(workload)[:-4]
            args_list.append(copy.deepcopy(args))
            args_list[-1].workload = workload
            args_list[-1].output = args.output + "/" + name + ".json"
            args_list[-1].gpu_output = args.output + \
                "/" + name + "_gpu-info.json"
        with multiprocessing.Pool(processes=1) as pool:
            ret_list = pool.map(simulate, args_list)
        summary = {"jcts": {}, "avgs": {}}
        for args_item, (_, jct_dict, gpu_util_dict) in zip(args_list, ret_list):
            name = os.path.basename(args_item.workload)[:-4]
            summary["jcts"][name] = jct_dict
            summary["avgs"][name] = sum(jct_dict.values()) / len(jct_dict)
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])

        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
    else:
        LOG.info("single workload")
        # exit()
        args.gpu_output = args.output + '/gpu.log'
        args.metric_output = args.output + '/metric.log'
        # simulator_logs, result_jcts, simulator_gpu_util_dict = simulate(args)

        summary = {"jcts": {}, "avgs": {}}
        logs, jct_dict, gpu_util_dict, metric_dict = simulate(args)
        summary["jcts"] = jct_dict
        if len(jct_dict) == 0:
            summary["avgs"] = 0
        else:
            summary["avgs"] = sum(jct_dict.values()) / len(jct_dict)

        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)

    LOG.info("schedule over")
