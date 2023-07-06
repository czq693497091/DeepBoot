# from typing_extensions import runtime


class JobInfo(object):
    def __init__(self, job,resources, speedup_fn, creation_timestamp, attained_service,
                 min_replicas, max_replicas, preemptible=True, inference=False, duration=-1,run_time=0):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            min_replicas (int): Minimum number of replicas job's guaranteed.
            max_replicas (int): Maximum number of replicas. Maximum should be
                                greater or equal to Minimum
            preemptible (bool): Is the job preemptible?
        """
        assert max_replicas > 0
        assert max_replicas >= min_replicas
        self.resources = resources
        self.speedup_fn = speedup_fn
        self.creation_timestamp = creation_timestamp
        self.attained_service = attained_service
        self.max_replicas = max_replicas
        self.min_replicas = min_replicas
        self.preemptible = preemptible
        self.run_time = run_time
        self.job = job 
        # self.inference = inference
        self.duration = duration

        self.application = job.application
        self.name = job.name
        self.inference = job.inference
        # self.protect_time = job.protect_time


    
        
        

class NodeInfo(object):
    def __init__(self, resources, preemptible):
        """
        Args:
            resources (dict): Available resources (eg. GPUs) on this node.
            preemptible (bool): Whether this node is pre-emptible.
        """
        self.resources = resources
        self.preemptible = preemptible