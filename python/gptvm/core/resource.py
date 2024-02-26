from .device import *

import os, sys
from typing import List, Dict, Tuple, Optional
import json

import ray
from ray.util.state import list_actors

has_cuda = True
try:
    from cuda import cuda
except ImportError:
    has_cuda = False


class _NodeResource:
    def __init__(self, id=0):
        self.node_id = id
        self.devices: Dict[GVDeviceType, List[GVDevice]] = {}
        self.memory_map: Dict[GVDeviceType, int] = {}
        self.compute_cap: Dict[GVDevice, Dict[str, float]] = {}

        self._find_cpu()
        if has_cuda:
            self._find_gpu()

    def __str__(self):
        return f"NodeResource(node_id={self.node_id}, devices={self.devices}, memory_map={self.memory_map}, compute_cap={self.compute_cap})"

    def __repr__(self):
        return self.__str__()

    def _find_cpu(self):
        # get the cpu number of the node
        self.devices[GVDeviceType.GV_CPU] = [
            GVDevice([x, GVDeviceType.GV_CPU, self.node_id])
            for x in range(os.cpu_count())
        ]
        self.memory_map[GVDeviceType.GV_CPU] = os.sysconf("SC_PAGE_SIZE") * os.sysconf(
            "SC_PHYS_PAGES"
        )

    def _find_gpu(self):
        # get the gpu number of the node
        cuda.cuInit(0)
        ret, count = cuda.cuDeviceGetCount()
        if ret == cuda.CUresult.CUDA_SUCCESS:
            self.devices[GVDeviceType.GV_NV_GPU] = [
                GVDevice([x, GVDeviceType.GV_NV_GPU, self.node_id])
                for x in range(count)
            ]
            for i in range(count):
                # dev = cuda.cuDeviceGet(i)
                _, mem = cuda.cuDeviceTotalMem(i)
                self.memory_map[GVDeviceType.GV_NV_GPU] = mem
                self._populate_compute_cap(i)

    def _populate_compute_cap(self, device, device_type=GVDeviceType.GV_NV_GPU):
        if device_type == GVDeviceType.GV_CPU:
            return
        # get the current directory
        cwd = os.path.dirname(os.path.realpath(__file__))
        # the path to the compute capability json file
        cc_path = os.path.join(cwd, os.pardir, os.pardir, "config/nv_spec.json")
        log.debug(f"get GV_NV_GPU compute capability from {cc_path}")
        # read the compute capability json file
        with open(cc_path) as f:
            cc = json.load(f)
        # get the list of GV_NV_GPU compute capability
        gpu_caps = cc["NV_GPU"]
        # get the current gpu name
        _, gpu_name = cuda.cuDeviceGetName(32, device)
        gpu_name = gpu_name.decode("utf-8")
        # if gpu_name contains any of the keys, then we found the compute capability
        for cap in gpu_caps:
            for x in cap.keys():
                print(f"checking {x} in {gpu_name}: {x in gpu_name}")
                if x in gpu_name:
                    self.compute_cap[device] = cap[x]
                    return
        # if we reach here, we did not find the compute capability
        log.warning(f"Failed to find compute capability for {gpu_name}")

    def get_devices(self, resource_type: GVDeviceType):
        return self.devices[resource_type]

    def get_cpus(self):
        return self.get_devices(GVDeviceType.GV_CPU)

    def get_gpus(self):
        return self.get_devices(GVDeviceType.GV_NV_GPU)

    def get_memory(self, resource_type: GVDeviceType):
        return self.memory_map[resource_type]

    def get_compute_cap(self, device: int):
        return self.compute_cap[device]

    def get_node_resource(self):
        return self.devices, self.memory_map, self.compute_cap


# define the resource type for the cluster
class _ClusterResourceImpl:
    def __init__(self):
        self.__node_id = 0
        # map ray node id to internal node id
        self.node_map: Dict[str, int] = {}
        self.workers: Dict[str, ray.actor.ActorHandle] = {}
        self.pg_bundle = {"CPU": 1}

        nodes = ray.nodes()
        for node in nodes:
            if node["alive"]:
                ray_id = node["NodeID"]
                my_id = self.node_id
                self.node_map[ray_id] = my_id
                sched = ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                    node["NodeID"], False
                )
                self.workers[ray_id] = ray.remote(
                    num_cpus=1,
                    num_gpus=0,
                    scheduling_strategy=sched,
                    namespace="gptvm",
                    name=f"node_resource_{my_id}",
                )(_NodeResource).remote(my_id)

    def __str__(self):
        return f"_ClusterResourceImpl(nodes={self.__node_id}, node_map={self.node_map}, workers={self.workers})"

    def __repr__(self):
        return self.__str__()

    @property
    def node_id(self):
        self.__node_id += 1
        if self.__node_id > 255:
            log.error("Node id overflow")
        return self.__node_id

    def get_node_id(self, ray_id):
        return self.node_map[ray_id]

    def get_ray_id(self, node_id):
        for k, v in self.node_map.items():
            if v == node_id:
                return k
        return None

    def get_node_resource(self, node_id):
        return self.workers[self.get_ray_id(node_id)].get_node_resource.remote()

    def get_cluster_resources(self):
        return [
            ray.get(self.workers[k].get_node_resource.remote())
            for k in self.workers.keys()
        ]

    def cleanup(self):
        for k, v in self.workers.items():
            ray.kill(v)


# The real cluster resource which exported to user
class ClusterResource:
    _cr_ref = ray.remote(
        name="cluster_resource",
        namespace="gptvm",
        lifetime="detached",
        get_if_exists=True,
    )(_ClusterResourceImpl).remote()

    @staticmethod
    def get_node_id():
        # first get the current ray id
        ray_id = ray.get_runtime_context().node_id
        # then get the internal node id
        return ray.get(ClusterResource._cr_ref.get_node_id.remote(ray_id))

    @staticmethod
    def get_ray_id(node_id):
        return ray.get(ClusterResource._cr_ref.get_ray_id.remote(node_id))

    @staticmethod
    def get_node_resource(node_id):
        return ray.get(ClusterResource._cr_ref.get_node_resource.remote(node_id))

    @staticmethod
    def get_cluster_resources():
        return ray.get(ClusterResource._cr_ref.get_cluster_resources.remote())

    @staticmethod
    def cleanup():
        ClusterResource._cr_ref.cleanup.remote()
        ray.kill(ClusterResource._cr_ref)
