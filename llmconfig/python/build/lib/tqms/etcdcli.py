import enum
from typing import List, Union
import json
import etcd3
from server_info import *

task_prefix = "/tqms/batchTask"
server_center = "/tqms/servers"
chat_history = "/tqms/chatHistory"


class TaskStatus(enum.Enum):
    """Status of a sequence."""
    WAITING = enum.auto()
    RUNNING = enum.auto()
    SWAPPED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH_CAPPED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_IGNORED = enum.auto()

    @staticmethod
    def is_finished(status: "TaskStatus") -> bool:
        return status in [
            TaskStatus.FINISHED_STOPPED,
            TaskStatus.FINISHED_LENGTH_CAPPED,
            TaskStatus.FINISHED_ABORTED,
            TaskStatus.FINISHED_IGNORED,
        ]

    @staticmethod
    def get_finished_reason(status: "TaskStatus") -> Union[str, None]:
        if status == TaskStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == TaskStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == TaskStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == TaskStatus.FINISHED_IGNORED:
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


class SubTask:
    def __init__(self, workerId: str, imageList: List[str]):
        self.worker_id = workerId
        self.input = imageList


class Task:
    def __init__(self, uuid: str, imageList: List[str]):
        self.sub_inputs: List[SubTask] = []
        self.output: List[str] = []
        self.uuid = uuid
        self.input = imageList
        self.size = len(imageList)

    def split_task(self):
        input_length = len(self.input)
        nums_per_worker = int(input_length / num_workers)
        additional = input_length % num_workers
        for id, worker_id in enumerate(g_worker_list):
            if id == num_workers - 1:
                self.sub_inputs.append(SubTask(worker_id, imageList=self.input[id * nums_per_worker: (id + 1) * nums_per_worker + additional]))
                continue
            self.sub_inputs.append(SubTask(worker_id, imageList=self.input[id * nums_per_worker: (id + 1) * nums_per_worker]))

    def collect_task(self, results: List[str]):
        self.output.extend(results)
        if len(self.output) == self.size:
            return True
        else:
            return False


class EtcdConfig(object):
    def __init__(self, hosts: str):
        index = hosts.rfind(":")
        self.trickHost = "ipv4:" + hosts[:index]
        self.trickPort = int(hosts[index + 1:])
        print(
            f'init etcd hosts is \"{hosts}\", trick hosts is \"{self.trickHost}\", trick port is \"{self.trickPort}\"')


class EtcdClient:

    def __init__(self, etcdCfg: EtcdConfig):
        self.client = etcd3.client(etcdCfg.trickHost, etcdCfg.trickPort)
        self.lease = None
        self.lease_chat = self.client.lease(3600 * 24 * 30)

    def add_task_data(self, task: Task, watchCallback):
        for subInput in task.sub_inputs:
            self.client.put(task_prefix + "/" + task.uuid + "/input/" + subInput.worker_id, json.dumps(subInput.input))
            self.add_watcher_on_task_output(task, subInput.worker_id, watchCallback)

    def put_results(self, taskId, results):
        self.client.put(task_prefix + "/" + taskId + "/output/" + worker_id, results)

    def add_watcher_on_task_output(self, task: Task, taskWorkerId, watchCallback):
        self.client.put(task_prefix + "/" + task.uuid + "/output/" + taskWorkerId, "")
        self.client.add_watch_callback(task_prefix + "/" + task.uuid + "/output/" + taskWorkerId, watchCallback)

    def delete_task_data(self, taskId: str):
        self.client.delete_prefix(task_prefix + "/" + taskId)

    def add_watch_on_task_root(self, watchCallback):
        self.client.add_watch_prefix_callback(task_prefix, watchCallback)

    def get_lease(self, timeout: int, leaseId: int):
        if self.lease != None:
            self.lease = self.client.lease(timeout, leaseId)
        return self.lease

    def get_all_workers_id(self):
        kvs = self.client.get_prefix(server_center + "/")
        results = []
        for key, value in kvs:
            results.append(value.key.decode('utf8').split("/")[-1])
        return results

    def get_lease_info(self, leaseId: int):
        return self.client.get_lease_info(leaseId)

    def add_worker_info_with_lease(self, key, value, leaseId):
        self.client.put(server_center + "/" + key, value, leaseId)

    def add_callback_with_servers_node(self, callback):
        self.client.add_watch_prefix_callback(server_center, callback)

    def put_kv_with_timeout(self, key, value):
        self.client.put(key, value, self.lease_chat)

    def get_by_key(self, key):
        value = self.client.get(key)
        if value[0] is None and value[1] is None:
            return None
        return value[0].decode("utf8")

    def get_by_key_prefix(self, keyPrefix):
        kvs = self.client.get_prefix(keyPrefix)
        results: list[tuple[str, str]] = []
        for key, value in kvs:
            rk = value.key.decode("utf8").split("/")[-1]
            rv = value.value.decode("utf8")
            result = (rk, rv)
            results.append(result)
        return results

    def delete_by_key(self, key):
        self.client.delete(key)

    def delete_by_key_prefix(self, keyPrefix):
        self.client.delete_prefix(keyPrefix)
