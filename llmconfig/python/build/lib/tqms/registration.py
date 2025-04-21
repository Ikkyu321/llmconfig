import sched
import uuid

from readerwriterlock import rwlock
import time
import threading

from server_info import *
from etcdcli import EtcdClient

lease_timeout = 5
is_lease = False
lease_id = 10011

rwmutex = rwlock.RWLockFair()


class Register:
    def __init__(self, etcdClient: EtcdClient):
        self.etcd_client = etcdClient
        self.schedule = sched.scheduler(time.time, time.sleep)
        self.lease = self.etcd_client.get_lease(lease_timeout, lease_id)

    def set_all_workers(self):
        global g_worker_list
        write_lock = rwmutex.gen_wlock()
        write_lock.acquire()
        ids = self.etcd_client.get_all_workers_id()
        g_worker_list = ids
        write_lock.release()

    def get_worker_id(self):
        global worker_id
        read_lock = rwmutex.gen_rlock()
        read_lock.acquire()
        workerId = worker_id
        read_lock.release()
        return workerId

    def get_worker_nums(self):
        global g_worker_list
        read_lock = rwmutex.gen_rlock()
        read_lock.acquire()
        length = len(g_worker_list)
        read_lock.release()
        return length

    def watch_callback(self, event):
        self.set_all_workers()

    def find_all_workers(self):
        self.set_all_workers()
        self.etcd_client.add_callback_with_servers_node(self.watch_callback())

    def schedule(self):
        t = threading.Thread(target=self.schedule_job())
        t.setDaemon(True)
        t.start()

    def schedule_job(self):
        inc = 3
        self.schedule.enter(0, 0, self.lease_worker, inc)
        self.schedule.run()

    def lease_worker(self, inc: int):
        global is_lease
        global worker_id
        if is_lease:
            self.lease.refresh()
        else:
            is_lease = True
            worker_id = uuid.uuid4()
            self.etcd_client.add_worker_info_with_lease(worker_id, "", self.lease)
        self.schedule.enter(inc, 0, self.lease_worker, inc)
