import asyncio
import json
import os
import queue
import traceback
import urllib
import uuid
from functools import partial
from threading import Lock

import boto3
import requests

from etcdcli import EtcdClient, EtcdConfig, Task, worker_id
#import etcd3
from handler import Handler

from registration import Register
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from models import *
from llmconfig.logger import *

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')


logging = init_logger("tqms_logger")

def process_async_reqs(taskId: str, results, requestBody: AsyncRequestBody):
    logging.info(f'process task {taskId}, task size {len(results)}')
    # send all subtask to etcd queue
    errMsg = None
    try:
        tmp_filename = f'caches/image_task/{taskId}.txt'
        with open(tmp_filename, 'w') as fh:
            for r in results:
                rstr = json.dumps(r)
                fh.write(f'{rstr}\n')
        logging.info(f'results of tasks {taskId} saved to {tmp_filename}')

        # write back to oss
        client = boto3.client('s3',
                              aws_access_key_id=requestBody.outputOss.accessId,
                              aws_secret_access_key=requestBody.outputOss.accessSecret,
                              endpoint_url=requestBody.outputOss.endpoint,
                              verify=False)
        client.upload_file(tmp_filename, requestBody.outputOss.bucket, requestBody.outputOss.key)
        logging.info(f'resutls of tasks {taskId} uploaded to {requestBody.outputOss.ossPath}')
    except Exception as e:
        errMsg = f'process async task {taskId} failed: {e}'
        logging.info(errMsg)
    finally:
        logging.info(f'callback of tasks {taskId} to {requestBody.callbackUrl}')
        resp = None
        try:
            if errMsg is not None:
                resp = requests.post(requestBody.callbackUrl,
                                     json={'msg': 'error', 'data': {'taskId': taskId, 'detail': errMsg}})
            else:
                resp = requests.post(requestBody.callbackUrl, json={'msg': 'success', 'data': {'taskId': taskId}})
        except Exception as e:
            logging.info(f'callback of tasks {taskId} to {requestBody.callbackUrl} error, {e}')
        if resp is not None and resp.status_code != 200:
            logging.info(f'callback of tasks {taskId} to {requestBody.callbackUrl} failed: {resp.text}')
        os.remove(tmp_filename)


async def process_llm_async_reqs(taskId: str, results_queue: queue.PriorityQueue, requestBody: AsyncLLMRequestBody):
    if requestBody.inputs is not None:
        logging.info(f'process task {taskId}, task size {len(requestBody.inputs)}')
    errMsg = None

    try:
        tmp_filename = f'caches/llm_task/{taskId}.txt'
        with open(tmp_filename, 'w') as fh:
            while not results_queue.empty():
                seq, res = results_queue.get()
                rstr = json.dumps(res, ensure_ascii=False)
                fh.write(f'{rstr}\n')

        logging.info(f'results of tasks {taskId} saved to {tmp_filename}')

        # write back to oss
        client = boto3.client('s3',
                              aws_access_key_id=requestBody.outputOss.accessId,
                              aws_secret_access_key=requestBody.outputOss.accessSecret,
                              endpoint_url=requestBody.outputOss.endpoint,
                              verify=False)

        # @sync_to_async
        # def upload_file(file, bucket, ossKey):
        #     client.upload_file(file, bucket, ossKey)
        #
        # async def async_upload_file(file, bucket, ossKey):
        #     await upload_file(file, bucket, ossKey)
        #
        # await async_upload_file(tmp_filename, requestBody.outputOss.bucket, requestBody.outputOss.key)

        async def async_upload_file(file, bucket, ossKey):
            loop = asyncio.get_event_loop()
            put_object_partial = partial(
                client.upload_file,
                file, bucket, ossKey)
            await loop.run_in_executor(None, put_object_partial)
        await async_upload_file(tmp_filename, requestBody.outputOss.bucket, requestBody.outputOss.key)
        logging.info(f'resutls of tasks {taskId} uploaded to {requestBody.outputOss.ossPath}')
    except Exception as e:
        errMsg = f'process async task {taskId} failed: {e}'
        logging.info(errMsg)
    finally:
        logging.info(f'callback of tasks {taskId} to {requestBody.callbackUrl}')
        resp = None
        try:
            if errMsg is not None:
                resp = requests.post(requestBody.callbackUrl,
                                     json={'msg': 'error', 'data': {'taskId': taskId, 'detail': errMsg}})
            else:
                resp = requests.post(requestBody.callbackUrl, json={'msg': 'success', 'data': {'taskId': taskId}})
        except Exception as e:
            logging.info(f'callback of tasks {taskId} to {requestBody.callbackUrl} error, {e}')
        if resp is not None and resp.status_code != 200:
            logging.info(f'callback of tasks {taskId} to {requestBody.callbackUrl} failed: {resp.text}')
        os.remove(tmp_filename)


class ProgressCenter:
    def __init__(self):
        self.progress_cache: Dict[str, Progress] = {}

    def get_progress(self, taskId: str):
        if self.progress_cache.get(taskId, None) is not None:
            progress = self.progress_cache.get(taskId)
        else:
            raise HTTPException(status_code=404, detail="task id not exists")
        progress.caculate()
        # self.progress_cache.pop()
        return progress

    def delete_progress(self, taskId: str):
        if self.progress_cache.get(taskId, None) is not None:
            progress = self.progress_cache.pop(taskId)
            progress.stop_event.set()
        else:
            raise HTTPException(status_code=404, detail="task id not exists")
        return progress

    def get_certain_progress(self, taskId: str):
        return self.progress_cache.get(taskId)

    def add_progress(self, taskId: str):
        progress = Progress()
        self.progress_cache[taskId] = progress

    def update_progress(self, taskId: str, total: int, processed: int):
        progress = self.progress_cache.get(taskId)
        progress.total = total
        progress.processed = processed

    def update_progress_total(self, taskId: str, total: int):
        progress = self.progress_cache.get(taskId)
        progress.total = total

    def update_progress_proccessed(self, taskId: str, processed: int):
        progress = self.progress_cache.get(taskId)
        progress.processed = processed

    def update_processed_reason(self, taskId: str, processReason: str):
        progress = self.progress_cache.get(taskId)
        progress.reason = processReason


class Progress:
    def __init__(self):
        self.total = -1
        self.processed = 0
        self.finished = False
        self.progress = 0.0
        self.reason = "success"
        self.stop_event = asyncio.Event()



    def caculate(self):
        if self.total == self.processed:
            self.finished = True
        if self.total != -1 and self.total != 0:
            self.progress = self.processed * 1.0 / self.total


class LLMTaskServer:
    def __init__(self, progressCenter: ProgressCenter):
        self.process_center = progressCenter
        self.semaphore = asyncio.Semaphore(PARALLEL_SIZE)

    async def process(self, taskId: str, requestBody: AsyncLLMRequestBody):
        self.process_center.add_progress(taskId)
        stop_event = self.process_center.get_progress(taskId).stop_event
        s3_path = urllib.parse.urlparse(requestBody.cloudFile)
        if s3_path.scheme == "http" or s3_path.scheme == "https":
            try:
                response = requests.get(requestBody.cloudFile, stream=True)
                if response.status_code == 200:
                    lines = response.iter_lines()
                else:
                    self.process_center.update_processed_reason(taskId,
                                                                f"http file download failed, message is {response}")
                    raise RuntimeError(f'http file download failed, message is {response}')
            except Exception as e:
                self.process_center.update_processed_reason(taskId,
                                                            f"http file download failed, message is {e}")
                traceback.print_exc()
                raise RuntimeError(f'http file download failed, message is {e}')
        elif s3_path.scheme == "oss" or s3_path.scheme == "s3":
            try:
                bucket_name = s3_path.netloc
                object_key = s3_path.path.lstrip('/')
                client = boto3.client('s3',
                                      aws_access_key_id=requestBody.outputOss.accessId,
                                      aws_secret_access_key=requestBody.outputOss.accessSecret,
                                      endpoint_url=requestBody.outputOss.endpoint,
                                      verify=False)

                response = client.get_object(Bucket=bucket_name, Key=object_key)

                lines = response['Body'].iter_lines()
            except Exception as e:
                self.process_center.update_processed_reason(taskId, f"s3 client init failed, message is {e}")
                traceback.print_exc()
                raise RuntimeError(f's3 client init failed, message is {e}')
        else:
            raise RuntimeError(f"cloud file obtain failed, check file type, only support http/https/s3/oss")

        concurrent_request_list_queue = asyncio.Queue()
        concurrent_result_queue = asyncio.PriorityQueue()
        results_priority_queue = queue.PriorityQueue()

        prefix = ""
        suffix = ""
        if requestBody.promptTemplate is not None:
            if requestBody.promptTemplate.get("prompt_input", None):
                prompt_input = requestBody.promptTemplate.get("prompt_input")
                if not (prompt_input == "{input}"):
                    start_index = prompt_input.find("{input}")
                    prefix = prompt_input[:start_index]
                    suffix = prompt_input[start_index + len("{input}"):]


        data_count = 0
        try:
            for line in lines:
                decoded_line = line.decode('utf-8')
                json_line = json.loads(decoded_line)
                prompt = json_line["question"]
                choices = json_line.get("choices", None)
                images = json_line.get("images", None)
                request = LLMRequestBody()
                if choices is not None:
                    request.choices = choices
                if images is not None:
                    request.images = images
                request.input = prompt
                if requestBody.temperature is not None:
                    request.params["temperature"] = requestBody.temperature
                if requestBody.repetition_penalty is not None:
                    request.params["repetition_penalty"] = requestBody.repetition_penalty
                if requestBody.frequency_penalty is not None:
                    request.params["frequency_penalty"] = requestBody.frequency_penalty
                if requestBody.presence_penalty is not None:
                    request.params["presence_penalty"] = requestBody.presence_penalty
                if requestBody.top_p is not None:
                    request.params["top_p"] = requestBody.top_p
                if requestBody.top_k is not None:
                    request.params["top_k"] = requestBody.top_k

                if requestBody.maxLength is not None:
                    request.maxLength = requestBody.maxLength
                if requestBody.summaryLlmUrl is not None:
                    request.summaryLlmUrl = requestBody.summaryLlmUrl
                if requestBody.requestId is not None:
                    request.requestId = requestBody.requestId + "#" + str(uuid.uuid4())
                if requestBody.promptTemplate is not None:
                    request.promptTemplate = requestBody.promptTemplate
                    wrapper_input = request.promptTemplate["prompt_input"].format(input=request.input)
                    request.input = wrapper_input

                request.type = "Async"
                request.params["stream"] = False
                data = {"data": request}
                await concurrent_request_list_queue.put((data_count, data))
                data_count = data_count + 1
        except Exception as e:
            self.process_center.update_processed_reason(taskId, f"input file format error, message is {e}")
            traceback.print_exc()
            raise RuntimeError(f'input file format error, message is {e}')
        self.process_center.update_progress_total(taskId, data_count)

        async def post_to_vllm(seq: int, data):
            try:
                text = await handler(data, {"stream": False})
            except Exception as e:
                traceback.print_exc()
                logging.error(f'inference error, {e}, request body is {data["data"].input}')
                raise RuntimeError("inference data error, throw exception")
            return (seq, text)

        async def parallel_process(queue, stop_event:asyncio.Event):
            async with self.semaphore:
                while not queue.empty() and not stop_event.is_set():
                    task = await queue.get()
                    try:
                        seq, text = await post_to_vllm(task[0], task[1])

                        if text["output"] == "":
                            if task[1]["data"].retryTimes < 3:
                                task[1]["data"].retryTimes = task[1]["data"].retryTimes + 1
                                logging.info(
                                    f'inference result with some errors happen, output is empty, retry {task[1]["data"].retryTimes}')
                                await queue.put(task)
                                continue
                            else:
                                logging.error("after 3 times retry, result still empty, write empty")
                                text = {}
                                text["input"] = data["data"].input
                                text["output"] = "error"
                                text["choicesProb"] = []
                                seq = task[0]
                    except Exception as e:
                        logging.info(f"Task {task} failed: {e}")
                        # 将任务重新加入队列
                        if task[1]["data"].retryTimes < 3:
                            task[1]["data"].retryTimes = task[1]["data"].retryTimes + 1
                            await queue.put(task)
                            continue
                        else:
                            logging.error("after 3 times retry, result still have exption, write empty")
                            text = {}
                            text["input"] = data["data"].input
                            text["output"] = "error"
                            text["choicesProb"] = []
                            seq = task[0]
                    await concurrent_result_queue.put((seq, text))
                logging.info(f"Task {taskId} is interrupted by handle, stop coroutine")

        async def await_result(event: asyncio.Event):
            await asyncio.gather(
                *[parallel_process(concurrent_request_list_queue, event) for _ in range(PARALLEL_SIZE)])  # 启动5个worker协程

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            logging.info('Async event loop already running. Adding coroutine to the event loop.')
            tsk = loop.create_task(await_result(stop_event))
            tsk.add_done_callback(
                lambda t: logging.info(f'Task done with result={t.result()}  << return val '))
        else:
            logging.info('Starting new event loop')
            asyncio.run(await_result(stop_event))

        async def get_request_result(stop_event: asyncio.Event):
            parallel_fail_count = 0
            parallel_total_count = 0
            for _ in range(data_count):
                if stop_event.is_set():
                    logging.info(f"Task {taskId} is interrupted by handle, break result gather")
                    break
                seq, text = await concurrent_result_queue.get()
                if seq % 32 == 0:
                    self.process_center.update_progress_proccessed(taskId, seq)
                dct = {}
                for k, v in text.items():
                    if k == "input":
                        raw_input = text["input"]
                        raw_input = raw_input.lstrip(prefix)
                        raw_input = raw_input.rstrip(suffix)
                        dct["question"] = raw_input
                    elif k == "output":
                        dct["answer"] = text["output"]
                    elif k == "choicesProb":
                        dct["choicesProb"] = text["choicesProb"]
                    else:
                        dct[k] = v
                results_priority_queue.put((seq, dct))
                if text["output"] == "":
                    parallel_fail_count = parallel_fail_count + 1
                    logging.debug(f"fail data count {parallel_fail_count}>>>>>>>>>>>>total {parallel_total_count}>>>>>>>>")
                parallel_total_count = parallel_total_count + 1

        await get_request_result(stop_event)
        if stop_event.is_set():
            logging.info(f"task {taskId} is interrupted by client, finally break data upload")
            return
        await process_llm_async_reqs(taskId, results_priority_queue, requestBody)
        self.process_center.update_progress_proccessed(taskId, data_count)


class ImageTaskServer:
    def __init__(self, etcdClient: EtcdClient, pregressCenter: ProgressCenter):
        self.task_cache: Dict[str, (Task, AsyncRequestBody)] = {}
        self.progress_center = pregressCenter
        self.etcd_client = etcdClient

    def buildTask(self, taskId, taskData: List[str]):
        task = Task(taskId, taskData)
        return task

    def addTask(self, task: Task, reqs: AsyncRequestBody):
        self.task_cache[task.uuid] = (task, reqs)
        self.etcd_client.add_task_data(task, self.subOutputCallback)
        self.progress_center.add_progress(task.uuid)

    def deleteTask(self, taskId):
        del self.task_cache[taskId]
        self.etcd_client.delete_task_data(taskId)

    def subOutputCallback(self, eventResponse):
        for event in eventResponse.events:
            if type(event) == etcd3.events.PutEvent:
                key = event.key.decode('utf8')
                value = event.value.decode('utf8')
                taskId, typ, workerId = self.getOutputTaskIdAndWorkerId(key)
                if typ != "output":
                    return None
                listValue = json.loads(value)
                logging.info(listValue)
                if self.task_cache[taskId][0].collect_task(listValue):
                    # TODO 任务结束
                    task = self.task_cache[taskId][0]
                    results = task.output
                    reqs = self.task_cache[taskId][1]
                    process_async_reqs(taskId, results, reqs)
                    self.progress_center.update_progress_proccessed(taskId, len(results))
                    self.progress_center.update_progress_total(taskId, len(results))
                else:
                    task = self.task_cache[taskId][0]
                    results = task.output
                    self.progress_center.update_progress_proccessed(taskId,
                                                                    self.progress_center.get_certain_progress(
                                                                        taskId).processed + len(results))

    def taskCallback(self, eventReponse):
        for event in eventReponse.events:
            if type(event) == etcd3.events.PutEvent:
                key = event.key.decode('utf8')
                value = event.value.decode('utf8')
                task_id, typ, worker_id = self.getOutputTaskIdAndWorkerId(key)
                ## get task from etcd
                if typ == "input" and worker_id == worker_id:
                    data_list = json.loads(value)
                    results = handler(data_list, {})
                    self.etcd_client.put_results(task_id, json.dumps(results))

    def getOutputTaskIdAndWorkerId(self, etcdPath):
        seq = etcdPath.split("/")
        return seq[3], seq[4], seq[5]


class LLMHistory():
    def __init__(self, etcdClient: EtcdClient):
        self.etcd_client = etcdClient
        self.chat_history_prefix = "/llmconfig/chatHistory"

    def get_all_history(self, userId):
        histories = self.etcd_client.get_by_key_prefix(self.chat_history_prefix + "/" + userId)
        results: dict = {}
        for history in histories:
            results[history[0]] = json.loads(history[1], object_hook=list[RoundChat])
        return results

    def get_history_by_session(self, userId, sessionId) -> List[RoundChat]:
        data = self.etcd_client.get_by_key(self.chat_history_prefix + "/" + userId + "/" + sessionId)
        if data is not None:
            return RoundChatDecoder().decode(data)
        else:
            return []

    def delete_history_by_session(self, userId, sessionId):
        self.etcd_client.delete_by_key(self.chat_history_prefix + "/" + userId + "/" + sessionId)

    def put_one_round_chat(self, userId, sessionId, roundChat):
        chat_history_body = self.get_history_by_session(userId, sessionId)
        if len(chat_history_body) > default_round:
            chat_history_body = chat_history_body[1:]
        chat_history_body.append(roundChat)
        self.etcd_client.put_kv_with_timeout(self.chat_history_prefix + "/" + userId + "/" + sessionId,
                                             json.dumps(chat_history_body, indent=4, cls=JsonEncoder))

    def update_round_chat(self, userId, sessionId, listRoundChat):
        logging.info(json.dumps(listRoundChat, indent=4, cls=JsonEncoder))
        self.etcd_client.put_kv_with_timeout(self.chat_history_prefix + "/" + userId + "/" + sessionId,
                                             json.dumps(listRoundChat, indent=4, cls=JsonEncoder))

    def delete_all_history(self, userId):
        self.etcd_client.delete_by_key_prefix(self.chat_history_prefix + "/" + userId)


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        websocket.close()

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)


class PromptTemplate:
    def __init__(self, prompt_input: str, prompt_history: str):
        self.prompt_input = prompt_input
        self.history = prompt_history


class PromptTemplateServer:
    def __init__(self, prompt_file_path: str):
        self.lock = Lock()
        self.prompt_file_path = prompt_file_path
        print("prompt_file_path:", prompt_file_path)
        self.prompt_templates: Dict[str, dict] = {}
        self.template_params: Dict[str, dict] = {}

        if self.prompt_file_path is not None:
            self.prompt_templates = self.load_consistence_data()

    def load_consistence_data(self) -> Dict[str, dict]:
        try:
            if not os.path.exists(self.prompt_file_path):
                print(f"File '{self.prompt_file_path}' not exists")
                data = {
                        "qwen_default": {
                            "templateName":"qwen_default",
                            "systemPrompt": "You are a helpful assistant.",
                            "query": "{input}",
                            "chatHistory": {
                                "historyInput": "{input}",
                                "historyOutput": "{output}"
                            }
                        },
                        "geogpt": {
                            "templateName": "geogpt",
                            "systemPrompt": "You are a helpful assistant named GeoGPT. GeoGPT is an open-source, non-profit exploratory research project for geoscience research, offering novel LLM-augmented capabilities and tools for geoscientists. Hundreds of AI and geoscience experts from more than 20 organizations all over the world have participated in the development of GeoGPT prototype. GeoGPT utilizes exclusively open-access training data, with no private data involved. If you do not have sufficient information or certainty to answer correctly, respond with 'Sorry, I need more details to provide an answer', and explains the reason step by step.",
                            "query": "{input}",
                            "chatHistory": {
                                "historyInput": "{input}",
                                "historyOutput": "{output}"
                            }
                        },
                        "geogpt_customized": {
                            "templateName": "geogpt_customized",
                            "systemPrompt": "You are a helpful assistant.",
                            "query": "{input}",
                            "chatHistory": {
                                "historyInput": "{input}",
                                "historyOutput": "{output}"
                            }
                        },
                        "geogpt_doc_extract": {
                            "templateName": "geogpt_doc_extract",
                            "systemPrompt": "You are a helpful assistant.",
                            "query": "{input}\n The answer is ",
                            "chatHistory": {
                                "historyInput": "{input}",
                                "historyOutput": "{output}"
                            }
                        }
                    }
                with open(self.prompt_file_path, "w") as file:
                    json.dump(data, file)
                    print(f"File '{self.prompt_file_path}' created and prompt data written.")
        except FileNotFoundError:
            print(f'prompt local file {self.prompt_file_path} not found')
            self.prompt_file_path = "llm_template.json"

        try:
            with open(self.prompt_file_path, "r") as file:
                content = file.read()
                return json.loads(content)
        except Exception:
            raise RuntimeError(f'json file {self.prompt_file_path} loads error, check format')

    def write_consistence_data(self, data) -> bool:
        try:
            with open(self.prompt_file_path, "w") as file:
                json.dump(data, file, indent=4)
            return True
        except Exception:
            raise RuntimeError(f'json file {self.prompt_file_path} dumps error, check format')

    def get_prompt_template_by_name(self, name: str) -> dict:
        self.lock.acquire()
        try:
            template_template_body = self.prompt_templates.get(name, None)
            if template_template_body is not None:
                return template_template_body
            else:
                raise RuntimeError(f'template {name} not exists')
        finally:
            self.lock.release()

    def get_all_prompt_template(self) -> Dict[str, dict]:
        self.lock.acquire()
        try:
            return self.prompt_templates
        finally:
            self.lock.release()

    def delete_prompt_template_by_name(self, name: str) -> bool:
        self.lock.acquire()

        try:
            local_prompt_templates = self.load_consistence_data()
            del_value = local_prompt_templates.pop(name, None)
            if del_value is None:
                raise RuntimeError(f'to be delete data {name} not exist')
            else:
                self.prompt_templates.pop(name)
                self.write_consistence_data(local_prompt_templates)
                return True
        finally:
            self.lock.release()

    def add_new_prompt_template(self, name: str, prompt_template_body: Dict) -> bool:
        self.lock.acquire()
        try:
            if self.prompt_templates.get(name, None) is None:

                local_prompt_templates = self.load_consistence_data()
                local_prompt_templates[name] = prompt_template_body

                self.write_consistence_data(local_prompt_templates)
                self.prompt_templates[name] = prompt_template_body
                return True
            else:
                raise RuntimeError(f'prompt template {name} exists')
        finally:
            self.lock.release()

    def add_template_params(self, name:str, params:Dict) -> bool:
        self.lock.acquire()
        try:
            if params is None:
                raise RuntimeError(f"prompt template params {params} is empty")
            
            # add new template params
            if self.template_params.get(name, None) is None:    
                self.template_params[name] = params    
            # update template params
            else:                 
                self.template_params[name] = params
            
            return True
        finally:
            self.lock.release()    

    def get_prompt_template_params_by_name(self, name:str) -> Dict:
        
        cur_params = self.template_params.get(name, None)    
        return cur_params



# setup_logging()
logging = init_logger("llmconfig_logger")

handler = Handler()
logging.info(handler)

progressCenter = ProgressCenter()
logging.info(progressCenter)

# etcdHosts = handler.read_meta_field("etcdHosts")
# etcdClient = EtcdClient(EtcdConfig(etcdHosts))
# logging.info(etcdClient)

# register = Register(etcdClient)
# logging.info(register)

# taskServer = ImageTaskServer(etcdClient, pregressCenter=progressCenter)
# taskServer.etcd_client.add_watch_on_task_root(taskServer.taskCallback)
# logging.info(taskServer)

llmTaskServer = LLMTaskServer(progressCenter=progressCenter)
logging.info(llmTaskServer)

# llmHistory = LLMHistory(etcdClient)
# logging.info(llmHistory)

manager = ConnectionManager()
logging.info(manager)

promptServer = PromptTemplateServer(WORK_DIR+"/llm_template.json")
logging.info(promptServer)
