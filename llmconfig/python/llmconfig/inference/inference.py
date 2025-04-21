import json
from llmconfig.logger import init_logger
import time
import numpy as np
import requests
import httpx
import os
from fastapi.encoders import jsonable_encoder
from llmconfig.models import *


logging = init_logger("llmconfig_logger")


class ConfigClass:

    def __init__(self):
        self.meta = self.load_meta()

    def load_meta(self):
        if not os.path.exists('meta.json'):
            raise IOError('meta.json not found')

        with open('meta.json', 'r') as fh:
            return json.loads(fh.read())

    def get_url(self, algoType):
        try:
            url = os.environ["LLM_INFERENCE_URL"]
        except Exception as e:
            url = None
        if url is None or url == "":
            url = self.meta.get(algoType, None)

        logging.info(f"llm inference url is {url}")
        return url


class ViewData(ConfigClass):

    def __init__(self):
        super().__init__()
        self.url = super().get_url("imageInferenceUrl")

    def post_for_object(self, data):
        response = requests.post(self.url, json=data)
        if response.status_code == 200:
            result = json.loads(response.text)
            if result['code'] == "200":
                my_list = result['data']
                return list(map(lambda x: np.array(x), my_list))
            else:
                print(f'post error, error is {result["message"]}, details is {result["detail"]}')
        else:
            raise RuntimeError("post error")

class ViewLLM(ConfigClass):

    def __init__(self):
        super().__init__()
        self.url = super().get_url("llmInferenceUrl")
        self.stream_url = super().get_url("llmInferenceUrl")
        self.async_client = httpx.AsyncClient()

    async def post_for_object(self, data, reqUrl):
        params = jsonable_encoder(data)

        try:
            response = await self.async_client.post(reqUrl, json=params, timeout=300)
        except httpx.ReadTimeout:
            raise RuntimeError(f"connection error, error is read timeout in 300s")
        except Exception as e:
            raise RuntimeError(f"connection error, error is {e}")

        if response.status_code == 200:
            result = json.loads(response.text)    
            return result          
        else:
            if response.text:
                raise RuntimeError((f'post error, error is {response.text}'))
            else:
                raise RuntimeError((f'post error, error is {response}'))

    def post_for_object_stream(self, data, reqUrl):
        params = jsonable_encoder(data)

        try:
            response = requests.post(reqUrl, json=params, stream=True)
        except Exception as e:
            raise RuntimeError(f"connection error, error is {e}")
        if response.status_code == 200:
            #return response.iter_content(chunk_size=1024)
            return response
        else:
            if response.text:
                raise RuntimeError((f'post error, error is {json.loads(response.text)}'))
            else:
                raise RuntimeError((f'post error, error is {response}'))

    def get_for_object(self, data, reqUrl):
        params = jsonable_encoder(data.query_params)

        try:
            response = requests.get(reqUrl, json=params, timeout=300)
        except Exception as e:
            raise RuntimeError(f"connection error, error is {e}")
        
        if response.status_code == 200: 
            result = json.loads(response.text)        
            return result
        else:
            if response.text:
                raise RuntimeError((f'get error, error is {response.text}'))
            else:
                raise RuntimeError((f'get error, error is {response}'))
    
    def health_check(self, reqUrl):
        """
        Performs a health check by sending a GET request to the specified URL.
        Args:
            reqUrl (str): The URL to send the GET request to.
        Returns:
            Response: The response object returned by the GET request.
        Raises:
            RuntimeError: If there is a connection error or any exception occurs during the request.
        """
        try:
            response = requests.get(reqUrl, timeout=300)
        except Exception as e:
            raise RuntimeError(f"connection error, error is {e}")
        
        return response

class Detect(ViewData):
    def initialize(self, workDir):
        super().__init__()
        pass

    def process(self, inputs):
        response = self.post_for_object(inputs)
        return response


class LLM_Chat(ViewLLM):
    def initialize(self, workDir):
        super().__init__()
        pass

    def process(self, inputs, reqType):
        """
        Processes the input request based on the specified request type.
        Args:
            inputs (object): The input data required for processing the request.
            reqType (RequestType): The type of request to process. It must be an instance of the `RequestType` enum.
        Returns:
            object: The result of the processed request, which could vary depending on the request type.
        Raises:
            RuntimeError: If the provided request type is not supported.
        Supported Request Types:
            - RequestType.health_check: Performs a health check by sending a request to the specified URL.
            - RequestType.get_models: Retrieves models by sending a GET request to the specified URL.
            - RequestType.llm_generate: Generates a response using LLM configuration. Supports both streaming and non-streaming modes.
            - RequestType.completions: Handles completion requests. Supports both streaming and non-streaming modes.
            - RequestType.chat_completions: Handles chat completion requests. Supports both streaming and non-streaming modes.
        """
        
        reqUrl = self.url+reqType.value
        reqType = RequestType(reqType.value)
        if reqType == RequestType.health_check:
            return self.health_check(reqUrl)    
        elif  reqType == RequestType.get_models:
            return self.get_for_object(inputs, reqUrl)   
        elif reqType == RequestType.llm_generate:
            llm_request = LLMconfigArgs2LLMRequest(inputs)
            if llm_request.data.stream is False:           
                return self.post_for_object(llm_request, reqUrl)
            else:        
                return self.post_for_object_stream(llm_request, reqUrl)          
        elif reqType == RequestType.completions or reqType == RequestType.chat_completions:
            if inputs.stream is False: 
                return self.post_for_object(inputs, reqUrl)
            else: 
                return self.post_for_object_stream(inputs, reqUrl)    
        else:
            raise RuntimeError((f'Request type is not supported, type is {reqType}'))

def LLMconfigArgs2LLMRequest(llmconfigArgs:LLMConfigRequest) -> LLMRequestWrapper: 
    """
        Converts an LLMConfigRequest object into an LLMRequestWrapper object.

        Args:
            llmconfigArgs (LLMConfigRequest): The input configuration request.

        Returns:
            LLMRequestWrapper: The converted request wrapper.
    """
    pload = {
        "data":{
            "requestId" : llmconfigArgs.requestId,
            "input" : llmconfigArgs.input,
            "stream" : llmconfigArgs.serviceParams.stream,
            "system" : llmconfigArgs.serviceParams.system,
            "history" : llmconfigArgs.history,
            "maxWindowSize" : llmconfigArgs.serviceParams.maxWindowSize,
            "maxContentRound" : llmconfigArgs.serviceParams.maxContentRound,
            "maxLength" : llmconfigArgs.serviceParams.maxOutputLength,
            "params" : llmconfigArgs.modelParams.dict(),
        },
    }

    return LLMRequestWrapper(**pload)

class Cls:

    def initialize(self, workDir):
        pass

    def process(self, inputs):
        print("start process")
        time.sleep(1)
        print("process finish")
        return [np.array([[0, 0.5, 10, 10, 20, 20]]) for i in range(len(inputs))]

class Seg:
    def initialize(self, workDir):
        pass

    def process(self, inputs):
        print("start process")
        time.sleep(1)
        print("process finish")

        return [np.array([[0, 0.5, 10, 10, 20, 20]]) for i in range(len(inputs))]
