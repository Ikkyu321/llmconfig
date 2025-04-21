#!python3.10
import argparse
import asyncio
import json
import sys
import traceback
#import codecs
import os


from fastapi import (
    FastAPI,
    Cookie,
    Query,
    WebSocketException,
    HTTPException,
    status,
    BackgroundTasks,
    Request
)

from typing import Annotated, AsyncGenerator

from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import uuid

from llmconfig import *
from models import *

print(sys.path)
MODEL_PATH=os.environ["MODEL_DIR"]
Default_Template="geogpt"

app = FastAPI(timeout_client=300)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

lock = asyncio.Lock()
sys.path.append(os.getcwd())
if not os.path.exists('caches'):
    os.makedirs('caches')

semaphore = asyncio.Semaphore(PARALLEL_SIZE)
logging = init_logger("llmconfig_logger")

@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler_1(request, exc):
    traceback.print_exc()
    error_message = {"error": "Parameter validation error", "detail": exc.errors()}
    return JSONResponse(status_code=422, content={"code":"422", "message":error_message["error"], "detail":error_message,
                                                  "data":None})

@app.exception_handler(RuntimeError)
async def request_validation_exception_handler_2(request, exc):
    traceback.print_exc()
    error_message = {"error": "runtime error", "detail": str(exc)}
    return JSONResponse(status_code=522, content={"code":"522", "message":error_message["error"], "detail":error_message,
                                                  "data":None})

@app.exception_handler(TypeError)
async def request_validation_exception_handler_3(request, exc):
    traceback.print_exc()
    error_message = {"error": "type error", "detail": str(exc)}
    return JSONResponse(status_code=512, content={"code":"512", "message":error_message["error"], "detail":error_message,
                                                  "data":None})

@app.exception_handler(HTTPException)
async def request_validation_exception_handler_4(request, exc):
    traceback.print_exc()
    error_message = {"error": "http exception", "detail": exc.detail}
    return JSONResponse(status_code=exc.status_code, content={"code":f'{exc.status_code}', "message":error_message["error"],
                                                              "detail":exc.detail, "data":None})



@app.get('/meta')
def meta():
    return Response(message='success', data=handler.meta)

class Response(BaseModel):
    code:str = "200"
    message: str = "success"
    data: Any


@app.post('/async/predict', status_code=201)
async def async_predict(reqW: AsyncRequestWrapper):
    lines = None
    try:
        output_s3_path = urllib.parse.urlparse(reqW.data.outputOss.ossPath)
        if len(output_s3_path.scheme) == 0 or len(output_s3_path.netloc) == 0:
            raise RuntimeError(f'invalid output oss path: {reqW.data.outputOss.ossPath}')

        reqW.data.outputOss.isS3 = True if output_s3_path.scheme == 's3' else False
        reqW.data.outputOss.bucket = output_s3_path.netloc
        reqW.data.outputOss.key = output_s3_path.path.lstrip('/')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'invalid param: {e}')

    reqs = reqW.data
    if reqW.data.imageUrls is not None:
        data_list = reqW.data.imageUrls
    elif reqW.data.inputListUri is not None:
        try:
            input_s3_path = urllib.parse.urlparse(reqW.data.inputListUri)
            client = boto3.client('s3',
                                  aws_access_key_id=reqW.data.inputOss.accessId,
                                  aws_secret_access_key=reqW.data.inputOss.accessSecret,
                                  endpoint_url=reqW.data.inputOss.endpoint,
                                  verify=False)
            if input_s3_path.scheme == "http":
                response = requests.get(reqW.data.inputListUri, stream=True)
                lines = response.iter_lines()
            elif input_s3_path.scheme == "oss":
                bucket_name = input_s3_path.netloc
                object_key = input_s3_path.path.lstrip('/')
                response = client.get_object(Bucket=bucket_name, Key=object_key)
                lines = response['Body'].iter_lines()
            data_list = []
            for line in lines:
                decoded_line = line.decode('utf-8')
                json_line = json.loads(decoded_line)
                image_key = json_line["imageKey"]
                presigned_url = client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': reqW.data.inputOss.bucket, 'Key': image_key},
                    ExpiresIn=3600
                )
                data_list.append(presigned_url)
        except Exception as e:
            raise RuntimeError(f'input http file or oss file read error, message is {e}')
    else:
        raise RuntimeError(f'invalid inputListUri : {reqW.data.inputListUri} or imageUrls : {reqW.data.imageUrls}')

    task_id = str(uuid.uuid4())
    task = taskServer.buildTask(task_id, data_list)
    task.split_task()
    taskServer.addTask(task, reqs)
    return Response(message="async tasks accepted", data={'taskId': task_id})


# sync request
@app.post('/predict')
def predict(reqW: RequestWrapper):
    req = reqW.data

    try:
        img = req.imageUrl
        data = handler([img], req.params)
        return Response(message='success', data=data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'process request failed: {e}')


async def get_cookie_or_token(
        session: Annotated[str | None, Cookie()] = None,
        token: Annotated[str | None, Query()] = None,
):
    if session is None and token is None:
        raise WebSocketException(code=status.WS_1008_POLICY_VIOLATION)
    return session or token


@app.post('/llm/generate')
async def llm_generate(request: LLMConfigRequest,          
                       sessionId: str = "",
                       requestId: str = ""
                       ):
    """
    Asynchronous function to handle LLM (Large Language Model) generation requests.
    This function processes the input request, manages session history, applies prompt templates, 
    and interacts with an inference service to generate responses. It supports both standard 
    and streaming response modes.
    Args:
        request (LLMConfigRequest): The request object containing input, service parameters, 
            and other metadata for the LLM generation.
        sessionId (str, optional): The session ID for tracking user interactions. Defaults to an empty string.
        requestId (str, optional): The unique request ID for logging and tracking. Defaults to an empty string.
    Returns:
        Response or StreamingResponse: 
            - A standard response containing the generated data if `stream` is False.
            - A streaming response generator if `stream` is True.
    Behavior:
        - Generates a unique `requestId` if not provided.
        - Logs the request details for debugging and tracking.
        - Retrieves and applies a prompt template based on the `promptTemplateName` in the request.
        - Manages session history, truncating it to the maximum allowed rounds (`maxContentRound`).
        - Wraps the input using the prompt template's query format.
        - Sends the processed request to the inference service.
        - Handles streaming responses by buffering and decoding chunks of data, ensuring proper UTF-8 handling.
    Raises:
        UnicodeDecodeError: If a chunk of streaming data cannot be decoded properly.
        Exception: If a streaming message is not a complete JSON object.
    Notes:
        - The function uses a semaphore to limit concurrent access to the inference service.
        - Streaming responses are returned as an asynchronous generator, yielding JSON-encoded chunks.
    """
    # print("request_body:")
    # print(request)
    if request.requestId == "" or request.requestId == None:
        if requestId == "":
            requestId = str(uuid.uuid4())
        request.requestId = requestId
    logging.info(f'requestId <<<{requestId}>>> receive request body is {request}')    
    user_id = request.userId

    if request.serviceParams.promptTemplateName is None:    
            # if system is none | "", and promptTemplate is none
        request.serviceParams.promptTemplateName = Default_Template
    
    prompt_template = promptServer.get_prompt_template_by_name(request.serviceParams.promptTemplateName)

    if request.serviceParams.system is None or request.serviceParams.system == "":
        request.serviceParams.system = prompt_template["systemPrompt"]

    if request.history is None:
        if user_id is not None and request.serviceParams.maxContentRound > 0:
            session_history = llmHistory.get_history_by_session(user_id, sessionId)
            session_history = session_history[-request.serviceParams.maxContentRound:]
            history = []
            for round_history in session_history:
                if prompt_template is not None:
                    history.append(( prompt_template["chatHistory"]["historyInput"]
                                     .format(input=round_history["input"]),
                                     prompt_template["chatHistory"]["historyOutput"]
                                     .format(output=round_history["output"])
                                     ))
                else:
                    history.append(round_history)
            request.history = history
        else:
            request.history = []
    else:
        if len(request.history) > request.serviceParams.maxContentRound:
            request.history = request.history[-request_body.serviceParams.maxContentRound:]
            history = []
            for round_history in request.history:                   
                if prompt_template is not None:
                    history.append(( prompt_template["chatHistory"]["historyInput"]
                                        .format(input=round_history["input"]),
                                        prompt_template["chatHistory"]["historyOutput"]
                                        .format(output=round_history["output"])
                                        ))
                else:
                    history.append(round_history)

            request.history = history

    if prompt_template is not None:
        #original_input = request_body.input
        wrapper_input = prompt_template["query"].format(input=request.input)
        request.input = wrapper_input

    if not request.serviceParams.stream:
        #stream = False
        logging.info(f"requestId <<<#{requestId}>>> post to inference service: body is {request}")
        async with semaphore:
            data = await handler(request, RequestType.llm_generate)
            if data:
                data = data['data']
                logging.info(f"requestId <<<#{data['id']}>>> receive the response: body is {data}")
    else:
        #stream = True   
        logging.info(f"requestId <<<#{requestId}>>> post to inference service: body is {request}")
        data = handler(request, RequestType.llm_generate).iter_content(chunk_size=8192)
        def stream_results_buffer() -> AsyncGenerator[bytes, None]:          
            buf = bytearray()   # 创建一个字节缓冲区
            remaining = ''
            for i,request_output in enumerate(data):   
                # 流式输出时，iter_content 方法返回的每个块是独立的字节串，不保证字符的完整性
                # 每个块（chunk）可能包含不完整的 UTF-8 字符，
                # 导致解码时出现 UnicodeDecodeError。 
                buf.extend(request_output)  # 将块内容添加到缓冲区
                try:
                    # 尝试解码缓冲区中节点
                    out_msgs=buf.decode("utf-8").split('\x00')
                    buf.clear()  # 解码成功则清空缓存区
                except  UnicodeDecodeError:
                    logging.info(f"catch UnicodeDecodeError, buf.decode fail, buffer not completed<<<#{buf}<<<end")
                    continue
                    
                for msg in out_msgs:
                    if msg:
                        if remaining:
                            msg = remaining + msg
                            remaining = ''
                        try:  
                            # msg可能不是完整的json
                            ret = {"output": json.loads(msg)["output"]}
                            yield (json.dumps(ret) + "\0")
                        except Exception:
                            #buf.extend(msg.encode('utf-8'))
                            remaining = msg
                            logging.debug(f'catch json_not_completed exception, json.loads fail, msg<<#{msg}<<end')
            logging.info(f'stream requestId <<<#{requestId}>>> receive result text is {ret["output"]}')        
        return StreamingResponse(stream_results_buffer()) 

    return Response(message='success', data=data)


@app.get("/v1/models")
async def show_available_models(raw_request: Request):
    """
    Handles an incoming request to retrieve and display the list of available models.

    Args:
        raw_request (Request): The incoming HTTP request containing necessary parameters.

    Returns:
        JSONResponse: A JSON response containing the list of available models.
    """
    reqType = RequestType.get_models
    data = handler(raw_request, reqType)
    return JSONResponse(content=data)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request,
                                 sessionId: str = "",
                                 requestId: str = ""):
    """
    Asynchronous function to handle chat completion requests.
    This function processes a chat completion request, applies default configurations, 
    and sends the request to an inference service. It supports both streaming and 
    non-streaming responses.
    Args:
        request (ChatCompletionRequest): The chat completion request object containing 
            the input parameters for the chat model.
        raw_request (Request): The raw HTTP request object.
        sessionId (str, optional): The session ID for the request. Defaults to an empty string.
        requestId (str, optional): The request ID for tracking the request. If not provided, 
            a new UUID will be generated. Defaults to an empty string.
    Returns:
        JSONResponse or StreamingResponse: The response from the inference service. 
            Returns a JSONResponse for non-streaming requests and a StreamingResponse 
            for streaming requests.
    Raises:
        Exception: Propagates any exceptions raised during the processing of the request.
    Notes:
        - If `requestId` is not provided, a new UUID is generated and assigned to the request.
        - Default configurations such as temperature, penalties, and top-k values are 
            applied using a predefined prompt template.
        - If no system message is found in the request, a default system message is 
            inserted at the beginning of the messages list.
        - The function uses a semaphore to limit concurrent access to the inference service 
            for non-streaming requests.
    """

    if request.requestId == "" or request.requestId == None:
        if requestId == "":
            requestId = str(uuid.uuid4())
        request.requestId = requestId
    logging.info(f'requestId <<<{requestId}>>> receive request body is {request}')    
    #user_id = request.userId
    
    request.model = MODEL_PATH

    # default use geogpt template
    prompt_template = promptServer.get_prompt_template_by_name(Default_Template)
    template_params = promptServer.get_prompt_template_params_by_name(Default_Template)
    if template_params:
        request.temperature = template_params['temperature']
        request.best_of = template_params['best_of']
        request.presence_penalty = template_params['presence_penalty']
        request.frequency_penalty = template_params['frequency_penalty']
        request.repetition_penalty = template_params['repetition_penalty']
        request.top_p = template_params['top_p']
        request.top_k = template_params['top_k']
        request.length_penalty = template_params['length_penalty']
        
    system_message_found = False
    for message in request.messages:
        if message['role'] == 'system' or message['role'] == 'developer':
           #message['content'] = prompt_template["systemPrompt"]
           system_message_found = True
           break
    if not system_message_found:
        request.messages.insert(0, {"role": "system", "content": prompt_template["systemPrompt"]})

    if not request.stream:
        #stream = False
        logging.info(f"requestId <<<#{requestId}>>> post to inference service: body is {request}")
        async with semaphore:
            data = await handler(request, RequestType.chat_completions)
        logging.info(f"requestId <<<#{data['id']}>>> post to inference service: body is {data}")
    else:
        #stream = True   
        logging.info(f"requestId <<<#{requestId}>>> post to inference service: body is {request}")      
        data = handler(request, RequestType.chat_completions)
        logging.info(f"requestId <<<#{requestId}>>> received response: {data}")
        async def stream_results_openai():
            for chunk in data:
                yield chunk

        return StreamingResponse(stream_results_openai(), media_type="text/event-stream")
    
    return JSONResponse(content=data)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, 
                            raw_request: Request,
                            sessionId: str = "",
                            requestId: str = ""):
    """
    Handles the creation of a completion request by interacting with an inference service.
    Args:
        request (CompletionRequest): The completion request object containing the input data.
        raw_request (Request): The raw HTTP request object.
        sessionId (str, optional): The session identifier for the request. Defaults to an empty string.
        requestId (str, optional): The unique identifier for the request. Defaults to an empty string.
    Returns:
        JSONResponse: A JSON response containing the completion result if streaming is disabled.
        StreamingResponse: A streaming response yielding completion results if streaming is enabled.
    Notes:
        - If `request.stream` is False, the function sends the request to the inference service 
            and returns the result as a JSON response.
        - If `request.stream` is True, the function streams the results back to the client 
            using a `StreamingResponse` with the "text/event-stream" media type.
        - The function uses a semaphore to limit concurrent access to the handler when streaming is disabled.
    """
    
    request.model = MODEL_PATH

    if not request.stream:
        #stream = False
        logging.info(f"requestId <<<#{requestId}>>> post to inference service: body is {request}")
        async with semaphore:
            data = await handler(request, RequestType.completions)
    else:
        #stream = True   
        logging.info(f"requestId <<<#{requestId}>>> post to inference service: body is {request}")
        data = handler(request, RequestType.completions)
        
        async def stream_results_openai():
            for chunk in data:
                yield chunk

        return StreamingResponse(stream_results_openai(), media_type="text/event-stream")
    
    return JSONResponse(content=data)     


@app.get("/prompt/templates")
async def prompt_templates_get(templateName:str = None):
    if templateName is not None:
        prompt_template = promptServer.get_prompt_template_by_name(templateName)
        async with lock:
            Default_Template = templateName
        logging.info(f"Set default template, ##{Default_Template}")
        return Response(data=prompt_template)
    else:
        prompt_templates = promptServer.get_all_prompt_template()
        return Response(data=prompt_templates)

@app.get("/prompt/templates/names")
async def prompt_templates_get_names():
    prompt_templates = promptServer.get_all_prompt_template()
    keys = []
    for key in prompt_templates.keys():
        keys.append(key)
    return Response(data=keys)

@app.post("/prompt/templates")
async def prompt_templates_post(promptTemplateBodyWarpper:PromptTemplateBodyWarpper,
                                promptTemplateName:str | None = None):
    print(promptTemplateBodyWarpper)
    print(promptTemplateName)
    if promptTemplateName:
        template_body = jsonable_encoder(promptTemplateBodyWarpper.data)
        if promptServer.add_new_prompt_template(name=promptTemplateName,
                                                prompt_template_body=template_body):
            return Response(data=None)
    else:
        return Response(code="405", message=f"promptTemplateName data value wrong")

@app.delete(path='/prompt/templates')
async def prompt_template_delete(templateName:str = None):
    if promptServer.delete_prompt_template_by_name(templateName):
        return Response(data=None)

@app.get('/templates/params/name')
async def prompt_template_get_params(templateName:str = None):
    print(templateName)
    if templateName:
        params = promptServer.get_prompt_template_params_by_name(templateName)
        return Response(data=params)
    else:
        return Response(code="405", message=f"promptTemplateName data value wrong")   
      
@app.post('/templates/params')
async def prompt_template_set_params(templateParamsBodyWarpper:TenmplateParamsWrapper,
                                     templateName:str | None = None):
    print(templateParamsBodyWarpper)
    print(templateName) 
    if templateName and templateParamsBodyWarpper:
        params_body = jsonable_encoder(templateParamsBodyWarpper.data)
        if promptServer.add_template_params(templateName, params_body):
            return Response(data=None, message="success")
    else:
        return Response(code="405", message=f"TemplateName or params data value wrong,{templateName},{templateParamsBodyWarpper}")

@app.get(path='/health')
async def health_check():
    """
    Performs a health check for the inference service.

    This asynchronous function logs a health check message, attempts to invoke the 
    handler for a health check request, and returns an appropriate JSON response 
    based on the outcome.

    Returns:
        JSONResponse: A JSON response indicating the health status of the service.
            - If the service is healthy, returns a 200 status code with a success message.
            - If an exception occurs, logs the exception traceback and returns a 512 
              status code with a "not ready" message.
    """

    try:
        response_data = handler(None, RequestType.health_check)
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return JSONResponse(status_code=512,
                            content={"code": "503", "message": "not ready", "data": None})
    return JSONResponse(status_code=200,
                        content={"code": "200", "message": "success", "data": None})



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run("llmconfig.llmconfigcli:app", host=args.host, port=args.port, log_level="info", timeout_keep_alive=600)


if __name__ == '__main__':
    main()
