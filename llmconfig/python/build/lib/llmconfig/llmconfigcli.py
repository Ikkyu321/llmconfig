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
        request.serviceParams.promptTemplateName = "qwen_default"
    
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
        data = handler(request, RequestType.llm_generate)
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
                            logging.info(f'catch json_not_completed exception, json.loads fail, msg<<#{msg}<<end')
            logging.info(f'stream requestId <<<#{requestId}>>> receive result text is {ret["output"]}')        
        return StreamingResponse(stream_results_buffer()) 

    return Response(message='success', data=data)


@app.get("/v1/models")
async def show_available_models(raw_request: Request):
    reqType = RequestType.get_models
    data = handler(raw_request, reqType)
    return JSONResponse(content=data)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest,
                                 raw_request: Request,
                                 sessionId: str = "",
                                 requestId: str = ""):

    if request.requestId == "" or request.requestId == None:
        if requestId == "":
            requestId = str(uuid.uuid4())
        request.requestId = requestId
    logging.info(f'requestId <<<{requestId}>>> receive request body is {request}')    
    #user_id = request.userId
    
    request.model = MODEL_PATH

    # default use geogpt template
    prompt_template = promptServer.get_prompt_template_by_name('geogpt')
    
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
        def stream_results() -> AsyncGenerator[bytes, None]:
            for request_output in data:
                out_msgs=request_output.decode("utf-8").rstrip('\n\n').split('\n\n')
                for msg in out_msgs:
                    if msg.startswith('data: '):
                        msg = msg[6:]  # 去除前缀 'data: '                       
                    if msg == '[DONE]':
                        yield(json.dumps(msg))
                    else:
                        yield (json.dumps(json.loads(msg)) + "\0") 
                #logging.info(f'stream requestId <<<#{requestId}>>> receive result text is {msg["output"]}')
        return StreamingResponse(stream_results(), media_type='text/event-stream') 
    
    return JSONResponse(content=data)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, 
                            raw_request: Request,
                            sessionId: str = "",
                            requestId: str = ""):
    
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
        def stream_results() -> AsyncGenerator[bytes, None]:
            for request_output in data:
                out_msgs=request_output.decode("utf-8").rstrip('\n\n').split('\n\n')
                for msg in out_msgs:
                    if msg.startswith('data: '):
                        msg = msg[6:]  # 去除前缀 'data: '                       
                    if msg == '[DONE]':
                        yield(json.dumps(msg) + "\0")
                    else:
                        yield (json.dumps(json.loads(msg)) + "\0") 
            #logging.info(f'stream requestId <<<#{requestId}>>> receive result text is {msg["output"]}')
        return StreamingResponse(stream_results())
    
    return JSONResponse(content=data)     

@app.post('/async/llm/generate')
async def async_llm_generate(reqW: AsyncLLMRequestWrapper, background_tasks: BackgroundTasks):
    logging.info(f'request body is {reqW}')
    request_body = reqW.data
    if request_body.cloudFile is None and request_body.inputs is None:
        return JSONResponse(status_code=422, content={"code": "422", "message": "cloudFile or inputs is musted", "data": None})

    try:
        s3_path = urllib.parse.urlparse(reqW.data.outputOss.ossPath)
        if len(s3_path.scheme) == 0 or len(s3_path.netloc) == 0:
            raise RuntimeError(f'invalid output oss path: {reqW.data.outputOss.ossPath}')
        reqW.data.outputOss.isS3 = True if s3_path.scheme == 's3' else False
        reqW.data.outputOss.bucket = s3_path.netloc
        reqW.data.outputOss.key = s3_path.path.lstrip('/')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f'invalid param: {e}')
    task_id = str(uuid.uuid4())
    request_body.requestId = task_id

    if request_body.promptTemplateName is not None:
        prompt_template = promptServer.get_prompt_template_by_name(request_body.promptTemplateName)
        request_body.promptTemplate = prompt_template

    async def bg_llm_request():
        await llmTaskServer.process(task_id, reqW.data)
    logging.info(f"this batch task id is {task_id}")
    background_tasks.add_task(bg_llm_request)
    return Response(data={"taskId": task_id})

@app.get('/asynctasks/{taskId}')
async def asynctasks_get(taskId: str):
    progress = progressCenter.get_progress(taskId)
    return Response(data=progress)

@app.delete('/asynctasks/{taskId}')
def asynctasks_delete(taskId: str):
    progress = progressCenter.delete_progress(taskId)
    return Response(data=progress)

@app.get("/prompt/templates")
async def prompt_templates_get(templateName:str = None):
    if templateName is not None:
        prompt_template = promptServer.get_prompt_template_by_name(templateName)
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

@app.get(path='/health')
async def health_check():
    data = {
             "data":{
                 "input":"你是谁",
                 "stream":False,
                 "type":"Sync",
                 "requestId":str(uuid.uuid4())
             }
           }
    logging.info(f"requestId <<<HealthyCheck>>> post to inference service: body is {data}")
    try:
        data = await handler(data, {"stream":False})
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=512,
                            content={"code": "512", "message": "not ready", "data": None})
    return JSONResponse(status_code=200,
                        content={"code": "200", "message": "success", "data": None})



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    uvicorn.run("llmconfig.llmconfigcli:app", host=args.host, port=args.port, log_level="info", timeout_keep_alive=300)


if __name__ == '__main__':
    main()
