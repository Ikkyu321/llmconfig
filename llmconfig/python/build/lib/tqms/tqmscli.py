#!python3.10
import argparse
import asyncio
import json
import sys
import traceback


from fastapi import (
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

from tqms import *


print(sys.path)

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
logging = init_logger("tqms_logger")

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
async def llm_generate(reqW: LLMRequestWrapper,
                       sessionId: str = "",
                       requestId: str = ""
                       ):
    request_body = reqW.data

    if requestId == "":
        requestId = str(uuid.uuid4())
    logging.info(f'requestId <<<{request_body.type}#{requestId}>>> receive request body is {reqW}')

    request_body.requestId = requestId
    user_id = request_body.userId

    if request_body.system is not None and request_body.system != "":
        if request_body.promptTemplate is None:
            request_body.promptTemplate = {
                "prompt_header":"",
                "prompt_history": {
                    "input": "{input}",
                    "output": "{output}"
                },
                "prompt_input": "{input}",
                "prompt_footer": None
            }
        request_body.promptTemplate["prompt_header"] = request_body.system

    if request_body.promptTemplateName is not None:
        prompt_template = promptServer.get_prompt_template_by_name(request_body.promptTemplateName)
        request_body.promptTemplate = prompt_template

    if request_body.history is None:
        if user_id is not None and request_body.maxContentRound > 0:
            session_history = llmHistory.get_history_by_session(user_id, sessionId)
            session_history = session_history[-request_body.maxContentRound:]
            history = []
            for round_history in session_history:
                if request_body.promptTemplate is not None:
                    history.append(( request_body.promptTemplate["prompt_history"]["input"]
                                     .format(input=round_history["input"]),
                                     request_body.promptTemplate["prompt_history"]["output"]
                                     .format(output=round_history["output"])
                                     ))
                else:
                    history.append(round_history)
            request_body.history = history
        else:
            request_body.history = []
    else:
        if len(request_body.history) > request_body.maxContentRound:
            request_body.history = request_body.history[-request_body.maxContentRound:]
        history = []
        for round_history in request_body.history:
            if request_body.promptTemplate is not None:
                history.append(( request_body.promptTemplate["prompt_history"]["input"]
                                 .format(input=round_history[0]),
                                 request_body.promptTemplate["prompt_history"]["output"]
                                 .format(output=round_history[1])
                                 ))
            else:
                history.append(round_history)
        request_body.history = history

    if request_body.promptTemplate is not None:
        original_input = request_body.input
        wrapper_input = request_body.promptTemplate["prompt_input"].format(input=request_body.input)
        request_body.input = wrapper_input

    if request_body.maxLength > 1500:
        request_body.maxLength = 1500

    if not request_body.stream:
        request_body.params["stream"] = False
        logging.info(f"requestId <<<{request_body.type}#{requestId}>>> post to inference service: body is {reqW}")
        async with semaphore:
            data = await handler(reqW, request_body.params)
    else:
        request_body.params["stream"] = True
        logging.info(f"requestId <<<{request_body.type}#{requestId}>>> post to inference service: body is {reqW}")
        data = handler(reqW, request_body.params)
        def stream_results() -> AsyncGenerator[bytes, None]:
            for request_output in data:
                ret = {"output": json.loads(request_output.decode("utf-8").rstrip('\x00'))["output"]}
                yield (json.dumps(ret) + "\0")
            logging.info(f'stream requestId <<<{request_body.type}#{requestId}>>> receive result text is {ret["output"]}')
            if user_id is not None and request_body.maxContentRound > 0:
                round_chat = RoundChat(request_body.input, ret["output"])
                llmHistory.put_one_round_chat(user_id, sessionId, round_chat)
        return StreamingResponse(stream_results())

    if user_id is not None and request_body.maxContentRound > 0:
        round_chat = RoundChat(original_input, data["output"])
        llmHistory.put_one_round_chat(user_id, sessionId, round_chat)
    return Response(message='success', data=data)

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

    uvicorn.run("tqms.tqmscli:app", host=args.host, port=args.port, log_level="info", timeout_keep_alive=300)


if __name__ == '__main__':
    main()
