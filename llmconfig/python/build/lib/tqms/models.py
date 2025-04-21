import time
from typing import Union, Any, List, Dict, Tuple, Optional
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
from json import JSONEncoder, JSONDecoder


class OssConfig(BaseModel):
    endpoint: str
    accessId: str
    accessSecret: str
    ossPath: Union[str, None] = None
    isS3: Union[bool, None] = None
    bucket: Union[str, None] = None
    key: Union[str, None] = None


class AsyncRequestBody(BaseModel):
    imageUrls: Union[List[str], None] = None
    inputListUri: Union[str, None] = None
    inputOss: OssConfig
    outputOss: OssConfig
    callbackUrl: str = Field(None, example="http://example.com/callback")
    params: dict = {}


class AsyncLLMRequestBody(BaseModel):
    inputs: Union[List[str], None] = None
    type: str = Field(default="Async")
    cloudFile: str = Field(default=None, example="http://example.com/input.txt")

    temperature: float = Field(default=None, ge=0.0, examples=0.8, le=1.0)
    presence_penalty: Union[float, None] = Field(default=None, ge=0.0, examples=0.8, le=2.0)
    repetition_penalty: Union[float, None] =  Field(default=None, gt=0.0, examples=0.8, le=2.0)
    frequency_penalty: Union[float, None] =  Field(default=None, ge=0.0, examples=0.8, le=2.0)
    top_p: Union[float, None] =  Field(default=None, ge=0.0, examples=0.8, le=1.0)
    top_k: Union[int, None] =  Field(default=None, ge=-1, examples=1, le=100)

    maxLength: int = Field(default=500, gt=0, lt=5000)
    promptTemplateName: str = Field(default="qwen_default", example="qwen_default")
    promptTemplate: dict|None = Field(default=None)
    requestId: str|None = Field(default=None)

    outputOss: OssConfig
    callbackUrl: str = Field(default=None, example="http://example.com/callback")
    summaryLlmUrl: str = Field(default=None, examples="http://example.com/callback")

class RoundChat():
    def __init__(self, input, output, createTime=None):
        self.createTime = time.time()
        if createTime is not None:
            self.createTime = createTime
        self.input = input
        self.output = output


## 对象转化为json str
class JsonEncoder(JSONEncoder):
    def default(self, o: Any) -> Any:
        return o.__dict__

## json str转化为对象
class RoundChatDecoder(JSONDecoder):
    def object_hook(self, dct):
        if "createTime" in dct and "input" in dct and "output" in dct:
            return RoundChat(dct["input"], dct["output"], dct["createTime"])
        return dct


class AsyncRequestWrapper(BaseModel):
    data: AsyncRequestBody


class RequestBody(BaseModel):
    imageB64: Union[str, None] = None
    imageUrl: Union[str, None] = None
    params: dict = {}


class LLMRequestBody(BaseModel):
    ### outer
    requestId: Union[str, None] = None
    input: Union[str, None] = None
    images: Union[List[str], None] = None
    userId: Union[str, None] = None
    maxContentRound: Union[int, None] = 0
    maxLength: Union[int, None] = 800
    maxWindowSize: Union[int, None] = 800
    summaryLlmUrl: str = Field(None, examples="http://example.com/callback")
    stream: bool = False
    draw: bool = False

    ### sample params
    best_of: Optional[int] = None
    presence_penalty: Union[float, None] = 0.0
    frequency_penalty: Union[float, None] = 0.0
    temperature: Union[float, None] = 1.0
    top_p: Union[float, None] = 1.0
    top_k: Union[int, None] = -1
    use_beam_search: Union[bool, None] = False
    length_penalty: Union[float, None] = 1.0

    ### inner and default
    type: str = Field(default="Sync")
    promptTemplateName: Union[str, None] = None
    generateStyle: Union[str, None] = "chat"
    system: Union[str, None] = None
    retryTimes: int = 0

    ### inner
    choices: Union[List[str], None] = None
    history: Union[List[Tuple[str, str]], None] = None
    promptTemplate: Union[Dict, None] = None
    params: dict = {
    }

class PromptTemplateHistory(BaseModel):
    input:str|None = Field(default="{input}")
    output:str|None = Field(default="{output}")

class PromptTemplateBody(BaseModel):
    prompt_header: str|None = Field(default="You are a helpful assistant.")
    prompt_history: PromptTemplateHistory
    prompt_input: str|None = Field(default="{input}")
    prompt_footer: str|None = Field(default=None)

class PromptTemplateBodyWarpper(BaseModel):
    data: PromptTemplateBody

class RequestWrapper(BaseModel):
    data: RequestBody

class LLMRequestWrapper(BaseModel):
    data: LLMRequestBody

class AsyncLLMRequestWrapper(BaseModel):
    data: AsyncLLMRequestBody
