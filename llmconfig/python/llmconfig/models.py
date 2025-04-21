import time
from enum import Enum
from typing import Union, Any, List, Dict, Tuple, Optional, Literal
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
from json import JSONEncoder, JSONDecoder

class RequestType(Enum):
    # API request
    # get_models       = 1    # v1/models
    # llm_generate     = 2    # llm/generate
    # completions      = 3    # v1/completions
    # chat_completions = 4    # v1/chat/completions    
    get_models       = '/v1/models'    # v1/models
    llm_generate     = '/llm/generate'    # llm/generate
    completions      = '/v1/completions'    # v1/completions
    chat_completions = '/v1/chat/completions'    # v1/chat/completions
    health_check     = '/health'  # health check


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

class ServiceParams(BaseModel):
    ### input/output control params
    system: Union[str, None] = None
    promptTemplateName: Union[str, None] = None
    maxWindowSize: Union[int, None] = 3000
    maxContentRound: Union[int, None] = 0
    maxOutputLength: Union[int, None] = 8192
    draw: bool = False
    stream: bool = False
    useTool: bool = False
    summarize: bool = False
    generateStyle: Union[str, None] = "chat"

class ModelParams(BaseModel):
    ### sample params
    best_of: Optional[int] = None
    presence_penalty: Union[float, None] = 0.0
    frequency_penalty: Union[float, None] = 0.0
    temperature: Union[float, None] = 1.0
    top_p: Union[float, None] = 1.0
    top_k: Union[int, None] = -1
    use_beam_search: Union[bool, None] = False
    length_penalty: Union[float, None] = 1.0

class LLMConfigRequest(BaseModel):
    ## llmconfig request params
    ### outer
    requestId: Union[str, None] = None
    input: str
    images: Union[List[str], None] = None
    userId: Union[str, None] = None
    history: Union[List[Tuple[str, str]], None] = None
    
    serviceParams: ServiceParams = Field(default_factory=ServiceParams)
    modelParams: ModelParams = Field(default_factory=ModelParams)

class LLMRequestBody(BaseModel):

    requestId: Union[str, None] = None
    input: str
    stream: bool = False
    
    system: Union[str, None] = None
    history: Union[List[Tuple[str, str]], None] = None

    maxWindowSize: Union[int, None] = 3000
    maxContentRound: Union[int, None] = 0
    maxLength: Union[int, None] = 8192
    params: dict = {
    }
    stopWords: Union[str|None] = None

class PromptTemplateHistory(BaseModel):
    historyInput:str|None = Field(default="{input}")
    historyOutput:str|None = Field(default="{output}")

class PromptTemplateBody(BaseModel):
    templateName: str|None = Field(default="qwen_default")
    systemPrompt: str|None = Field(default="You are a helpful assistant.")
    query: str|None = Field(default="{input}")
    chatHistory: PromptTemplateHistory

class PromptTemplateBodyWarpper(BaseModel):
    data: PromptTemplateBody

class RequestWrapper(BaseModel):
    data: RequestBody

class LLMRequestWrapper(BaseModel):
    data: LLMRequestBody

class AsyncLLMRequestWrapper(BaseModel):
    data: AsyncLLMRequestBody

class JsonSchemaResponseFormat(BaseModel):
    name: str
    description: Optional[str] = None
    # schema is the field in openai but that causes conflicts with pydantic so
    # instead use json_schema with an alias
    json_schema: Optional[Dict[str, Any]] = Field(default=None, alias='schema')
    strict: Optional[bool] = None

class ResponseFormat(BaseModel):
    # type must be "json_schema", "json_object" or "text"
    type: Literal["text", "json_object", "json_schema"]
    json_schema: Optional[JsonSchemaResponseFormat] = None

class FunctionDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ChatCompletionToolsParam(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDefinition


class ChatCompletionNamedFunction(BaseModel):
    name: str


class ChatCompletionNamedToolChoiceParam(BaseModel):
    function: ChatCompletionNamedFunction
    type: Literal["function"] = "function"

class ChatCompletionRequest(BaseModel):
    # openai chat params
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/chat/create
    requestId: Union[str, None] = None
    messages: List
    model: Optional[str] = None
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: bool = False
    top_logprobs: Optional[int] = None
    max_completion_tokens: Optional[int] = 0
    n: Optional[int] = 1
    presence_penalty: Optional[float] = 0.0
    response_format: Optional[ResponseFormat] = None

    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    tools: Optional[List[ChatCompletionToolsParam]] = None
    tool_choice: Optional[Union[Literal["none"], Literal["auto"],
                                ChatCompletionNamedToolChoiceParam]] = "none"

    # NOTE this will be ignored by VLLM -- the model determines the behavior
    parallel_tool_calls: Optional[bool] = False
    user: Optional[str] = None

    # doc: begin-chat-completion-sampling-params
    best_of: Optional[int] = None
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    prompt_logprobs: Optional[int] = None


class CompletionRequest(BaseModel):
    # Ordered by official OpenAI API documentation
    # https://platform.openai.com/docs/api-reference/completions/create
    #promptTemplateName: Union[str, None] = "geogpt"
    model: Optional[str] = None
    prompt: Union[List[int], List[List[int]], str, List[str]]
    best_of: Optional[int] = None
    echo: Optional[bool] = False
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 50
    n: int = 1
    presence_penalty: Optional[float] = 0.0
    stop: Optional[Union[str, List[str]]] = Field(default_factory=list)
    stream: Optional[bool] = False
    suffix: Optional[str] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    user: Optional[str] = None

    # doc: begin-completion-sampling-params
    use_beam_search: bool = False
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    stop_token_ids: Optional[List[int]] = Field(default_factory=list)
    include_stop_str_in_output: bool = False
    ignore_eos: bool = False
    min_tokens: int = 0
    skip_special_tokens: bool = True
    spaces_between_special_tokens: bool = True
    allowed_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[int] = None
    # doc: end-completion-sampling-params


class TemplateParams(BaseModel):
    temperature: float = 0.6
    best_of: int = 1
    length_penalty: float = 1.0
    presence_penalty: float = 2.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    top_p: float = 0.8
    top_k: int =-1

class TenmplateParamsWrapper(BaseModel):
    data: TemplateParams