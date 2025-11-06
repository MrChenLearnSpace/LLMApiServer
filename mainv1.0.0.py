# main.py
# 一个与 OpenAI API 标准完全兼容的、同时支持普通和流式响应的 FastAPI 服务器

import os
import time
import json
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict, AsyncGenerator

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from llama_cpp import Llama
from fastapi.responses import StreamingResponse
from typing import List, Dict, AsyncGenerator, Optional
# --- 1. 配置 (Configuration) ---
MODEL_PATH = os.getenv("MODEL_PATH", "./Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama")
VALID_API_KEYS = {
    "aa1234567",
    "another-valid-key-for-testing",
}

model_container = {}

# --- 2. 应用生命周期管理 (Lifespan Management) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("服务器启动，开始加载模型...")
    model_container["llm"] = Llama(
        model_path=MODEL_PATH,
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False,
    )
    print(f"模型 '{MODEL_PATH}' 加载成功。")
    yield
    model_container.clear()
    print("模型已卸载，服务器关闭。")


# --- 3. 安全与认证 (Security & Authentication) ---
auth_scheme = HTTPBearer()
def validate_api_key(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    if credentials.scheme != "Bearer" or credentials.credentials not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key 或认证格式错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


# --- 4. Pydantic 数据模型 (与 OpenAI API 完全对齐) ---

# --- 请求模型 ---
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(0.7, gt=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0)
    stream: bool = False # 客户端可以指定是否启用流式响应

# --- 非流式响应模型 ---
class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Usage

# --- 流式响应模型 ---
class DeltaMessage(BaseModel):
    role: str | None = None         # <--- 修改这里！
    content: str | None = None      # <--- 最好也一起修改，保持健壮性

class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None 

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]


# --- 5. FastAPI 应用实例与 API 路由 ---

app = FastAPI(lifespan=lifespan)

# --- 流式响应生成器 ---
async def stream_generator(llm_generator, model_name: str) -> AsyncGenerator[str, None]:
    """
    一个异步生成器，用于处理 llama-cpp-python 的流式输出，
    并将其格式化为 OpenAI 兼容的 Server-Sent Events (SSE)。
    """
    # 第一个数据块通常只包含角色信息
    first_chunk = True
    for chunk in llm_generator:
        if first_chunk:
            response_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            first_chunk = False
        
        # 构造符合 OpenAI 流式标准的 chunk
        stream_choice = StreamChoice(
            index=0,
            delta=DeltaMessage(
                role=chunk["choices"][0]["delta"].get("role"),
                content=chunk["choices"][0]["delta"].get("content", ""),
            ),
            finish_reason=chunk["choices"][0].get("finish_reason")
        )
        stream_response = ChatCompletionStreamResponse(
            id=response_id,
            model=model_name,
            choices=[stream_choice]
        )
        
        # 格式化为 SSE
        sse_data = f"data: {stream_response.model_dump_json()}\n\n"
        yield sse_data

    # 流结束时，发送一个特殊的 [DONE] 消息
    yield "data: [DONE]\n\n"


@app.post(
    "/v1/chat/completions",
    # 移除 response_model，因为我们要根据情况返回两种不同的模型
    tags=["兼容OpenAI"],
    dependencies=[Depends(validate_api_key)]
)
async def create_chat_completion(request: ChatCompletionRequest):
    """
    生成聊天补全。此端点现在同时支持标准的 JSON 响应和流式响应。
    """
    llm = model_container.get("llm")
    if not llm:
        raise HTTPException(status_code=503, detail="模型当前不可用或正在加载中")

    messages_for_llm = [msg.model_dump() for msg in request.messages]

    # 调用 llama-cpp-python，直接传递 stream 参数
    try:
        llm_response_generator = llm.create_chat_completion(
            messages=messages_for_llm,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream, # <--- 关键！
        )
    except Exception as e:
        print(f"在模型推理过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {e}")

    # --- 根据 stream 参数决定返回类型 ---
    if request.stream:
        # 如果是流式请求，返回一个 StreamingResponse
        return StreamingResponse(
            stream_generator(llm_response_generator, MODEL_NAME),
            media_type="text/event-stream"
        )
    else:
        # 如果是非流式请求，行为和之前一样
        llm_response = llm_response_generator # 在非流式模式下，这就是一个完整的字典
        response = ChatCompletionResponse(
            model=MODEL_NAME,
            choices=[
                Choice(
                    index=choice['index'],
                    message=ChatMessage(
                        role=choice['message']['role'],
                        content=choice['message']['content']
                    ),
                    finish_reason=choice['finish_reason']
                ) for choice in llm_response['choices']
            ],
            usage=Usage.model_validate(llm_response['usage'])
        )
        return response


# --- 运行服务器的命令 ---
# uvicorn main:app --host 0.0.0.0 --port 8000