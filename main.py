# main.py

import os
from contextlib import asynccontextmanager
from typing import Union

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse

# 导入我们的封装类
from model_wrapper import LLMWrapper, VLWrapper
# 导入我们抽离出去的 Pydantic 模型
from api_models import ChatCompletionRequest

# --- 1. 配置 (Configuration) ---
MODEL_TYPE = os.getenv("MODEL_TYPE", "VL").upper()
MODEL_PATH = os.getenv("MODEL_PATH", "./Qwen3-VL-4B-Instruct-UD-IQ1_M.gguf")
MMPROJ_PATH = os.getenv("MMPROJ_PATH","./mmproj-BF16.gguf")
MODEL_NAME = os.getenv("MODEL_NAME", "default-model")
VALID_API_KEYS = {"aa1234567", "another-valid-key-for-testing"}

model_container = {}

# --- 2. 应用生命周期管理 (Lifespan Management) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"服务器启动，准备加载模型 (类型: {MODEL_TYPE})...")
    
    model_kwargs = {
        "n_ctx": 8192,
        "n_gpu_layers": -1,
        "verbose": False
    }

    if MODEL_TYPE == "VL":
        if not MMPROJ_PATH:
            raise ValueError("错误: MODEL_TYPE 设置为 'VL' 但未提供 MMPROJ_PATH 环境变量。")
        wrapper = VLWrapper(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME,
            mmproj_path=MMPROJ_PATH,
            **model_kwargs
        )
    elif MODEL_TYPE == "LLM":
        wrapper = LLMWrapper(
            model_path=MODEL_PATH,
            model_name=MODEL_NAME,
            **model_kwargs
        )
    else:
        raise ValueError(f"不支持的模型类型: {MODEL_TYPE}")

    model_container["wrapper"] = wrapper
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

# --- 4. FastAPI 应用实例与 API 路由 ---
app = FastAPI(lifespan=lifespan)

@app.post(
    "/v1/chat/completions",
    tags=["兼容OpenAI"],
    dependencies=[Depends(validate_api_key)],
    response_model=None
)
async def create_chat_completion(request: ChatCompletionRequest) -> Union[JSONResponse, StreamingResponse]:
    """
    生成聊天补全。此端点现在同时支持标准的 JSON 响应和流式响应，
    并且能够根据加载的模型类型处理纯文本或多模态输入。
    """
    wrapper = model_container.get("wrapper")
    if not wrapper:
        raise HTTPException(status_code=503, detail="模型当前不可用或正在加载中")

    try:
        return await wrapper.create_chat_completion(request)
    except Exception as e:
        print(f"在模型推理过程中发生错误: {e}")
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

# --- 运行服务器的命令 ---
# uvicorn main:app --host 0.0.0.0 --port 8000