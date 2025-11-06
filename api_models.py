# api_models.py

import time
import uuid
from typing import List, Union, Literal, Optional

from pydantic import BaseModel, Field

# --- 请求模型 ---
class TextContentPart(BaseModel):
    type: Literal["text"]
    text: str

class ImageUrl(BaseModel):
    url: str  # 预期格式: "data:image/jpeg;base64,{base64_string}"

class ImageContentPart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl

class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[Union[TextContentPart, ImageContentPart]]]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: float = Field(0.7, gt=0.0, le=2.0)
    max_tokens: int = Field(2048, gt=0)
    stream: bool = False

# --- 响应模型 ---
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
    role: Optional[str] = None
    content: Optional[str] = None

class StreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]