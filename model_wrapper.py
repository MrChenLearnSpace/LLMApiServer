# model_wrapper.py

import time
import uuid
from abc import ABC, abstractmethod
from typing import AsyncGenerator, Union, List

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
from fastapi.responses import StreamingResponse, JSONResponse

# --- 修改这里：从新的 api_models.py 导入 ---
from api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    ChatMessage,
    StreamChoice,
    DeltaMessage,
    Usage,
    TextContentPart,
    ImageContentPart
)

# --- 辅助函数：流式生成器 ---
async def stream_generator(llm_generator, model_name: str, response_id: str) -> AsyncGenerator[str, None]:
    created_time = int(time.time())
    
    for chunk in llm_generator:
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
            choices=[stream_choice],
            created=created_time
        )
        yield f"data: {stream_response.model_dump_json()}\n\n"

    yield "data: [DONE]\n\n"

# --- 模型封装 (这部分代码无需改动) ---

class ModelWrapper(ABC):
    """模型封装的抽象基类"""
    def __init__(self, model_path: str, model_name: str, **kwargs):
        self.model_path = model_path
        self.model_name = model_name
        self.llm = self._load_model(**kwargs)
    
    @abstractmethod
    def _load_model(self, **kwargs) -> Llama:
        """加载模型的具体实现"""
        pass

    @abstractmethod
    def _prepare_messages(self, messages: List[Union[ChatMessage, dict]]) -> List[dict]:
        """准备发送给模型的最终消息格式"""
        pass

    async def create_chat_completion(self, request: ChatCompletionRequest) -> Union[JSONResponse, StreamingResponse]:
        """处理聊天补全请求，支持流式和非流式"""
        
        messages_for_llm = self._prepare_messages(request.messages)
        
        response_generator = self.llm.create_chat_completion(
            messages=messages_for_llm,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream,
        )

        if request.stream:
            response_id = f"chatcmpl-{uuid.uuid4()}"
            return StreamingResponse(
                stream_generator(response_generator, self.model_name, response_id),
                media_type="text/event-stream"
            )
        else:
            full_response = response_generator
            response = ChatCompletionResponse(
                id=full_response.get('id', f"chatcmpl-{uuid.uuid4()}"),
                model=self.model_name,
                created=full_response.get('created', int(time.time())),
                choices=[
                    Choice(
                        index=choice['index'],
                        message=ChatMessage.model_validate(choice['message']),
                        finish_reason=choice['finish_reason']
                    ) for choice in full_response['choices']
                ],
                usage=Usage.model_validate(full_response['usage'])
            )
            return JSONResponse(content=response.model_dump())


class LLMWrapper(ModelWrapper):
    """标准大型语言模型的封装"""
    def _load_model(self, **kwargs) -> Llama:
        print("正在加载标准 LLM...")
        return Llama(model_path=self.model_path, **kwargs)

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[dict]:
        return [msg.model_dump() for msg in messages]


class VLWrapper(ModelWrapper):
    """视觉语言模型的封装"""
    def __init__(self, model_path: str, model_name: str, mmproj_path: str, **kwargs):
        self.mmproj_path = mmproj_path
        super().__init__(model_path, model_name, **kwargs)

    def _load_model(self, **kwargs) -> Llama:
        print(f"正在加载 VL 模型及 MMProj 文件 '{self.mmproj_path}'...")
        if not self.mmproj_path:
            raise ValueError("VL 模型必须提供 MMProj 文件路径 (MMPROJ_PATH)")
        
        chat_handler = Llava15ChatHandler(clip_model_path=self.mmproj_path, verbose=False)
        return Llama(model_path=self.model_path, chat_handler=chat_handler, **kwargs)

    def _prepare_messages(self, messages: List[ChatMessage]) -> List[dict]:
        processed_messages = []
        for msg in messages:
            if isinstance(msg.content, str):
                processed_messages.append(msg.model_dump())
                continue

            content_parts = []
            for part in msg.content:
                if isinstance(part, TextContentPart):
                    content_parts.append({"type": "text", "text": part.text})
                elif isinstance(part, ImageContentPart):
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": part.image_url.url}
                    })
            processed_messages.append({"role": msg.role, "content": content_parts})
        return processed_messages