# LLM API Server

一个基于 FastAPI 的大语言模型 API 服务器，提供与 OpenAI API 兼容的接口，支持标准 LLM 和视觉语言模型（VL）。

## ✨ 特性

- 🔌 **OpenAI API 兼容**：完全兼容 OpenAI Chat Completions API 规范
- 🎯 **双模型支持**：同时支持纯文本 LLM 和多模态视觉语言模型（VL）
- 🚀 **流式响应**：支持流式（Streaming）和非流式响应模式
- 🔒 **API 密钥认证**：内置 Bearer Token 认证机制
- 🏗️ **模块化设计**：清晰的代码架构，易于扩展和维护
- ⚡ **高性能**：基于 llama-cpp-python 和 FastAPI 构建
- 🎨 **多模态支持**：VL 模式下支持图文混合输入

## 📋 目录

- [快速开始](#快速开始)
- [安装](#安装)
- [配置](#配置)
- [使用方法](#使用方法)
- [API 文档](#api-文档)
- [项目结构](#项目结构)
- [示例](#示例)

## 🚀 快速开始

### 环境要求

- Python 3.8+
- CUDA（可选，用于 GPU 加速）
- 足够的内存和存储空间用于加载模型

### 安装

1. 克隆仓库：

```bash
git clone https://github.com/MrChenLearnSpace/LLMApiServer.git
cd LLMApiServer
```

2. 安装依赖：

```bash
# 基础依赖
pip install "fastapi[all]" uvicorn

# 安装 llama-cpp-python（CPU 版本）
pip install llama-cpp-python

# 或者安装 GPU 版本（支持 CUDA）
CMAKE_ARGS="-DLLAMA_CUDA=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

3. 下载模型文件：

将您的 GGUF 格式模型文件放置到项目目录中，或者指定模型路径。

### 启动服务

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## ⚙️ 配置

服务器通过环境变量进行配置：

### 环境变量

| 变量名 | 说明 | 默认值 | 必需 |
|--------|------|--------|------|
| `MODEL_TYPE` | 模型类型：`LLM` 或 `VL` | `VL` | 否 |
| `MODEL_PATH` | GGUF 模型文件路径 | `./Qwen3-VL-4B-Instruct-UD-IQ1_M.gguf` | 否 |
| `MMPROJ_PATH` | VL 模型的 MMProj 文件路径 | `./mmproj-BF16.gguf` | VL 模式时必需 |
| `MODEL_NAME` | 模型名称（API 响应中使用） | `default-model` | 否 |

### 配置示例

#### 启动 LLM 模式：

```bash
export MODEL_TYPE=LLM
export MODEL_PATH=/path/to/your/model.gguf
export MODEL_NAME=my-llm-model
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 启动 VL 模式（多模态）：

```bash
export MODEL_TYPE=VL
export MODEL_PATH=/path/to/your/vl-model.gguf
export MMPROJ_PATH=/path/to/your/mmproj.gguf
export MODEL_NAME=my-vl-model
uvicorn main:app --host 0.0.0.0 --port 8000
```

### API 密钥配置

默认的 API 密钥定义在 `main.py` 中的 `VALID_API_KEYS` 集合中：

```python
VALID_API_KEYS = {"aa1234567", "another-valid-key-for-testing"}
```

**⚠️ 生产环境建议**：
- 修改默认密钥
- 使用环境变量管理密钥
- 实施更严格的认证机制

## 📖 使用方法

### Python 示例

#### 基础对话（非流式）

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer aa1234567",
    "Content-Type": "application/json"
}

data = {
    "model": "default-model",
    "messages": [
        {"role": "user", "content": "你好，请介绍一下你自己。"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": False
}

response = requests.post(url, headers=headers, json=data)
print(response.json()["choices"][0]["message"]["content"])
```

#### 流式响应

```python
import requests

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer aa1234567",
    "Content-Type": "application/json"
}

data = {
    "model": "default-model",
    "messages": [
        {"role": "user", "content": "讲个故事"}
    ],
    "stream": True
}

response = requests.post(url, headers=headers, json=data, stream=True)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            chunk = line[6:]
            if chunk != '[DONE]':
                import json
                data = json.loads(chunk)
                content = data["choices"][0]["delta"].get("content", "")
                print(content, end="", flush=True)
```

#### 多模态输入（VL 模式）

```python
import base64
import requests

# 读取并编码图片
with open("image.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Authorization": "Bearer aa1234567",
    "Content-Type": "application/json"
}

data = {
    "model": "default-model",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "这张图片里有什么？"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    }
                }
            ]
        }
    ],
    "temperature": 0.7,
    "max_tokens": 2048
}

response = requests.post(url, headers=headers, json=data)
print(response.json()["choices"][0]["message"]["content"])
```

### cURL 示例

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer aa1234567" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default-model",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7,
    "max_tokens": 2048,
    "stream": false
  }'
```

## 📚 API 文档

### 端点

#### POST `/v1/chat/completions`

创建聊天补全。

**请求头：**
- `Authorization: Bearer <api_key>` - 必需

**请求体参数：**

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `model` | string | 是 | 模型名称 |
| `messages` | array | 是 | 消息数组 |
| `temperature` | float | 否 | 温度参数（0.0-2.0），默认 0.7 |
| `max_tokens` | integer | 否 | 最大生成 token 数，默认 2048 |
| `stream` | boolean | 否 | 是否启用流式响应，默认 false |

**消息格式：**

纯文本消息：
```json
{
  "role": "user",
  "content": "消息内容"
}
```

多模态消息（VL 模式）：
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "文本内容"},
    {
      "type": "image_url",
      "image_url": {
        "url": "data:image/jpeg;base64,<base64_string>"
      }
    }
  ]
}
```

**响应格式：**

非流式响应：
```json
{
  "id": "chatcmpl-xxxxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "default-model",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "回复内容"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

流式响应（SSE 格式）：
```
data: {"id":"chatcmpl-xxxxx","object":"chat.completion.chunk","created":1234567890,"model":"default-model","choices":[{"index":0,"delta":{"content":"内容"},"finish_reason":null}]}

data: [DONE]
```

## 🗂️ 项目结构

```
LLMApiServer/
├── main.py              # 主应用入口，包含 FastAPI 应用和路由
├── api_models.py        # Pydantic 数据模型定义
├── model_wrapper.py     # 模型封装类（LLMWrapper 和 VLWrapper）
├── mainv1.0.0.py       # v1.0.0 版本的主文件（历史版本）
├── deploy.ipynb        # 部署笔记本
├── code-server.sh      # Code Server 安装脚本
├── .gitignore          # Git 忽略文件配置
└── README.md           # 项目说明文档
```

### 核心模块说明

#### `main.py`
- FastAPI 应用初始化
- 生命周期管理（模型加载/卸载）
- API 路由定义
- 认证中间件

#### `api_models.py`
- 请求/响应数据模型
- 符合 OpenAI API 规范的 Pydantic 模型
- 支持流式和非流式响应格式

#### `model_wrapper.py`
- `ModelWrapper`: 抽象基类
- `LLMWrapper`: 标准 LLM 封装
- `VLWrapper`: 视觉语言模型封装
- 流式生成器实现

## 🔧 高级配置

### 模型参数调整

在 `main.py` 中可以调整模型加载参数：

```python
model_kwargs = {
    "n_ctx": 8192,        # 上下文窗口大小
    "n_gpu_layers": -1,   # GPU 层数（-1 表示全部使用 GPU）
    "verbose": False      # 是否输出详细日志
}
```

### 自定义模型封装

如需支持其他模型类型，可以继承 `ModelWrapper` 类：

```python
class CustomModelWrapper(ModelWrapper):
    def _load_model(self, **kwargs) -> Llama:
        # 自定义模型加载逻辑
        pass
    
    def _prepare_messages(self, messages: List[ChatMessage]) -> List[dict]:
        # 自定义消息预处理逻辑
        pass
```

## 🔍 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认有足够的内存
   - 查看服务器日志获取详细错误信息

2. **CUDA 相关错误**
   - 确认 CUDA 版本兼容
   - 重新安装支持 CUDA 的 llama-cpp-python

3. **认证失败**
   - 确认使用正确的 API 密钥
   - 检查 Authorization 头格式：`Bearer <api_key>`

4. **VL 模式下图片无法识别**
   - 确保提供了 MMPROJ_PATH
   - 检查图片 base64 编码格式正确
   - 确认模型支持视觉输入

## 📝 版本历史

- **v1.0.0**: 基础版本，支持标准 LLM
- **当前版本**: 增加多模态支持，模块化重构，完善 API 兼容性

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用开源许可证，具体请查看 LICENSE 文件。

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代化的 Python Web 框架
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) - llama.cpp 的 Python 绑定
- OpenAI - API 规范参考

## 📞 联系方式

如有问题或建议，请通过 GitHub Issues 联系。

---

**注意**：本项目仅供学习和研究使用，请遵守相关模型的使用协议和限制。
