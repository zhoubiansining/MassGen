import os
import json
from typing import List, Optional, Any
from openai import OpenAI
from .schema import LLMMessage, ToolCall

# LLM 调用封装

class LLMClient:
    """
    LLM 客户端封装。
    支持 OpenAI, Qwen, DeepSeek 等所有兼容 OpenAI 接口的模型。
    """
    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        # 1. 确定 API Key: 优先使用参数，其次环境变量
        final_api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        
        # 2. 强校验：如果没有 Key，直接报错，而不是打印警告
        if not final_api_key:
            raise ValueError(
                "❌ Critical Error: No API Key provided for LLMClient. "
                "Please pass 'api_key' explicitly or set 'OPENAI_API_KEY'/'DEEPSEEK_API_KEY' in environment variables."
            )

        # 3. 确定 Base URL
        final_base_url = base_url or os.getenv("OPENAI_BASE_URL")

        self.client = OpenAI(
            api_key=final_api_key,
            base_url=final_base_url
        )
        self.model = model

    def chat(self, messages: List[LLMMessage], tools: Optional[List[Any]] = None) -> LLMMessage:
        """
        发送消息给 LLM,并处理工具定义
        """
        # 1. 将内部消息格式转换为 OpenAI SDK 格式
        openai_msgs = []
        for msg in messages:
            m = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    } for tc in msg.tool_calls
                ]
            if msg.role == "tool":
                m["tool_call_id"] = msg.tool_call_id
                m["name"] = msg.name
            openai_msgs.append(m)

        openai_tools = [t.to_openai_schema() for t in tools] if tools else None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=openai_msgs,
                tools=openai_tools if openai_tools else None,
                temperature=0.7
            )

            choice = response.choices[0].message
            
            parsed_tool_calls = []
            if choice.tool_calls:
                for tc in choice.tool_calls:
                    parsed_tool_calls.append(ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments)
                    ))

            return LLMMessage(
                role="assistant",
                content=choice.content,
                tool_calls=parsed_tool_calls if parsed_tool_calls else None
            )
            
        except Exception as e:
            print(f"LLM API Error: {e}")
            raise e # 抛出异常让上层处理