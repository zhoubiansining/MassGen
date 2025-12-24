import asyncio
import traceback
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Union

from .schema import LLMMessage, ToolResult, ToolCall, AgentStep
from .tool import BaseTool
from .llm import LLMClient

class BaseAgent(ABC):
    """
    通用 Agent 基类。
    职责：
    1. 管理 Infrastructure (LLM 连接, 工具注册, 历史记录)。
    2. 提供原子操作 (Atomic Capabilities): `think()` 和 `act()`。
    3. 不强制定义工作流 (Workflow)，把 `run()` 留给子类实现。
    """

    def __init__(
        self, 
        name: str, 
        tools: List[BaseTool], 
        system_prompt: str, 
        model: str,                  
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.llm = LLMClient(model=model, base_url=base_url, api_key=api_key)
        self.tools = {t.name: t for t in tools}
        self.history: List[LLMMessage] = []
    
    @abstractmethod
    async def run(self, task: str, **kwargs) -> Any:
        """
        [必须实现] 子类定义自己的工作流。
        """
        pass

    # --- 原子能力 1: 思考 (Think) ---
    async def think(self) -> LLMMessage:
        """
        将当前 history 发送给 LLM, 获取回复。
        """
        try:
            # 异步调用 LLM (防止阻塞)
            response = await asyncio.to_thread(
                self.llm.chat, 
                self.history, 
                list(self.tools.values())
            )
            # 自动追加 Assistant 消息到历史
            self.history.append(response)
            return response
        except Exception as e:
            print(f"[{self.name}] Thinking Error: {e}")
            raise e

    # --- 原子能力 2: 行动 (Act) ---
    async def act(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """
        执行工具调用列表，支持并发执行。
        """
        if not tool_calls:
            return []

        tasks = []
        for call in tool_calls:
            print(f"🛠️ [{self.name}] Calling: {call.name}")
            tasks.append(self._execute_single_tool(call))
        
        # 并发执行
        results = await asyncio.gather(*tasks)
        
        # 自动追加 Tool 结果到历史 (闭环)
        for res in results:
            self.history.append(LLMMessage(
                role="tool",
                tool_call_id=res.call_id,
                name=res.name,
                content=res.output
            ))
            # 简单日志
            preview = res.output[:100].replace("\n", " ") + "..."
            print(f"   -> Result: {preview}")
            
        return results

    async def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """内部 helper: 执行单个工具"""
        if tool_call.name not in self.tools:
            return ToolResult(call_id=tool_call.id, name=tool_call.name, output="Tool not found", success=False)
        
        tool = self.tools[tool_call.name]
        try:
            # 兼容同步和异步工具
            if asyncio.iscoroutinefunction(tool.run):
                return await tool.run(call_id=tool_call.id, **tool_call.arguments)
            else:
                return await asyncio.to_thread(tool.run, call_id=tool_call.id, **tool_call.arguments)
        except Exception as e:
            return ToolResult(call_id=tool_call.id, name=tool_call.name, output=str(e), success=False)

    def init_history(self, task: str):
        """标准初始化"""
        self.history = [
            LLMMessage(role="system", content=self.system_prompt),
            LLMMessage(role="user", content=task)
        ]