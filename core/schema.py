from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, Union
from enum import Enum

# 基础数据结构（消息、状态、工具结果）

# 1. 定义 Agent 的运行状态 (对应 trae_agent 中的 AgentState)
class AgentState(Enum):
    IDLE = "idle"                   # 未启动
    THINKING = "thinking"           # LLM 正在生成回复
    CALLING_TOOL = "calling_tool"   # 正在执行工具
    COMPLETED = "completed"         # 任务完成
    ERROR = "error"                 # 发生错误

# 2. 定义工具调用的结构 (对应 OpenAI Function Call)
@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

# 3. 定义工具执行的结果
@dataclass
class ToolResult:
    call_id: str
    name: str
    output: str                     # 工具执行后的文本结果
    error: Optional[str] = None
    success: bool = True

# 4. 定义发给 LLM 的消息结构 (User/System/Assistant/Tool)
@dataclass
class LLMMessage:
    role: str                       # system, user, assistant, tool
    content: Optional[str] = None
    # 如果是 assistant role 且调用了工具：
    tool_calls: Optional[List[ToolCall]] = None 
    # 如果是 tool role (返回结果)：
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

# 5. 定义 Agent 的“一步”操作 (用于记录轨迹/日志)
@dataclass
class AgentStep:
    step_number: int
    state: AgentState
    thought: Optional[str] = None   # LLM 的思考过程
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    error: Optional[str] = None