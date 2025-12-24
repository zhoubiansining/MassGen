import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict
from .schema import ToolResult

# 工具的父类定义

class BaseTool(ABC):
    """
    所有工具的基类。
    自定义工具需要继承此类，并实现 name, description, get_parameters 和 execute。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称 (例如: 'google_search')"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述，告诉 LLM 这个工具是干什么的"""
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        返回 JSON Schema 格式的参数定义。
        例如:
        {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"}
            },
            "required": ["query"]
        }
        """
        pass

    @abstractmethod
    def execute(self, **kwargs) -> Any:
        """工具的具体执行逻辑"""
        pass

    def to_openai_schema(self) -> Dict[str, Any]:
        """将工具定义转换为 OpenAI API 要求的格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.get_parameters(),
            },
        }

    def run(self, call_id: str, **kwargs) -> ToolResult:
        """
        统一的运行入口，包含错误捕获。
        它会调用子类的 execute() 并包装成 ToolResult。
        """
        try:
            # 执行具体逻辑
            result_data = self.execute(**kwargs)
            
            # 确保结果是字符串格式
            output_str = str(result_data)
            
            return ToolResult(
                call_id=call_id,
                name=self.name,
                output=output_str,
                success=True
            )
            
        except Exception as e:
            return ToolResult(
                call_id=call_id,
                name=self.name,
                output=f"Error executing tool {self.name}: {str(e)}",
                error=str(e),
                success=False
            )