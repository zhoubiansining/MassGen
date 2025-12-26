"""Writing Agent - English prompts + multi-temperature candidate generation."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import openai


# ==================== Config ====================


@dataclass
class ModelConfig:
    """Model configuration"""

    name: str
    api_key: str
    base_url: str
    temperature: float = 0.5
    max_tokens: int = 8192
    top_p: float = 0.9


MODEL_CONFIGS = {
    "qwen-max": ModelConfig(
        name="qwen-max",
        api_key="your-dashscope-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192,
    ),
    "deepseek-chat": ModelConfig(
        name="deepseek-chat",
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com/v1",
        temperature=0.5,
        max_tokens=8192,
    ),
    "glm-4": ModelConfig(
        name="glm-4",
        api_key="your-zhipu-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192,
    ),
    "kimi-k2": ModelConfig(
        name="Kimi-K2",
        api_key="key-here",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192,
    ),
    "doubao-pro": ModelConfig(
        name="doubao-pro-32k",
        api_key="your-doubao-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192,
    ),
}


# ==================== Prompts ====================

WRITING_AGENT_SYSTEM_PROMPT = """You are an expert academic survey writer.

**Task**: Based on the provided structured cluster summaries, write a coherent academic survey section.

**Guidelines**
1. Use formal academic English with consistent style.
2. Cite using [paper_id] in-text markers.
3. Organize with clear topic sentences and smooth transitions.
4. Include comparative analysis where appropriate.
5. Every factual claim must be supported by a citation.

**Output Requirements**
- Length: 400–700 words per section
- Structure: Background → Methods → Comparison → Challenges → Future directions
- Citations: In-text [paper_id]; list references at the end
- Style: {style_type}

**Quality Checklist**
□ All claims have citations
□ Logical flow between paragraphs
□ No contradictions
□ Consistent terminology
□ Clear section structure
"""

REVIEWER_PROMPT = """You are a senior academic reviewer. Review the draft focusing on:

1. Coverage: important papers and concepts covered?
2. Factuality: statements accurate and supported by citations?
3. Coherence: logical flow and structure?
4. Academic style: formal tone and citation format?
5. Novelty: trends, gaps, and future directions identified?

Return JSON:
{
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvements": [
    {"issue": "...", "suggestion": "...", "location": "..."}
  ]
}
"""


# ==================== LLM 客户端封装 ====================

class LLMClient:
    """统一的 LLM 客户端，支持多种国产模型"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    def chat(self, messages: List[Dict], temperature: Optional[float] = None) -> str:
        """同步调用"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] LLM 调用失败: {e}")
            raise

    async def chat_async(self, messages: List[Dict], temperature: Optional[float] = None) -> str:
        """异步调用（使用 asyncio 包装同步调用）"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chat,
            messages,
            temperature
        )


# ==================== Writing Agent 核心类 ====================

class WritingAgent:
    """
    Writing Agent - 负责生成高质量综述草稿

    功能：
    1. 生成多个候选草稿（不同风格/温度）
    2. 自我修正（reflection）
    3. 引用追踪
    """

    def __init__(self, model_config: ModelConfig, style: str = "narrative"):
        """
        初始化 Writing Agent

        Args:
            model_config: 模型配置
            style: 写作风格 (narrative/table-driven/timeline/taxonomy)
        """
        self.llm_client = LLMClient(model_config)
        self.model_config = model_config
        self.style = style

        # 系统提示词
        self.system_prompt = WRITING_AGENT_SYSTEM_PROMPT.format(
            style_type=style
        )

    def _build_writing_prompt(self, cluster_summaries: Dict) -> str:
        """Build the writing prompt"""
        prompt = f"""Write a survey section based on the structured cluster summaries below.

# Input
{json.dumps(cluster_summaries, ensure_ascii=False, indent=2)}

# Requirements
1. Length 400–700 words
2. Structure: Background → Methods → Comparison → Challenges → Future directions
3. Use [paper_id] for in-text citations
4. Every factual statement must have citation support

# Output
1. Survey text with in-text citations
2. Reference list

Begin writing:
"""
        return prompt

    def generate_draft(
        self,
        cluster_summaries: Dict,
        temperature: Optional[float] = None
    ) -> Dict:
        """Generate a single draft"""
        print(f"[INFO] Generating draft (temperature={temperature or self.model_config.temperature}) ...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_writing_prompt(cluster_summaries)}
        ]

        draft_content = self.llm_client.chat(messages, temperature)

        # Extract citations
        citations = self._extract_citations(draft_content)

        result = {
            "content": draft_content,
            "citations": citations,
            "temperature": temperature or self.model_config.temperature,
            "style": self.style,
            "timestamp": datetime.now().isoformat()
        }

        print(f"[SUCCESS] Draft generated with {len(citations)} citations")
        return result

    async def generate_draft_async(
        self,
        cluster_summaries: Dict,
        temperature: Optional[float] = None
    ) -> Dict:
        """Generate draft asynchronously"""
        print(f"[INFO] Async draft generation (temperature={temperature or self.model_config.temperature}) ...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_writing_prompt(cluster_summaries)}
        ]

        draft_content = await self.llm_client.chat_async(messages, temperature)
        citations = self._extract_citations(draft_content)

        result = {
            "content": draft_content,
            "citations": citations,
            "temperature": temperature or self.model_config.temperature,
            "style": self.style,
            "timestamp": datetime.now().isoformat()
        }

        print(f"[SUCCESS] Async draft generated with {len(citations)} citations")
        return result

    def generate_multiple_candidates(
        self,
        cluster_summaries: Dict,
        num_candidates: int = 3,
        temperature_range: Tuple[float, float] = (0.3, 0.9)
    ) -> List[Dict]:
        """Generate multiple candidates with a temperature range"""
        print(f"\n[INFO] Generating {num_candidates} candidates ...")

        candidates = []
        temp_min, temp_max = temperature_range
        temp_step = (temp_max - temp_min) / (num_candidates - 1) if num_candidates > 1 else 0

        for i in range(num_candidates):
            temperature = temp_min + i * temp_step
            draft = self.generate_draft(cluster_summaries, temperature)
            draft["candidate_id"] = i + 1
            candidates.append(draft)

        print(f"[SUCCESS] Generated {num_candidates} candidates\n")
        return candidates

    async def generate_multiple_candidates_async(
        self,
        cluster_summaries: Dict,
        num_candidates: int = 3,
        temperature_range: Tuple[float, float] = (0.3, 0.9)
    ) -> List[Dict]:
        """
        并行生成多个候选草稿（更快）

        Args:
            cluster_summaries: 研究摘要
            num_candidates: 候选数量
            temperature_range: 温度范围

        Returns:
            候选草稿列表
        """
        print(f"\n[INFO] 并行生成 {num_candidates} 个候选草稿...")

        temp_min, temp_max = temperature_range
        temp_step = (temp_max - temp_min) / (num_candidates - 1) if num_candidates > 1 else 0

        # 创建并行任务
        tasks = [
            self.generate_draft_async(cluster_summaries, temp_min + i * temp_step)
            for i in range(num_candidates)
        ]

        # 并行执行
        candidates = await asyncio.gather(*tasks)

        # 添加候选 ID
        for i, draft in enumerate(candidates):
            draft["candidate_id"] = i + 1

        print(f"[SUCCESS] 所有 {num_candidates} 个候选草稿并行生成完成\n")
        return candidates

    def refine_draft(self, draft: str, feedback: Dict) -> Dict:
        """
        基于反馈改进草稿（Reflection）

        Args:
            draft: 原始草稿
            feedback: 反馈信息（来自 self-critique 或 judge）

        Returns:
            改进后的草稿
        """
        print("[INFO] 基于反馈改进草稿...")

        improvement_points = "\n".join([
            f"- {imp['issue']}: {imp['suggestion']}"
            for imp in feedback.get("improvements", [])
        ])

        refine_prompt = f"""请改进以下综述草稿，解决下列问题：

# 反馈问题
{improvement_points}

# 原始草稿
{draft}

# 任务
请根据反馈修改草稿，保持学术风格和引用格式。输出改进后的完整草稿。
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": refine_prompt}
        ]

        refined_content = self.llm_client.chat(messages)
        citations = self._extract_citations(refined_content)

        result = {
            "content": refined_content,
            "citations": citations,
            "refined": True,
            "timestamp": datetime.now().isoformat()
        }

        print("[SUCCESS] 草稿改进完成")
        return result

    def self_critique(self, draft: str) -> Dict:
        """
        自我批评（self-reflection）

        Args:
            draft: 草稿内容

        Returns:
            批评和改进建议
        """
        print("[INFO] 执行自我批评...")

        messages = [
            {"role": "system", "content": REVIEWER_PROMPT},
            {"role": "user", "content": f"请审阅以下综述草稿：\n\n{draft}"}
        ]

        critique_text = self.llm_client.chat(messages, temperature=0.2)

        # 尝试解析 JSON（如果模型返回格式正确）
        try:
            # 提取 JSON 部分（处理可能的 markdown 格式）
            if "```json" in critique_text:
                json_start = critique_text.find("```json") + 7
                json_end = critique_text.find("```", json_start)
                json_text = critique_text[json_start:json_end].strip()
            elif "{" in critique_text:
                json_start = critique_text.find("{")
                json_end = critique_text.rfind("}") + 1
                json_text = critique_text[json_start:json_end]
            else:
                json_text = critique_text

            critique = json.loads(json_text)
        except json.JSONDecodeError:
            # 如果解析失败，返回原始文本
            critique = {
                "strengths": [],
                "weaknesses": [],
                "improvements": [{"issue": "解析失败", "suggestion": critique_text}]
            }

        print(f"[SUCCESS] 自我批评完成：发现 {len(critique.get('improvements', []))} 个改进点")
        return critique

    def refine_with_reflection(
        self,
        draft: str,
        max_iterations: int = 3
    ) -> Dict:
        """
        多轮自我修正

        Args:
            draft: 初始草稿
            max_iterations: 最大迭代次数

        Returns:
            最终改进后的草稿和迭代历史
        """
        print(f"\n[INFO] 开始多轮自我修正（最大 {max_iterations} 轮）...\n")

        current_draft = draft
        history = []

        for iteration in range(max_iterations):
            print(f"--- 第 {iteration + 1} 轮修正 ---")

            # Step 1: 自我批评
            critique = self.self_critique(current_draft)

            # 记录历史
            history.append({
                "iteration": iteration + 1,
                "critique": critique,
                "draft": current_draft
            })

            # 如果没有改进建议，停止迭代
            if not critique.get("improvements"):
                print("[INFO] 没有更多改进建议，停止迭代")
                break

            # Step 2: 基于批评改进
            refined = self.refine_draft(current_draft, critique)
            current_draft = refined["content"]

        print(f"\n[SUCCESS] 自我修正完成，共进行 {len(history)} 轮\n")

        return {
            "final_draft": current_draft,
            "iterations": len(history),
            "history": history,
            "timestamp": datetime.now().isoformat()
        }

    def _extract_citations(self, text: str) -> List[str]:
        """提取文中的引用 [paper_id]"""
        import re
        citations = re.findall(r'\[([^\]]+)\]', text)
        return list(set(citations))  # 去重

    def validate_citations(
        self,
        draft: str,
        available_papers: Dict
    ) -> Dict:
        """
        验证引用的有效性

        Args:
            draft: 草稿内容
            available_papers: 可用的论文字典 {paper_id: paper_info}

        Returns:
            验证结果
        """
        citations = self._extract_citations(draft)
        available_ids = set(available_papers.keys())

        invalid = [c for c in citations if c not in available_ids]
        missing_important = available_ids - set(citations)

        result = {
            "valid": len(invalid) == 0,
            "total_citations": len(citations),
            "invalid_citations": invalid,
            "missing_papers": list(missing_important),
            "coverage_rate": len(set(citations) & available_ids) / len(available_ids) if available_ids else 0
        }

        if invalid:
            print(f"[WARNING] 发现 {len(invalid)} 个无效引用: {invalid}")
        if missing_important:
            print(f"[WARNING] 有 {len(missing_important)} 篇重要论文未被引用")

        return result
