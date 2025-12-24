"""
Writing Agent 实现 - 支持国产模型 API
支持模型：Qwen、DeepSeek、GLM、Kimi、Doubao 等
"""

import openai
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
from datetime import datetime


# ==================== 配置类 ====================

@dataclass
class ModelConfig:
    """模型配置"""
    name: str                    # 模型名称，如 "qwen-max", "$Kimi-K2"
    api_key: str                 # API Key
    base_url: str                # API Base URL
    temperature: float = 0.5     # 温度参数
    max_tokens: int = 8192       # 最大 token 数
    top_p: float = 0.9           # Top-p 采样


# 国产模型配置示例
MODEL_CONFIGS = {
    # 通义千问（阿里云）
    "qwen-max": ModelConfig(
        name="qwen-max",
        api_key="your-dashscope-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192
    ),

    # DeepSeek
    "deepseek-chat": ModelConfig(
        name="deepseek-chat",
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com/v1",
        temperature=0.5,
        max_tokens=8192
    ),

    # 智谱 GLM
    "glm-4": ModelConfig(
        name="glm-4",
        api_key="your-zhipu-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192
    ),

    # Kimi (通过代理)
    "kimi-k2": ModelConfig(
        name="Kimi-K2",  # 注意：不要使用 $Kimi-K2
        api_key="key-here",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192
    ),

    # 字节豆包（火山引擎）
    "doubao-pro": ModelConfig(
        name="doubao-pro-32k",
        api_key="your-doubao-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192
    ),
}


# ==================== Prompt 模板 ====================

WRITING_AGENT_SYSTEM_PROMPT = """你是一位专业的学术综述撰写专家，擅长撰写高质量的文献综述。

**你的任务：**
基于提供的结构化研究摘要（cluster summaries），生成学术规范的综述文章段落。

**写作准则：**
1. 使用正式的学术语言，保持风格一致
2. 使用 [paper_id] 格式进行文内引用
3. 组织内容时使用清晰的主题句和过渡句
4. 适当进行比较分析
5. 每个事实性陈述都必须有引用支持

**输出要求：**
- 长度：每个部分 400-700 字
- 结构：背景介绍 → 方法概述 → 方法比较 → 挑战分析 → 未来方向
- 引用：文内使用 [paper_id] 标注，段落末尾列出参考文献
- 风格：{style_type}

**质量检查清单：**
□ 所有陈述都有引用
□ 段落之间逻辑流畅
□ 没有矛盾之处
□ 术语使用一致
□ 章节结构清晰
"""

REVIEWER_PROMPT = """你是一位资深学术审稿人。请审阅下面的综述草稿，重点关注：

1. **覆盖度**：是否包含了所有重要的论文和概念？
2. **准确性**：陈述是否准确反映了引用论文的内容？
3. **连贯性**：叙述是否逻辑清晰、结构合理？
4. **风格**：是否符合学术写作规范？
5. **新颖性**：是否识别出研究趋势和空白？

请提供：
- 优点列表（3-5 条）
- 缺点列表（3-5 条）
- 具体改进建议（带行号引用）

以 JSON 格式返回：
{
  "strengths": ["优点1", "优点2", ...],
  "weaknesses": ["缺点1", "缺点2", ...],
  "improvements": [
    {"issue": "问题描述", "suggestion": "改进建议", "location": "位置"},
    ...
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
        """构建写作任务的提示词"""
        prompt = f"""请基于以下结构化的研究摘要生成一段学术综述。

# 输入数据

## Cluster Summaries（研究主题分组）
{json.dumps(cluster_summaries, ensure_ascii=False, indent=2)}

# 任务要求

1. 生成 400-700 字的综述段落
2. 遵循结构：背景 → 方法 → 比较 → 挑战 → 未来方向
3. 使用 [paper_id] 格式标注引用
4. 确保所有事实性陈述都有引用支持

# 输出格式

请输出：
1. 综述正文（带引用标注）
2. 参考文献列表

开始写作：
"""
        return prompt

    def generate_draft(
        self,
        cluster_summaries: Dict,
        temperature: Optional[float] = None
    ) -> Dict:
        """
        生成单个草稿

        Args:
            cluster_summaries: 结构化的研究摘要
            temperature: 可选的温度参数（覆盖配置）

        Returns:
            包含草稿内容和元信息的字典
        """
        print(f"[INFO] 生成草稿 (temperature={temperature or self.model_config.temperature})...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_writing_prompt(cluster_summaries)}
        ]

        draft_content = self.llm_client.chat(messages, temperature)

        # 提取引用
        citations = self._extract_citations(draft_content)

        result = {
            "content": draft_content,
            "citations": citations,
            "temperature": temperature or self.model_config.temperature,
            "style": self.style,
            "timestamp": datetime.now().isoformat()
        }

        print(f"[SUCCESS] 草稿生成完成，包含 {len(citations)} 个引用")
        return result

    async def generate_draft_async(
        self,
        cluster_summaries: Dict,
        temperature: Optional[float] = None
    ) -> Dict:
        """异步生成草稿"""
        print(f"[INFO] 异步生成草稿 (temperature={temperature or self.model_config.temperature})...")

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

        print(f"[SUCCESS] 异步草稿生成完成，包含 {len(citations)} 个引用")
        return result

    def generate_multiple_candidates(
        self,
        cluster_summaries: Dict,
        num_candidates: int = 3,
        temperature_range: Tuple[float, float] = (0.3, 0.9)
    ) -> List[Dict]:
        """
        生成多个候选草稿（使用不同温度）

        Args:
            cluster_summaries: 研究摘要
            num_candidates: 候选数量
            temperature_range: 温度范围 (min, max)

        Returns:
            候选草稿列表
        """
        print(f"\n[INFO] 开始生成 {num_candidates} 个候选草稿...")

        candidates = []
        temp_min, temp_max = temperature_range
        temp_step = (temp_max - temp_min) / (num_candidates - 1) if num_candidates > 1 else 0

        for i in range(num_candidates):
            temperature = temp_min + i * temp_step
            draft = self.generate_draft(cluster_summaries, temperature)
            draft["candidate_id"] = i + 1
            candidates.append(draft)

        print(f"[SUCCESS] 所有 {num_candidates} 个候选草稿生成完成\n")
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


# ==================== 使用示例 ====================

def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("示例 1: 基础使用 - 生成单个草稿")
    print("=" * 60)

    # 1. 配置模型（使用 Kimi）
    config = MODEL_CONFIGS["kimi-k2"]

    # 2. 创建 Writing Agent
    agent = WritingAgent(config, style="narrative")

    # 3. 准备输入数据（模拟 cluster summaries）
    cluster_summaries = {
        "cluster_1": {
            "topic": "Transformer 架构改进",
            "summary": "本主题聚焦于对 Transformer 架构的各种改进方法，包括注意力机制优化、位置编码改进等。",
            "papers": [
                {
                    "paper_id": "paper_001",
                    "title": "Attention is All You Need",
                    "authors": ["Vaswani et al."],
                    "year": 2017,
                    "key_contribution": "提出了原始 Transformer 架构"
                },
                {
                    "paper_id": "paper_002",
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                    "authors": ["Devlin et al."],
                    "year": 2019,
                    "key_contribution": "基于 Transformer 的双向预训练"
                }
            ]
        }
    }

    # 4. 生成草稿
    draft = agent.generate_draft(cluster_summaries)

    # 5. 输出结果
    print("\n" + "=" * 60)
    print("生成的草稿：")
    print("=" * 60)
    print(draft["content"])
    print("\n" + "=" * 60)
    print(f"引用的论文: {draft['citations']}")
    print("=" * 60)


def example_multiple_candidates():
    """生成多个候选草稿示例"""
    print("\n\n" + "=" * 60)
    print("示例 2: 生成多个候选草稿")
    print("=" * 60)

    config = MODEL_CONFIGS["kimi-k2"]
    agent = WritingAgent(config)

    cluster_summaries = {
        "cluster_1": {
            "topic": "大语言模型",
            "summary": "关于大语言模型的最新进展",
            "papers": [
                {"paper_id": "paper_101", "title": "GPT-3", "year": 2020},
                {"paper_id": "paper_102", "title": "LLaMA", "year": 2023}
            ]
        }
    }

    # 生成 3 个候选（不同温度）
    candidates = agent.generate_multiple_candidates(
        cluster_summaries,
        num_candidates=3,
        temperature_range=(0.3, 0.9)
    )

    # 输出所有候选
    for i, candidate in enumerate(candidates):
        print(f"\n--- 候选草稿 {i+1} (temperature={candidate['temperature']:.2f}) ---")
        print(candidate["content"][:200] + "...")  # 只显示前 200 字符


async def example_parallel_generation():
    """并行生成示例（更快）"""
    print("\n\n" + "=" * 60)
    print("示例 3: 并行生成多个候选草稿（异步）")
    print("=" * 60)

    config = MODEL_CONFIGS["kimi-k2"]
    agent = WritingAgent(config)

    cluster_summaries = {
        "cluster_1": {
            "topic": "计算机视觉",
            "papers": [
                {"paper_id": "paper_201", "title": "ResNet"},
                {"paper_id": "paper_202", "title": "Vision Transformer"}
            ]
        }
    }

    # 并行生成（速度更快）
    candidates = await agent.generate_multiple_candidates_async(
        cluster_summaries,
        num_candidates=5
    )

    print(f"\n生成了 {len(candidates)} 个候选草稿")


def example_refinement():
    """自我修正示例"""
    print("\n\n" + "=" * 60)
    print("示例 4: 自我修正（Reflection）")
    print("=" * 60)

    config = MODEL_CONFIGS["kimi-k2"]
    agent = WritingAgent(config)

    cluster_summaries = {
        "cluster_1": {
            "topic": "强化学习",
            "papers": [
                {"paper_id": "paper_301", "title": "DQN"},
                {"paper_id": "paper_302", "title": "PPO"}
            ]
        }
    }

    # 生成初始草稿
    initial_draft = agent.generate_draft(cluster_summaries)

    # 多轮自我修正
    refined_result = agent.refine_with_reflection(
        initial_draft["content"],
        max_iterations=2
    )

    print(f"\n经过 {refined_result['iterations']} 轮修正")
    print("\n最终草稿：")
    print("=" * 60)
    print(refined_result["final_draft"])
    print("=" * 60)


def example_citation_validation():
    """引用验证示例"""
    print("\n\n" + "=" * 60)
    print("示例 5: 引用验证")
    print("=" * 60)

    config = MODEL_CONFIGS["kimi-k2"]
    agent = WritingAgent(config)

    # 模拟已有的草稿
    draft_text = """
    Transformer 架构 [paper_001] 已经成为自然语言处理的基础。
    BERT [paper_002] 进一步改进了预训练方法。
    但是引用了一个不存在的论文 [paper_999]。
    """

    available_papers = {
        "paper_001": {"title": "Attention is All You Need"},
        "paper_002": {"title": "BERT"},
        "paper_003": {"title": "GPT-3"}  # 这篇没被引用
    }

    # 验证引用
    validation = agent.validate_citations(draft_text, available_papers)

    print("\n引用验证结果：")
    print(f"- 总引用数: {validation['total_citations']}")
    print(f"- 引用有效: {validation['valid']}")
    print(f"- 无效引用: {validation['invalid_citations']}")
    print(f"- 遗漏论文: {validation['missing_papers']}")
    print(f"- 覆盖率: {validation['coverage_rate']:.1%}")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    # 示例 1: 基础使用
    example_basic_usage()

    # 示例 2: 多候选生成
    example_multiple_candidates()

    # 示例 3: 并行生成（需要 asyncio）
    # asyncio.run(example_parallel_generation())

    # 示例 4: 自我修正
    # example_refinement()

    # 示例 5: 引用验证
    example_citation_validation()


if __name__ == "__main__":
    main()
