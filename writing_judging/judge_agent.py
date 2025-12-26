"""
Judge Agent 实现 - 评估和择优综述草稿
支持模型：Qwen、DeepSeek、GLM、Kimi、Doubao 等
"""

import openai
import json
import asyncio
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


# ==================== 复用 Writing Agent 的配置 ====================

@dataclass
class ModelConfig:
    """模型配置"""
    name: str                    # 模型名称，如 "qwen-max", "$Kimi-K2"
    api_key: str                 # API Key
    base_url: str                # API Base URL
    temperature: float = 0.2     # 温度参数（Judge Agent 使用低温度）
    max_tokens: int = 4096       # 最大 token 数
    top_p: float = 0.9           # Top-p 采样


# 国产模型配置示例（与 Writing Agent 保持一致）
MODEL_CONFIGS = {
    "kimi-k2": ModelConfig(
        name="Kimi-K2",  # 注意：不要使用 $Kimi-K2
        api_key="sk-aRG9iu2Hy9--oPxrG-5faA",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.2,  # Judge 使用更低的温度以提高一致性
        max_tokens=4096
    ),
    "qwen-max": ModelConfig(
        name="qwen-max",
        api_key="your-dashscope-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0.2,
        max_tokens=4096
    ),
    "deepseek-chat": ModelConfig(
        name="deepseek-chat",
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com/v1",
        temperature=0.2,
        max_tokens=4096
    ),
}


# ==================== 评分维度定义 ====================

@dataclass
class ScoringDimension:
    """评分维度"""
    name: str
    description: str
    weight: float
    max_score: int = 100


SCORING_DIMENSIONS = {
    "coverage": ScoringDimension(
        name="覆盖度",
        description="关键问题、方法、代表论文是否被覆盖",
        weight=0.25
    ),
    "factuality": ScoringDimension(
        name="准确性",
        description="事实准确性，是否有幻觉或错误引用",
        weight=0.30
    ),
    "coherence": ScoringDimension(
        name="连贯性",
        description="逻辑连贯性、段落衔接",
        weight=0.20
    ),
    "academic_style": ScoringDimension(
        name="学术性",
        description="是否符合学术写作规范",
        weight=0.15
    ),
    "novelty": ScoringDimension(
        name="新颖性",
        description="是否指出新颖趋势或研究空白",
        weight=0.10
    )
}


# ==================== Prompt 模板 ====================

JUDGE_SYSTEM_PROMPT = """你是一位资深的学术综述评审专家，擅长多维度评估综述质量。

**你的任务：**
对提供的学术综述草稿进行全面、客观、一致的评估。

**评分维度（0-100分制）：**

1. **覆盖度 (Coverage) - 25%**
   - 是否包含了所有重要论文？
   - 是否涵盖了主要研究方向？
   - 范围是否适当？

2. **准确性 (Factuality) - 30%**
   - 所有陈述是否准确？
   - 引用是否正确且相关？
   - 是否有幻觉或无依据的陈述？

3. **连贯性 (Coherence) - 20%**
   - 逻辑流程是否清晰？
   - 段落过渡是否流畅？
   - 结构是否组织良好？

4. **学术性 (Academic Style) - 15%**
   - 语言是否正式准确？
   - 引用格式是否正确？
   - 术语使用是否一致？

5. **新颖性 (Novelty) - 10%**
   - 是否识别出研究趋势？
   - 是否指出文献空白？
   - 是否提出未来方向？

**评分标准：**
- 90-100: 优秀，可发表
- 80-89: 良好，需小幅修改
- 70-79: 合格，需中等修改
- 60-69: 尚可，需大幅修改
- <60: 不合格，需重写

**输出要求：**
必须返回有效的 JSON 格式，包含：
- scores: 各维度分数
- overall_score: 加权总分
- strengths: 优点列表（3-5条）
- weaknesses: 缺点列表（3-5条）
- improvement_suggestions: 改进建议（具体且可操作）
"""


FACTUALITY_CHECK_PROMPT = """你是一位事实核验专家。

**任务：**
验证综述草稿中的每个事实性陈述是否得到引用论文的支持。

**验证流程：**
1. 提取草稿中的所有事实性陈述
2. 检查每个陈述是否有引用支持
3. 对比陈述与引用论文的内容是否一致
4. 识别任何可能的幻觉或误解

**输出 JSON 格式：**
{
  "total_claims": <数量>,
  "verified_claims": <已验证数量>,
  "unverified_claims": [
    {
      "claim": "陈述内容",
      "issue": "问题描述",
      "cited_papers": ["paper_id1", ...]
    }
  ],
  "accuracy_rate": <准确率 0-1>,
  "hallucinations": [
    {
      "text": "幻觉内容",
      "reason": "原因"
    }
  ]
}
"""


# ==================== LLM 客户端封装 ====================

class LLMClient:
    """统一的 LLM 客户端（复用 Writing Agent 的实现）"""

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
        """异步调用"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chat,
            messages,
            temperature
        )


# ==================== Judge Agent 核心类 ====================

class JudgeAgent:
    """
    Judge Agent - 负责评估和择优综述草稿

    功能：
    1. 多维度评分（coverage, factuality, coherence, academic_style, novelty）
    2. 提供详细反馈和改进建议
    3. 对多个候选草稿进行排序和择优
    4. 事实性验证
    """

    def __init__(self, model_config: ModelConfig):
        """
        初始化 Judge Agent

        Args:
            model_config: 模型配置
        """
        self.llm_client = LLMClient(model_config)
        self.model_config = model_config
        self.scoring_dimensions = SCORING_DIMENSIONS

    def _build_evaluation_prompt(
        self,
        draft: str,
        reference: Optional[Dict] = None
    ) -> str:
        """构建评估提示词"""

        reference_section = ""
        if reference:
            reference_section = f"""
# 参考资料（用于验证覆盖度和准确性）

{json.dumps(reference, ensure_ascii=False, indent=2)}
"""

        prompt = f"""请评估以下学术综述草稿。

# 待评估的草稿

{draft}

{reference_section}

# 评估要求

请按照系统提示词中的评分维度进行评估，返回 JSON 格式的结果。

**JSON 格式示例：**
{{
  "scores": {{
    "coverage": 85,
    "factuality": 90,
    "coherence": 80,
    "academic_style": 88,
    "novelty": 75
  }},
  "overall_score": 84.5,
  "strengths": [
    "引用全面，覆盖了所有重要论文",
    "逻辑结构清晰，过渡自然",
    "学术语言规范，术语使用准确"
  ],
  "weaknesses": [
    "部分段落过于冗长，可以更简洁",
    "对未来方向的讨论略显不足",
    "某些引用的相关性不够强"
  ],
  "improvement_suggestions": [
    {{
      "issue": "第二段引用密度过高，影响可读性",
      "suggestion": "将部分次要引用移到段落末尾，保持叙述流畅性",
      "priority": "medium"
    }},
    {{
      "issue": "缺少对研究空白的明确指出",
      "suggestion": "在结论部分增加一段专门讨论当前研究的局限和未来机会",
      "priority": "high"
    }}
  ]
}}

请开始评估：
"""
        return prompt

    def evaluate_draft(
        self,
        draft: str,
        reference: Optional[Dict] = None
    ) -> Dict:
        """
        评估单个草稿

        Args:
            draft: 草稿内容
            reference: 参考资料（cluster summaries）

        Returns:
            包含评分和反馈的字典
        """
        print(f"[INFO] 评估草稿（长度: {len(draft)} 字符）...")

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_evaluation_prompt(draft, reference)}
        ]

        response = self.llm_client.chat(messages)

        # 解析 JSON 响应
        evaluation = self._parse_evaluation_response(response)

        # 计算加权总分（如果模型没有计算）
        if "overall_score" not in evaluation or evaluation["overall_score"] is None:
            evaluation["overall_score"] = self._calculate_weighted_score(
                evaluation.get("scores", {})
            )

        evaluation["timestamp"] = datetime.now().isoformat()
        evaluation["draft_length"] = len(draft)

        print(f"[SUCCESS] 评估完成，总分: {evaluation['overall_score']:.1f}/100")
        return evaluation

    async def evaluate_draft_async(
        self,
        draft: str,
        reference: Optional[Dict] = None
    ) -> Dict:
        """异步评估草稿"""
        print(f"[INFO] 异步评估草稿（长度: {len(draft)} 字符）...")

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self._build_evaluation_prompt(draft, reference)}
        ]

        response = await self.llm_client.chat_async(messages)
        evaluation = self._parse_evaluation_response(response)

        if "overall_score" not in evaluation or evaluation["overall_score"] is None:
            evaluation["overall_score"] = self._calculate_weighted_score(
                evaluation.get("scores", {})
            )

        evaluation["timestamp"] = datetime.now().isoformat()
        evaluation["draft_length"] = len(draft)

        print(f"[SUCCESS] 异步评估完成，总分: {evaluation['overall_score']:.1f}/100")
        return evaluation

    def _parse_evaluation_response(self, response: str) -> Dict:
        """解析评估响应（提取 JSON）"""
        try:
            # 尝试提取 JSON 部分
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_text = response[json_start:json_end].strip()
            elif "{" in response:
                json_start = response.find("{")
                json_end = response.rfind("}") + 1
                json_text = response[json_start:json_end]
            else:
                json_text = response

            evaluation = json.loads(json_text)
            return evaluation

        except json.JSONDecodeError as e:
            print(f"[WARNING] JSON 解析失败，返回默认结构: {e}")
            # 返回默认结构
            return {
                "scores": {
                    "coverage": 50,
                    "factuality": 50,
                    "coherence": 50,
                    "academic_style": 50,
                    "novelty": 50
                },
                "overall_score": 50,
                "strengths": [],
                "weaknesses": ["JSON 解析失败"],
                "improvement_suggestions": [],
                "raw_response": response
            }

    @staticmethod
    def _normalize_ratio(value: Optional[float]) -> Optional[float]:
        """将 0-1、1-5 或 0-100 区间的得分统一映射到 0-100。"""
        if value is None:
            return None
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None

        if val <= 1.0:
            return val * 100
        if 1.0 < val <= 5.0:
            return min(100.0, val / 5.0 * 100)
        return min(val, 100.0)

    @staticmethod
    def _avg(scores: List[Optional[float]]) -> Optional[float]:
        """安全均值"""
        valid = [s for s in scores if s is not None]
        return round(sum(valid) / len(valid), 2) if valid else None

    @staticmethod
    def _summarize_auto_metrics(
        auto_metrics: Dict,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Optional[float]]:
        """
        将自动评估指标（引文/内容/结构）汇总为 0-100 评分。

        参数 auto_metrics 预期结构：
        {
          "citation": {"precision": 0.8, "recall": 0.7, "f1": 0.74, "accuracy": 0.9},
          "content": {"semantic_similarity": 0.82, "rouge_l": 0.61, "kpr": 0.55},
          "structure": {"overlap": 0.76, "relevance_llm": 4.2}
        }
        """

        weights = weights or {"citation": 0.4, "content": 0.35, "structure": 0.25}

        citation = auto_metrics.get("citation", {}) if auto_metrics else {}
        content = auto_metrics.get("content", {}) if auto_metrics else {}
        structure = auto_metrics.get("structure", {}) if auto_metrics else {}

        citation_score = JudgeAgent._avg([
            JudgeAgent._normalize_ratio(citation.get(k))
            for k in ["f1", "precision", "recall", "accuracy"]
        ])

        content_score = JudgeAgent._avg([
            JudgeAgent._normalize_ratio(content.get(k))
            for k in ["semantic_similarity", "rouge_l", "kpr"]
        ])

        structure_raw = [
            structure.get("overlap"),
            structure.get("relevance_llm")
        ]
        # relevance_llm 1-5，overlap/结构 F1 视为 0-1
        structure_score = JudgeAgent._avg([
            JudgeAgent._normalize_ratio(structure_raw[0]),
            JudgeAgent._normalize_ratio(structure_raw[1])
        ])

        overall = None
        components = []
        for key, score in [("citation", citation_score), ("content", content_score), ("structure", structure_score)]:
            weight = weights.get(key, 0)
            if score is not None:
                components.append((score, weight))

        if components:
            total_weight = sum(w for _, w in components) or 1.0
            overall = round(sum(s * w for s, w in components) / total_weight, 2)

        return {
            "citation_score": citation_score,
            "content_score": content_score,
            "structure_score": structure_score,
            "overall_score": overall,
        }

    @staticmethod
    def _blend_scores(
        llm_score: float,
        auto_score: Optional[float],
        auto_weight: float = 0.5
    ) -> float:
        """融合 LLM 主观评分与自动指标得分。"""
        auto_weight = min(max(auto_weight, 0.0), 1.0)
        if auto_score is None:
            return round(llm_score, 2)
        return round(llm_score * (1 - auto_weight) + auto_score * auto_weight, 2)

    @staticmethod
    def multi_model_vote_with_auto_metrics(
        draft: str,
        model_configs: List[ModelConfig],
        reference: Optional[Dict] = None,
        auto_metrics: Optional[Dict] = None,
        auto_metric_weights: Optional[Dict[str, float]] = None,
        auto_weight: float = 0.5
    ) -> Dict:
        """
        多模型评分 + 自动评估指标融合。

        Args:
            draft: 待评估草稿
            model_configs: 参与投票的模型配置列表
            reference: 参考资料，用于覆盖度/准确性评估
            auto_metrics: 自动指标（引文/内容/结构）
            auto_metric_weights: 自动指标汇总权重
            auto_weight: 最终融合时自动指标所占权重（0-1）
        """

        evaluations = []
        for cfg in model_configs:
            agent = JudgeAgent(cfg)
            result = agent.evaluate_draft(draft, reference)
            result["model_name"] = cfg.name
            evaluations.append(result)

        llm_average = round(sum(e["overall_score"] for e in evaluations) / len(evaluations), 2) if evaluations else 0.0

        auto_summary = JudgeAgent._summarize_auto_metrics(auto_metrics, auto_metric_weights) if auto_metrics else None
        auto_overall = auto_summary.get("overall_score") if auto_summary else None

        final_score = JudgeAgent._blend_scores(llm_average, auto_overall, auto_weight)

        return {
            "llm_evaluations": evaluations,
            "llm_average": llm_average,
            "auto_evaluation": auto_summary,
            "final_score": final_score,
            "weights": {
                "auto_weight": auto_weight,
                "llm_weight": 1 - auto_weight
            }
        }

    def _calculate_weighted_score(self, scores: Dict[str, float]) -> float:
        """计算加权总分"""
        total_score = 0.0
        for dimension, score in scores.items():
            if dimension in self.scoring_dimensions:
                weight = self.scoring_dimensions[dimension].weight
                total_score += score * weight
        return round(total_score, 2)

    def rank_drafts(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None
    ) -> List[Dict]:
        """
        对多个草稿进行排序

        Args:
            drafts: 草稿列表
            reference: 参考资料

        Returns:
            排序后的评估结果列表（按分数从高到低）
        """
        print(f"\n[INFO] 对 {len(drafts)} 个草稿进行排序...")

        evaluations = []
        for i, draft in enumerate(drafts):
            print(f"\n--- 评估第 {i+1}/{len(drafts)} 个草稿 ---")
            evaluation = self.evaluate_draft(draft, reference)
            evaluation["draft_id"] = i
            evaluation["draft"] = draft
            evaluations.append(evaluation)

        # 按总分排序
        evaluations.sort(key=lambda x: x["overall_score"], reverse=True)

        print(f"\n[SUCCESS] 排序完成")
        print("\n排序结果：")
        for i, eval_result in enumerate(evaluations):
            print(f"  {i+1}. 草稿 {eval_result['draft_id']} - 得分: {eval_result['overall_score']:.1f}")

        return evaluations

    async def rank_drafts_async(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None
    ) -> List[Dict]:
        """异步并行对多个草稿进行排序（更快）"""
        print(f"\n[INFO] 并行评估 {len(drafts)} 个草稿...")

        # 创建并行任务
        tasks = [
            self.evaluate_draft_async(draft, reference)
            for draft in drafts
        ]

        # 并行执行
        evaluations = await asyncio.gather(*tasks)

        # 添加 draft_id 和原始内容
        for i, evaluation in enumerate(evaluations):
            evaluation["draft_id"] = i
            evaluation["draft"] = drafts[i]

        # 按总分排序
        evaluations.sort(key=lambda x: x["overall_score"], reverse=True)

        print(f"\n[SUCCESS] 并行评估完成")
        print("\n排序结果：")
        for i, eval_result in enumerate(evaluations):
            print(f"  {i+1}. 草稿 {eval_result['draft_id']} - 得分: {eval_result['overall_score']:.1f}")

        return evaluations

    def rejection_sampling(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None,
        threshold: float = 70.0,
        max_keep: int = 3
    ) -> List[Dict]:
        """
        拒绝采样：过滤低质量草稿

        Args:
            drafts: 草稿列表
            reference: 参考资料
            threshold: 最低分数阈值
            max_keep: 最多保留数量

        Returns:
            高质量草稿列表（已排序）
        """
        print(f"\n[INFO] 执行拒绝采样（阈值: {threshold}, 最多保留: {max_keep}）...")

        # 先排序
        ranked_drafts = self.rank_drafts(drafts, reference)

        # 过滤低于阈值的
        high_quality = [
            d for d in ranked_drafts
            if d["overall_score"] >= threshold
        ]

        # 保留 top-k
        selected = high_quality[:max_keep]

        print(f"\n[SUCCESS] 拒绝采样完成")
        print(f"  原始数量: {len(drafts)}")
        print(f"  通过阈值: {len(high_quality)}")
        print(f"  最终保留: {len(selected)}")

        return selected

    async def rejection_sampling_async(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None,
        threshold: float = 70.0,
        max_keep: int = 3
    ) -> List[Dict]:
        """异步拒绝采样（更快）"""
        print(f"\n[INFO] 并行执行拒绝采样（阈值: {threshold}, 最多保留: {max_keep}）...")

        # 并行排序
        ranked_drafts = await self.rank_drafts_async(drafts, reference)

        # 过滤和选择
        high_quality = [
            d for d in ranked_drafts
            if d["overall_score"] >= threshold
        ]
        selected = high_quality[:max_keep]

        print(f"\n[SUCCESS] 拒绝采样完成")
        print(f"  原始数量: {len(drafts)}")
        print(f"  通过阈值: {len(high_quality)}")
        print(f"  最终保留: {len(selected)}")

        return selected

    def verify_factuality(
        self,
        draft: str,
        evidence: Dict
    ) -> Dict:
        """
        验证事实准确性

        Args:
            draft: 草稿内容
            evidence: 证据材料（论文信息）

        Returns:
            验证结果
        """
        print("[INFO] 执行事实准确性验证...")

        # 提取引用
        citations = self._extract_citations(draft)

        prompt = f"""请验证以下综述草稿的事实准确性。

# 草稿内容
{draft}

# 可用证据（论文信息）
{json.dumps(evidence, ensure_ascii=False, indent=2)}

# 任务
1. 提取草稿中的所有事实性陈述
2. 检查每个陈述是否有引用支持
3. 验证陈述与引用论文的内容是否一致
4. 识别可能的幻觉或误解

请按照 JSON 格式返回验证结果。
"""

        messages = [
            {"role": "system", "content": FACTUALITY_CHECK_PROMPT},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.chat(messages)

        # 解析响应
        try:
            verification = self._parse_evaluation_response(response)
        except:
            verification = {
                "total_claims": len(citations),
                "verified_claims": 0,
                "accuracy_rate": 0.0,
                "unverified_claims": [],
                "hallucinations": []
            }

        # 添加引用检查
        available_ids = set(evidence.keys()) if evidence else set()
        invalid_citations = [c for c in citations if c not in available_ids]

        verification["citation_check"] = {
            "total_citations": len(citations),
            "invalid_citations": invalid_citations,
            "citation_validity_rate": 1.0 - (len(invalid_citations) / len(citations)) if citations else 0.0
        }

        print(f"[SUCCESS] 事实验证完成")
        print(f"  准确率: {verification.get('accuracy_rate', 0):.1%}")
        print(f"  引用有效率: {verification['citation_check']['citation_validity_rate']:.1%}")

        return verification

    def _extract_citations(self, text: str) -> List[str]:
        """提取文中的引用 [paper_id]"""
        citations = re.findall(r'\[([^\]]+)\]', text)
        return list(set(citations))  # 去重

    def select_best_draft(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None
    ) -> Dict:
        """
        选择最佳草稿（简化版）

        Args:
            drafts: 草稿列表
            reference: 参考资料

        Returns:
            最佳草稿及其评估结果
        """
        print(f"\n[INFO] 从 {len(drafts)} 个草稿中选择最佳...")

        ranked = self.rank_drafts(drafts, reference)
        best = ranked[0]

        print(f"\n[SUCCESS] 最佳草稿已选择")
        print(f"  草稿 ID: {best['draft_id']}")
        print(f"  得分: {best['overall_score']:.1f}/100")

        return best


# ==================== 使用示例 ====================

def example_basic_evaluation():
    """示例 1: 基础评估"""
    print("=" * 80)
    print("示例 1: 基础草稿评估")
    print("=" * 80)

    # 1. 配置 Judge Agent
    config = MODEL_CONFIGS["kimi-k2"]
    judge = JudgeAgent(config)

    # 2. 模拟草稿
    draft = """
    Transformer 架构 [paper_001] 自 2017 年提出以来，已经成为自然语言处理领域的基础架构。
    该架构完全基于自注意力机制，摒弃了传统的循环神经网络和卷积神经网络。
    随后，BERT [paper_002] 提出了双向预训练方法，进一步提升了预训练模型的性能。
    GPT-3 [paper_003] 则展示了大规模预训练模型的少样本学习能力，参数量达到 175B。
    这些工作共同推动了大语言模型的快速发展。
    """

    # 3. 评估
    evaluation = judge.evaluate_draft(draft)

    # 4. 输出结果
    print("\n" + "=" * 80)
    print("评估结果：")
    print("=" * 80)
    print(f"\n总分: {evaluation['overall_score']:.1f}/100\n")

    print("各维度得分:")
    for dim, score in evaluation.get("scores", {}).items():
        print(f"  - {dim}: {score}/100")

    print(f"\n优点:")
    for strength in evaluation.get("strengths", []):
        print(f"  ✓ {strength}")

    print(f"\n缺点:")
    for weakness in evaluation.get("weaknesses", []):
        print(f"  ✗ {weakness}")

    print(f"\n改进建议:")
    for suggestion in evaluation.get("improvement_suggestions", []):
        print(f"  → {suggestion.get('issue', '')}")
        print(f"    建议: {suggestion.get('suggestion', '')}")


def example_ranking():
    """示例 2: 多草稿排序"""
    print("\n\n" + "=" * 80)
    print("示例 2: 多个草稿排序")
    print("=" * 80)

    config = MODEL_CONFIGS["kimi-k2"]
    judge = JudgeAgent(config)

    # 模拟 3 个不同质量的草稿
    drafts = [
        "Transformer [paper_001] 是一个重要的架构。BERT [paper_002] 也很重要。",  # 低质量
        "Transformer 架构 [paper_001] 提出了自注意力机制，BERT [paper_002] 改进了预训练方法。",  # 中等
        "Transformer 架构 [paper_001] 自 2017 年提出以来已成为 NLP 基础。它完全基于注意力机制，摒弃了 RNN。BERT [paper_002] 通过双向预训练进一步提升性能。"  # 高质量
    ]

    # 排序
    ranked = judge.rank_drafts(drafts)

    # 输出
    print("\n" + "=" * 80)
    print("排序结果详情：")
    print("=" * 80)
    for i, result in enumerate(ranked):
        print(f"\n第 {i+1} 名:")
        print(f"  草稿 ID: {result['draft_id']}")
        print(f"  得分: {result['overall_score']:.1f}/100")
        print(f"  草稿内容: {result['draft'][:100]}...")


def example_rejection_sampling():
    """示例 3: 拒绝采样"""
    print("\n\n" + "=" * 80)
    print("示例 3: 拒绝采样")
    print("=" * 80)

    config = MODEL_CONFIGS["kimi-k2"]
    judge = JudgeAgent(config)

    drafts = [
        "简短草稿 [paper_001]。",
        "Transformer [paper_001] 是重要架构，BERT [paper_002] 改进了预训练。",
        "完整的综述内容，包含详细分析 [paper_001][paper_002][paper_003]。",
    ]

    # 拒绝采样（只保留得分 >= 70 的，最多 2 个）
    selected = judge.rejection_sampling(
        drafts,
        threshold=70.0,
        max_keep=2
    )

    print("\n" + "=" * 80)
    print("保留的高质量草稿：")
    print("=" * 80)
    for result in selected:
        print(f"\n草稿 {result['draft_id']}: {result['overall_score']:.1f}/100")


def example_factuality_check():
    """示例 4: 事实验证"""
    print("\n\n" + "=" * 80)
    print("示例 4: 事实准确性验证")
    print("=" * 80)

    config = MODEL_CONFIGS["kimi-k2"]
    judge = JudgeAgent(config)

    draft = """
    Transformer [paper_001] 于 2017 年提出。
    BERT [paper_002] 使用了双向预训练。
    GPT-5 [paper_999] 有 1000B 参数。  # 这是错误的引用
    """

    evidence = {
        "paper_001": {"title": "Attention is All You Need", "year": 2017},
        "paper_002": {"title": "BERT", "year": 2019},
        # paper_999 不存在
    }

    verification = judge.verify_factuality(draft, evidence)

    print("\n" + "=" * 80)
    print("事实验证结果：")
    print("=" * 80)
    print(f"准确率: {verification.get('accuracy_rate', 0):.1%}")
    print(f"引用有效率: {verification['citation_check']['citation_validity_rate']:.1%}")
    print(f"无效引用: {verification['citation_check']['invalid_citations']}")


# ==================== 主函数 ====================

def main():
    """运行所有示例"""
    # 示例 1: 基础评估
    example_basic_evaluation()

    # 示例 2: 多草稿排序
    # example_ranking()

    # 示例 3: 拒绝采样
    # example_rejection_sampling()

    # 示例 4: 事实验证
    # example_factuality_check()


if __name__ == "__main__":
    main()
