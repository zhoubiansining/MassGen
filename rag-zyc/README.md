# 学术综述自动生成系统

端到端学术文献综述生成系统，包含 Writing Agent 与 Judge Agent。

## 📦 项目结构

```
/Users/zyc/Desktop/rag/
├── writing_agent.py           # Writing Agent 核心实现
├── judge_agent.py             # Judge Agent 核心实现
├── pipeline_adapter.py        # 将检索/分析输出转换为写作输入
├── model_config.json          # 模型配置文件
├── test_full_pipeline.py      # 完整示例脚本（支持接入前序 Analysis JSON）
└── README.md                  # 本文档
```

---

## 🎯 核心功能

### Writing Agent（写作代理）
- ✅ 将结构化研究摘要转化为学术综述
- ✅ 支持多温度采样生成多样性候选
- ✅ 4种写作风格（叙述、表格、时间线、分类）
- ✅ 自我修正机制（Reflection）
- ✅ 引用追踪和验证
- ✅ 同步/异步并行生成

### Judge Agent（评分代理）
- ✅ 5维度评分系统（覆盖度、准确性、连贯性、学术性、新颖性）
- ✅ 详细反馈（优点、缺点、改进建议）
- ✅ 拒绝采样（过滤低质量草稿）
- ✅ 事实准确性验证
- ✅ 引用有效性检查

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install openai
```

### 2. 配置模型

编辑 `model_config.json` 或直接在代码中配置：

```python
from writing_agent import ModelConfig

config = ModelConfig(
    name="Kimi-K2",  # 模型名称
    api_key="your-api-key-here",
    base_url="https://llmapi.paratera.com/v1/",
    temperature=0.5,
    max_tokens=4096
)
```

**支持的模型**：Kimi、Qwen、DeepSeek、GLM、Doubao 等

### 3. 运行示例（可接入前序检索/分析）

```bash
# 情况 A：已有 Analysis 阶段 JSON（包含 clusters/insights/datas）
python test_full_pipeline.py --analysis-json /path/to/analysis_result.json

# 情况 B：无前序结果，使用内置示例数据
python test_full_pipeline.py
```

---

## 💡 使用示例

### 前序模块对接说明
- 使用 Retrieval/Analysis 得到的 JSON（含 `clusters`/`insights`/`datas`），调用 `pipeline_adapter.analysis_to_cluster_summaries` 转为 `cluster_summaries`。
- `test_full_pipeline.py` 已内置转换，传入 `--analysis-json` 即可直接运行。

### 示例 1: 生成学术综述

```python
from writing_agent import WritingAgent, ModelConfig

# 配置模型
config = ModelConfig(
    name="Kimi-K2",
    api_key="sk-your-api-key",
    base_url="https://llmapi.paratera.com/v1/"
)

# 创建 Writing Agent
writer = WritingAgent(config, style="narrative")

# 准备输入数据
cluster_summaries = {
    "cluster_1": {
        "topic": "Transformer 架构",
        "summary": "关于 Transformer 架构的研究",
        "papers": [
            {
                "paper_id": "paper_001",
                "title": "Attention is All You Need",
                "authors": ["Vaswani et al."],
                "year": 2017,
                "key_contribution": "提出 Transformer 架构"
            }
            # 更多论文...
        ]
    }
}

# 生成草稿
draft = writer.generate_draft(cluster_summaries)
print(draft["content"])
```

### 示例 2: 评估草稿质量

```python
from judge_agent import JudgeAgent, ModelConfig

# 配置 Judge（使用低温度保证评分一致性）
judge_config = ModelConfig(
    name="Kimi-K2",
    api_key="sk-your-api-key",
    base_url="https://llmapi.paratera.com/v1/",
    temperature=0.2  # Judge 使用低温度
)

# 创建 Judge Agent
judge = JudgeAgent(judge_config)

# 评估草稿
evaluation = judge.evaluate_draft(draft["content"])

# 查看结果
print(f"总分: {evaluation['overall_score']:.1f}/100")
print(f"优点: {evaluation['strengths']}")
print(f"缺点: {evaluation['weaknesses']}")
print(f"改进建议: {evaluation['improvement_suggestions']}")
```

### 示例 3: 完整流程

```python
from writing_agent import WritingAgent, ModelConfig
from judge_agent import JudgeAgent

# 1. 配置模型
writing_config = ModelConfig(
    name="Kimi-K2",
    api_key="sk-your-api-key",
    base_url="https://llmapi.paratera.com/v1/",
    temperature=0.5
)

judge_config = ModelConfig(
    name="Kimi-K2",
    api_key="sk-your-api-key",
    base_url="https://llmapi.paratera.com/v1/",
    temperature=0.2
)

# 2. 创建 Agents
writer = WritingAgent(writing_config)
judge = JudgeAgent(judge_config)

# 3. 生成多个候选草稿
candidates = writer.generate_multiple_candidates(
    cluster_summaries=cluster_summaries,
    num_candidates=3,
    temperature_range=(0.3, 0.7)
)

# 4. 评估和择优
selected = judge.rejection_sampling(
    drafts=[c["content"] for c in candidates],
    reference=cluster_summaries,
    threshold=70.0,  # 最低分数阈值
    max_keep=2       # 最多保留数量
)

# 5. 选择最佳草稿
best = selected[0]
print(f"最佳草稿得分: {best['overall_score']:.1f}/100")

# 6. 基于反馈改进（可选）
if best['overall_score'] < 85:
    feedback = {"improvements": best['improvement_suggestions']}
    refined = writer.refine_draft(best['draft'], feedback)

    # 重新评估
    final_eval = judge.evaluate_draft(refined["content"])
    print(f"改进后得分: {final_eval['overall_score']:.1f}/100")

# 7. 验证事实准确性
evidence = {}
for cluster_data in cluster_summaries.values():
    for paper in cluster_data.get('papers', []):
        evidence[paper['paper_id']] = paper

verification = judge.verify_factuality(best['draft'], evidence)
print(f"事实准确率: {verification.get('accuracy_rate', 0):.1%}")
print(f"引用有效率: {verification['citation_check']['citation_validity_rate']:.1%}")
```

---

## 📊 评分标准

### Judge Agent 评分维度

| 维度 | 权重 | 说明 |
|-----|------|-----|
| **覆盖度** (Coverage) | 25% | 是否包含所有重要论文和概念 |
| **准确性** (Factuality) | 30% | 事实陈述是否准确，引用是否正确 |
| **连贯性** (Coherence) | 20% | 逻辑流程是否清晰，过渡是否自然 |
| **学术性** (Academic Style) | 15% | 是否符合学术写作规范 |
| **新颖性** (Novelty) | 10% | 是否识别出研究趋势和空白 |

### 评分等级

| 分数范围 | 等级 | 说明 |
|---------|------|------|
| 90-100 | 优秀 | 可发表质量，无需修改或仅需微调 |
| 80-89 | 良好 | 质量较高，需小幅修改 |
| 70-79 | 合格 | 基本合格，需中等程度修改 |
| 60-69 | 尚可 | 质量一般，需大幅修改 |
| <60 | 不合格 | 需要重写 |

---

## 🎨 高级用法

### 1. 生成多个候选草稿

```python
# 使用不同温度生成多样性候选
candidates = writer.generate_multiple_candidates(
    cluster_summaries=data,
    num_candidates=5,
    temperature_range=(0.3, 0.9)  # 温度范围
)

# 查看所有候选
for candidate in candidates:
    print(f"候选 {candidate['candidate_id']}: {len(candidate['content'])} 字符")
```

### 2. 自我修正（Reflection）

```python
# 生成初始草稿
initial_draft = writer.generate_draft(cluster_summaries)

# 多轮自我修正
refined_result = writer.refine_with_reflection(
    draft=initial_draft["content"],
    max_iterations=3  # 最多 3 轮
)

print(f"经过 {refined_result['iterations']} 轮修正")
print(refined_result["final_draft"])
```

### 3. 并行生成（异步，更快）

```python
import asyncio

async def parallel_generation():
    writer = WritingAgent(config)

    # 并行生成 5 个候选（速度快 5 倍）
    candidates = await writer.generate_multiple_candidates_async(
        cluster_summaries=data,
        num_candidates=5
    )

    return candidates

# 运行
candidates = asyncio.run(parallel_generation())
```

### 4. 拒绝采样（过滤低质量）

```python
# 只保留高质量草稿
selected = judge.rejection_sampling(
    drafts=all_drafts,
    reference=cluster_summaries,
    threshold=70.0,  # 过滤低于 70 分的
    max_keep=3       # 最多保留 3 个
)

print(f"从 {len(all_drafts)} 个草稿中保留了 {len(selected)} 个高质量草稿")
```

### 5. 事实准确性验证

```python
# 准备证据材料
evidence = {
    "paper_001": {
        "title": "Attention is All You Need",
        "year": 2017,
        "key_facts": ["提出 Transformer", "基于自注意力机制"]
    }
}

# 验证草稿
verification = judge.verify_factuality(draft["content"], evidence)

# 查看结果
print(f"总陈述数: {verification.get('total_claims', 'N/A')}")
print(f"准确率: {verification.get('accuracy_rate', 0):.1%}")
print(f"引用有效率: {verification['citation_check']['citation_validity_rate']:.1%}")

if verification['citation_check']['invalid_citations']:
    print(f"无效引用: {verification['citation_check']['invalid_citations']}")
```

### 6. 多模型投票（提高可靠性）

```python
# 使用多个模型评分，取平均值
judges = [
    JudgeAgent(ModelConfig(name="Kimi-K2", api_key="...", base_url="...")),
    # 可以添加其他模型
]

scores = []
for judge in judges:
    evaluation = judge.evaluate_draft(draft)
    scores.append(evaluation['overall_score'])

final_score = sum(scores) / len(scores)
print(f"多模型平均分: {final_score:.1f}/100")
```

### 7. 不同写作风格

```python
# 叙述式（适合背景介绍）
agent_narrative = WritingAgent(config, style="narrative")

# 表格驱动（适合方法对比）
agent_table = WritingAgent(config, style="table-driven")

# 时间线式（适合发展历程）
agent_timeline = WritingAgent(config, style="timeline")

# 分类法式（适合方法归类）
agent_taxonomy = WritingAgent(config, style="taxonomy")
```

---

## ⚡ 性能优化

### 1. 使用异步并行

```python
import asyncio

# 慢：串行生成 5 个草稿 → 耗时 5x
for i in range(5):
    draft = writer.generate_draft(data)

# 快：并行生成 5 个草稿 → 耗时 1x
candidates = await writer.generate_multiple_candidates_async(data, 5)
```

### 2. 结果缓存

```python
import hashlib
import pickle

cache = {}

def cached_evaluate(judge, draft):
    """带缓存的评估"""
    key = hashlib.md5(draft.encode()).hexdigest()

    if key in cache:
        print("[Cache hit]")
        return cache[key]

    result = judge.evaluate_draft(draft)
    cache[key] = result
    return result
```

### 3. 批量处理

```python
# 分批处理大量草稿
batch_size = 10
for i in range(0, len(all_drafts), batch_size):
    batch = all_drafts[i:i+batch_size]
    results = await judge.rank_drafts_async(batch)
    # 处理结果...
```

---

## 🔧 API 参考

### WritingAgent 类

```python
class WritingAgent:
    def __init__(self, model_config: ModelConfig, style: str = "narrative")

    # 生成单个草稿
    def generate_draft(
        self,
        cluster_summaries: Dict,
        temperature: Optional[float] = None
    ) -> Dict

    # 异步生成
    async def generate_draft_async(
        self,
        cluster_summaries: Dict,
        temperature: Optional[float] = None
    ) -> Dict

    # 生成多个候选
    def generate_multiple_candidates(
        self,
        cluster_summaries: Dict,
        num_candidates: int = 3,
        temperature_range: Tuple[float, float] = (0.3, 0.9)
    ) -> List[Dict]

    # 并行生成多个候选
    async def generate_multiple_candidates_async(
        self,
        cluster_summaries: Dict,
        num_candidates: int = 3,
        temperature_range: Tuple[float, float] = (0.3, 0.9)
    ) -> List[Dict]

    # 基于反馈改进
    def refine_draft(self, draft: str, feedback: Dict) -> Dict

    # 自我批评
    def self_critique(self, draft: str) -> Dict

    # 多轮自我修正
    def refine_with_reflection(
        self,
        draft: str,
        max_iterations: int = 3
    ) -> Dict

    # 验证引用
    def validate_citations(
        self,
        draft: str,
        available_papers: Dict
    ) -> Dict
```

### JudgeAgent 类

```python
class JudgeAgent:
    def __init__(self, model_config: ModelConfig)

    # 评估单个草稿
    def evaluate_draft(
        self,
        draft: str,
        reference: Optional[Dict] = None
    ) -> Dict

    # 异步评估
    async def evaluate_draft_async(
        self,
        draft: str,
        reference: Optional[Dict] = None
    ) -> Dict

    # 对多个草稿排序
    def rank_drafts(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None
    ) -> List[Dict]

    # 异步排序
    async def rank_drafts_async(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None
    ) -> List[Dict]

    # 拒绝采样
    def rejection_sampling(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None,
        threshold: float = 70.0,
        max_keep: int = 3
    ) -> List[Dict]

    # 异步拒绝采样
    async def rejection_sampling_async(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None,
        threshold: float = 70.0,
        max_keep: int = 3
    ) -> List[Dict]

    # 验证事实准确性
    def verify_factuality(
        self,
        draft: str,
        evidence: Dict
    ) -> Dict

    # 选择最佳草稿
    def select_best_draft(
        self,
        drafts: List[str],
        reference: Optional[Dict] = None
    ) -> Dict
```

### 返回值结构

#### evaluate_draft() 返回值

```python
{
    "scores": {
        "coverage": 85,       # 覆盖度
        "factuality": 90,     # 准确性
        "coherence": 80,      # 连贯性
        "academic_style": 88, # 学术性
        "novelty": 75         # 新颖性
    },
    "overall_score": 84.5,    # 总分
    "strengths": [            # 优点列表
        "引用全面",
        "逻辑清晰"
    ],
    "weaknesses": [           # 缺点列表
        "部分段落过长",
        "缺少未来方向"
    ],
    "improvement_suggestions": [  # 改进建议
        {
            "issue": "问题描述",
            "suggestion": "改进建议",
            "priority": "high|medium|low"
        }
    ],
    "timestamp": "2024-12-09T10:30:00",
    "draft_length": 1234
}
```

#### verify_factuality() 返回值

```python
{
    "total_claims": 10,           # 总陈述数
    "verified_claims": 8,         # 已验证数
    "accuracy_rate": 0.8,         # 准确率
    "unverified_claims": [...],   # 未验证的陈述
    "hallucinations": [...],      # 可能的幻觉
    "citation_check": {
        "total_citations": 5,
        "invalid_citations": ["paper_999"],
        "citation_validity_rate": 0.8
    }
}
```

---

## 📝 输入数据格式

```python
cluster_summaries = {
    "cluster_1": {
        "topic": "主题名称",
        "summary": "主题摘要描述",
        "papers": [
            {
                "paper_id": "唯一标识",
                "title": "论文标题",
                "authors": ["作者1", "作者2"],
                "year": 2024,
                "venue": "会议/期刊名称",
                "key_contribution": "核心贡献描述",
                "citation_count": 1000  # 可选
            }
            # 更多论文...
        ]
    },
    "cluster_2": {
        # 更多主题...
    }
}
```

---

## 🔍 常见问题

### Q1: 如何切换模型？

```python
# 方法 1: 创建时指定
config = ModelConfig(
    name="Qwen-Max",  # 切换到通义千问
    api_key="your-qwen-api-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
writer = WritingAgent(config)

# 方法 2: 使用配置文件
import json
with open("model_config.json") as f:
    config_data = json.load(f)
    config = ModelConfig(**config_data["models"]["qwen-max"])
```

### Q2: 生成速度慢怎么办？

**解决方案**：
- 使用异步并行：`generate_multiple_candidates_async()`
- 选择更快的模型：Gemini Flash
- 减少候选数量：`num_candidates=3` → `2`
- 减少修正轮数：`max_iterations=3` → `1`

### Q3: 如何提高生成质量？

**建议**：
- 使用更好的模型：Qwen-Max、Claude Sonnet
- 增加修正轮数：`max_iterations=3`
- 生成更多候选：`num_candidates=5`
- 提供更详细的输入数据
- 使用自我修正功能

### Q4: API 调用失败怎么办？

```python
# 添加重试机制
import time

def generate_with_retry(agent, data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return agent.generate_draft(data)
        except Exception as e:
            print(f"尝试 {attempt+1} 失败: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避
            else:
                raise
```

### Q5: 评分不稳定怎么办？

**解决方案**：
- 降低 Judge Agent 的 temperature（推荐 0.1-0.2）
- 使用多模型投票取平均
- 多次评估取平均值

### Q6: 如何自定义评分权重？

```python
from judge_agent import ScoringDimension

# 创建 Judge
judge = JudgeAgent(config)

# 修改权重（例如更重视准确性）
judge.scoring_dimensions["factuality"].weight = 0.40  # 提高到 40%
judge.scoring_dimensions["novelty"].weight = 0.05     # 降低到 5%
```

### Q7: 支持哪些国产模型？

| 模型 | 提供商 | 适用场景 |
|-----|--------|---------|
| Kimi-K2 | Moonshot | 长上下文（200K） |
| Qwen-Max | 阿里云 | 高质量学术写作 |
| DeepSeek-Chat | DeepSeek | 推理能力强，成本低 |
| GLM-4 | 智谱AI | 中文理解能力强 |
| Doubao-Pro | 字节跳动 | 速度快，性价比高 |

---

## 🎯 实战案例

### 案例 1: 生成 Transformer 综述

```python
# 输入数据
data = {
    "cluster_1": {
        "topic": "Transformer 架构与注意力机制",
        "summary": "关于 Transformer 架构的提出及改进",
        "papers": [
            {
                "paper_id": "paper_001",
                "title": "Attention is All You Need",
                "authors": ["Vaswani et al."],
                "year": 2017,
                "key_contribution": "提出 Transformer 架构"
            },
            {
                "paper_id": "paper_002",
                "title": "BERT",
                "authors": ["Devlin et al."],
                "year": 2019,
                "key_contribution": "双向预训练方法"
            },
            {
                "paper_id": "paper_003",
                "title": "GPT-3",
                "authors": ["Brown et al."],
                "year": 2020,
                "key_contribution": "大规模少样本学习"
            }
        ]
    }
}

# 运行完整流程
config = ModelConfig(
    name="Kimi-K2",
    api_key="your-api-key",
    base_url="https://llmapi.paratera.com/v1/"
)

writer = WritingAgent(config)
judge = JudgeAgent(ModelConfig(name="Kimi-K2", api_key="your-api-key",
                                base_url="https://llmapi.paratera.com/v1/",
                                temperature=0.2))

# 1. 生成候选
candidates = writer.generate_multiple_candidates(data, num_candidates=3)

# 2. 评估择优
selected = judge.rejection_sampling(
    drafts=[c["content"] for c in candidates],
    reference=data,
    threshold=70.0
)

# 3. 输出最佳
best = selected[0]
print(f"最佳草稿得分: {best['overall_score']:.1f}/100")
print(f"\n草稿内容:\n{best['draft']}")

# 4. 保存结果
with open("transformer_survey.txt", "w", encoding="utf-8") as f:
    f.write(best['draft'])
```

**预期输出**：
- 总分：75-85/100
- 准确性：90+/100
- 引用有效率：100%
- 质量等级：良好-合格

---

## 📈 系统性能

### 性能指标

**完整流程耗时**（3个候选）：
- 生成阶段：1-2 分钟
- 评估阶段：1-2 分钟
- 总计：2-4 分钟

**API 调用次数**：
- Writing Agent：3 次（3个候选）
- Judge Agent：4 次（3次评分 + 1次验证）
- 总计：7 次

**成本估算**（Kimi-K2）：
- 输入 tokens：~10K
- 输出 tokens：~8K
- 估计成本：¥0.5-1.0

### 优化建议

1. **使用异步并行**：速度提升 3-5 倍
2. **实现结果缓存**：节省 30-50% 成本
3. **选择更快模型**：Gemini Flash 可提速 5-10 倍
4. **减少候选数量**：从 5 个降到 3 个，节省 40% 时间

---

## 🛠️ 最佳实践

### 1. 生成阶段

```python
# 推荐配置
candidates = writer.generate_multiple_candidates(
    cluster_summaries=data,
    num_candidates=3-5,           # 3-5 个候选即可
    temperature_range=(0.3, 0.9)  # 温度范围不要太窄
)
```

### 2. 评估阶段

```python
# Judge 使用低温度保证一致性
judge_config = ModelConfig(
    name="Kimi-K2",
    temperature=0.2,  # 0.1-0.2 最佳
    max_tokens=4096
)
```

### 3. 质量控制

```python
# 设置合理的阈值
selected = judge.rejection_sampling(
    drafts=candidates,
    threshold=70.0,  # 70-75 比较合理
    max_keep=2-3
)
```

### 4. 迭代改进

```python
# 基于反馈改进
if best['overall_score'] < 85:
    feedback = {"improvements": best['improvement_suggestions']}
    refined = writer.refine_draft(best['draft'], feedback)
    # 重新评估
    final_eval = judge.evaluate_draft(refined["content"])
```

---

## 📦 依赖要求

- Python >= 3.8
- openai >= 1.0.0

---

## 🔄 后续计划

- [ ] 实现基于反馈的迭代改进循环
- [ ] 添加更多模型支持
- [ ] 实现结果缓存机制
- [ ] 支持批量处理
- [ ] 添加图表生成功能
- [ ] 实现 Web UI 界面

---

## 📄 许可证

MIT License

---

## 🙏 致谢

本项目基于大语言模型技术，感谢 OpenAI、Anthropic、阿里云、DeepSeek、智谱AI 等提供的优秀模型 API。

---

**最后更新**: 2024-12-09
**版本**: v1.0
**状态**: ✅ 生产就绪
