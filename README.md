# MassGen: Multi-Agent System for Survey Generation

**MassGen**（Multi-Agent System for Survey Generation）是一个基于多智能体协作的系统，旨在自动生成高质量的学术综述文章。该系统通过多个专业化智能体的协作，实现从文献检索、内容分析到最终撰写和评审的全流程自动化。

## 🎯 系统概述

MassGen旨在解决以下核心问题：

- 手动撰写调研报告耗时且容易遗漏关键信息
- 需要高效整合大量文献资源并进行结构化输出
- 不同任务（如检索、分析、写作）需由专业化模块分工完成

系统通过四个核心智能体的协作，实现从原始学术文献到结构化综述文章的自动转换。

## 🏗️ 系统架构

### 系统架构图

下图展示了MassGen系统的整体架构和数据流：

![MassGen Framework](assets/MassGen.png)

### 智能体组成

MassGen由四个主要智能体组成，每个智能体负责特定的任务：

#### 1. Retrieval Agent（检索智能体）

- **职责**：使用"Scout-and-Strike"策略收集高质量学术论文
- **策略**：
  - **Scout阶段**：使用Google搜索找到"Awesome lists"、综述博客或GitHub仓库
  - **Strike阶段**：使用Hybrid Graph Search进行精确检索
- **目标**：收集30+篇高质量学术论文

#### 2. Analysis Agent（分析智能体）

采用三阶段工作流设计：

- **信息提取模块**：通过结构化提示词从学术论文中抽取关键信息，包括研究问题、方法论、数据集、实验结果、限制和关键词
- **聚类与组织模块**：使用SciBERT生成嵌入表示，采用X-means/K-means进行自适应聚类
- **综合与全局推理模块**：生成簇摘要并构建整体研究领域视图

#### 3. Writing Agent（写作智能体）

- **职责**：基于分析结果生成结构化综述文章
- **特点**：
  - 支持多种写作风格
  - 可生成多个候选草稿
  - 实现引用验证功能
  - 支持自我修正机制

#### 4. Judge Agent（评审智能体）

- **职责**：对生成的草稿进行质量评估
- **评估维度**：准确性、清晰性、完整性、引用质量等
- **功能**：支持多模型投票和阈值筛选

### 技术方法

#### 信息提取与聚类

- 使用SciBERT对学术文献进行语义嵌入
- 采用X-means算法自动确定最优聚类数，或使用K-means作为备选
- 聚类数量采用启发式方法：$k \approx \sqrt{n/2}$，约束在2到8之间

#### 自适应工具集成

- Analysis Agent通过[ClusterContextTool]实现动态工具集成
- 允许智能体在推理过程中动态访问和检查聚类信息

#### 多候选生成与评估

- Writing Agent可生成多个不同温度设置的候选草稿
- Judge Agent结合自动指标和多LLM评分进行综合评估
- 采用拒绝采样方法选择最佳草稿

## 🚀 使用方法

### 完整流程示例

```python
from writing_agent import WritingAgent, ModelConfig
from judge_agent import JudgeAgent
from pipeline_adapter import analysis_to_cluster_summaries

# 配置模型
config = ModelConfig(
    name="Kimi-K2",
    api_key="your-api-key",
    base_url="https://llmapi.paratera.com/v1/",
    temperature=0.5
)

# 创建写作和评审智能体
writer = WritingAgent(config, style="narrative")
judge = JudgeAgent(ModelConfig(
    name="Kimi-K2", 
    api_key="your-api-key",
    base_url="https://llmapi.paratera.com/v1/",
    temperature=0.2  # 评审使用低温度以保证一致性
))

# 1. 生成多个候选草稿
cluster_summaries = {
    "cluster_1": {
        "topic": "Transformer 架构与注意力机制",
        "summary": "本主题涵盖了 Transformer 架构的提出及其注意力机制的改进方法。",
        "papers": [
            {
                "paper_id": "paper_001",
                "title": "Attention is All You Need",
                "authors": ["Vaswani, A.", "Shazeer, N."],
                "year": 2017,
                "key_contribution": "提出了原始 Transformer 架构，基于自注意力机制",
            },
            # 更多论文...
        ]
    }
}

candidates = writer.generate_multiple_candidates(
    cluster_summaries=cluster_summaries,
    num_candidates=3
)

# 2. 评审和选择最佳草稿
selected = judge.rejection_sampling(
    drafts=[c["content"] for c in candidates],
    reference=cluster_summaries,
    threshold=70.0
)

# 3. 输出最佳结果
best = selected[0]
print(f"最佳草稿得分: {best['overall_score']:.1f}/100")
print(f"\n草稿内容:\n{best['content']}")

# 4. 保存结果
with open("survey_output.txt", "w", encoding="utf-8") as f:
    f.write(best['content'])
```

### 主要工作流程

1. **文献检索**：Retrieval Agent使用Scout-and-Strike策略收集学术文献
2. **内容分析**：Analysis Agent执行三阶段分析（信息提取→聚类→综合推理）
3. **草稿生成**：Writing Agent基于分析结果生成多个候选草稿
4. **质量评估**：Judge Agent对候选草稿进行评分和筛选
5. **最终输出**：选择最佳草稿作为最终结果

## 🛠️ 技术特点

### 优势

- **模块化设计**：各智能体职责分离，便于维护和扩展
- **自适应聚类**：使用科学领域专用模型和自适应聚类算法
- **多候选生成**：生成多个候选并选择最优结果
- **质量控制**：多维度评估和引用验证机制

### 性能指标

- **完整流程耗时**（3个候选）：2-4分钟
- **API调用次数**：约7次（写作3次+评审4次）
- **成本估算**：约¥0.5-1.0（基于Kimi-K2模型）

## 📊 评估指标

系统采用多种指标评估生成质量：

- **自动指标**：引用质量（精度/召回/F1）、内容和结构指标
- **主观评估**：准确性、清晰性、完整性、新颖性
- **多模型投票**：结合多个LLM的评分进行综合评估

## 🤝 贡献

欢迎提交Issue和Pull Request来帮助改进MassGen系统。

## 📄 许可证

本项目采用MIT许可证。
