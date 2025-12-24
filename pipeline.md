# 学术综述自动生成与评估流水线

本仓库提供端到端流程：Writing Agent 生成多温度、多样性的候选综述草稿，Judge Agent 将 SurveyGen 的自动评估指标与多 LLM 打分融合，对所有草稿排序并选出最佳结果。

## 目录
- `writing_agent.py`：生成综述草稿，支持固定温度批量生成。
- `judge_agent.py`：多维度评分，支持自动指标 + 多模型融合。
- `pipeline_adapter.py`：将检索/分析阶段 JSON 转成写作输入。
- `test_full_pipeline.py`：一键运行完整流程。

## 快速开始
1. 安装依赖
   ```bash
   pip install openai
   ```
2. 配置模型（示例使用 Kimi，支持自行替换）
   - 修改 `CONFIG`（Writing Agent & Judge Agent）中的 `api_key`、`base_url`、`name`。
3. 运行全流程
   ```bash
   python test_full_pipeline.py              # 使用内置示例数据
   python test_full_pipeline.py --analysis-json /path/to/analysis.json  # 接入前序检索/分析结果
   ```

## 写作阶段（Writing Agent）
- 输入：`cluster_summaries`（可由 `pipeline_adapter.analysis_to_cluster_summaries` 生成）。
- 生成策略：温度 0.3 / 0.4 / 0.5，各生成 3 篇，共 9 篇候选。
- 关键接口：`WritingAgent.generate_candidates_by_temps(cluster_summaries, temps=[0.3,0.4,0.5], per_temp=3)`。
- 输出：每篇草稿包含正文、引用列表、温度、候选 ID。

## 评估阶段（Judge Agent）
- 评分维度（LLM 主观）：覆盖度、准确性、连贯性、学术性、新颖性。
- 自动指标（SurveyGen 对齐）：
  - 引文质量：Precision / Recall / F1 / Accuracy（基于候选引用与 `cluster_summaries` 的 paper_id 对齐）。
  - 内容、结构指标可在有金标时扩展（占位保留）。
- 融合方式：`JudgeAgent.multi_model_vote_with_auto_metrics` 将多模型 LLM 均分与自动指标（0-100 归一化）按权重合成最终得分。
- 默认模型列表：`[CONFIG]`，可在 `test_full_pipeline.py` 中追加更多模型配置实现多 LLM 投票。

## 流水线输出
- 排序表：显示每个候选的最终分、LLM 均分、自动分、温度。
- 最佳草稿：输出 ID、温度、最终分、引文指标，以及正文前 500 字预览。

## 自定义与扩展
- **添加更多模型**：在 `test_full_pipeline.py` 的 `model_configs` 列表中追加 `ModelConfig`。
- **调整融合权重**：调用 `multi_model_vote_with_auto_metrics(..., auto_weight=0.3~0.7)`。
- **对接真实金标**：在 `compute_auto_metrics` 中加入语义相似度、ROUGE-L、KPR、结构 overlap 等指标。

## 注意事项
- 需要有效的 LLM API Key 与可访问的 `base_url`。
- 生成和评分调用均消耗 token，请在批量运行前确认配额。

