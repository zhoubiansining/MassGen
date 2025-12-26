"""
完整流程：按固定温度批量生成候选草稿 + SurveyGen 自动指标 + 多 LLM 评分融合，输出最佳草稿。
"""

import argparse
import json
import os
from typing import Dict, List, Optional

from writing_agent import WritingAgent, ModelConfig
from judge_agent import JudgeAgent
from pipeline_adapter import analysis_to_cluster_summaries


CONFIG = ModelConfig(
    name="Kimi-K2",
    api_key="sk-aRG9iu2Hy9--oPxrG-5faA",
    base_url="https://llmapi.paratera.com/v1/",
    temperature=0.5,
    max_tokens=4096,
)


DEFAULT_TEST_DATA = {
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
            {
                "paper_id": "paper_002",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Devlin, J.", "Chang, M. W."],
                "year": 2019,
                "key_contribution": "双向预训练方法，Masked Language Model",
            },
            {
                "paper_id": "paper_003",
                "title": "GPT-3: Language Models are Few-Shot Learners",
                "authors": ["Brown, T.", "Mann, B."],
                "year": 2020,
                "key_contribution": "大规模预训练模型的少样本学习能力",
            },
        ],
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="写作 + 评分融合流水线")
    parser.add_argument("--analysis-json", dest="analysis_json", help="可选，前序 analysis 输出 JSON 路径")
    return parser.parse_args()


def load_cluster_summaries(analysis_json: Optional[str]):
    if analysis_json:
        if not os.path.exists(analysis_json):
            raise FileNotFoundError(f"找不到 analysis_json 文件: {analysis_json}")
        with open(analysis_json, "r", encoding="utf-8") as f:
            analysis_result = json.load(f)
        print(f"\n📥 已读取 {analysis_json}，转换为 cluster_summaries")
        return analysis_to_cluster_summaries(analysis_result)
    print("\n📥 未提供 analysis_json，使用内置示例数据")
    return DEFAULT_TEST_DATA


def compute_auto_metrics(candidate: Dict, cluster_summaries: Dict) -> Dict:
    """基于 SurveyGen 指标：目前实现引文质量（精度/召回/F1/准确率），内容与结构指标留空待对接。"""
    human_refs = {
        paper.get("paper_id")
        for cluster in cluster_summaries.values()
        for paper in cluster.get("papers", [])
        if paper.get("paper_id")
    }
    pred_refs = set(candidate.get("citations", []))

    matches = len(pred_refs & human_refs)
    prec = matches / len(pred_refs) if pred_refs else 0.0
    rec = matches / len(human_refs) if human_refs else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

    return {
        "citation": {
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "accuracy": prec,
        },
        "content": {},
        "structure": {},
    }


def main():
    args = parse_args()
    cluster_summaries = load_cluster_summaries(args.analysis_json)

    # 1) 生成候选草稿：温度 0.3/0.4/0.5 各 3 篇，共 9 篇
    writer = WritingAgent(CONFIG, style="narrative")
    temps = [0.3, 0.4, 0.5]
    candidates = writer.generate_candidates_by_temps(
        cluster_summaries=cluster_summaries,
        temps=temps,
        per_temp=3,
    )

    print(f"生成完成：共 {len(candidates)} 篇候选草稿")

    # 2) 评分融合：SurveyGen 自动指标 + 多 LLM 主观评分
    model_configs: List[ModelConfig] = [CONFIG]  # 可在此处追加更多模型
    ranked = []
    for cand in candidates:
        auto_metrics = compute_auto_metrics(cand, cluster_summaries)
        report = JudgeAgent.multi_model_vote_with_auto_metrics(
            draft=cand["content"],
            model_configs=model_configs,
            reference=cluster_summaries,
            auto_metrics=auto_metrics,
            auto_weight=0.5,
        )
        ranked.append({
            "candidate_id": cand.get("candidate_id"),
            "temperature": cand.get("temperature"),
            "citations": cand.get("citations", []),
            "final_score": report["final_score"],
            "llm_average": report.get("llm_average"),
            "auto_metrics": auto_metrics,
            "auto_score": report.get("auto_evaluation", {}).get("overall_score"),
            "details": report,
            "content": cand["content"],
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)

    # 3) 输出结果
    print("\n排序结果 (前 5)：")
    for i, item in enumerate(ranked[:5], 1):
        print(
            f"{i}. 草稿ID={item['candidate_id']}, temp={item['temperature']}, "
            f"最终分={item['final_score']:.2f}, LLM均分={item['llm_average']:.2f}, 自动分={item['auto_score'] or 0:.2f}"
        )

    best = ranked[0]
    print("\n🏆 最佳草稿摘要:")
    print(f"- 草稿ID: {best['candidate_id']}")
    print(f"- 温度: {best['temperature']}")
    print(f"- 最终分: {best['final_score']:.2f}")
    print(f"- 自动引文指标: P={best['auto_metrics']['citation']['precision']:.2f}, "
          f"R={best['auto_metrics']['citation']['recall']:.2f}, F1={best['auto_metrics']['citation']['f1']:.2f}")
    print("\n前 500 字预览:\n" + best["content"][:500])


if __name__ == "__main__":
    main()
