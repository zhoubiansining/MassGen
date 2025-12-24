"""
完整测试流程：Writing Agent + Judge Agent

新增：支持从前序检索/分析步骤的 JSON 输出加载 cluster_summaries，与 Retrieval/Analysis 模块对齐。
"""

import argparse
import json
import os

from writing_agent import WritingAgent, ModelConfig
from judge_agent import JudgeAgent
from pipeline_adapter import analysis_to_cluster_summaries

# 配置（使用 Kimi）
CONFIG = ModelConfig(
    name="Kimi-K2",  # 修正：去掉 $ 前缀
    api_key="sk-aRG9iu2Hy9--oPxrG-5faA",
    base_url="https://llmapi.paratera.com/v1/",
    temperature=0.5,
    max_tokens=4096
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
                "key_contribution": "提出了原始 Transformer 架构，基于自注意力机制"
            },
            {
                "paper_id": "paper_002",
                "title": "BERT: Pre-training of Deep Bidirectional Transformers",
                "authors": ["Devlin, J.", "Chang, M. W."],
                "year": 2019,
                "key_contribution": "双向预训练方法，Masked Language Model"
            },
            {
                "paper_id": "paper_003",
                "title": "GPT-3: Language Models are Few-Shot Learners",
                "authors": ["Brown, T.", "Mann, B."],
                "year": 2020,
                "key_contribution": "大规模预训练模型的少样本学习能力"
            }
        ]
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run writing + judging pipeline with optional analysis JSON")
    parser.add_argument(
        "--analysis-json",
        dest="analysis_json",
        help="前序 Analysis 输出的 JSON 路径（包含 clusters/insights/datas）"
    )
    return parser.parse_args()


def load_cluster_summaries(analysis_json: str = None):
    """若提供 analysis_json，则转换为 WritingAgent 所需输入，否则回退到默认示例数据。"""
    if analysis_json:
        if not os.path.exists(analysis_json):
            raise FileNotFoundError(f"找不到 analysis_json 文件: {analysis_json}")
        with open(analysis_json, "r", encoding="utf-8") as f:
            analysis_result = json.load(f)
        print(f"\n📥 已从 {analysis_json} 读取 Analysis 结果，转换为写作输入格式...")
        return analysis_to_cluster_summaries(analysis_result)

    print("\n📥 未提供 Analysis 输出文件，使用内置示例数据")
    return DEFAULT_TEST_DATA


def main():
    """完整测试流程"""

    args = parse_args()
    cluster_summaries = load_cluster_summaries(args.analysis_json)

    print("=" * 80)
    print("完整测试流程：Writing Agent + Judge Agent")
    print("=" * 80)

    # ==================== Phase 1: 生成草稿 ====================
    print("\n" + "=" * 80)
    print("Phase 1: 使用 Writing Agent 生成草稿")
    print("=" * 80)

    try:
        writer = WritingAgent(CONFIG, style="narrative")
        print("\n✓ Writing Agent 初始化成功")

        print("\n[Step 1.1] 生成 3 个候选草稿...")
        candidates = writer.generate_multiple_candidates(
            cluster_summaries=cluster_summaries,
            num_candidates=3,
            temperature_range=(0.3, 0.7)
        )

        print(f"\n✅ 成功生成 {len(candidates)} 个候选草稿")

        # 显示候选草稿摘要
        print("\n候选草稿摘要:")
        for i, candidate in enumerate(candidates, 1):
            content_preview = candidate['content'][:150].replace('\n', ' ')
            print(f"\n  草稿 {i}:")
            print(f"    温度: {candidate['temperature']:.2f}")
            print(f"    引用数: {len(candidate['citations'])}")
            print(f"    长度: {len(candidate['content'])} 字符")
            print(f"    预览: {content_preview}...")

    except Exception as e:
        print(f"\n❌ Phase 1 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ==================== Phase 2: 评估草稿 ====================
    print("\n\n" + "=" * 80)
    print("Phase 2: 使用 Judge Agent 评估草稿")
    print("=" * 80)

    try:
        # 创建 Judge Agent（使用更低的温度以提高评分一致性）
        judge_config = ModelConfig(
            name="GLM-4.6",  # 修正：去掉 $ 前缀
            api_key="sk-aRG9iu2Hy9--oPxrG-5faA",
            base_url="https://llmapi.paratera.com/v1/",
            temperature=0.2,  # Judge 使用低温度
            max_tokens=4096
        )
        judge = JudgeAgent(judge_config)
        print("\n✓ Judge Agent 初始化成功")

        print("\n[Step 2.1] 对所有草稿进行评分和排序...")

        # 提取草稿内容
        drafts = [c['content'] for c in candidates]

        # 排序
        ranked = judge.rank_drafts(drafts, reference=cluster_summaries)

        print(f"\n✅ 评估完成")

        # 显示排序结果
        print("\n排序结果:")
        print("-" * 80)
        print(f"{'排名':<6} {'草稿ID':<10} {'总分':<10} {'覆盖度':<10} {'准确性':<10} {'连贯性':<10}")
        print("-" * 80)

        for i, result in enumerate(ranked, 1):
            scores = result.get('scores', {})
            print(f"{i:<6} "
                  f"{result['draft_id']:<10} "
                  f"{result['overall_score']:<10.1f} "
                  f"{scores.get('coverage', 0):<10.1f} "
                  f"{scores.get('factuality', 0):<10.1f} "
                  f"{scores.get('coherence', 0):<10.1f}")

    except Exception as e:
        print(f"\n❌ Phase 2 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ==================== Phase 3: 选择最佳草稿 ====================
    print("\n\n" + "=" * 80)
    print("Phase 3: 选择最佳草稿")
    print("=" * 80)

    try:
        best = ranked[0]

        print(f"\n🏆 最佳草稿：草稿 {best['draft_id']}")
        print(f"   总分: {best['overall_score']:.1f}/100")
        print(f"   长度: {best['draft_length']} 字符")

        # 显示各维度得分
        print(f"\n📊 各维度得分:")
        scores = best.get('scores', {})
        for dim, score in scores.items():
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            print(f"   {dim:20s}: {score:3.0f}/100  {bar}")

        # 显示优点
        print(f"\n✅ 优点 ({len(best.get('strengths', []))}):")
        for i, strength in enumerate(best.get('strengths', []), 1):
            print(f"   {i}. {strength}")

        # 显示缺点
        print(f"\n⚠️  缺点 ({len(best.get('weaknesses', []))}):")
        for i, weakness in enumerate(best.get('weaknesses', []), 1):
            print(f"   {i}. {weakness}")

        # 显示改进建议
        print(f"\n💡 改进建议 ({len(best.get('improvement_suggestions', []))}):")
        for i, suggestion in enumerate(best.get('improvement_suggestions', []), 1):
            print(f"   {i}. 问题: {suggestion.get('issue', 'N/A')}")
            print(f"      建议: {suggestion.get('suggestion', 'N/A')}")

    except Exception as e:
        print(f"\n❌ Phase 3 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ==================== Phase 4: 事实验证 ====================
    print("\n\n" + "=" * 80)
    print("Phase 4: 事实准确性验证")
    print("=" * 80)

    try:
        # 准备证据材料
        evidence = {}
        for cluster_data in cluster_summaries.values():
            for paper in cluster_data.get('papers', []):
                evidence[paper['paper_id']] = paper

        print(f"\n[Step 4.1] 验证最佳草稿的事实准确性...")
        verification = judge.verify_factuality(best['draft'], evidence)

        print(f"\n✅ 验证完成")

        # 显示验证结果
        print(f"\n📊 验证结果:")
        print(f"   总陈述数: {verification.get('total_claims', 'N/A')}")
        print(f"   已验证数: {verification.get('verified_claims', 'N/A')}")
        print(f"   准确率: {verification.get('accuracy_rate', 0):.1%}")

        citation_check = verification.get('citation_check', {})
        print(f"\n📎 引用检查:")
        print(f"   总引用数: {citation_check.get('total_citations', 0)}")
        print(f"   无效引用数: {len(citation_check.get('invalid_citations', []))}")
        print(f"   引用有效率: {citation_check.get('citation_validity_rate', 0):.1%}")

        if citation_check.get('invalid_citations'):
            print(f"\n⚠️  无效引用:")
            for citation in citation_check['invalid_citations']:
                print(f"      - [{citation}]")
        else:
            print(f"\n✅ 所有引用均有效")

    except Exception as e:
        print(f"\n❌ Phase 4 失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # ==================== Phase 5: 生成最终报告 ====================
    print("\n\n" + "=" * 80)
    print("Phase 5: 生成最终报告")
    print("=" * 80)

    try:
        final_report = {
            "summary": {
                "best_draft_id": best['draft_id'],
                "overall_score": best['overall_score'],
                "factuality_rate": verification.get('accuracy_rate', 0),
                "citation_validity": citation_check.get('citation_validity_rate', 0)
            },
            "best_draft": {
                "content": best['draft'],
                "scores": best.get('scores', {}),
                "length": best['draft_length']
            },
            "feedback": {
                "strengths": best.get('strengths', []),
                "weaknesses": best.get('weaknesses', []),
                "improvements": best.get('improvement_suggestions', [])
            },
            "verification": {
                "accuracy_rate": verification.get('accuracy_rate', 0),
                "citation_check": citation_check
            },
            "alternatives": [
                {
                    "draft_id": alt['draft_id'],
                    "score": alt['overall_score']
                }
                for alt in ranked[1:]
            ]
        }

        # 保存报告
        with open("full_pipeline_report.json", "w", encoding="utf-8") as f:
            json.dump(final_report, f, ensure_ascii=False, indent=2)

        # 保存最佳草稿
        with open("best_draft.txt", "w", encoding="utf-8") as f:
            f.write(best['draft'])

        print("\n✅ 报告已生成")
        print("\n文件已保存:")
        print("   - full_pipeline_report.json  (完整评估报告)")
        print("   - best_draft.txt             (最佳草稿)")

        # 显示摘要
        print("\n" + "=" * 80)
        print("测试摘要")
        print("=" * 80)
        print(f"\n🎯 综合评估:")
        print(f"   总分: {final_report['summary']['overall_score']:.1f}/100")
        print(f"   事实准确率: {final_report['summary']['factuality_rate']:.1%}")
        print(f"   引用有效率: {final_report['summary']['citation_validity']:.1%}")

        print(f"\n📝 质量等级:")
        score = final_report['summary']['overall_score']
        if score >= 90:
            grade = "优秀（可发表）"
        elif score >= 80:
            grade = "良好（需小幅修改）"
        elif score >= 70:
            grade = "合格（需中等修改）"
        elif score >= 60:
            grade = "尚可（需大幅修改）"
        else:
            grade = "不合格（需重写）"
        print(f"   等级: {grade}")

        print(f"\n✅ 完整测试流程成功完成！")

        return final_report

    except Exception as e:
        print(f"\n❌ Phase 5 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
