"""
完整测试流程：Retrieval -> Analysis -> Writing -> Judging -> Verification
版本：Final Optimized (Detailed Logs + File Save)
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

from agents.retrieval_agent import RetrievalAgent
from agents.analysis_agent import AnalysisAgent
from writing_judging.writing_agent import WritingAgent, ModelConfig
from writing_judging.judge_agent import JudgeAgent
from pipeline_adapter import analysis_to_cluster_summaries
from core.schema import LLMMessage


# ==================== 配置与环境 ====================
load_dotenv()
API_KEY = os.getenv("PARATERA_API_KEY") 
BASE_URL = "https://llmapi.paratera.com/v1/"
MODEL_NAME = "Kimi-K2"

if not API_KEY:
    print("⚠️  警告: 未检测到 API Key，请检查 .env 文件。")

COMMON_AGENT_ARGS = {
    "model": MODEL_NAME,
    "base_url": BASE_URL,
    "api_key": API_KEY
}

WRITER_CONFIG = ModelConfig(
    name=MODEL_NAME, api_key=API_KEY, base_url=BASE_URL, temperature=0.7, max_tokens=8192
)

JUDGE_CONFIG = ModelConfig(
    name="GLM-4.6", api_key=API_KEY, base_url=BASE_URL, temperature=0.2, max_tokens=4096
)

# ==================== 日志与目录工具 ====================

class DualLogger(object):
    """同时将输出写入终端和文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush() # 确保实时写入

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_run_directory(run_id: str):
    base_dir = os.path.join("output", run_id)
    traj_dir = os.path.join(base_dir, "trajectory")
    os.makedirs(traj_dir, exist_ok=True)
    return base_dir, traj_dir

def serialize_history(history: List[Any]) -> List[Dict]:
    serialized = []
    for msg in history:
        if hasattr(msg, 'role'):
            m_dict = {"role": msg.role, "content": msg.content}
            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                m_dict["tool_calls"] = [{"name": tc.name, "arguments": tc.arguments} for tc in msg.tool_calls]
            if hasattr(msg, 'tool_call_id') and msg.tool_call_id:
                m_dict["tool_call_id"] = msg.tool_call_id
                m_dict["name"] = msg.name
            serialized.append(m_dict)
        else:
            serialized.append(msg if isinstance(msg, dict) else str(msg))
    return serialized

def save_json(data: Any, folder: str, filename: str):
    if not filename.endswith(".json"):
        filename += ".json"
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path

def parse_args():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--query", type=str, help="从头开始：指定研究主题")
    group.add_argument("--analysis-json", type=str, help="从中间开始：指定 analysis 结果文件")
    parser.add_argument("--max-search-steps", type=int, default=3)
    return parser.parse_args()

# ==================== 主流程 ====================

async def main():
    args = parse_args()
    
    # 1. 目录初始化
    RUN_ID = get_timestamp()
    RUN_DIR, TRAJ_DIR = create_run_directory(RUN_ID)

    # 2. 【关键】重定向 print 到日志文件
    log_file_path = os.path.join(RUN_DIR, "execution_log.txt")
    sys.stdout = DualLogger(log_file_path)
    
    print("=" * 80)
    print(f"🚀 全自动文献综述流水线启动")
    print(f"📁 本次运行目录: {RUN_DIR}")
    print(f"📝 完整日志文件: {log_file_path}")
    print("=" * 80)

    analysis_result = {}
    cluster_summaries = {}
    papers = []

    # ------------------------------------------------------------------
    # Phase 1 & 2: 数据获取 (Retrieval + Analysis)
    # ------------------------------------------------------------------
    if args.query:
        # >>> Phase 1: Retrieval <<<
        print("\n" + "=" * 80)
        print("Phase 1: Retrieval Agent (文献检索)")
        print("=" * 80)
        
        retriever = RetrievalAgent(**COMMON_AGENT_ARGS)
        papers = await retriever.run(args.query, max_steps=args.max_search_steps)
        
        save_json(serialize_history(retriever.history), TRAJ_DIR, "retrieval_traj")
        
        if not papers:
            print("❌ 检索失败，未找到论文。")
            return
        save_json(papers, RUN_DIR, "1_retrieval_papers")
        print(f"✅ 检索完成，获取 {len(papers)} 篇论文。")

        # >>> Phase 2: Analysis <<<
        print("\n" + "=" * 80)
        print("Phase 2: Analysis Agent (深度分析)")
        print("=" * 80)
        
        analyzer = AnalysisAgent(datas=papers, **COMMON_AGENT_ARGS)
        analysis_result = await analyzer.run(args.query)
        
        save_json(serialize_history(analyzer.history), TRAJ_DIR, "analysis_traj")
        
        if not analysis_result:
            print("❌ 分析失败。")
            return
        save_json(analysis_result, RUN_DIR, "2_analysis_result")
        print(f"✅ 分析完成，生成 {len(analysis_result.get('clusters', []))} 个研究聚类。")
        
        print("\n🔄 Adapting data format...")
        cluster_summaries = analysis_to_cluster_summaries(analysis_result, papers)
        
    else:
        print(f"\n📂 Loading analysis from {args.analysis_json}")
        with open(args.analysis_json, "r", encoding="utf-8") as f:
            analysis_result = json.load(f)
        papers = analysis_result.get("datas", [])
        cluster_summaries = analysis_to_cluster_summaries(analysis_result)

    if not cluster_summaries:
        print("❌ 无法获取 Cluster Summaries，流程终止。")
        return
    
    save_json(cluster_summaries, RUN_DIR, "3_adapter_input")

    # ------------------------------------------------------------------
    # Phase 3: 写作 (Writing)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Phase 3: 使用 Writing Agent 生成草稿")
    print("=" * 80)

    try:
        writer = WritingAgent(WRITER_CONFIG, style="narrative")
        print("\n✓ Writing Agent 初始化成功")
        
        print("\n[Step 3.1] 生成 3 个候选草稿 (异步并行)...")
        candidates = await writer.generate_multiple_candidates_async(
            cluster_summaries=cluster_summaries,
            num_candidates=3
        )
        
        if not candidates:
            print("❌ 写作失败。")
            return
        save_json(candidates, RUN_DIR, "4_writing_candidates")
        
        print(f"\n✅ 成功生成 {len(candidates)} 个候选草稿")

        # === 详细打印 ===
        print("\n候选草稿摘要:")
        for i, candidate in enumerate(candidates, 1):
            content_preview = candidate['content'][:150].replace('\n', ' ')
            print(f"\n  草稿 {i}:")
            print(f"    温度: {candidate.get('temperature', 'N/A'):.2f}")
            print(f"    引用数: {len(candidate.get('citations', []))}")
            print(f"    长度: {len(candidate.get('content', ''))} 字符")
            print(f"    预览: {content_preview}...")
        # ===============

    except Exception as e:
        print(f"\n❌ Phase 3 Error: {e}")
        import traceback; traceback.print_exc()
        return

    # ------------------------------------------------------------------
    # Phase 4: 评审与择优 (Judging)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("Phase 4: 使用 Judge Agent 评估草稿")
    print("=" * 80)

    try:
        judge = JudgeAgent(JUDGE_CONFIG)
        print("\n✓ Judge Agent 初始化成功")
        
        reference_material = {
            "task": args.query or "Analysis",
            "clusters": cluster_summaries
        }
        draft_texts = [c['content'] for c in candidates]
        
        print("\n[Step 4.1] 对所有草稿进行评分和排序...")
        ranked_results = await judge.rank_drafts_async(draft_texts, reference=reference_material)
        
        # 恢复元数据
        for rank_item in ranked_results:
            idx = rank_item['draft_id']
            if idx < len(candidates):
                rank_item['meta'] = {
                    "temp": candidates[idx].get("temperature"),
                    "citations": len(candidates[idx].get("citations", []))
                }

        best = ranked_results[0]
        
        save_json(ranked_results, RUN_DIR, "5_judging_result")
        
        print(f"\n✅ 评估完成")

        # === 详细打印 ===
        print("\n排序结果:")
        print("-" * 80)
        print(f"{'排名':<6} {'草稿ID':<10} {'总分':<10} {'覆盖度':<10} {'准确性':<10} {'连贯性':<10}")
        print("-" * 80)

        for i, result in enumerate(ranked_results, 1):
            scores = result.get('scores', {})
            print(f"{i:<6} "
                  f"{result['draft_id']:<10} "
                  f"{result['overall_score']:<10.1f} "
                  f"{scores.get('coverage', 0):<10.1f} "
                  f"{scores.get('factuality', 0):<10.1f} "
                  f"{scores.get('coherence', 0):<10.1f}")
        
        # 最佳草稿详情
        print("\n\n" + "=" * 80)
        print("Phase 4.5: 最佳草稿详情")
        print("=" * 80)
        
        print(f"\n🏆 最佳草稿：草稿 {best['draft_id']}")
        print(f"   总分: {best['overall_score']:.1f}/100")
        print(f"   长度: {best.get('draft_length', 0)} 字符")

        print(f"\n📊 各维度得分:")
        scores = best.get('scores', {})
        for dim, score in scores.items():
            bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
            print(f"   {dim:20s}: {score:3.0f}/100  {bar}")

        print(f"\n✅ 优点 ({len(best.get('strengths', []))}):")
        for i, strength in enumerate(best.get('strengths', []), 1):
            print(f"   {i}. {strength}")

        print(f"\n⚠️  缺点 ({len(best.get('weaknesses', []))}):")
        for i, weakness in enumerate(best.get('weaknesses', []), 1):
            print(f"   {i}. {weakness}")

        print(f"\n💡 改进建议 ({len(best.get('improvement_suggestions', []))}):")
        for i, suggestion in enumerate(best.get('improvement_suggestions', []), 1):
            print(f"   {i}. 问题: {suggestion.get('issue', 'N/A')}")
            print(f"      建议: {suggestion.get('suggestion', 'N/A')}")
        # ===============

    except Exception as e:
        print(f"\n❌ Phase 4 Error: {e}")
        import traceback; traceback.print_exc()
        return

    # ------------------------------------------------------------------
    # Phase 5: 验证与交付 (Verification)
    # ------------------------------------------------------------------
    print("\n\n" + "=" * 80)
    print("Phase 5: 事实准确性验证与最终报告")
    print("=" * 80)

    try:
        evidence = {}
        for p in papers:
            if p.get("id"): evidence[p["id"]] = p
            if p.get("url"): evidence[p["url"]] = p
        if not evidence:
            for c in cluster_summaries.values():
                for p in c.get("papers", []):
                    evidence[p["paper_id"]] = p

        print(f"\n[Step 5.1] 验证最佳草稿的事实准确性...")
        verification = judge.verify_factuality(best['draft'], evidence)
        
        print(f"\n✅ 验证完成")

        # === 详细打印 ===
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
        # ===============

        # 构建最终报告
        final_report = {
            "meta": {"run_id": RUN_ID, "timestamp": datetime.now().isoformat()},
            "best_draft_content": best['draft'],
            "evaluation": {
                "score": best['overall_score'],
                "details": best.get("scores"),
                "feedback": {
                    "strengths": best.get("strengths"),
                    "weaknesses": best.get("weaknesses"),
                    "improvements": best.get("improvement_suggestions")
                }
            },
            "verification": verification
        }
        
        save_json(final_report, RUN_DIR, "FINAL_REPORT")
        
        md_path = os.path.join(RUN_DIR, "FINAL_PAPER.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# Literature Review: {args.query or 'Auto Survey'}\n\n")
            f.write(f"> Run ID: {RUN_ID} | Score: {best['overall_score']:.1f}\n\n")
            f.write(best['draft'])
            f.write("\n\n---\n*Verified by Judge Agent*")
        
        print("\n\n" + "=" * 80)
        print("测试摘要")
        print("=" * 80)
        print(f"\n🎯 综合评估:")
        print(f"   总分: {best['overall_score']:.1f}/100")
        print(f"   事实准确率: {verification.get('accuracy_rate', 0):.1%}")
        print(f"   引用有效率: {citation_check.get('citation_validity_rate', 0):.1%}")

        print(f"\n📝 质量等级:")
        score = best['overall_score']
        if score >= 90: grade = "优秀（可发表）"
        elif score >= 80: grade = "良好（需小幅修改）"
        elif score >= 70: grade = "合格（需中等修改）"
        elif score >= 60: grade = "尚可（需大幅修改）"
        else: grade = "不合格（需重写）"
        print(f"   等级: {grade}")

        print(f"\n✅ 完整测试流程成功完成！")
        print(f"   结果目录: {RUN_DIR}")

    except Exception as e:
        print(f"\n❌ Phase 5 Error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

    # python main.py --query "Coding Benchmark for LLM in Repo-Level" --max-search-steps 5
    # python main.py --analysis-json "你的分析结果json路径"
