"""
完整测试流程：Retrieval -> Analysis -> Writing -> Judging -> Verification
版本：Final Optimized (Detailed Logs + File Save)
"""

import asyncio
import argparse
import json
import re
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv
import pandas as pd
import json as jsonlib

# 确保当前目录与父目录在搜索路径中，避免相对导入报错
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
for _p in (CURRENT_DIR, PARENT_DIR):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from agents.retrieval_agent import RetrievalAgent
from agents.analysis_agent import AnalysisAgent
from writing_judging.writing_agent import WritingAgent, ModelConfig
from writing_judging.judge_agent import JudgeAgent
from writing_judging.pipeline_adapter import analysis_to_cluster_summaries
from core.schema import LLMMessage


# ==================== 配置与环境 ====================
load_dotenv()
API_KEY = "sk-aRG9iu2Hy9--oPxrG-5faA"
BASE_URL = os.getenv("PARATERA_BASE_URL", "https://llmapi.paratera.com/v1/")
MODEL_NAME = os.getenv("PARATERA_MODEL", "Qwen3-235B-A22B-Instruct-2507")

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


def load_judge_models(config_path: str) -> List[ModelConfig]:
    """从 model_config.json 载入模型列表（中文注释便于阅读）。"""
    with open(config_path, "r", encoding="utf-8") as f:
        data = jsonlib.load(f)
    models = data.get("models", {})
    judge_models = []
    for _, info in models.items():
        judge_models.append(
            ModelConfig(
                name=info.get("name"),
                api_key=info.get("api_key"),
                base_url=info.get("base_url", BASE_URL),
                temperature=info.get("temperature", 0.2),
                max_tokens=info.get("max_tokens", 4096)
            )
        )
    return judge_models

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
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--query", type=str, help="从头开始：指定研究主题")
    group.add_argument("--analysis-json", type=str, help="从中间开始：指定 analysis 结果文件")
    group.add_argument("--dataset-pkl", type=str, help="使用本地 pkl 数据集作为“检索种子”（仅提取标题发起检索）")
    parser.add_argument("--max-search-steps", type=int, default=3)
    parser.add_argument("--sample-n", type=int, default=-1, help="dataset-pkl 模式下采样条数，-1 表示全部")
    parser.add_argument("--skip-n", type=int, default=0, help="dataset-pkl 模式下跳过前 N 条样本")
    parser.add_argument("--use-test-only", action="store_true", help="dataset-pkl 模式下仅使用 split == 'test' 的样本")
    parser.add_argument("--per-paper", action="store_true", help="dataset-pkl 模式下逐篇处理，每篇分配一个 API Key（分析/写作/多模型评审共用该 Key）")
    parser.add_argument("--api-keys-file", type=str, help="存放多个 API Key 的文件（每行一个或 JSON 数组）")
    parser.add_argument("--retrieval-only", action="store_true", help="仅执行检索并保存结果，不进入分析/写作/评审")
    parser.add_argument("--reuse-retrieval-dir", type=str, help="使用指定目录下的 title_<idx>_retrieval.json 作为检索结果，跳过在线检索")
    return parser.parse_args()


def load_papers_from_pkl(pkl_path: str, sample_n: int = -1, use_test_only: bool = False) -> List[Dict[str, Any]]:
    """从本地 pkl 加载论文列表，模拟检索结果（中文注释方便阅读）。"""
    df = pd.read_pickle(pkl_path)
    if use_test_only and "split" in df.columns:
        df = df[df["split"] == "test"]
    if sample_n and sample_n > 0:
        df = df.head(sample_n)
    papers = []
    for idx, row in df.iterrows():
        papers.append({
            "id": str(idx),
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "authors": [],
            "year": None,
            "summary": row.get("abstract", "")
        })
    return papers


def load_titles_from_pkl(pkl_path: str, sample_n: int = -1, use_test_only: bool = False) -> List[str]:
    """从本地 pkl 加载标题列表（用于发起检索）。"""
    df = pd.read_pickle(pkl_path)
    if use_test_only and "split" in df.columns:
        df = df[df["split"] == "test"]
    if sample_n and sample_n > 0:
        df = df.head(sample_n)
    return [str(t) for t in df.get("title", []) if str(t).strip()]


def log_progress(step: int, total: int, label: str):
    """简单进度条打印（中文注释便于阅读）。"""
    bar_len = 30
    filled = int(bar_len * step / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"[进度 {step}/{total}] {bar} {label}")


def render_progress(current: int, total: int, start_ts: float, label: str):
    """单行进度条，附剩余时间预估（中文注释便于阅读）。"""
    elapsed = time.time() - start_ts
    rate = elapsed / current if current else 0
    remaining = rate * (total - current) if rate else 0
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    eta = datetime.fromtimestamp(time.time() + remaining).strftime("%H:%M:%S") if current else "--:--:--"
    sys.stdout.write(f"\r[{label}] {current}/{total} {bar} ETA {eta}")
    sys.stdout.flush()
    if current == total:
        sys.stdout.write("\n")


def load_api_keys(path: str) -> List[str]:
    """从文件读取 API Key 列表，支持每行或 JSON 数组（中文注释便于阅读）。"""
    if not path or not os.path.exists(path):
        return []
    text = open(path, "r", encoding="utf-8").read().strip()
    keys: List[str] = []
    try:
        data = jsonlib.loads(text)
        if isinstance(data, list):
            keys = [str(k).strip() for k in data]
    except Exception:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        parsed = []
        for line in lines:
            if "=" in line:
                line = line.split("=", 1)[1].strip()
            parsed.append(line)
        keys = parsed
    # 仅保留 sk- 开头的 key
    keys = [k for k in keys if k.startswith("sk-")]
    return keys


def build_configs(api_key: str) -> Dict[str, Any]:
    """按给定 key 构造检索/写作/评审配置（中文注释便于阅读）。"""
    common = {
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "api_key": api_key
    }
    writer_conf = ModelConfig(
        name=MODEL_NAME, api_key=api_key, base_url=BASE_URL, temperature=0.7, max_tokens=8192
    )
    judge_conf = ModelConfig(
        name="GLM-4.6", api_key=api_key, base_url=BASE_URL, temperature=0.2, max_tokens=4096
    )
    return {"common": common, "writer": writer_conf, "judge": judge_conf}

# ==================== 主流程 ====================

async def main():
    args = parse_args()
    if not (args.query or args.analysis_json or args.dataset_pkl or args.reuse_retrieval_dir):
        print("❌ 需要提供 --query / --analysis-json / --dataset-pkl / --reuse-retrieval-dir 之一。")
        return
    
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
    if args.per_paper and not (args.dataset_pkl or args.reuse_retrieval_dir):
        print("❌ per-paper 模式需提供 --dataset-pkl 或 --reuse-retrieval-dir。")
        return

    if args.per_paper:
        api_keys = load_api_keys(args.api_keys_file)
        if not api_keys:
            print("❌ 未找到有效的 API Key 列表，请检查 --api-keys-file。")
            return

        # 使用测试集标题作为检索起点（强制只取 test）
        # 处理 reuse_retrieval_dir（直接复用已下载的检索结果）
        reuse_entries = []
        reuse_ids: List[int] = []
        if args.reuse_retrieval_dir:
            import glob
            pattern = os.path.join(args.reuse_retrieval_dir, "title_*_retrieval.json")
            files = sorted(
                glob.glob(pattern), 
                key=lambda x: int(re.search(r"title_(\d+)_retrieval", os.path.basename(x)).group(1)) 
                              if re.search(r"title_(\d+)_retrieval", os.path.basename(x)) else 0
            )
            for path in files:
                try:
                    data = json.load(open(path, "r", encoding="utf-8"))
                    fname = os.path.basename(path)
                    m = re.search(r"title_(\d+)_retrieval", fname)
                    file_id = int(m.group(1)) if m else len(reuse_entries) + 1
                    if isinstance(data, dict):
                        title_val = data.get("title") or os.path.basename(path)
                        papers_val = data.get("papers") or []
                    else:
                        title_val = os.path.basename(path)
                        papers_val = data
                    reuse_entries.append((title_val, papers_val))
                    reuse_ids.append(file_id)
                except Exception as e:
                    print(f"⚠️ 无法读取 {path}: {e}")

        if args.reuse_retrieval_dir and not reuse_entries:
            print("❌ reuse_retrieval_dir 未找到任何 title_*_retrieval.json")
            return

        if args.reuse_retrieval_dir:
            titles = [t for t, _ in reuse_entries]
            preloaded_papers = [p for _, p in reuse_entries]
        else:
            titles = load_titles_from_pkl(
                args.dataset_pkl,
                sample_n=args.sample_n,
                use_test_only=True,
            )
            preloaded_papers = None

        if args.skip_n and args.skip_n > 0:
            titles = titles[args.skip_n:]
            if preloaded_papers:
                preloaded_papers = preloaded_papers[args.skip_n:]
            if args.reuse_retrieval_dir and reuse_ids:
                reuse_ids = reuse_ids[args.skip_n:]
        if not titles:
            print("❌ 数据集为空或未找到标题。")
            return

        total_titles = len(titles)
        start_index = args.skip_n or 0
        run_start = time.time()
        print(f"📑 从测试集提取 {total_titles} 个标题，准备启动检索")

        task_queue = asyncio.Queue()
        for idx, title in enumerate(titles):
            logical_idx = idx + start_index
            item_id = reuse_ids[idx] if args.reuse_retrieval_dir and reuse_ids and idx < len(reuse_ids) else logical_idx + 1
            papers_prefill = preloaded_papers[idx] if preloaded_papers and idx < len(preloaded_papers) else None
            task_queue.put_nowait((logical_idx, item_id, title, papers_prefill))
        completed = 0
        completed_lock = asyncio.Lock()

        async def process_title(display_idx: int, item_id: int, title: str, papers_prefill: Any, key: str):
            cfgs = build_configs(key)
            display_current = display_idx + 1
            display_total = total_titles + start_index
            print(f"\n=== 处理第 {display_current}/{display_total} 个标题，使用模型 Key: {key[:6]}... ===")

            # Phase 1: 检索或复用
            if papers_prefill is not None:
                papers_local = papers_prefill
            else:
                retriever = RetrievalAgent(**cfgs["common"])
                papers_local = await retriever.run(title, max_steps=args.max_search_steps)
                if not papers_local:
                    print("❌ 检索无结果，跳过该标题。")
                    return

            # 补齐 ID
            for i, p in enumerate(papers_local):
                if not p.get("id"):
                    p["id"] = f"p{i+1}"
                if not p.get("paper_id"):
                    p["paper_id"] = p["id"]

            # 保存检索结果，带上标题元数据
            save_json({"title": title, "papers": papers_local}, RUN_DIR, f"title_{item_id}_retrieval")

            # 仅检索模式，提前返回
            if args.retrieval_only:
                return

            # Phase 2: 分析
            analyzer = AnalysisAgent(datas=papers_local, **cfgs["common"])
            analysis_result = await analyzer.run(title)
            if isinstance(analysis_result, list):
                analysis_result = analysis_result[0] if analysis_result else {}
            if not analysis_result:
                print("❌ 分析失败，跳过该标题。")
                return
            cluster_summaries_local = analysis_to_cluster_summaries(analysis_result, papers_local)

            # 如果生成的 summaries 仍然缺失 paper_id 或包含 tool call 残留，做一次简单兜底
            def _invalid_cluster(cs: Dict[str, Any]) -> bool:
                if not cs:
                    return True
                topics = list(cs.values())
                if not topics:
                    return True
                topic_txt = topics[0].get("summary") or ""
                return "<function_calls>" in topic_txt or any(p.get("paper_id") is None for c in topics for p in c.get("papers", []))

            if _invalid_cluster(cluster_summaries_local):
                print("⚠️ 分析结果异常，使用兜底 cluster_summaries。")
                cluster_summaries_local = {
                    "cluster_0": {
                        "topic": f"Survey seed: {title}",
                        "summary": f"Papers retrieved for: {title}",
                        "papers": [
                            {
                                "paper_id": p.get("paper_id"),
                                "title": p.get("title"),
                                "authors": p.get("authors", []),
                                "year": p.get("year") or p.get("published"),
                                "key_contribution": p.get("summary") or p.get("abstract", ""),
                                "abstract": p.get("abstract") or p.get("summary"),
                                "url": p.get("url") or p.get("link"),
                            }
                            for p in papers_local
                        ],
                    }
                }
            save_json(analysis_result, RUN_DIR, f"title_{item_id}_analysis")
            save_json(cluster_summaries_local, RUN_DIR, f"title_{item_id}_adapter_input")

            # Phase 3: 写作（温度 0.3/0.4/0.5 各 1 篇）
            writer = WritingAgent(cfgs["writer"], style="narrative")
            candidates, cid = [], 1
            temps = [0.3, 0.4, 0.5]
            writing_start = time.time()
            for t in temps:
                draft = await writer.generate_draft_async(
                    cluster_summaries=cluster_summaries_local,
                    temperature=t,
                )
                draft["candidate_id"] = cid
                cid += 1
                candidates.append(draft)
                render_progress(len(candidates), len(temps), writing_start, "写作进度")

            if not candidates:
                print("❌ 写作失败，跳过该标题。")
                return
            save_json(candidates, RUN_DIR, f"title_{item_id}_writing_candidates")

            # Phase 4: 评审（多模型评分 + 引文指标融合）
            judge_model_configs = load_judge_models(os.path.join(CURRENT_DIR, "model_config.json"))
            if not judge_model_configs:
                print("❌ 无可用评审模型（model_config.json 为空）。")
                return
            judge_model_configs = [
                ModelConfig(
                    name=mc.name,
                    api_key=key,
                    base_url=mc.base_url,
                    temperature=mc.temperature,
                    max_tokens=mc.max_tokens,
                    top_p=mc.top_p,
                )
                for mc in judge_model_configs
            ]

            draft_texts = [c['content'] for c in candidates]
            eval_start = time.time()
            evals = []
            human_refs = {
                paper.get("paper_id")
                for cluster in cluster_summaries_local.values()
                for paper in cluster.get("papers", [])
                if paper.get("paper_id")
            }

            async def eval_one(idx_c: int, draft: str, cand_meta: Dict):
                # 自动指标（仅引文）
                pred_refs = set(cand_meta.get("citations", []))
                matches = len(pred_refs & human_refs)
                prec = matches / len(pred_refs) if pred_refs else 0.0
                rec = matches / len(human_refs) if human_refs else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0

                auto_metrics = {
                    "citation": {
                        "precision": prec,
                        "recall": rec,
                        "f1": f1,
                        "accuracy": prec,
                    },
                    "content": {},
                    "structure": {},
                }

                # 多模型评分并取均值
                async def eval_model(cfg: ModelConfig):
                    agent = JudgeAgent(cfg)
                    res = await agent.evaluate_draft_async(draft, {"task": title, "clusters": cluster_summaries_local})
                    return cfg.name, res.get("overall_score"), res.get("scores")

                results = await asyncio.gather(*[eval_model(cfg) for cfg in judge_model_configs])
                model_scores = {
                    name: {
                        "overall": overall,
                        "dimensions": dims or {}
                    }
                    for name, overall, dims in results
                    if overall is not None
                }
                avg_score = round(sum(ms.get("overall", 0) for ms in model_scores.values()) / len(model_scores), 2) if model_scores else 0.0

                # 融合：70% 模型均分 + 30% 引文 F1（0-100）
                citation_score = auto_metrics["citation"]["f1"] * 100
                final_score = round(avg_score * 0.7 + citation_score * 0.3, 2)

                return {
                    "draft_id": idx_c,
                    "draft": draft,
                    "model_scores": model_scores,
                    "auto_metrics": auto_metrics,
                    "vote_final_score": final_score,
                }

            tasks = [eval_one(j, draft, candidates[j]) for j, draft in enumerate(draft_texts)]

            for i, coro in enumerate(asyncio.as_completed(tasks), 1):
                evaluation = await coro
                evals.append(evaluation)
                render_progress(i, len(draft_texts), eval_start, "评审进度")

            ranked_results = sorted(evals, key=lambda x: x.get("vote_final_score") or 0, reverse=True)
            save_json(ranked_results, RUN_DIR, f"title_{item_id}_judging_result")

            best = ranked_results[0]
            save_json(best, RUN_DIR, f"title_{item_id}_best")
            with open(os.path.join(RUN_DIR, f"title_{item_id}_best.txt"), "w", encoding="utf-8") as f:
                f.write(best.get("draft", ""))

            print(f"\n第 {display_current} 个标题最佳草稿得分: {best['vote_final_score']} (模型均分 {best.get('model_scores')})")

        async def worker(key: str):
            nonlocal completed
            while not task_queue.empty():
                try:
                    display_idx, item_id, title, papers_prefill = task_queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                await process_title(display_idx, item_id, title, papers_prefill, key)
                task_queue.task_done()
                async with completed_lock:
                    completed += 1
                    render_progress(completed, total_titles, run_start, "标题进度")

        await asyncio.gather(*(worker(k) for k in api_keys))
        print("\n✅ 所有标题处理完成。")
        return

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
        log_progress(1, 5, "完成检索")

    elif args.dataset_pkl:
        print(f"\n📂 Loading papers from dataset: {args.dataset_pkl}")
        papers = load_papers_from_pkl(args.dataset_pkl, sample_n=args.sample_n, use_test_only=args.use_test_only)
        if not papers:
            print("❌ 数据集为空或解析失败。")
            return
        save_json(papers, RUN_DIR, "1_retrieval_papers_dataset")
        print(f"✅ 成功载入 {len(papers)} 篇论文（本地数据集）。")
        log_progress(1, 5, "完成数据载入")

    else:
        print(f"\n📂 Loading analysis from {args.analysis_json}")
        with open(args.analysis_json, "r", encoding="utf-8") as f:
            analysis_result = json.load(f)
        papers = analysis_result.get("datas", [])
        cluster_summaries = analysis_to_cluster_summaries(analysis_result)

    # >>> Phase 2: Analysis <<<
    if not cluster_summaries:
        print("\n" + "=" * 80)
        print("Phase 2: Analysis Agent (深度分析)")
        print("=" * 80)

        analyzer = AnalysisAgent(datas=papers, **COMMON_AGENT_ARGS)
        analysis_task = args.query or (f"Dataset-{os.path.basename(args.dataset_pkl)}" if args.dataset_pkl else "Analysis")
        analysis_result = await analyzer.run(analysis_task)

        save_json(serialize_history(analyzer.history), TRAJ_DIR, "analysis_traj")

        if not analysis_result:
            print("❌ 分析失败。")
            return
        save_json(analysis_result, RUN_DIR, "2_analysis_result")
        print(f"✅ 分析完成，生成 {len(analysis_result.get('clusters', []))} 个研究聚类。")

        print("\n🔄 Adapting data format...")
        cluster_summaries = analysis_to_cluster_summaries(analysis_result, papers)
        log_progress(2, 5, "完成分析")

    if not cluster_summaries:
        print("❌ 无法获取 Cluster Summaries，流程终止。")
        return
    
    save_json(cluster_summaries, RUN_DIR, "3_adapter_input")
    log_progress(3, 5, "完成格式适配")

    # ------------------------------------------------------------------
    # Phase 3: 写作 (Writing)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("Phase 3: 使用 Writing Agent 生成草稿")
    print("=" * 80)

    try:
        writer = WritingAgent(WRITER_CONFIG, style="narrative")
        print("\n✓ Writing Agent 初始化成功")
        
        print("\n[Step 3.1] 按温度生成 9 个候选草稿 (顺序异步，带进度)...")
        temps = [0.3, 0.4, 0.5]
        candidates = []
        cid = 1
        total_writing = len(temps) * 3
        writing_start = time.time()
        for t in temps:
            for _ in range(3):
                draft = await writer.generate_draft_async(
                    cluster_summaries=cluster_summaries,
                    temperature=t
                )
                draft["candidate_id"] = cid
                candidates.append(draft)
                cid += 1
                render_progress(len(candidates), total_writing, writing_start, "写作进度")
        
        if not candidates:
            print("❌ 写作失败。")
            return
        save_json(candidates, RUN_DIR, "4_writing_candidates")
        
        print(f"\n✅ 成功生成 {len(candidates)} 个候选草稿")
        log_progress(4, 5, "完成写作")

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
        reference_material = {
            "task": args.query or "Analysis",
            "clusters": cluster_summaries
        }
        draft_texts = [c['content'] for c in candidates]
        
        print("\n[Step 4.1] 对所有草稿进行评分和排序 (文章级并行，每篇分配一个模型)...")
        judge_model_configs = load_judge_models(os.path.join(CURRENT_DIR, "model_config.json"))
        if not judge_model_configs:
            print("❌ 无可用评审模型（未配置有效 sk- 开头的 api_key）。")
            return

        # 预创建 Agent，轮询分配草稿
        judge_agents = [JudgeAgent(cfg) for cfg in judge_model_configs]
        print(f"✓ 加载 {len(judge_agents)} 个评审模型")

        evals = []
        total_eval = len(draft_texts)
        eval_start = time.time()

        async def eval_one(idx: int, draft: str, agent: JudgeAgent):
            res = await agent.evaluate_draft_async(draft, reference_material)
            return {
                "draft_id": idx,
                "draft": draft,
                "model_scores": {agent.model_config.name: res.get("overall_score")},
                "vote_final_score": res.get("overall_score")
            }

        tasks = []
        for idx, draft in enumerate(draft_texts):
            agent = judge_agents[idx % len(judge_agents)]
            tasks.append(eval_one(idx, draft, agent))

        for i, coro in enumerate(asyncio.as_completed(tasks), 1):
            evaluation = await coro
            evals.append(evaluation)
            render_progress(i, total_eval, eval_start, "评审进度")

        ranked_results = sorted(evals, key=lambda x: x.get("vote_final_score") or 0, reverse=True)

        # 写入模型打分明细
        model_scores = [
            {
                "draft_id": res["draft_id"],
                "model_scores": res.get("model_scores", {}),
                "final_score": res.get("vote_final_score")
            }
            for res in ranked_results
        ]
        save_json(model_scores, RUN_DIR, "5_model_scores")
        print("\n模型打分明细:")
        for item in model_scores:
            print(f"草稿 {item['draft_id']} 模型评分: {item['model_scores']} 最终: {item['final_score']}")
        
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
        log_progress(5, 5, "完成评审")

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
