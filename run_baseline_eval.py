#!/usr/bin/env python3
"""批量对 baseline 目录下的 JSON 综述进行多模型评分（仅 LLM，无参考文献）。

用法：在 /Users/zyc/Desktop/rag-zyc 目录下执行：
    python run_baseline_eval.py

说明：
- baseline 目录中的每个 *.json 视为一篇综述，可为 list[str]、list[dict{content}] 或 dict{content}。
- 每篇文章分配一个 API Key（来自 api_keys.txt），同一 Key 串行处理多篇；并发度 = Key 数。
- 使用 survey-agents/writing_judging/model_config.json 的模型列表，多模型均分；记录每模型各维度分数。
- 评分前会移除正文中的参考文献部分（根据常见“References/参考文献/Bibliography”标题截断）。
- 输出 baseline/<原文件名>_judging_result.json。
"""

import asyncio
import glob
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

ROOT_DIR = "/Users/xxx/Desktop/xxx"
BASELINE_DIR = os.path.join(ROOT_DIR, "baseline")
API_KEYS_FILE = os.path.join(ROOT_DIR, "api_keys.txt")
WRITING_DIR = os.path.join(ROOT_DIR, "survey-agents", "writing_judging")
MODEL_CONFIG_PATH = os.path.join(WRITING_DIR, "model_config.json")

# 确保能够导入 writing_judging 下的 judge_agent
if WRITING_DIR not in sys.path:
    sys.path.insert(0, WRITING_DIR)

from judge_agent import JudgeAgent, ModelConfig  # type: ignore


def load_api_keys(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"API key 文件不存在: {path}")
    keys = [k.strip() for k in open(path, encoding="utf-8") if k.strip() and k.strip().startswith("sk-")]
    if not keys:
        raise ValueError("未找到有效的 sk- API Key")
    return keys


def load_model_templates(path: str) -> List[ModelConfig]:
    data = json.load(open(path, "r", encoding="utf-8"))
    models = data.get("models", {})
    cfgs = []
    for _, info in models.items():
        cfgs.append(
            ModelConfig(
                name=info.get("name"),
                api_key="dummy",
                base_url=info.get("base_url"),
                temperature=info.get("temperature", 0.2),
                max_tokens=info.get("max_tokens", 4096),
                top_p=info.get("top_p", 0.9),
            )
        )
    if not cfgs:
        raise ValueError("model_config.json 中 models 为空")
    return cfgs


def strip_references(text: str) -> str:
    """简单移除参考文献：找到常见标题 References/参考文献/Bibliography 后截断。"""
    patterns = [r"^references\b", r"^bibliography\b", r"^参考文献", r"^参考资料"]
    lines = text.splitlines()
    cut = len(lines)
    for i, line in enumerate(lines):
        low = line.strip().lower()
        if any(re.match(pat, low) for pat in patterns):
            cut = i
            break
    if cut < len(lines):
        lines = lines[:cut]
    return "\n".join(lines).strip()


def load_drafts(path: str) -> List[str]:
    data = json.load(open(path, "r", encoding="utf-8"))
    drafts: List[str] = []
    if isinstance(data, list):
        if all(isinstance(x, str) for x in data):
            drafts = list(data)
        else:
            drafts = [d.get("content", "") for d in data if isinstance(d, dict) and d.get("content")]
    elif isinstance(data, dict):
        if data.get("content"):
            drafts = [data.get("content")]
    if not drafts:
        drafts = [json.dumps(data, ensure_ascii=False)]
    return [strip_references(d) for d in drafts if isinstance(d, str)]


def list_json_files() -> List[str]:
    paths = sorted(glob.glob(os.path.join(BASELINE_DIR, "*.json")))
    if not paths:
        raise FileNotFoundError("baseline 目录下未找到任何 json 文件")
    return paths


def build_agents(model_templates: List[ModelConfig], api_key: str) -> List[JudgeAgent]:
    models = [
        ModelConfig(
            name=tpl.name,
            api_key=api_key,
            base_url=tpl.base_url,
            temperature=tpl.temperature,
            max_tokens=tpl.max_tokens,
            top_p=tpl.top_p,
        )
        for tpl in model_templates
    ]
    return [JudgeAgent(cfg) for cfg in models]


def eval_drafts_sync(drafts: List[str], agents: List[JudgeAgent]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for j, draft in enumerate(drafts):
        per_model = []
        for agent in agents:
            res = agent.evaluate_draft(draft, reference=None)
            per_model.append(
                {
                    "model": agent.model_config.name,
                    "overall": res.get("overall_score"),
                    "dimensions": res.get("scores"),
                }
            )
        scores = [m.get("overall") for m in per_model if m.get("overall") is not None]
        avg_score = round(sum(scores) / len(scores), 2) if scores else None
        results.append({"draft_id": j, "per_model": per_model, "avg_overall": avg_score})
    return results


async def worker(queue: asyncio.Queue, model_templates: List[ModelConfig]):
    while True:
        try:
            idx, path, api_key = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        prefix = os.path.splitext(os.path.basename(path))[0]
        try:
            drafts = load_drafts(path)
            agents = build_agents(model_templates, api_key)
            evals = await asyncio.to_thread(eval_drafts_sync, drafts, agents)
            out_path = os.path.join(BASELINE_DIR, f"{prefix}_judging_result.json")
            json.dump(
                {
                    "source": path,
                    "api_key_used": api_key[:6] + "...",
                    "num_drafts": len(drafts),
                    "evaluations": evals,
                },
                open(out_path, "w", encoding="utf-8"),
                ensure_ascii=False,
                indent=2,
            )
            print(f"[done] {prefix} -> {out_path} (drafts={len(drafts)})")
        except Exception as e:
            print(f"[error] {path}: {e}")
        finally:
            queue.task_done()


async def main():
    api_keys = load_api_keys(API_KEYS_FILE)
    model_templates = load_model_templates(MODEL_CONFIG_PATH)
    files = list_json_files()

    q: asyncio.Queue[Tuple[int, str, str]] = asyncio.Queue()
    for i, path in enumerate(files):
        q.put_nowait((i, path, api_keys[i % len(api_keys)]))

    workers = [worker(q, model_templates) for _ in api_keys]
    await asyncio.gather(*workers)


if __name__ == "__main__":
    asyncio.run(main())

