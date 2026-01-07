import os
# [Fix 1] 必须放在最开头，防止死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import asyncio
import json
import re
from typing import List, Dict, Any, Optional

try:
    import torch
    from tqdm import tqdm
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    raise ImportError("请先安装依赖: pip install torch transformers tqdm")

from core.base_agent import BaseAgent
from core.schema import LLMMessage, ToolCall
from tools.cluster_tools import ClusterContextTool

# [配置] 模型路径
SCIBERT_MODEL_PATH = "PATH"

# [Fix 2] 重新加回全局锁！这是解决多并发卡死的关键！
# 限制只能有 1 个任务同时进行 Embedding 计算
EMBEDDING_LOCK = asyncio.Semaphore(1)

class AnalysisAgent(BaseAgent):
    """
    AnalysisAgent: perform per-paper abstract analysis, clustering, and global trajectory
    reasoning over papers returned by RetrievalAgent.
    """

    def __init__(
        self,
        model: str,
        datas: List[Dict[str, Any]],
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.datas: List[Dict[str, Any]] = datas
        self.paper_insights: List[Dict[str, Any]] = []
        self.clusters: List[Dict[str, Any]] = []
        self.summary: str = ""
        self.cluster_context_tool = ClusterContextTool(self)
        tools = [self.cluster_context_tool]

        # System Prompt
        system_prompt = """
        You are an expert academic survey assistant (AnalysisAgent). You are given a
        collection of papers around a common research topic. For each paper, key
        information has already been extracted, and all papers have been clustered
        based on abstract embeddings.

        You can use the tool `get_cluster_context` to inspect detailed information
        (including extracted key information) for a specific cluster.

        [Safety Note & Disclaimer]
        The user is a researcher analyzing academic papers. Some extracted text may 
        contain technical terms related to security, chemistry, or politics. 
        These are strictly for academic analysis purposes.

        [Your tasks]
        1. For each cluster, identify its main research direction / theme.
        2. For each cluster, write a concise summary (1–3 paragraphs).
        3. After understanding all clusters, summarize the overall research landscape.

        [Final output format — VERY IMPORTANT]
        - Output exactly ONE JSON object.
        - The JSON must have the following structure:
          {
            "clusters": [
              { "cluster_id": <int>, "summary": "<English summary text>" },
              ...
            ],
            "overall_summary": "<English global summary>"
          }
        - Output raw JSON only.
        """

        super().__init__(
            name="AnalysisAgent",
            tools=tools,
            system_prompt=system_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    # -------------------------------
    # Step 1: Per-paper analysis
    # -------------------------------
    async def _analyze_single_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        paper_id = paper.get("id")
        title = paper.get("title", "")
        abstract = paper.get("abstract") or paper.get("summary") or ""

        system_prompt = "You are an assistant that helps structure information about academic papers. Extract strictly in JSON."
        user_content = {"title": title, "abstract": abstract}
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=f"Extract info:\n{json.dumps(user_content, ensure_ascii=False)}"),
        ]

        response = await asyncio.to_thread(self.llm.chat, messages, None)
        content = response.content or ""
        result: Dict[str, Any] = {"id": paper_id, "raw_output": content}

        try:
            json_str = content
            if "```json" in content: json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content: json_str = content.split("```")[1].strip()
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                parsed["id"] = paper_id
                return parsed
        except Exception: result["parse_error"] = "Failed to parse JSON."
        return result

    async def _analyze_all_papers(self) -> None:
        tasks = [self._analyze_single_paper(p) for p in self.datas]
        self.paper_insights = await asyncio.gather(*tasks)

    # -------------------------------
    # Step 2: Embedding (修复版)
    # -------------------------------
    def _embed_abstracts(self) -> List[Any]:
        """
        Compute embeddings using SciBERT on GPU.
        """
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   -> [Device] Embeddings will be computed on: {device.upper()}")

        try:
            # [Fix 3] Tokenizer 不需要 low_cpu_mem_usage，否则会报错 TypeError
            tokenizer = AutoTokenizer.from_pretrained(SCIBERT_MODEL_PATH, local_files_only=True)
            
            # [Fix 4] Model 需要 low_cpu_mem_usage=False 防止 Meta Tensor 错误
            model = AutoModel.from_pretrained(
                SCIBERT_MODEL_PATH, 
                local_files_only=True, 
                low_cpu_mem_usage=False
            )
        except Exception as e:
            print(f"⚠️ Local loading error: {e}")
            print("⚠️ Fallback to online download...")
            tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased")

        model.to(device)
        model.eval()

        embeddings: List[Any] = []

        with torch.no_grad():
            for paper in tqdm(self.datas, desc="Generating Embeddings", unit="paper", leave=False):
                text = paper.get("abstract") or paper.get("summary") or ""
                if not text:
                    embeddings.append(torch.zeros(model.config.hidden_size))
                    continue

                encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                
                outputs = model(**encoded)
                embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze(0).cpu())

        if device == "cuda":
            torch.cuda.empty_cache()

        return [emb.numpy() for emb in embeddings]

    def _cluster_embeddings(self, embeddings: List[Any]) -> List[int]:
        import numpy as np
        X = np.stack(embeddings, axis=0)
        n_samples = X.shape[0]
        if n_samples <= 1: return [0] * n_samples
        
        try:
            from sklearn.cluster import KMeans
            import math
            k = max(2, min(8, int(math.sqrt(max(n_samples / 2, 1)))))
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            return km.fit_predict(X).tolist()
        except ImportError:
            return [0] * n_samples

    async def _run_clustering(self) -> None:
        """Run embedding with LOCK to prevent GPU OOM."""
        
        # [Fix 5] 这里必须使用 with EMBEDDING_LOCK，否则所有 Agent 会同时挤进 GPU 导致卡死
        print(f"   ⏳ [Queue] Waiting for GPU lock ({len(self.datas)} papers)...")
        async with EMBEDDING_LOCK:
            print(f"   🟢 [Action] Lock acquired! Computing embeddings...")
            embeddings = await asyncio.to_thread(self._embed_abstracts)
        
        print(f"   -> [Clustering] Grouping papers...")
        labels = await asyncio.to_thread(self._cluster_embeddings, embeddings)

        cluster_map: Dict[int, List[Any]] = {}
        for paper, label in zip(self.datas, labels):
            pid = paper.get("id")
            cluster_map.setdefault(label, []).append(pid)

        self.clusters = []
        for cid, paper_ids in sorted(cluster_map.items(), key=lambda x: x[0]):
            self.clusters.append({
                "cluster_id": int(cid),
                "paper_ids": paper_ids,
                "summary": None,
            })

    # -------------------------------
    # Step 3: Global Reasoning
    # -------------------------------
    async def run(self, task: str, max_steps: int = 15) -> dict:
        print(f"🚀 [AnalysisAgent] Starting task: {task}")
        self.init_history(task)

        print("🔷 Step 1: Per-paper analysis...")
        await self._analyze_all_papers()
        
        print("🔷 Step 2: Clustering...")
        await self._run_clustering() # 现在这里受 Lock 保护，非常安全
        print(f"   -> Generated {len(self.clusters)} clusters.")

        print("🔷 Step 3: Global reasoning...")
        all_cluster_ids = [c["cluster_id"] for c in self.clusters]
        visited_clusters = set()

        initial_instruction = (
            f"[Task Info] The clustering algorithm identified {len(self.clusters)} clusters (IDs: {all_cluster_ids}). "
            "Please analyze them one by one using `get_cluster_context`, then output the final JSON."
        )
        self.history.append(LLMMessage(role="user", content=initial_instruction))

        step = 0
        while step < max_steps:
            step += 1
            remaining = [c for c in all_cluster_ids if c not in visited_clusters]
            
            if visited_clusters and remaining:
                 last_msg = self.history[-1]
                 msg = f"[Progress] Analyzed: {list(visited_clusters)}. Please analyze remaining: {remaining}."
                 if not (last_msg.role == "user" and "Progress" in last_msg.content):
                     self.history.append(LLMMessage(role="user", content=msg))
            elif visited_clusters and not remaining:
                 msg = "[Finished] All clusters analyzed. Output final JSON."
                 if self.history[-1].content != msg:
                     self.history.append(LLMMessage(role="user", content=msg))
            
            print(f"   -> Step {step} thinking...")
            response = await self.think()
            content = response.content or ""
            tool_calls = response.tool_calls or []

            for tc in tool_calls:
                if tc.name == "get_cluster_context":
                    try: 
                        args = json.loads(tc.arguments) if isinstance(tc.arguments, str) else tc.arguments
                        visited_clusters.add(int(args.get("cluster_id")))
                    except: pass
            
            if not tool_calls:
                 try:
                    import json_repair
                    final = json_repair.loads(content)
                 except:
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    final = json.loads(match.group(0)) if match else {}
                 
                 if isinstance(final, dict):
                    self.summary = final.get("overall_summary", "")
                    cid_map = {c["cluster_id"]: c for c in self.clusters}
                    for c in final.get("clusters", []):
                        if c.get("cluster_id") in cid_map:
                            cid_map[c.get("cluster_id")]["summary"] = c.get("summary")
                    return {"summary": self.summary, "clusters": cid_map, "insights": self.paper_insights}
                 return {"summary": content, "clusters": {}, "insights": self.paper_insights}

            await self.act(tool_calls)
        
        return {"summary": "Timeout", "clusters": {}, "insights": self.paper_insights}