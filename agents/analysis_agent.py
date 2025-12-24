import asyncio
import json
import re
from typing import List, Dict, Any, Optional

from core.base_agent import BaseAgent
from core.schema import LLMMessage
from tools.cluster_tools import ClusterContextTool


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
        # Raw data: list of papers returned by RetrievalAgent
        self.datas: List[Dict[str, Any]] = datas

        # Per-paper analysis results: list[dict]
        self.paper_insights: List[Dict[str, Any]] = []

        # Clustering results
        self.clusters: List[Dict[str, Any]] = []

        # Global summary after integrating all cluster-level summaries
        self.summary: str = ""

        # Tool: cluster-level context provider
        self.cluster_context_tool = ClusterContextTool(self)
        tools = [self.cluster_context_tool]

        # System prompt: describe tool usage and final JSON format
        system_prompt = """
        You are an expert academic survey assistant (AnalysisAgent). You are given a
        collection of papers around a common research topic. For each paper, key
        information has already been extracted, and all papers have been clustered
        based on abstract embeddings.

        You can use the tool `get_cluster_context` to inspect detailed information
        (including extracted key information) for a specific cluster.

        [Your tasks]
        1. For each cluster, identify its main research direction / theme.
           - Consider aspects such as: problem formulation, methodological patterns,
             data/tasks, typical experimental setups, key findings, limitations, etc.
        2. For each cluster, write a concise summary (1–3 paragraphs) that explains:
           - The main research problem and background;
           - Typical methods/models and their commonalities or differences;
           - Representative experimental setups or application scenarios;
           - Important findings, strengths, and key limitations.
        3. After understanding all clusters, summarize the overall research landscape:
           - Relationships and differences between clusters;
           - Possible temporal or technical evolution paths;
           - Open problems and promising future research directions.

        [Tool usage hints]
        - When you need to analyze a given cluster, first call `get_cluster_context`
          and carefully read the provided information.
        - You may call the tool multiple times for different clusters.

        [Final output format — VERY IMPORTANT]
        - When you are done with all cluster-level analysis and the global reasoning,
          output exactly ONE JSON object (no extra natural language outside JSON).
        - The JSON must have the following structure:
          {
            "clusters": [
              {
                "cluster_id": <int>,
                "summary": "<English summary text for this cluster>"
              },
              ...
            ],
            "overall_summary": "<A coherent English paragraph (or several paragraphs) describing the overall research landscape across all clusters>"
          }
        - Make sure the JSON can be parsed by json.loads (no comments, no trailing commas).
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
    # Step 1: Per-paper key information extraction (fixed workflow)
    # -------------------------------
    async def _analyze_single_paper(self, paper: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use the LLM to extract key information from a single paper abstract.
        This step DOES NOT write into the main Agent history; instead it creates
        standalone messages and calls self.llm.chat directly.
        Returns a structured dict that always contains the original paper id.
        """
        paper_id = paper.get("id")
        title = paper.get("title", "")
        abstract = paper.get("abstract") or paper.get("summary") or ""

        system_prompt = """
        You are an assistant that helps structure information about academic papers.
        Given the title and abstract of a paper, extract the following key information
        and output STRICTLY in JSON:
        {
          "problem": "... what is the main research problem or task?",
          "method": "... what is the core method / model / framework proposed?",
          "data_or_task": "... what datasets, tasks, or application scenarios are used? "
                          "If the abstract does not name them explicitly, summarize them "
                          "at a high level.",
          "key_results": "... what key experimental results or conclusions are claimed?",
          "limitations": "... what limitations are mentioned or can be reasonably inferred? "
                          "If the abstract does not clearly mention limitations, say so.",
          "keywords": ["keyword1", "keyword2", ...]   // 3-8 English keywords
        }
        Requirements:
        - The answer MUST be in English.
        - Stay faithful to the abstract; light summarization is allowed.
        - Output STRICTLY valid JSON: no comments, no extra fields, no trailing commas,
          and no additional natural language outside the JSON.
        """

        user_content = {
            "title": title,
            "abstract": abstract,
        }

        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(
                role="user",
                content=(
                    "Please extract the key information from the following paper:\n"
                    f"{json.dumps(user_content, ensure_ascii=False)}"
                ),
            ),
        ]

        # Use LLMClient directly, bypassing BaseAgent.think (so history is not touched)
        response = await asyncio.to_thread(self.llm.chat, messages, None)

        content = response.content or ""
        result: Dict[str, Any] = {
            "id": paper_id,
            "raw_output": content,
        }

        try:
            # Robust JSON extraction
            json_str = content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].strip()
            
            parsed = json.loads(json_str)
            if isinstance(parsed, dict):
                parsed["id"] = paper_id
                return parsed
        except Exception:
            # If parsing fails, keep the raw output for potential manual inspection
            result["parse_error"] = "Failed to parse JSON from LLM output."

        return result

    async def _analyze_all_papers(self) -> None:
        """Run per-paper key information extraction in parallel for all papers."""
        tasks = [self._analyze_single_paper(p) for p in self.datas]
        self.paper_insights = await asyncio.gather(*tasks)

    # -------------------------------
    # Step 2: Abstract embeddings + clustering (fixed workflow)
    # -------------------------------
    def _embed_abstracts(self) -> List[Any]:
        """
        Compute embeddings for each paper abstract using a scientific-domain model
        (e.g., SciBERT).
        """
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
        except ImportError as e:
            raise ImportError(
                "transformers and torch are required for abstract embeddings. "
                "Please install them with: pip install transformers torch"
            ) from e

        model_name = "allenai/scibert_scivocab_uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        embeddings: List[Any] = []

        with torch.no_grad():
            for paper in self.datas:
                text = paper.get("abstract") or paper.get("summary") or ""
                if not text:
                    embeddings.append(torch.zeros(model.config.hidden_size))
                    continue

                encoded = tokenizer(
                    text,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
                outputs = model(**encoded)
                # Use the [CLS] token representation as a simple sentence embedding
                cls_emb = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_emb.squeeze(0))

        # Move to CPU and convert to numpy
        return [emb.cpu().numpy() for emb in embeddings]

    def _cluster_embeddings(self, embeddings: List[Any]) -> List[int]:
        """
        Cluster the embedding vectors.
        Prefer X-means (from pyclustering); if unavailable, fall back to KMeans.
        Returns a list of cluster_id (length N, one per paper).
        """
        import numpy as np

        X = np.stack(embeddings, axis=0)
        n_samples = X.shape[0]

        # Edge case: very small number of samples
        if n_samples <= 1:
            return [0] * n_samples

        # Try X-means first
        try:
            from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer

            initial_centers = kmeans_plusplus_initializer(
                X, 2 if n_samples >= 2 else 1
            ).initialize()
            xm = xmeans(X.tolist(), initial_centers)
            xm.process()
            clusters = xm.get_clusters()

            labels = [0] * n_samples
            for cid, indices in enumerate(clusters):
                for idx in indices:
                    labels[idx] = cid
            return labels
        except ImportError:
            # Fall back to KMeans
            try:
                from sklearn.cluster import KMeans
            except ImportError as e:
                raise ImportError(
                    "pyclustering or scikit-learn is required for clustering. "
                    "Install with: pip install pyclustering or pip install scikit-learn"
                ) from e

            import math
            k = max(2, min(8, int(math.sqrt(max(n_samples / 2, 1)))))
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(X)
            return labels.tolist()

    def _run_clustering(self) -> None:
        """Run embeddings + clustering for all abstracts and populate self.clusters."""
        embeddings = self._embed_abstracts()
        labels = self._cluster_embeddings(embeddings)

        # Build mapping: cluster_id -> paper_ids
        cluster_map: Dict[int, List[Any]] = {}
        for paper, label in zip(self.datas, labels):
            pid = paper.get("id")
            cluster_map.setdefault(label, []).append(pid)

        # Normalize into a list[dict]
        self.clusters = []
        for cid, paper_ids in sorted(cluster_map.items(), key=lambda x: x[0]):
            self.clusters.append(
                {
                    "cluster_id": int(cid),
                    "paper_ids": paper_ids,
                    "summary": None,
                }
            )

    # -------------------------------
    # Step 3: Agent-based cluster theme reasoning and global summary
    # -------------------------------
    async def run(self, task: str, max_steps: int = 10) -> dict:
        """
        Full workflow of AnalysisAgent:
        1. Run per-paper key information extraction.
        2. Run abstract embeddings and clustering.
        3. Use the standard Agent workflow to summarize clusters and global landscape.
        """
        print(f"🚀 [AnalysisAgent] Starting task: {task}")

        # Initialize main history for Step 3
        # 【关键修改】在 run 的时候初始化 history，而不是在 init
        self.init_history(task)

        # ---- Step 1: per-paper analysis ----
        print("🔷 Step 1: Per-paper key information extraction (per-paper analysis)...")
        await self._analyze_all_papers()
        print(f"   -> Completed key information extraction for {len(self.paper_insights)} papers.")

        # ---- Step 2: embeddings + clustering ----
        print("🔷 Step 2: Abstract embeddings + clustering...")
        self._run_clustering()
        print(f"   -> Generated {len(self.clusters)} clusters.")

        # ---- Step 3: Agent-based cluster and global summary ----
        print("🔷 Step 3: Agent workflow to generate cluster summaries and global overview...")

        # 【关键修改】
        # 显式地告诉 Agent 只有哪些 Cluster ID 是有效的，防止它幻觉去查不存在的 Cluster。
        # 我们把这段信息作为 System Notification 注入到 history 中。
        cluster_ids = [c["cluster_id"] for c in self.clusters]
        cluster_info_str = (
            f"System Notification: The clustering algorithm has identified {len(self.clusters)} clusters. "
            f"The valid Cluster IDs are: {cluster_ids}. "
            "Please use the tool `get_cluster_context` for EACH of these valid IDs to analyze them. "
            "Do NOT try to access other IDs."
        )
        self.history.append(LLMMessage(role="user", content=cluster_info_str))

        step = 0
        final_json: Optional[Dict[str, Any]] = None

        while step < max_steps:
            step += 1
            print(f"\n🔹 --- Analysis step {step} ---")

            # 1. Think
            response = await self.think()

            if response.content:
                print(f"🤖 Thought: {response.content[:200]}...\n")

            # 2. Termination condition: no tool calls -> treat as final JSON summary
            if not response.tool_calls:
                print("✅ Analysis completed (no further tool calls).")
                cid_to_cluster = {c["cluster_id"]: c for c in self.clusters}

                content = response.content or ""
                try:
                    # 【关键修改】增加鲁棒的 JSON 提取逻辑
                    json_str = content
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        json_str = content.split("```")[1].strip()
                    
                    final_json = json.loads(json_str)
                except Exception as e:
                    print(f"⚠️ Failed to parse final JSON output: {e}")
                    # Even if parsing fails, still return the raw text
                    self.summary = content
                    return {"summary": self.summary, "clusters": cid_to_cluster, "insights": self.paper_insights}

                # Parse JSON and write back into self.clusters and self.summary
                clusters_info = final_json.get("clusters", [])
                overall_summary = final_json.get("overall_summary", "")

                # Update cluster summaries                
                for item in clusters_info:
                    cid = item.get("cluster_id")
                    summary_text = item.get("summary")
                    if cid is not None and cid in cid_to_cluster:
                        cid_to_cluster[cid]["summary"] = summary_text

                self.summary = overall_summary

                return {"summary": self.summary, "clusters": cid_to_cluster, "insights": self.paper_insights}

            # 3. If there are tool calls, execute them (Act)
            await self.act(response.tool_calls)

        # If max steps are reached without a final JSON
        print("⚠️ Max steps reached. The agent failed to generate a final JSON summary.")
        self.summary = "⚠️ Max steps reached. The agent failed to generate a final JSON summary."
        return {"summary": self.summary, "clusters": {c["cluster_id"]: c for c in self.clusters}, "insights": self.paper_insights}
