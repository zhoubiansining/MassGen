import json
from typing import List, Optional, Dict, Any
from core.base_agent import BaseAgent
from core.schema import LLMMessage
from tools.search_tools import GoogleSearchTool, HybridGraphSearchTool

class RetrievalAgent(BaseAgent):
    """
    Retrieval Agent (Final Version)
    策略：Scout (Google) -> Strike (Hybrid v3.0)
    特性：
    1. 强制多轮搜索 (Multiple Strikes)。
    2. 区分侦察信息 (Context) 和 最终数据 (Storage)。
    3. 目标：收集 30+ 篇高质量论文。
    """
    
    def __init__(self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None):
        # 注册 Google 和 Hybrid v3.0
        tools = [GoogleSearchTool(), HybridGraphSearchTool()]
        
        system_prompt = """
        You are an expert Research Strategist and Librarian. Your goal is to curate a comprehensive list of high-quality academic papers (Target: 30+ papers).

        ### WORKFLOW STRATEGY (Scout-and-Strike)

        **PHASE 1: SCOUT (Google Search)**
        - Use `Google Search` to find "Awesome lists", survey blogs, or GitHub repos.
        - **GOAL**: Identify at least 3-5 DISTINCT sub-topics, benchmarks, or specific paper titles .
        - **NOTE**: Do NOT treat Google results as final papers. Use them ONLY to formulate better queries.

        **PHASE 2: STRIKE (Hybrid Graph Search)**
        - You MUST perform MULTIPLE strikes (at least 3 different queries).
        - For EACH distinct topic found in Phase 1, run a separate `hybrid_graph_search`.
        - **CRITICAL**: Use the `max_results=20` argument to get more papers per search.
        - Example:
          1. `hybrid_graph_search(query="RepoBench repository level", max_results=20)`
          2. `hybrid_graph_search(query="SWE-bench software engineering agent", max_results=20)`
          3. `hybrid_graph_search(query="LLM repository code completion", max_results=20)`
        
        ### DATA COLLECTION RULES
        - Only `hybrid_graph_search` outputs are collected into the final database.
        - **DO NOT STOP** until you have collected **at least 30 unique papers**.
        - If you have fewer than 30 papers, you MUST search for a new topic or refine your query.
        """
        
        super().__init__(
            name="RetrievalAgent",
            tools=tools,
            system_prompt=system_prompt,
            model=model,
            base_url=base_url,
            api_key=api_key
        )
        
        self.raw_papers: List[Dict[str, Any]] = []

    async def run(self, task: str, max_steps: int = 15) -> List[Dict[str, Any]]:
        print(f"🚀 [{self.name}] Starting Strategy: Scout (Google) -> Strike (Hybrid v3) for: {task}")
        self.init_history(task)
        self.raw_papers = [] 
        
        step = 0
        while step < max_steps:
            step += 1
            print(f"\n🔹 --- Retrieval Step {step} ---")
            
            # 1. Think
            response = await self.think()
            
            if response.content:
                print(f"🤖 Thought: {response.content[:200]}...\n")

            # 2. Act
            if response.tool_calls:
                tool_results = await self.act(response.tool_calls)
                
                # 3. Collect Data
                for res in tool_results:
                    if not res.success: continue
                    
                    try:
                        data = json.loads(res.output)
                        if not isinstance(data, list): continue
                        
                        # [Scout] Google 结果只看不存
                        if res.name == "google_search":
                            print(f"   👀 [Scout] Analyzed {len(data)} Google results for context.")
                            continue
                        
                        # [Strike] Hybrid 结果存入数据库
                        if res.name == "hybrid_graph_search":
                            new_count = 0
                            for item in data:
                                # 查重 (基于 title 或 paper_id)
                                if not any(p.get("title") == item.get("title") for p in self.raw_papers):
                                    # 必须有摘要才收录
                                    if item.get("abstract") and len(item.get("abstract")) > 50:
                                        self.raw_papers.append(item)
                                        new_count += 1
                            print(f"   📥 [Strike] Collected {new_count} new papers via Hybrid Graph.")
                            
                    except json.JSONDecodeError:
                        pass
            
            # 4. Termination Check (关键逻辑)
            else:
                # 阈值提高到 30 篇，强制多轮搜索
                if len(self.raw_papers) >= 30: 
                    print(f"✅ Retrieval Process Finished. Collected {len(self.raw_papers)} papers.")
                    break
                
                else:
                    # 引导 Agent 继续搜索
                    if len(self.raw_papers) == 0:
                        hint = "System Notification: You have 0 papers. Move to Phase 2 immediately. Pick a benchmark name you found (e.g. SWE-bench) and use `hybrid_graph_search` with `max_results=20`."
                    else:
                        hint = (f"System Notification: You have {len(self.raw_papers)} papers, but the target is 30+. "
                                "Please perform another `hybrid_graph_search` on a DIFFERENT sub-topic or benchmark you found in Google Search. "
                                "Do NOT stop yet.")
                    
                    print(f"⚠️ Insufficient papers ({len(self.raw_papers)}/30). Forcing next strike...")
                    self.history.append(LLMMessage(role="user", content=hint))
                    continue

        # 5. Final Deduplication
        final_papers = self._deduplicate(self.raw_papers)
        print(f"\n📦 Finalizing: {len(self.raw_papers)} -> {len(final_papers)} unique papers.")
        return final_papers

    def _deduplicate(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """优先保留信息更全的版本"""
        unique = {}
        for p in papers:
            # 优先用 paper_id (S2/Arxiv ID)，其次用 title
            uid = p.get("paper_id") or p.get("title")
            
            if uid not in unique:
                unique[uid] = p
            else:
                # 如果新来的数据摘要更长，替换旧的
                old_len = len(unique[uid].get("abstract", ""))
                new_len = len(p.get("abstract", ""))
                if new_len > old_len:
                    unique[uid] = p
                    
        return list(unique.values())
