import time
import requests
import json
import os
import xml.etree.ElementTree as ET
import math
from datetime import datetime
from typing import Any, Dict, List, Set, Optional

from core.tool import BaseTool
try:
    from sentence_transformers import SentenceTransformer, util
    HAS_EMBEDDING_MODEL = True
except ImportError:
    HAS_EMBEDDING_MODEL = False
    print("⚠️ Warning: sentence-transformers not found. Semantic ranking will be disabled.")

from core.tool import BaseTool

# all-MiniLM-L6-v2

# ==========================================
# 辅助函数：查询处理与评分 (源自你之前的项目)
# ==========================================

def process_query(query: str, max_length: int = 300) -> str:
    """防止查询过长，Arxiv API 对 URL 长度有限制"""
    if len(query) <= max_length:
        return query
    words = query.split()
    processed = []
    current_len = 0
    for word in words:
        if current_len + len(word) + 1 <= max_length:
            processed.append(word)
            current_len += len(word) + 1
        else:
            break
    return ' '.join(processed)

def calculate_paper_score(paper: dict, keywords: List[str] = None) -> float:
    """
    计算论文综合得分。
    权重策略: 相关性(0.5) + 时效性(0.2) + 顶会加分(0.3)
    """
    # 1. 相关性 (标题+摘要中关键词出现的次数)
    relevance_score = 0
    if keywords:
        # 将关键词和文本都转小写比较
        text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
        count = sum(1 for kw in keywords if kw.lower() in text)
        relevance_score = min(count, 5) # 上限 5 分

    # 2. 时效性 (越新分越高)
    pub_date = paper.get("published_date_obj")
    recency_score = 1
    if pub_date:
        days_old = (datetime.now() - pub_date).days
        if days_old <= 30: recency_score = 5
        elif days_old <= 90: recency_score = 4
        elif days_old <= 180: recency_score = 3
        elif days_old <= 365: recency_score = 2

    # 3. 顶会/期刊加分 (从 comment 字段判断)
    bonus_terms = ["cvpr", "iclr", "neurips", "icml", "acl", "emnlp", "ijcai", "aaai", "nature", "science", "accepted"]
    comment = paper.get("comment", "").lower()
    pub_bonus = 5 if any(term in comment for term in bonus_terms) else 0

    # 加权求和
    return (0.5 * relevance_score) + (0.2 * recency_score) + (0.3 * pub_bonus)

# ==========================================
# Google 搜索工具 (支持 Serper 和 DuckDuckGo)
# ==========================================

class GoogleSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "google_search"
    
    @property
    def description(self) -> str:
        return "Search the web using Google. Useful for getting recent news, blog posts, github repos, or broad context."

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string", 
                    "description": "The search query string."
                },
                "num_results": {
                    "type": "integer", 
                    "description": "Number of results to return.", 
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str, num_results: int = 5) -> str:
        """
        优先尝试使用 Serper API (需要 SERPER_API_KEY 环境变量)。
        如果失败或未配置，降级使用 duckduckgo_search (免费)。
        """
        api_key = os.getenv("SERPER_API_KEY")
        
        # 1. 尝试 Serper (推荐)
        if api_key:
            try:
                url = "https://google.serper.dev/search"
                
                # [修改点] 增加了 gl 和 hl 参数，确保搜索结果是 英文/美国地区
                payload = json.dumps({
                    "q": query,
                    "num": num_results,
                    "gl": "us", 
                    "hl": "en"
                })
                
                headers = {
                    'X-API-KEY': api_key, 
                    'Content-Type': 'application/json'
                }
                
                # 使用 requests 库发送请求（比 http.client 更稳健）
                response = requests.request("POST", url, headers=headers, data=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    # 我们只需要 organic (自然搜索结果)
                    for item in data.get("organic", []):
                        results.append({
                            "title": item.get("title"),
                            "link": item.get("link"),
                            "snippet": item.get("snippet"),
                            "date": item.get("date", "") # 有些新闻会有日期
                        })
                        
                    if not results:
                        return "Google Search returned no organic results."
                        
                    return json.dumps(results, indent=2, ensure_ascii=False)
                else:
                    print(f"[GoogleTool] Serper API Error: {response.status_code} - {response.text}")
                    # 如果 API 报错，不要返回错误文本，而是让它继续往下走尝试 DuckDuckGo
            except Exception as e:
                print(f"[GoogleTool] Serper API failed: {e}. Falling back to DuckDuckGo.")

        # 2. 降级尝试 DuckDuckGo (无需 Key)
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                # DDGS 返回的是迭代器
                ddg_gen = ddgs.text(query, max_results=num_results)
                for r in ddg_gen:
                    results.append({
                        "title": r.get("title"),
                        "link": r.get("href"),
                        "snippet": r.get("body")
                    })
            if not results:
                return "No results found on Google/DDG."
            return json.dumps(results, indent=2, ensure_ascii=False)
            
        except ImportError:
            return "Error: Please install `duckduckgo-search` library: pip install duckduckgo-search"
        except Exception as e:
            return f"Error executing Google Search: {str(e)}"

# ==========================================
# Arxiv 搜索工具 (集成评分与重排)
# ==========================================

class ArxivSearchTool(BaseTool):
    """
    增强版 Arxiv 搜索工具。
    Agent 调用时可以提供 'keywords'，工具内部会利用这些关键词对 Arxiv 返回的原始结果
    进行加权评分 (相关性 + 时效性 + 顶会)，从而返回质量更高的论文。
    """
    @property
    def name(self) -> str:
        return "arxiv_search"

    @property
    def description(self) -> str:
        return (
            "Search for academic papers on ArXiv. "
            "You MUST provide a 'query' for the API search. "
            "You SHOULD provide 'keywords' (list of strings) to help rank the importance of papers."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The main search query for ArXiv API (e.g. 'LLM memory')."
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of important keywords to score the relevance of papers (e.g. ['transformer', 'KV cache'])."
                },
                "max_results": {
                    "type": "integer",
                    "description": "Final number of top-ranked papers to return.",
                    "default": 10
                },
                "sort_by_date": {
                    "type": "boolean",
                    "description": "Force sort by date (ignores scoring).", 
                    "default": False
                }
            },
            "required": ["query"]
        }

    def execute(self, query: str, keywords: List[str] = None, max_results: int = 10, sort_by_date: bool = False) -> str:
        base_url = "http://export.arxiv.org/api/query"
        processed_query = process_query(query)
        
        # 策略：为了能进行有效的重排，我们需要从 API 拉取比最终需求更多的论文
        # 比如用户只要 10 篇，我们拉 30 篇，然后算出最好的 10 篇返回
        fetch_limit = max_results * 3 if (not sort_by_date and keywords) else max_results
        
        params = {
            "search_query": f"all:{processed_query}",
            "start": 0,
            "max_results": fetch_limit,
            "sortBy": "submittedDate" if sort_by_date else "relevance",
            "sortOrder": "descending"
        }

        # 1. 带有重试机制的请求
        response = None
        for attempt in range(3):
            try:
                r = requests.get(base_url, params=params, timeout=20)
                if r.status_code == 200:
                    response = r
                    break
            except Exception as e:
                print(f"[ArxivTool] Attempt {attempt+1} failed: {e}")
        
        if not response:
            return "Error: Unable to connect to ArXiv API."

        # 2. 解析 XML 数据
        try:
            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
            
            papers = []
            
            for entry in root.findall("atom:entry", ns):
                # 提取基础元数据
                paper = {
                    "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
                    "summary": entry.find("atom:summary", ns).text.strip().replace("\n", " "),
                    "url": entry.find("atom:id", ns).text.strip(),
                    "published": entry.find("atom:published", ns).text.strip(),
                    "authors": [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)],
                    "comment": entry.find("arxiv:comment", ns).text if entry.find("arxiv:comment", ns) is not None else ""
                }
                
                # 处理日期
                try:
                    paper["published_date_obj"] = datetime.strptime(paper["published"], "%Y-%m-%dT%H:%M:%SZ")
                    paper["published"] = paper["published_date_obj"].strftime("%Y-%m-%d")
                except:
                    paper["published_date_obj"] = datetime.now()

                # 3. 计算得分
                if not sort_by_date and keywords:
                    paper["score"] = calculate_paper_score(paper, keywords)
                else:
                    paper["score"] = 0 # 不评分
                
                # 清理临时对象
                del paper["published_date_obj"]
                papers.append(paper)

            # 4. 排序与截断
            if not sort_by_date and keywords:
                # 按我们计算的 score 降序排列
                papers.sort(key=lambda x: x["score"], reverse=True)
            
            # 只取前 max_results 个
            final_papers = papers[:max_results]
            
            if not final_papers:
                return "No papers found."

            # 构建返回结果
            optimized_results = []
            for p in final_papers:
                optimized_results.append({
                    "title": p["title"],
                    "authors": p["authors"][:5], # 作者保留前5个吧，有时候前3个不够
                    "published": p["published"],
                    "url": p["url"],
                    "score": round(p.get("score", 0), 2),
                    # [修改点]：保留完整摘要，不再截断
                    "summary": p["summary"] 
                })
                
            return json.dumps(optimized_results, indent=2, ensure_ascii=False)

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error parsing ArXiv data: {str(e)}"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error parsing ArXiv data: {str(e)}"

class SemanticScholarSearchTool(BaseTool):
    """
    Semantic Scholar 搜索工具。
    功能：
    1. 根据关键词搜索论文 (Seed Papers)。
    2. 自动扩展：获取这些论文的"参考文献 (References)"和"被引文献 (Citations)"。
    3. 返回包含摘要、作者、年份、引用数的详细信息。
    """

    def __init__(self):
        # 即使没有 API Key，S2 也有一定的免费额度，但在并发高时容易由 429 错误
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {"x-api-key": self.api_key} if self.api_key else {}

    @property
    def name(self) -> str:
        return "semantic_scholar_search"

    @property
    def description(self) -> str:
        return (
            "Search for academic papers using Semantic Scholar. "
            "Returns the search results PLUS their direct references and citations (1-hop expansion). "
            "Excellent for finding connected research and highly cited papers."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Research topic or keywords (e.g., 'Large Language Models memory')."
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of seed papers to search for initially (suggest 3-5).",
                    "default": 5
                }
            },
            "required": ["query"]
        }

    def _make_request(self, endpoint: str, params: Dict = None, method="GET", json_data=None) -> Any:
        """封装请求逻辑，包含简单的重试机制"""
        url = f"{self.base_url}/{endpoint}"
        for attempt in range(3):
            try:
                if method == "GET":
                    response = requests.get(url, headers=self.headers, params=params, timeout=30)
                else:
                    response = requests.post(url, headers=self.headers, json=json_data, params=params, timeout=30)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limit hit, wait and retry
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    print(f"[S2 Error] {response.status_code}: {response.text}")
            except Exception as e:
                print(f"[S2 Exception] {e}")
                time.sleep(1)
        return None

    def execute(self, query: str, limit: int = 5) -> str:
        # 1. 搜索种子论文 (Seed Papers)
        # 我们只获取 ID，稍后批量获取详细信息，这样流程更统一
        search_params = {"query": query, "limit": limit, "fields": "paperId,title"}
        search_data = self._make_request("paper/search", params=search_params)
        
        if not search_data or "data" not in search_data or not search_data["data"]:
            return "No papers found on Semantic Scholar."

        seed_papers = search_data["data"]
        all_paper_ids: Set[str] = set()
        
        # 记录种子论文 ID
        for p in seed_papers:
            if p.get("paperId"):
                all_paper_ids.add(p["paperId"])

        print(f"[S2] Found {len(seed_papers)} seed papers. Expanding graph...")

        # 2. 图扩展 (Graph Expansion) - 获取引用和被引
        # 为了效率，我们使用 Batch API 或者针对每个 seed paper 获取连接
        # 这里为了精准控制质量，我们对每个 seed paper 分别请求引用/被引，但限制数量
        
        # 我们希望获取 seed papers 的 references (它引用的) 和 citations (引用它的)
        # 注意：Semantic Scholar 的 Graph API 在 batch 模式下不支持直接展开所有引用，需要分步
        
        related_ids: Set[str] = set()
        
        for seed in seed_papers:
            pid = seed["paperId"]
            if not pid: continue

            # 获取该论文的详情，包含 references 和 citations 的简要列表
            # limit=10 表示只取前 10 个引用/被引，按影响力排序
            details_params = {
                "fields": "references.paperId,references.isInfluential,citations.paperId,citations.isInfluential",
                "limit": 10 
            }
            paper_graph = self._make_request(f"paper/{pid}", params=details_params)
            
            if paper_graph:
                # 收集 References
                refs = paper_graph.get("references", [])
                for ref in refs:
                    if ref.get("paperId"):
                        # 策略：优先保留 "isInfluential" (有影响力的引用) 或者全部保留
                        # 这里为了拓宽视野，全部保留前10个
                        related_ids.add(ref["paperId"])
                
                # 收集 Citations
                cites = paper_graph.get("citations", [])
                for cite in cites:
                    if cite.get("paperId"):
                        related_ids.add(cite["paperId"])

        # 合并所有 ID (种子 + 扩展)
        total_ids = list(all_paper_ids.union(related_ids))
        
        # 限制总数量，防止下文过长 (例如限制 50 篇)
        if len(total_ids) > 50:
            total_ids = total_ids[:50]

        print(f"[S2] Total unique papers after expansion: {len(total_ids)}. Fetching details...")

        # 3. 批量获取所有论文的详细元数据 (Batch Details Fetch)
        # 使用 POST /paper/batch
        batch_fields = "paperId,title,abstract,year,authors,citationCount,url,venue"
        batch_body = {"ids": total_ids}
        
        details_resp = self._make_request("paper/batch", params={"fields": batch_fields}, method="POST", json_data=batch_body)
        
        if not details_resp:
            return "Error fetching paper details."

        # 4. 格式化输出
        formatted_results = []
        for item in details_resp:
            if not item: continue # S2 可能会返回 null
            
            # 简单的清洗
            abstract = item.get("abstract")
            if not abstract: continue # 没有摘要的论文对 Agent 价值不大，过滤掉
            
            entry = {
                "title": item.get("title"),
                "year": item.get("year"),
                "citationCount": item.get("citationCount", 0),
                "authors": [a["name"] for a in item.get("authors", [])[:3]], # 只取前3作者
                "abstract": abstract,
                "url": item.get("url") or f"https://www.semanticscholar.org/paper/{item.get('paperId')}",
                "source": "semantic_scholar",
                # 标记它是种子论文还是扩展出来的
                "type": "seed" if item.get("paperId") in all_paper_ids else "related"
            }
            formatted_results.append(entry)

        # 按引用数降序排列，保证最重要的论文在前面
        formatted_results.sort(key=lambda x: (x.get("citationCount") or 0), reverse=True)

        return json.dumps(formatted_results, indent=2, ensure_ascii=False)


class HybridGraphSearchTool(BaseTool):
    """
    混合图谱搜索工具 v3.0 (ArXiv + S2 + Semantic Re-ranking)
    
    特性：
    1. 无 Key 模式适配 (Rate Limit + Fallback)。
    2. 混合排序：结合了 [语义相关性] + [引用影响力] + [种子加权]。
    3. 本地 Embedding：使用 all-MiniLM-L6-v2 进行语义重排。
    """

    def __init__(self):
        self.s2_api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self.s2_base_url = "https://api.semanticscholar.org/graph/v1"
        self.s2_headers = {"x-api-key": self.s2_api_key} if self.s2_api_key else {}
        self.sleep_time = 1.1 if not self.s2_api_key else 0.1
        
        # 初始化 Embedding 模型 (单例模式，避免重复加载)
        self.encoder = None
        if HAS_EMBEDDING_MODEL:
            # 这是一个非常轻量级且强大的模型 (约 80MB)
            print("[HybridTool] Loading local embedding model: all-MiniLM-L6-v2...")
            self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    @property
    def name(self) -> str:
        return "hybrid_graph_search"

    @property
    def description(self) -> str:
        return (
            "Advanced academic search with semantic re-ranking. "
            "Finds ArXiv seeds, expands via Semantic Scholar citation graph, "
            "and re-ranks results based on vector similarity to the query."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Research topic query string."},
                "num_seeds": {"type": "integer", "default": 3},
                "max_results": {"type": "integer", "default": 15}
            },
            "required": ["query"]
        }

    def _safe_request(self, method: str, endpoint: str, params: Dict = None, json_data: Dict = None) -> Any:
        # ... (保持之前的 _safe_request 代码不变) ...
        url = f"{self.s2_base_url}/{endpoint}"
        for attempt in range(3):
            try:
                time.sleep(self.sleep_time)
                if method == "GET":
                    r = requests.get(url, headers=self.s2_headers, params=params, timeout=15)
                else:
                    r = requests.post(url, headers=self.s2_headers, params=params, json=json_data, timeout=15)
                
                if r.status_code == 200: return r.json()
                elif r.status_code == 429: time.sleep(2 * (attempt + 1))
                elif r.status_code == 404: return None
                else: print(f"[HybridTool] Error {r.status_code}: {r.text[:50]}"); return None
            except Exception:
                time.sleep(1)
        return None

    def _search_arxiv_seeds(self, query: str, limit: int) -> List[Dict]:
        # ... (保持之前的 _search_arxiv_seeds 代码不变) ...
        base_url = "http://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query[:300]}",
            "start": 0, "max_results": limit, 
            "sortBy": "relevance", "sortOrder": "descending"
        }
        try:
            resp = requests.get(base_url, params=params, timeout=20)
            if resp.status_code != 200: return []
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            seeds = []
            for entry in root.findall("atom:entry", ns):
                arxiv_id = entry.find("atom:id", ns).text.strip().split("/")[-1].split("v")[0]
                seeds.append({
                    "arxiv_id": arxiv_id,
                    "title": entry.find("atom:title", ns).text.strip().replace("\n", " "),
                    "summary": entry.find("atom:summary", ns).text.strip().replace("\n", " "),
                    "url": entry.find("atom:id", ns).text.strip(),
                    "source": "arxiv_seed",
                    "year": int(entry.find("atom:published", ns).text[:4])
                })
            return seeds
        except: return []

    def _calculate_semantic_scores(self, query: str, papers: List[Dict]) -> List[Dict]:
        """
        核心升级：使用余弦相似度计算相关性
        """
        if not self.encoder or not papers:
            return papers # 无法计算，原样返回

        # 1. 准备 Corpus (Title + Abstract)
        # 标题权重加倍，因为标题最重要
        corpus = [f"{p.get('title', '')} {p.get('title', '')} {p.get('abstract', '')}" for p in papers]
        
        # 2. 批量编码 (Encoding)
        # query_emb: [1, 384]
        # doc_embs:  [N, 384]
        query_emb = self.encoder.encode(query, convert_to_tensor=True)
        doc_embs = self.encoder.encode(corpus, convert_to_tensor=True)

        # 3. 计算相似度
        # scores: [1, N]
        cosine_scores = util.cos_sim(query_emb, doc_embs)[0].cpu().tolist()

        # 4. 注入分数
        for i, paper in enumerate(papers):
            paper["_semantic_score"] = cosine_scores[i]
            
        return papers

    def execute(self, query: str, num_seeds: int = 3, max_results: int = 15) -> str:
        # --- Step 1: Seeds ---
        seeds = self._search_arxiv_seeds(query, num_seeds)
        if not seeds: return "No seeds found."
        
        print(f"[HybridTool] Found {len(seeds)} seeds. Extending graph...")
        
        collected_papers = {}
        for s in seeds:
            s_id = f"ARXIV:{s['arxiv_id']}"
            collected_papers[s_id] = {**s, "paperId": s_id, "citationCount": 0}

        # --- Step 2: Expansion (No Key Mode) ---
        for seed in seeds:
            s_id = f"ARXIV:{seed['arxiv_id']}"
            # 限制扩展数量，减少噪音
            for direction in ["references", "citations"]:
                data = self._safe_request("GET", f"paper/{s_id}/{direction}", params={"limit": 8, "fields": "paperId,title,citationCount,year"})
                if data and "data" in data:
                    for item in data["data"]:
                        node = item.get("citedPaper") if direction == "references" else item.get("citingPaper")
                        if node and node.get("paperId"):
                            pid = node["paperId"]
                            if pid not in collected_papers:
                                collected_papers[pid] = {
                                    "paperId": pid,
                                    "title": node.get("title"),
                                    "citationCount": node.get("citationCount", 0),
                                    "year": node.get("year"),
                                    "source": "expanded"
                                }

        # --- Step 3: Fetch Details ---
        missing_ids = [pid for pid, p in collected_papers.items() if "abstract" not in p][:40] # 稍微放宽一点上限，让重排发挥作用
        if missing_ids:
            print(f"[HybridTool] Fetching abstracts for {len(missing_ids)} papers to rank...")
            batch_data = self._safe_request("POST", "paper/batch", params={"fields": "paperId,abstract,url,authors"}, json_data={"ids": missing_ids})
            if batch_data:
                for item in batch_data:
                    if item and item.get("paperId") in collected_papers:
                        collected_papers[item["paperId"]].update(item)

        # --- Step 4: Semantic Re-Ranking ---
        candidate_list = []
        for p in collected_papers.values():
            if not p.get("abstract") and p["source"] != "arxiv_seed": continue # 过滤无摘要
            
            entry = {
                "title": p.get("title", ""),
                "abstract": p.get("abstract", ""),
                "citation_count": p.get("citationCount", 0),
                "year": p.get("year"),
                "url": p.get("url"),
                "source": "arxiv" if p.get("source") == "arxiv_seed" else "semantic_scholar",
                "is_seed": p.get("source") == "arxiv_seed",
                "authors": [a["name"] for a in p.get("authors", [])[:3]] if p.get("authors") else []
            }
            candidate_list.append(entry)

        # 计算语义分
        if self.encoder:
            candidate_list = self._calculate_semantic_scores(query, candidate_list)
        else:
            for p in candidate_list: p["_semantic_score"] = 0

        # 混合排序逻辑
        # Score = (语义分 * 10) + (log(引用数) * 0.5) + (种子奖励 * 5)
        for p in candidate_list:
            sem_score = p.get("_semantic_score", 0)
            cit_score = math.log(p["citation_count"] + 1)
            seed_bonus = 5.0 if p["is_seed"] else 0.0
            
            # 关键：如果语义太差（<0.3），强制降权，不管引用多高
            if sem_score < 0.25: 
                final_score = sem_score  # 直接沉底
            else:
                final_score = (sem_score * 10) + (cit_score * 0.5) + seed_bonus
            
            p["_final_score"] = final_score

        candidate_list.sort(key=lambda x: x["_final_score"], reverse=True)
        result = candidate_list[:max_results]
        
        # 清理内部字段
        for r in result:
            r.pop("_semantic_score", None)
            r.pop("_final_score", None)

        print(f"[HybridTool] Re-ranking done. Returning top {len(result)} semantically relevant papers.")
        return json.dumps(result, indent=2, ensure_ascii=False)
