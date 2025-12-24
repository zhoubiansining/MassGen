import json
import os
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, Any, List, Optional

from core.tool import BaseTool

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