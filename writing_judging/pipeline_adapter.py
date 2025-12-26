"""将检索/分析阶段的输出转换为 WritingAgent 可接受的 cluster_summaries 结构。"""

from typing import Any, Dict, List, Optional


def _normalize_authors(authors: Any) -> List[str]:
    if authors is None:
        return []
    if isinstance(authors, str):
        return [authors]
    if isinstance(authors, (list, tuple)):
        return [str(a) for a in authors]
    return [str(authors)]


def _extract_year(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        for token in value.split("-"):
            if token.isdigit() and len(token) == 4:
                try:
                    return int(token)
                except ValueError:
                    return None
    return None


def analysis_to_cluster_summaries(
    analysis_result: Dict[str, Any],
    papers: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    将 AnalysisAgent 的输出（clusters/insights/datas）映射为 WritingAgent 所需的 cluster_summaries。

    参数:
        analysis_result: 包含 clusters、insights、summary 等字段的分析结果。
        papers: 可选，显式传入检索阶段的原始论文列表；若为空则尝试从 analysis_result["datas"] 读取。

    返回:
        符合 WritingAgent 输入格式的字典：
        {
          "cluster_0": {
             "topic": "...",
             "summary": "...",
             "papers": [ {"paper_id": "...", "title": "...", ...}, ... ]
          },
          ...
        }
    """

    clusters_raw = analysis_result.get("clusters") or []
    if isinstance(clusters_raw, dict):
        clusters = list(clusters_raw.values())
    else:
        clusters = clusters_raw

    insights = analysis_result.get("insights") or []
    insight_map = {}
    for item in insights:
        pid = item.get("id")
        if pid is not None:
            insight_map[pid] = item

    papers_source = papers or analysis_result.get("datas") or analysis_result.get("papers") or []
    paper_map = {}
    for paper in papers_source:
        pid = paper.get("id") or paper.get("paper_id")
        if pid is not None:
            paper_map[pid] = paper

    cluster_summaries: Dict[str, Dict[str, Any]] = {}

    for cluster in clusters:
        cid = cluster.get("cluster_id")
        if cid is None:
            continue

        paper_ids = cluster.get("paper_ids") or []
        topic = cluster.get("summary") or analysis_result.get("summary") or f"Cluster {cid}"

        cluster_key = f"cluster_{cid}"
        cluster_entry = {
            "topic": topic,
            "summary": topic,
            "papers": []
        }

        for pid in paper_ids:
            paper_info = paper_map.get(pid, {})
            insight = insight_map.get(pid, {})

            title = paper_info.get("title") or insight.get("title") or f"Paper {pid}"
            authors = _normalize_authors(paper_info.get("authors") or paper_info.get("author"))
            year = _extract_year(paper_info.get("year") or paper_info.get("date"))
            abstract = paper_info.get("abstract") or paper_info.get("summary")
            url = paper_info.get("url") or paper_info.get("link")

            cluster_entry["papers"].append({
                "paper_id": pid,
                "title": title,
                "authors": authors,
                "year": year,
                "key_contribution": insight.get("method") or insight.get("problem") or "",
                "abstract": abstract,
                "url": url
            })

        cluster_summaries[cluster_key] = cluster_entry

    return cluster_summaries

