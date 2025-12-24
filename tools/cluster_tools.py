from typing import List, Dict, Any

from core.tool import BaseTool


class ClusterContextTool(BaseTool):
    """
    Provide cluster-level paper context for AnalysisAgent.
    The LLM can call this tool to fetch key information of all papers in a given cluster
    in order to write cluster themes and research trajectories.
    """

    def __init__(self, agent: Any):
        # agent is expected to be an AnalysisAgent instance with attributes:
        # datas, paper_insights, clusters
        self._agent = agent

    @property
    def name(self) -> str:
        return "get_cluster_context"

    @property
    def description(self) -> str:
        return (
            "Get detailed context of a specified cluster, including paper titles, authors, "
            "years, abstracts, and extracted key information. "
            "Useful for summarizing the research theme, methodological characteristics, "
            "and development trajectory of that cluster."
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "integer",
                    "description": "ID of the cluster to analyze (0-based integer).",
                }
            },
            "required": ["cluster_id"],
        }

    def execute(self, cluster_id: int) -> str:
        # Ensure clustering has been done
        if not self._agent.clusters:
            return "Cluster information is not ready yet. Please run clustering first."

        # Find the specified cluster
        cluster = None
        for c in self._agent.clusters:
            if c.get("cluster_id") == cluster_id:
                cluster = c
                break

        if cluster is None:
            return f"Cluster with id {cluster_id} not found."

        paper_id_list = cluster.get("paper_ids", [])
        id_to_paper = {p.get("id"): p for p in self._agent.datas}

        # Per-paper insight indexed by id
        id_to_insight = {}
        for item in self._agent.paper_insights:
            pid = item.get("id")
            if pid is not None:
                id_to_insight[pid] = item

        lines: List[str] = []
        lines.append(f"Cluster ID: {cluster_id}")
        lines.append("Paper list:")

        for pid in paper_id_list:
            paper = id_to_paper.get(pid, {})
            insight = id_to_insight.get(pid, {})
            title = paper.get("title", "N/A")
            date = paper.get("date", "")
            authors = ", ".join(paper.get("authors", []))
            abstract = paper.get("abstract") or paper.get("summary") or ""

            lines.append("-" * 60)
            lines.append(f"Paper ID: {pid}")
            lines.append(f"Title: {title}")
            if date:
                lines.append(f"Date/Year: {date}")
            if authors:
                lines.append(f"Authors: {authors}")
            if abstract:
                lines.append(f"Abstract: {abstract}")

            if insight:
                lines.append("Extracted key information:")
                for key in [
                    "problem",
                    "method",
                    "data_or_task",
                    "key_results",
                    "limitations",
                    "keywords",
                ]:
                    val = insight.get(key)
                    if val:
                        lines.append(f"{key}: {val}")

        return "\n".join(lines)


