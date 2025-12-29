"""Writing Agent - English prompts + multi-temperature candidate generation."""

import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import openai


# ==================== 配置区域 ====================


@dataclass
class ModelConfig:
    """Model configuration"""

    name: str
    api_key: str
    base_url: str
    temperature: float = 0.5
    max_tokens: int = 8192
    top_p: float = 0.9


MODEL_CONFIGS = {
    "qwen-max": ModelConfig(
        name="qwen-max",
        api_key="your-dashscope-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192,
    ),
    "deepseek-chat": ModelConfig(
        name="deepseek-chat",
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com/v1",
        temperature=0.5,
        max_tokens=8192,
    ),
    "glm-4": ModelConfig(
        name="glm-4",
        api_key="your-zhipu-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192,
    ),
    "kimi-k2": ModelConfig(
        name="Kimi-K2",
        api_key="key-here",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192,
    ),
    "doubao-pro": ModelConfig(
        name="doubao-pro-32k",
        api_key="your-doubao-api-key",
        base_url="https://llmapi.paratera.com/v1/",
        temperature=0.5,
        max_tokens=8192,
    ),
}


# ==================== 提示词模板 ====================

WRITING_AGENT_SYSTEM_PROMPT = """You are an expert academic survey writer. Always write in English only.

**Task**: Based on the provided structured cluster summaries, write a coherent academic survey section.

**Guidelines**
1. Use formal academic English with consistent style.
2. Cite using [paper_id] in-text markers.
3. Organize with clear topic sentences and smooth transitions.
4. Include comparative analysis where appropriate.
5. Every factual claim must be supported by a citation.

**Output Requirements**
- Length: 400–700 words per section
- Structure: Background → Methods → Comparison → Challenges → Future directions
- Citations: In-text [paper_id]; list references at the end
- Style: {style_type}

**Quality Checklist**
□ All claims have citations
□ Logical flow between paragraphs
□ No contradictions
□ Consistent terminology
□ Clear section structure
"""

REVIEWER_PROMPT = """You are a senior academic reviewer. Review the draft focusing on:

1. Coverage: important papers and concepts covered?
2. Factuality: statements accurate and supported by citations?
3. Coherence: logical flow and structure?
4. Academic style: formal tone and citation format?
5. Novelty: trends, gaps, and future directions identified?

Return JSON:
{
  "strengths": ["..."],
  "weaknesses": ["..."],
  "improvements": [
    {"issue": "...", "suggestion": "...", "location": "..."}
  ]
}
"""


# ==================== LLM 客户端封装 ====================

class LLMClient:
    """Unified LLM client for multiple providers"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )

    def chat(self, messages: List[Dict], temperature: Optional[float] = None) -> str:
        """Synchronous call"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                extra_body={"enable_thinking": False}
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] LLM call failed: {e}")
            raise

    async def chat_async(self, messages: List[Dict], temperature: Optional[float] = None) -> str:
        """Async call (wrap sync with executor)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.chat,
            messages,
            temperature
        )


# ==================== 写作代理核心 ====================

class WritingAgent:
    """
    Writing Agent - generate high-quality survey drafts

    Features:
    1. Generate multiple candidates (styles/temperatures)
    2. Self-reflection refinement
    3. Citation extraction
    """

    def __init__(self, model_config: ModelConfig, style: str = "narrative"):
        """Initialize Writing Agent"""
        self.llm_client = LLMClient(model_config)
        self.model_config = model_config
        self.style = style

        # System prompt
        self.system_prompt = WRITING_AGENT_SYSTEM_PROMPT.format(
            style_type=style
        )

    def _build_writing_prompt(self, cluster_summaries: Dict) -> str:
        """Build the writing prompt"""
        prompt = f"""Write a survey section based on the structured cluster summaries below.

# Input
{json.dumps(cluster_summaries, ensure_ascii=False, indent=2)}

# Requirements
1. Length 400–700 words
2. Structure: Background → Methods → Comparison → Challenges → Future directions
3. Use [paper_id] for in-text citations
4. Every factual statement must have citation support

# Output
1. Survey text with in-text citations
2. Reference list

Begin writing:
"""
        return prompt

    def generate_draft(
        self,
        cluster_summaries: Dict,
        temperature: Optional[float] = None
    ) -> Dict:
        """Generate a single draft"""
        print(f"[INFO] Generating draft (temperature={temperature or self.model_config.temperature}) ...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_writing_prompt(cluster_summaries)}
        ]

        draft_content = self.llm_client.chat(messages, temperature)

        # Extract citations
        citations = self._extract_citations(draft_content)

        result = {
            "content": draft_content,
            "citations": citations,
            "temperature": temperature or self.model_config.temperature,
            "style": self.style,
            "timestamp": datetime.now().isoformat()
        }

        print(f"[SUCCESS] Draft generated with {len(citations)} citations")
        return result

    async def generate_draft_async(
        self,
        cluster_summaries: Dict,
        temperature: Optional[float] = None
    ) -> Dict:
        """Generate draft asynchronously"""
        print(f"[INFO] Async draft generation (temperature={temperature or self.model_config.temperature}) ...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._build_writing_prompt(cluster_summaries)}
        ]

        draft_content = await self.llm_client.chat_async(messages, temperature)
        citations = self._extract_citations(draft_content)

        result = {
            "content": draft_content,
            "citations": citations,
            "temperature": temperature or self.model_config.temperature,
            "style": self.style,
            "timestamp": datetime.now().isoformat()
        }

        print(f"[SUCCESS] Async draft generated with {len(citations)} citations")
        return result

    def generate_multiple_candidates(
        self,
        cluster_summaries: Dict,
        num_candidates: int = 3,
        temperature_range: Tuple[float, float] = (0.3, 0.9)
    ) -> List[Dict]:
        """Generate multiple candidates with a temperature range"""
        print(f"\n[INFO] Generating {num_candidates} candidates ...")

        candidates = []
        temp_min, temp_max = temperature_range
        temp_step = (temp_max - temp_min) / (num_candidates - 1) if num_candidates > 1 else 0

        for i in range(num_candidates):
            temperature = temp_min + i * temp_step
            draft = self.generate_draft(cluster_summaries, temperature)
            draft["candidate_id"] = i + 1
            candidates.append(draft)

        print(f"[SUCCESS] Generated {num_candidates} candidates\n")
        return candidates

    def generate_candidates_by_temps(
        self,
        cluster_summaries: Dict,
        temps: List[float],
        per_temp: int = 3
    ) -> List[Dict]:
        """Generate candidates for fixed temperatures (e.g., 0.3/0.4/0.5)."""
        print(f"\n[INFO] Generating candidates for temps={temps}, {per_temp} each ...")
        candidates: List[Dict] = []
        cid = 1
        for temp in temps:
            for _ in range(per_temp):
                draft = self.generate_draft(cluster_summaries, temperature=temp)
                draft["candidate_id"] = cid
                candidates.append(draft)
                cid += 1
        print(f"[SUCCESS] Generated {len(candidates)} candidates\n")
        return candidates

    async def generate_multiple_candidates_async(
        self,
        cluster_summaries: Dict,
        num_candidates: int = 3,
        temperature_range: Tuple[float, float] = (0.3, 0.9)
    ) -> List[Dict]:
        """Generate multiple candidates asynchronously (faster)."""
        print(f"\n[INFO] Generating {num_candidates} candidates asynchronously ...")

        temp_min, temp_max = temperature_range
        temp_step = (temp_max - temp_min) / (num_candidates - 1) if num_candidates > 1 else 0

        # Build async tasks
        tasks = [
            self.generate_draft_async(cluster_summaries, temp_min + i * temp_step)
            for i in range(num_candidates)
        ]

        # Run in parallel
        candidates = await asyncio.gather(*tasks)

        # Add candidate ids
        for i, draft in enumerate(candidates):
            draft["candidate_id"] = i + 1

        print(f"[SUCCESS] Generated {num_candidates} candidates asynchronously\n")
        return candidates

    def refine_draft(self, draft: str, feedback: Dict) -> Dict:
        """Refine draft based on feedback (reflection)."""
        print("[INFO] Refining draft based on feedback ...")

        improvement_points = "\n".join([
            f"- {imp['issue']}: {imp['suggestion']}"
            for imp in feedback.get("improvements", [])
        ])

        refine_prompt = f"""Please refine the survey draft to fix the issues below:

# Issues
{improvement_points}

# Original draft
{draft}

# Task
Revise the draft per feedback, keep academic tone and citation format, and output the full revised draft.
"""

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": refine_prompt}
        ]

        refined_content = self.llm_client.chat(messages)
        citations = self._extract_citations(refined_content)

        result = {
            "content": refined_content,
            "citations": citations,
            "refined": True,
            "timestamp": datetime.now().isoformat()
        }

        print("[SUCCESS] Draft refinement completed")
        return result

    def self_critique(self, draft: str) -> Dict:
        """Self-critique and improvement suggestions"""
        print("[INFO] Running self-critique ...")

        messages = [
            {"role": "system", "content": REVIEWER_PROMPT},
            {"role": "user", "content": f"Please review the survey draft below:\n\n{draft}"}
        ]

        critique_text = self.llm_client.chat(messages, temperature=0.2)

        # Try to parse JSON (if model returns proper format)
        try:
            # Extract JSON part (may be wrapped in markdown)
            if "```json" in critique_text:
                json_start = critique_text.find("```json") + 7
                json_end = critique_text.find("```", json_start)
                json_text = critique_text[json_start:json_end].strip()
            elif "{" in critique_text:
                json_start = critique_text.find("{")
                json_end = critique_text.rfind("}") + 1
                json_text = critique_text[json_start:json_end]
            else:
                json_text = critique_text

            critique = json.loads(json_text)
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            critique = {
                "strengths": [],
                "weaknesses": [],
                "improvements": [{"issue": "parse failed", "suggestion": critique_text}]
            }

        print(f"[SUCCESS] Self-critique done: {len(critique.get('improvements', []))} improvements")
        return critique

    def refine_with_reflection(
        self,
        draft: str,
        max_iterations: int = 3
    ) -> Dict:
        """Multi-round self-refinement"""
        print(f"\n[INFO] Start multi-round refinement (max {max_iterations})...\n")

        current_draft = draft
        history = []

        for iteration in range(max_iterations):
            print(f"--- Iteration {iteration + 1} ---")

            # Step 1: self-critique
            critique = self.self_critique(current_draft)

            # Record history
            history.append({
                "iteration": iteration + 1,
                "critique": critique,
                "draft": current_draft
            })

            # Stop when no further improvements
            if not critique.get("improvements"):
                print("[INFO] No further improvements, stop.")
                break

            # Step 2: refine
            refined = self.refine_draft(current_draft, critique)
            current_draft = refined["content"]

        print(f"\n[SUCCESS] Refinement done after {len(history)} rounds\n")

        return {
            "final_draft": current_draft,
            "iterations": len(history),
            "history": history,
            "timestamp": datetime.now().isoformat()
        }

    def _extract_citations(self, text: str) -> List[str]:
        """Extract in-text citations of form [paper_id]"""
        import re
        citations = re.findall(r'\[([^\]]+)\]', text)
        return list(set(citations))  # deduplicate

    def validate_citations(
        self,
        draft: str,
        available_papers: Dict
    ) -> Dict:
        """Validate citations against available paper ids"""
        citations = self._extract_citations(draft)
        available_ids = set(available_papers.keys())

        invalid = [c for c in citations if c not in available_ids]
        missing_important = available_ids - set(citations)

        result = {
            "valid": len(invalid) == 0,
            "total_citations": len(citations),
            "invalid_citations": invalid,
            "missing_papers": list(missing_important),
            "coverage_rate": len(set(citations) & available_ids) / len(available_ids) if available_ids else 0
        }

        if invalid:
            print(f"[WARNING] Found {len(invalid)} invalid citations: {invalid}")
        if missing_important:
            print(f"[WARNING] {len(missing_important)} important papers not cited")

        return result
