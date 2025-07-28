import re
from typing import List, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

UBR5_KEYWORDS = [
    r"\bUBR5\b", r"\bUbr5\b", r"ubiquitin.*ligase", r"EDD protein", r"EDD1", r"EDD-1", r"EDD/UBR5"
]

def is_ubr5_related(text: str) -> bool:
    """
    Returns True if the text is related to UBR-5 based on keyword matching.
    """
    if not isinstance(text, str):
        return False
    for kw in UBR5_KEYWORDS:
        if re.search(kw, text, re.IGNORECASE):
            return True
    return False

class MetaHypothesisGenerator:
    """
    Meta-hypothesis generator that takes a user prompt and creates 5 different prompts
    to send to the actual hypothesis generator for diverse hypothesis generation.
    """
    def __init__(self, model=None):
        self.model = model  # Gemini client

    def build_meta_prompt(self, user_prompt: str) -> str:
        """Build prompt for generating 5 different meta-hypotheses from user input."""
        prompt = f"""
# Meta-Hypothesis Generator Prompt

## Role
You are an expert research strategist specializing in UBR-5 protein research and Dr. Xiaojing Ma's laboratory at Weill Cornell Medicine. Your task is to take a user's research query and break it down into 5 distinct, complementary research directions.

## Task
Given the user's research query, generate exactly 5 different meta-hypotheses that represent diverse angles and approaches to the same research area. Each meta-hypothesis should be:
- Focused on a specific aspect or mechanism
- Complementary to the others (not redundant)
- Feasible for Dr. Ma's lab to investigate
- Novel and scientifically interesting

## Guidelines
- Focus on different molecular mechanisms, cellular processes, or therapeutic approaches
- Consider different experimental methodologies
- Vary the scope from molecular to cellular to organismal levels
- Ensure each direction is distinct but related to the core topic

## Output Format
Provide exactly 5 meta-hypotheses, numbered 1-5. Each should be a clear, specific research direction.

User Query: {user_prompt}

Meta-Hypotheses:
1.
"""
        return prompt

    def generate_meta_hypotheses(self, user_prompt: str) -> List[str]:
        """Generate 5 meta-hypotheses from the user's prompt."""
        if not self.model:
            # Fallback meta-hypotheses
            return [
                f"Investigate the role of UBR-5 in {user_prompt} at the molecular level",
                f"Examine UBR-5's impact on {user_prompt} in cellular processes",
                f"Study UBR-5-mediated regulation of {user_prompt} in disease models",
                f"Explore therapeutic targeting of UBR-5 for {user_prompt}",
                f"Analyze UBR-5's interaction with {user_prompt} in immune responses"
            ]
        
        try:
            prompt = self.build_meta_prompt(user_prompt)
            response = self.model.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            text = response.text
            return self._parse_meta_hypotheses(text)
        except Exception as e:
            print(f"[MetaHypothesisGenerator] Error generating meta-hypotheses: {e}")
            # Return fallback meta-hypotheses
            return [
                f"Investigate the role of UBR-5 in {user_prompt} at the molecular level",
                f"Examine UBR-5's impact on {user_prompt} in cellular processes", 
                f"Study UBR-5-mediated regulation of {user_prompt} in disease models",
                f"Explore therapeutic targeting of UBR-5 for {user_prompt}",
                f"Analyze UBR-5's interaction with {user_prompt} in immune responses"
            ]

    def _parse_meta_hypotheses(self, text: str) -> List[str]:
        """Parse numbered meta-hypotheses from LLM output."""
        pattern = re.compile(r"\n?\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|$)", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # Return only the meta-hypothesis text, up to 5
            return [m[1].strip() for m in matches[:5]]
        # Fallback: split by lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[:5]

class HypothesisGenerator:
    """
    Generates scientific hypotheses tailored to UBR-5 and Dr. Xiaojing Ma's lab using provided literature context.
    Uses a Gemini LLM client for generation.
    """
    def __init__(self, model=None):
        self.model = model  # Gemini client

    def build_prompt(self, context_chunks: List[str], n: int = 3) -> str:
        context = "\n\n".join(context_chunks)
        prompt = f"""
# AI Research Assistant Prompt

## Role
You are a molecular biology research specialist with expertise in UBR-5 protein research, working within the context of Dr. Xiaojing Ma's laboratory at Weill Cornell Medicine.

## Primary Responsibilities

### 1. Hypothesis Generation
Analyze provided information \"batches\" to develop one novel, scientifically sound hypothesis related to UBR-5 protein function, regulation, or therapeutic applications.

### 2. Critical Evaluation
Assess generated hypotheses using three key criteria:
- Scientific accuracy and feasibility
- Novelty and originality in the field
- Relevance to Dr. Xiaojing Ma's research program and laboratory capabilities
- Ability for Dr. Xiaojing Ma's lab to execute said experiment, based on your previous reading of Dr. Xiaojing Ma's papers.

### 3. Experimental Design
Develop detailed research proposal for the hypothesis, including:
- Specific experimental approaches and methodologies
- Required resources and timelines
- Expected outcomes and potential limitations

## Communication Style
- Maintain a highly analytical, critical, and scientifically rigorous approach.
- Please limit your output to 1000 words.

Please generate a hypothesis and a research plan.
---

Literature Context:
{context}

Hypotheses:
1.
"""
        return prompt

    def generate(self, context_chunks: List[str], n: int = 3) -> List[str]:
        prompt = self.build_prompt(context_chunks, n)
        if not self.model:
            # Fallback placeholder
            return [f"Hypothesis {i+1} about UBR-5 (see prompt for details)." for i in range(n)]
        response = self.model.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text
        return self._parse_hypotheses(text, n)

    def _parse_hypotheses(self, text: str, n: int) -> List[str]:
        # Parse numbered hypotheses from LLM output
        pattern = re.compile(r"\n?\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|$)", re.DOTALL)
        matches = pattern.findall(text)
        if matches:
            # Return only the hypothesis text, up to n
            return [m[1].strip() for m in matches[:n]]
        # Fallback: split by lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines[:n]

LAB_GOALS = "UBR5, cancer immunology, protein ubiquitination, mechanistic and therapeutic hypotheses, Dr. Xiaojing Ma's lab at Weill Cornell Medicine. The lab focuses on post-transcriptional regulation, ubiquitination, cancer models, and translational control."

class HypothesisCritic:
    """
    Critiques scientific hypotheses in the context of UBR-5 and Dr. Xiaojing Ma's lab using provided literature.
    Uses a Gemini LLM client for critique and parses scores/verdict.
    """
    def __init__(self, model=None, embedding_fn=None):
        self.model = model  # Gemini client
        self.embedding_fn = embedding_fn  # Function to get embeddings

    def build_prompt(self, hypothesis: str, context_chunks: list) -> str:
        context = "\n\n".join(context_chunks)
        prompt = f"""
You are an expert scientific reviewer for Dr. Xiaojing Ma's lab, specializing in UBR-5 and related pathways. Critically evaluate the following hypothesis in light of the provided literature. Discuss its novelty, plausibility, and potential impact. Point out any supporting or conflicting evidence from the context.
Be extremely critical and professional.

After your critique, provide:
- A novelty score (0-100, where 100 is completely novel)
- An accuracy score (0-100, where 100 is fully supported by the context)
- A relevancy score (0-100, where 100 is maximally relevant to the prompt and lab goals)

Format your answer as:
Critique: <your critique>
Novelty Score: <number>
Accuracy Score: <number>
Relevancy Score: <number>

Literature Context:
{context}

Hypothesis:
{hypothesis}
"""
        return prompt

    def compute_relevancy(self, hypothesis: str, prompt: str, lab_goals: str) -> float:
        if not self.embedding_fn:
            return 100.0  # fallback
        hyp_emb = self.embedding_fn(hypothesis)
        prompt_emb = self.embedding_fn(prompt)
        goals_emb = self.embedding_fn(lab_goals)
        sim_prompt = cosine_similarity([hyp_emb], [prompt_emb])[0][0]
        sim_goals = cosine_similarity([hyp_emb], [goals_emb])[0][0]
        return float(np.round(100 * (sim_prompt + sim_goals) / 2, 1))

    def critique(self, hypothesis: str, context_chunks: list, prompt: str, lab_goals: str = LAB_GOALS) -> dict:
        prompt_text = self.build_prompt(hypothesis, context_chunks)
        relevancy = self.compute_relevancy(hypothesis, prompt, lab_goals)
        if not self.model:
            # Fallback placeholder
            return {
                "critique": f"Critique of hypothesis: '{hypothesis}'\n- This is a placeholder critique based on UBR-5 and Dr. Ma's lab context.",
                "novelty": 100,
                "accuracy": 100,
                "relevancy": relevancy
            }
        response = self.model.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text
        )
        text = response.text
        result = self._parse_critique(text)
        result["relevancy"] = relevancy
        return result

    def _parse_critique(self, text: str) -> dict:
        # Extract critique, novelty, accuracy, and relevancy (no verdict)
        critique_match = re.search(r"Critique:\s*(.*?)(?:\nNovelty Score:|\nAccuracy Score:|\nRelevancy Score:|$)", text, re.DOTALL)
        novelty_match = re.search(r"Novelty Score[:\s]+(\d+)", text)
        accuracy_match = re.search(r"Accuracy Score[:\s]+(\d+)", text)
        relevancy_match = re.search(r"Relevancy Score[:\s]+(\d+)", text)
        return {
            "critique": critique_match.group(1).strip() if critique_match else text,
            "novelty": int(novelty_match.group(1)) if novelty_match else None,
            "accuracy": int(accuracy_match.group(1)) if accuracy_match else None,
            "relevancy": int(relevancy_match.group(1)) if relevancy_match else None
        } 