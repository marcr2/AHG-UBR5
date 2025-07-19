import re
from typing import List, Optional

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
You are an expert scientific assistant for Dr. Xiaojing Ma's lab, specializing in UBR-5 (ubiquitin-protein ligase E3 component n-recognin 5) and its role in cancer immunology and protein ubiquitination. Based on the following literature excerpts, generate {n} novel, testable mechanistic or therapeutic hypotheses related to UBR-5. Each hypothesis should be clear, specific, and relevant to Dr. Ma's research interests.

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

class HypothesisCritic:
    """
    Critiques scientific hypotheses in the context of UBR-5 and Dr. Xiaojing Ma's lab using provided literature.
    Uses a Gemini LLM client for critique and parses scores/verdict.
    """
    def __init__(self, model=None):
        self.model = model  # Gemini client

    def build_prompt(self, hypothesis: str, context_chunks: List[str]) -> str:
        context = "\n\n".join(context_chunks)
        prompt = f"""
You are an expert scientific reviewer for Dr. Xiaojing Ma's lab, specializing in UBR-5 and related pathways. Critically evaluate the following hypothesis in light of the provided literature. Discuss its novelty, plausibility, and potential impact. Point out any supporting or conflicting evidence from the context.

After your critique, provide:
- A novelty score (0-100, where 100 is completely novel)
- An accuracy score (0-100, where 100 is fully supported by the context)
- A final verdict: ACCEPT or REJECT

Format your answer as:
Critique: <your critique>
Novelty Score: <number>
Accuracy Score: <number>
Verdict: <ACCEPT/REJECT>

Literature Context:
{context}

Hypothesis:
{hypothesis}
"""
        return prompt

    def critique(self, hypothesis: str, context_chunks: List[str]) -> dict:
        prompt = self.build_prompt(hypothesis, context_chunks)
        if not self.model:
            # Fallback placeholder
            return {
                "critique": f"Critique of hypothesis: '{hypothesis}'\n- This is a placeholder critique based on UBR-5 and Dr. Ma's lab context.",
                "novelty": 100,
                "accuracy": 100,
                "verdict": "ACCEPT"
            }
        response = self.model.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        text = response.text
        return self._parse_critique(text)

    def _parse_critique(self, text: str) -> dict:
        # Extract critique, novelty, accuracy, and verdict
        critique_match = re.search(r"Critique:\s*(.*?)(?:\nNovelty Score:|\nAccuracy Score:|\nVerdict:|$)", text, re.DOTALL)
        novelty_match = re.search(r"Novelty Score[:\s]+(\d+)", text)
        accuracy_match = re.search(r"Accuracy Score[:\s]+(\d+)", text)
        verdict_match = re.search(r"Verdict[:\s]+(ACCEPT|REJECT)", text, re.IGNORECASE)
        return {
            "critique": critique_match.group(1).strip() if critique_match else text,
            "novelty": int(novelty_match.group(1)) if novelty_match else None,
            "accuracy": int(accuracy_match.group(1)) if accuracy_match else None,
            "verdict": verdict_match.group(1).upper() if verdict_match else None
        } 