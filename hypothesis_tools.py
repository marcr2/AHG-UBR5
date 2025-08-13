import re
import json
import os
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

def validate_hypothesis_format(hypothesis: str) -> tuple[bool, str]:
    """
    Validate that a hypothesis contains all three required sections.
    
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    hypothesis_lower = hypothesis.lower()
    
    # Check for hypothesis section
    has_hypothesis = any(phrase in hypothesis_lower for phrase in [
        '1. hypothesis',
        '1. hypothesis:',
        'hypothesis:',
        'hypothesis ',
        'we hypothesize',
        'we propose',
        'we suggest',
        'our hypothesis',
        'the hypothesis'
    ])
    
    # Check for experimental design section
    has_experimental_design = any(phrase in hypothesis_lower for phrase in [
        '2. experimental design',
        '2. experimental design:',
        'experimental design:',
        'experimental design ',
        'experimental design',
        'methods',
        'methodology',
        'experimental methods',
        'experimental strategy',
        'approach:',
        'approach ',
        'we will',
        'we propose to',
        'we suggest to',
        'to test this',
        'to investigate',
        'to examine',
        'to study'
    ])
    
    # Check for rationale section
    has_rationale = any(phrase in hypothesis_lower for phrase in [
        '3. rationale',
        '3. rationale:',
        'rationale:',
        'rationale ',
        'reasoning',
        'scientific basis',
        'basis for',
        'because',
        'since',
        'as'
    ])
    
    if not has_hypothesis and not has_experimental_design and not has_rationale:
        return False, "Missing all three required sections: hypothesis, experimental design, and rationale"
    elif not has_hypothesis:
        return False, "Missing hypothesis section"
    elif not has_experimental_design:
        return False, "Missing experimental design section"
    elif not has_rationale:
        return False, "Missing rationale section"
    else:
        return True, "Format validation passed - all three sections present"

class MetaHypothesisGenerator:
    """
    Meta-hypothesis generator that takes a user prompt and creates 5 different prompts
    to send to the actual hypothesis generator for diverse hypothesis generation.
    """
    def __init__(self, model=None):
        self.model = model  # Gemini client

    def build_meta_prompt(self, user_prompt: str) -> str:
        """Build prompt for generating 5 different meta-hypotheses from user input."""
        config = get_lab_config()
        lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
        institution = config.get("institution", "Weill Cornell Medicine")
        
        prompt = f"""
# Meta-Hypothesis Generator Prompt

## Role
You are an expert research strategist specializing in UBR-5 protein research and {lab_name}'s laboratory at {institution}. Your task is to take a user's research query and break it down into 5 distinct, complementary research directions.

## Task
Given the user's research query, generate exactly 5 different meta-hypotheses that represent diverse angles and approaches to the same research area. Each meta-hypothesis should be:
- Focused on a specific aspect or mechanism
- Complementary to the others (not redundant)
- Feasible for {lab_name}'s lab to investigate
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
        config = get_lab_config()
        lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
        institution = config.get("institution", "Weill Cornell Medicine")
        
        prompt = f"""Your task is to formulate {n} scientific hypotheses and experimental designs based on provided source materials ("chunks"). Follow these rules precisely:

1. **Vary Source Chunks:** For each new hypothesis, you must select a different and non-overlapping set of source chunks to use as evidence.
2. **Cite All Sources:** You are forbidden from generating a hypothesis, rationale, or experimental design without citing the specific source chunks that informed your response.
3. **Adhere to Format:** Your final output must strictly follow this exact template. If any section is incomplete or missing, you must reject the response and try again until it meets the format.

<output_format>
1. Hypothesis
[Formulate a clear hypothesis, citing all relevant source chunks.]

2. Experimental Design
[Propose a brief, plausible experiment to test the hypothesis.]

3. Rationale
[Explain the reasoning and scientific basis for the hypothesis, citing all relevant source chunks.]
</output_format>

## Context
You are a molecular biology research specialist with expertise in UBR-5 protein research, working within the context of {lab_name}'s laboratory at {institution}.

## Source Materials
{context}

## Generated Hypotheses
Generate exactly {n} hypotheses following the format above."""
        return prompt

    def generate(self, context_chunks: List[str], n: int = 3) -> List[str]:
        prompt = self.build_prompt(context_chunks, n)
        if not self.model:
            # Fallback placeholder in new format
            return [f"1. Hypothesis: UBR-5 regulates protein stability in cellular pathways.\n\n2. Experimental Design: We will analyze UBR-5 knockout effects on protein degradation.\n\n3. Rationale: UBR-5 is an E3 ubiquitin ligase involved in protein turnover." for i in range(n)]
        
        try:
            response = self.model.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            text = response.text
            print(f"[HypothesisGenerator.generate] Raw response length: {len(text)} characters")
            
            # Parse the hypotheses
            hypotheses = self._parse_hypotheses(text, n)
            print(f"[HypothesisGenerator.generate] Parsed {len(hypotheses)} hypotheses")
            
            # Validate hypotheses
            valid_hypotheses = []
            for i, hyp in enumerate(hypotheses):
                if hyp and len(hyp) > 50:
                    # Validate format requirements
                    is_valid_format, format_reason = validate_hypothesis_format(hyp)
                    if is_valid_format:
                        valid_hypotheses.append(hyp)
                        print(f"[HypothesisGenerator.generate] Hypothesis {i+1} length: {len(hyp)} characters - FORMAT VALID (3 sections)")
                    else:
                        print(f"[HypothesisGenerator.generate] Hypothesis {i+1} REJECTED: {format_reason}")
                else:
                    print(f"[HypothesisGenerator.generate] Hypothesis {i+1} too short or empty: {len(hyp)} characters")
            
            if not valid_hypotheses:
                print(f"[HypothesisGenerator.generate] WARNING: No valid hypotheses generated. Raw text preview: {text[:200]}...")
                # Return a fallback hypothesis in the new format
                return [f"1. Hypothesis: UBR-5 plays a critical role in protein ubiquitination pathways based on provided literature context.\n\n2. Experimental Design: To test this hypothesis, we will examine UBR-5 expression and activity in cellular models.\n\n3. Rationale: The literature suggests UBR-5 is involved in protein degradation and regulation processes."]
            
            return valid_hypotheses
            
        except Exception as e:
            print(f"[HypothesisGenerator.generate] ERROR: Failed to generate hypotheses: {e}")
            return [f"Error generating hypothesis: {e}"]

    def _parse_hypotheses(self, text: str, n: int) -> List[str]:
        # First try to parse numbered hypotheses from LLM output with the new format
        # Look for patterns like "1. Hypothesis", "2. Experimental Design", "3. Rationale"
        pattern = re.compile(r"\n?\s*(\d+)\.\s+(?:Hypothesis|Experimental Design|Rationale)\s*:?\s*(.*?)(?=\n\s*\d+\.|$)", re.DOTALL)
        matches = pattern.findall(text)
        
        if matches and len(matches) >= n:
            # Return only the hypothesis text, up to n
            hypotheses = [m[1].strip() for m in matches[:n]]
            # Clean up any remaining formatting
            cleaned_hypotheses = []
            for hyp in hypotheses:
                # Remove any markdown formatting that might interfere
                hyp = re.sub(r'\*\*(.*?)\*\*', r'\1', hyp)  # Remove bold formatting
                hyp = re.sub(r'#+\s*', '', hyp)  # Remove headers
                hyp = hyp.strip()
                if hyp and len(hyp) > 50:  # Ensure minimum length
                    cleaned_hypotheses.append(hyp)
            return cleaned_hypotheses[:n]
        
        # Fallback: try to split by the new section markers
        if "Hypothesis" in text and "Experimental Design" in text and "Rationale" in text:
            # Split by the new section markers
            sections = re.split(r'\n\s*(?:1\.\s*Hypothesis|2\.\s*Experimental Design|3\.\s*Rationale)\s*:?\s*', text)
            if len(sections) > 1:
                # Take the first substantial section as the hypothesis
                hypothesis = sections[0].strip()
                if len(hypothesis) > 50:
                    return [hypothesis]
        
        # Additional fallback: try to split by common section markers
        if "Hypothesis Statement" in text or "Rationale" in text:
            # Split by potential section markers
            sections = re.split(r'\n\s*(?:Hypothesis Statement|Rationale|Experimental Approach|Expected Outcomes|Significance)\s*:', text)
            if len(sections) > 1:
                # Take the first substantial section as the hypothesis
                hypothesis = sections[0].strip()
                if len(hypothesis) > 50:
                    return [hypothesis]
        
        # Final fallback: split by lines and take substantial content
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            # Combine lines into a single hypothesis if they seem related
            combined = ' '.join(lines[:10])  # Take first 10 lines
            if len(combined) > 50:
                return [combined]
        
        # If all else fails, return the original text as a single hypothesis
        return [text.strip()] if text.strip() else []

def get_lab_config():
    """Get lab configuration from file or return default"""
    config_file = "lab_config.json"
    default_config = {
        "lab_name": "Dr. Xiaojing Ma",
        "institution": "Weill Cornell Medicine",
        "research_focus": "UBR5, cancer immunology, protein ubiquitination, mechanistic and therapeutic hypotheses"
    }
    
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"⚠️  Error reading lab config: {e}, using default")
    
    return default_config

def get_lab_goals():
    """Get lab goals based on current configuration"""
    config = get_lab_config()
    lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
    institution = config.get("institution", "Weill Cornell Medicine")
    research_focus = config.get("research_focus", "UBR5, cancer immunology, protein ubiquitination, mechanistic and therapeutic hypotheses")
    
    return f"{research_focus}, {lab_name}'s lab at {institution}. The lab focuses on post-transcriptional regulation, ubiquitination, cancer models, and translational control."

# Default LAB_GOALS for backward compatibility
LAB_GOALS = get_lab_goals()

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
        config = get_lab_config()
        lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
        
        prompt = f"""
You are an expert scientific reviewer for {lab_name}'s lab, specializing in UBR-5 and related pathways. Critically evaluate the following hypothesis in light of the provided literature. Discuss its novelty, plausibility, and potential impact. Point out any supporting or conflicting evidence from the context.
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
            config = get_lab_config()
            lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
            return {
                "critique": f"Critique of hypothesis: '{hypothesis}'\n- This is a placeholder critique based on UBR-5 and {lab_name}'s lab context.",
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