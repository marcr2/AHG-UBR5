import re
import json
import os
from typing import List
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from src.ai.optimized_prompts import get_optimized_meta_prompt, get_optimized_hypothesis_prompt, get_optimized_critique_prompt

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
    Uses more flexible pattern matching to handle various formats.
    
    Returns:
        tuple: (is_valid: bool, reason: str)
    """
    hypothesis_lower = hypothesis.lower()
    
    # More flexible pattern matching for hypothesis section
    has_hypothesis = any(phrase in hypothesis_lower for phrase in [
        '1. hypothesis',
        '1. hypothesis:',
        'hypothesis:',
        'hypothesis ',
        'we hypothesize',
        'we propose',
        'we suggest',
        'our hypothesis',
        'the hypothesis',
        'hypothesis that',
        'hypothesis is'
    ])
    
    # More flexible pattern matching for experimental design section
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
        'to study',
        'experiments will',
        'testing will',
        'analysis will'
    ])
    
    # More flexible pattern matching for rationale section
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
        'as',
        'this is based on',
        'previous studies',
        'literature shows',
        'evidence suggests',
        'supporting evidence'
    ])
    
    # Check if all three sections are present
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
            
            # Check rate limit before making API call
            if hasattr(self, 'rate_limiter') and self.rate_limiter:
                # Use improved token estimation with full text
                estimated_tokens = self.rate_limiter.estimate_tokens(prompt) + 1500  # Buffer for response
                self.rate_limiter.wait_if_needed(estimated_tokens)
            
            model = self.model.GenerativeModel("gemini-2.5-flash")
            
            # Use retry logic for quota exceeded errors
            def make_api_call():
                return model.generate_content(prompt)
            
            response = self.rate_limiter.execute_with_retry(make_api_call)
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

    def build_prompt(self, context_chunks: List[str], n: int = 3, meta_hypothesis: str = None) -> str:
        """Build optimized prompt for hypothesis generation."""
        config = get_lab_config()
        lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
        institution = config.get("institution", "Weill Cornell Medicine")
        
        # Use optimized prompt to reduce token usage
        return get_optimized_hypothesis_prompt(context_chunks, n, lab_name, institution, meta_hypothesis)

    def generate(self, context_chunks: List[str], n: int = 3, meta_hypothesis: str = None) -> List[str]:
        prompt = self.build_prompt(context_chunks, n, meta_hypothesis)
        if not self.model:
            print("[HypothesisGenerator.generate] ERROR: No model available for hypothesis generation")
            return []
        
        try:
            # Check rate limit before making API call
            if hasattr(self, 'rate_limiter') and self.rate_limiter:
                # Use improved token estimation with full text
                estimated_tokens = self.rate_limiter.estimate_tokens(prompt) + 2000  # Buffer for response
                self.rate_limiter.wait_if_needed(estimated_tokens)
            
            model = self.model.GenerativeModel("gemini-2.5-flash")
            
            # Use retry logic for quota exceeded errors
            def make_api_call():
                return model.generate_content(prompt)
            
            response = self.rate_limiter.execute_with_retry(make_api_call)
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
                # Return empty list to indicate failure - don't provide fallback
                return []
            
            return valid_hypotheses
        
        except Exception as e:
            print(f"[HypothesisGenerator.generate] ERROR: Failed to generate hypotheses: {e}")
            return []

    def refine_hypothesis(self, original_hypothesis: str, critique_feedback: dict, context_chunks: List[str], user_prompt: str) -> str:
        """Refine a hypothesis based on critique feedback."""
        if not self.model:
            print("[HypothesisGenerator.refine_hypothesis] ERROR: No model available for hypothesis refinement")
            return original_hypothesis
        
        config = get_lab_config()
        lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
        institution = config.get("institution", "Weill Cornell Medicine")
        
        # Extract text from chunks
        context_texts = []
        for chunk in context_chunks:
            if isinstance(chunk, dict):
                context_texts.append(chunk.get("document", str(chunk)))
            else:
                context_texts.append(str(chunk))
        context = "\n\n".join(context_texts)
        
        # Build refinement prompt
        refinement_prompt = f"""You are a senior research scientist at {lab_name}'s lab at {institution}.

TASK: Refine the following hypothesis based on the critique feedback to address the identified issues.

ORIGINAL HYPOTHESIS:
{original_hypothesis}

CRITIQUE FEEDBACK:
{critique_feedback.get('critique', '')}

SCORES FROM CRITIQUE:
- Novelty: {critique_feedback.get('novelty', 'N/A')}/5
- Accuracy: {critique_feedback.get('accuracy', 'N/A')}/5
- Relevancy: {critique_feedback.get('relevancy', 'N/A')}/5

LITERATURE CONTEXT:
{context}

INSTRUCTIONS:
1. Address the specific issues raised in the critique
2. Improve novelty by focusing on novel applications, mechanisms, or therapeutic approaches
3. Ensure scientific accuracy and feasibility
4. Maintain relevance to the research focus: {user_prompt}
5. Keep the same format: Hypothesis, Experimental Design, Rationale

REQUIRED FORMAT (follow exactly):
1. Hypothesis: [Refined hypothesis addressing critique issues]
2. Experimental Design: [Updated experimental approach]
3. Rationale: [Revised scientific reasoning]

Generate the refined hypothesis now:"""
        
        try:
            # Check rate limit before making API call
            if hasattr(self, 'rate_limiter') and self.rate_limiter:
                estimated_tokens = self.rate_limiter.estimate_tokens(refinement_prompt) + 2000
                self.rate_limiter.wait_if_needed(estimated_tokens)
            
            model = self.model.GenerativeModel("gemini-2.5-flash")
            
            def make_api_call():
                return model.generate_content(refinement_prompt)
            
            response = self.rate_limiter.execute_with_retry(make_api_call)
            text = response.text
            
            # Parse the refined hypothesis
            refined_hypotheses = self._parse_hypotheses(text, 1)
            if refined_hypotheses:
                return refined_hypotheses[0]
            else:
                print("[HypothesisGenerator.refine_hypothesis] ERROR: Failed to parse refined hypothesis")
                return original_hypothesis
                
        except Exception as e:
            print(f"[HypothesisGenerator.refine_hypothesis] ERROR: Failed to refine hypothesis: {e}")
            return original_hypothesis
            
    def _parse_hypotheses(self, text: str, n: int) -> List[str]:
        # First try to parse complete hypotheses with all three sections
        # Look for the pattern: "1. Hypothesis: ... 2. Experimental Design: ... 3. Rationale: ..."
        complete_pattern = re.compile(
            r"1\.\s*Hypothesis\s*:?\s*(.*?)(?=2\.\s*Experimental Design|$)", 
            re.DOTALL | re.IGNORECASE
        )
        
        # Try to find complete hypothesis blocks
        if "1. Hypothesis" in text and "2. Experimental Design" in text and "3. Rationale" in text:
            # Extract the complete hypothesis including all three sections
            sections = re.split(r'\n\s*(?:1\.\s*Hypothesis|2\.\s*Experimental Design|3\.\s*Rationale)\s*:?\s*', text, flags=re.IGNORECASE)
            if len(sections) >= 4:  # Should have intro + 3 sections
                # Reconstruct the complete hypothesis
                complete_hypothesis = f"1. Hypothesis: {sections[1].strip()}\n\n2. Experimental Design: {sections[2].strip()}\n\n3. Rationale: {sections[3].strip()}"
                return [complete_hypothesis]
        
        # Fallback: try to parse numbered hypotheses from LLM output with the new format
        # Look for patterns like "1. Hypothesis", "2. Experimental Design", "3. Rationale"
        pattern = re.compile(r"\n?\s*(\d+)\.\s+(?:Hypothesis|Experimental Design|Rationale)\s*:?\s*(.*?)(?=\n\s*\d+\.|$)", re.DOTALL)
        matches = pattern.findall(text)
        
        if matches and len(matches) >= 3:  # Need all three sections
            # Combine all three sections into a complete hypothesis
            hypothesis_text = ""
            for match in matches:
                section_num = match[0]
                section_content = match[1].strip()
                if section_num == "1":
                    hypothesis_text += f"1. Hypothesis: {section_content}\n\n"
                elif section_num == "2":
                    hypothesis_text += f"2. Experimental Design: {section_content}\n\n"
                elif section_num == "3":
                    hypothesis_text += f"3. Rationale: {section_content}\n\n"
            
            if hypothesis_text.strip():
                return [hypothesis_text.strip()]
        
        # More flexible fallback: look for any numbered list items that might be hypotheses
        numbered_pattern = re.compile(r"\n?\s*(\d+)\.\s+(.*?)(?=\n\s*\d+\.|$)", re.DOTALL)
        numbered_matches = numbered_pattern.findall(text)
        
        if numbered_matches and len(numbered_matches) >= n:
            # Take the first n matches as potential hypotheses
            hypotheses = [m[1].strip() for m in numbered_matches[:n]]
            cleaned_hypotheses = []
            for hyp in hypotheses:
                # Remove any markdown formatting
                hyp = re.sub(r'\*\*(.*?)\*\*', r'\1', hyp)
                hyp = re.sub(r'#+\s*', '', hyp)
                hyp = hyp.strip()
                if hyp and len(hyp) > 50:  # Ensure minimum length
                    cleaned_hypotheses.append(hyp)
            if cleaned_hypotheses:
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
    # Check for temporary config first (used by GUI)
    temp_config_file = "config/temp_lab_config.json"
    config_file = "lab_config.json"
    default_config = {
        "lab_name": "Dr. Xiaojing Ma",
        "institution": "Weill Cornell Medicine",
        "research_focus": "UBR5, cancer immunology, protein ubiquitination, mechanistic and therapeutic hypotheses"
    }
    
    # Try temporary config first (GUI override)
    if os.path.exists(temp_config_file):
        try:
            with open(temp_config_file, 'r') as f:
                config = json.load(f)
                return config
        except Exception as e:
            print(f"⚠️  Error reading temp lab config: {e}, trying main config")
    
    # Try main config file
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
    Supports customizable critique prompts via configuration file.
    """
    def __init__(self, model=None, embedding_fn=None, critique_config_file="config/critique_config.json"):
        self.model = model  # Gemini client
        self.embedding_fn = embedding_fn  # Function to get embeddings
        self.critique_config_file = critique_config_file
        self.critique_config = self._load_critique_config()

    def _load_critique_config(self) -> dict:
        """Load critique configuration from file or return default."""
        default_config = {
            "evaluation_criteria": [
                "Scientific rigor and testability",
                "Novelty and innovation",
                "Feasibility and experimental design",
                "Relevance to UBR-5 research",
                "Clinical/translational potential"
            ],
            "scoring_scale": "0-5 (0=poor, 5=excellent)",
            "focus_areas": [
                "Molecular mechanisms",
                "Experimental methodology", 
                "Literature alignment",
                "Lab expertise match"
            ],
            "detailed_feedback": True
        }
        
        if os.path.exists(self.critique_config_file):
            try:
                with open(self.critique_config_file, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_config.update(config)
                    return default_config
            except Exception as e:
                print(f"⚠️  Error reading critique config: {e}, using defaults")
        
        return default_config

    def build_prompt(self, hypothesis: str, context_chunks: list, meta_hypothesis: str = None) -> str:
        """Build optimized prompt for critique generation with custom configuration."""
        config = get_lab_config()
        lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
        
        # Use optimized prompt with custom critique configuration
        return get_optimized_critique_prompt(hypothesis, context_chunks, lab_name, self.critique_config, meta_hypothesis)

    def compute_relevancy(self, hypothesis: str, prompt: str, lab_goals: str) -> float:
        if not self.embedding_fn:
            return 5.0  # fallback
        hyp_emb = self.embedding_fn(hypothesis)
        prompt_emb = self.embedding_fn(prompt)
        goals_emb = self.embedding_fn(lab_goals)
        sim_prompt = cosine_similarity([hyp_emb], [prompt_emb])[0][0]
        sim_goals = cosine_similarity([hyp_emb], [goals_emb])[0][0]
        return float(np.round(5 * (sim_prompt + sim_goals) / 2, 1))

    def critique(self, hypothesis: str, context_chunks: list, prompt: str, lab_goals: str = LAB_GOALS, meta_hypothesis: str = None) -> dict:
        prompt_text = self.build_prompt(hypothesis, context_chunks, meta_hypothesis)
        relevancy = self.compute_relevancy(hypothesis, prompt, lab_goals)
        if not self.model:
            # Fallback placeholder
            config = get_lab_config()
            lab_name = config.get("lab_name", "Dr. Xiaojing Ma")
            return {
                "critique": f"Critique of hypothesis: '{hypothesis}'\n- This is a placeholder critique based on {lab_name}'s lab context and research focus.",
                "novelty": 5,
                "accuracy": 5,
                "relevancy": relevancy
            }
        # Check rate limit before making API call
        if hasattr(self, 'rate_limiter') and self.rate_limiter:
            # Use improved token estimation with full text
            estimated_tokens = self.rate_limiter.estimate_tokens(prompt_text) + 1500  # Buffer for response
            self.rate_limiter.wait_if_needed(estimated_tokens)
        
        model = self.model.GenerativeModel("gemini-2.5-flash")
        
        # Use retry logic for quota exceeded errors
        def make_api_call():
            return model.generate_content(prompt_text)
        
        response = self.rate_limiter.execute_with_retry(make_api_call)
        text = response.text
        result = self._parse_critique(text)
        result["relevancy"] = relevancy
        return result

    def update_critique_config(self, new_config: dict) -> None:
        """Update critique configuration and save to file."""
        # Merge with existing config
        self.critique_config.update(new_config)
        
        # Save to file
        try:
            with open(self.critique_config_file, 'w') as f:
                json.dump(self.critique_config, f, indent=2)
            print(f"✅ Critique configuration updated and saved to {self.critique_config_file}")
        except Exception as e:
            print(f"⚠️  Error saving critique config: {e}")

    def get_critique_config(self) -> dict:
        """Get current critique configuration."""
        return self.critique_config.copy()

    def _parse_critique(self, text: str) -> dict:
        """Enhanced critique parsing to handle detailed feedback sections."""
        # Extract main critique
        critique_match = re.search(r"Critique:\s*(.*?)(?:\nNovelty Score:|\nAccuracy Score:|\nRelevancy Score:|$)", text, re.DOTALL)
        
        # Extract detailed analysis sections if present
        strengths_match = re.search(r"Strengths:\s*(.*?)(?:\nWeaknesses:|\nSuggestions:|\nLiterature gaps:|$)", text, re.DOTALL)
        weaknesses_match = re.search(r"Weaknesses:\s*(.*?)(?:\nSuggestions:|\nLiterature gaps:|$)", text, re.DOTALL)
        suggestions_match = re.search(r"Suggestions:\s*(.*?)(?:\nLiterature gaps:|$)", text, re.DOTALL)
        gaps_match = re.search(r"Literature gaps:\s*(.*?)(?:\nNovelty Score:|$)", text, re.DOTALL)
        
        # Extract scores using enhanced parsing
        novelty = self._extract_score(text, ["novelty", "novelty score"])
        accuracy = self._extract_score(text, ["accuracy", "accuracy score"])
        relevancy = self._extract_score(text, ["relevancy", "relevancy score", "relevance", "relevance score"])
        
        result = {
            "critique": critique_match.group(1).strip() if critique_match else text,
            "novelty": novelty,
            "accuracy": accuracy,
            "relevancy": relevancy
        }
        
        # Add detailed analysis if present
        if strengths_match or weaknesses_match or suggestions_match or gaps_match:
            result["detailed_analysis"] = {
                "strengths": strengths_match.group(1).strip() if strengths_match else None,
                "weaknesses": weaknesses_match.group(1).strip() if weaknesses_match else None,
                "suggestions": suggestions_match.group(1).strip() if suggestions_match else None,
                "literature_gaps": gaps_match.group(1).strip() if gaps_match else None
            }
        
        return result
    
    def _extract_score(self, text: str, score_labels: list) -> int:
        """Extract numeric score from text using multiple patterns and labels."""
        # Try multiple patterns for each score type
        patterns = []
        for label in score_labels:
            patterns.extend([
                rf"{label}[:\s]+(\d+)",  # "Novelty: 4" or "Novelty 4"
                rf"{label}[:\s]+(\d+)/\d+",  # "Novelty: 4/5"
                rf"{label}[:\s]+(\d+)\.\d*",  # "Novelty: 4.0"
                rf"{label}[:\s]+(\d+)\s*out\s*of\s*\d+",  # "Novelty: 4 out of 5"
            ])
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    score = int(match.group(1))
                    # Ensure score is within valid range (0-5)
                    if 0 <= score <= 5:
                        return score
                except (ValueError, IndexError):
                    continue
        
        # If no numeric score found, try to extract from descriptive text
        return self._extract_score_from_description(text, score_labels)
    
    def _extract_score_from_description(self, text: str, score_labels: list) -> int:
        """Extract score from descriptive text when numeric scores aren't found."""
        text_lower = text.lower()
        
        # Look for descriptive words near score labels
        for label in score_labels:
            # Find text around the label
            label_pattern = rf"{label}[:\s]*(.*?)(?:\n|$)"
            match = re.search(label_pattern, text_lower)
            if match:
                context = match.group(1)
                
                # Map descriptive words to scores
                if any(word in context for word in ["excellent", "outstanding", "exceptional", "groundbreaking", "revolutionary"]):
                    return 5
                elif any(word in context for word in ["very good", "strong", "significant", "high", "substantial"]):
                    return 4
                elif any(word in context for word in ["good", "moderate", "adequate", "reasonable", "fair"]):
                    return 3
                elif any(word in context for word in ["poor", "weak", "limited", "low", "insufficient"]):
                    return 2
                elif any(word in context for word in ["very poor", "very weak", "minimal", "negligible"]):
                    return 1
                elif any(word in context for word in ["none", "no", "absent", "lacking"]):
                    return 0
        
        # Default to 3 (moderate) if no clear indication
        return 3 