#!/usr/bin/env python3
"""
Optimized prompts to reduce token usage and prevent TPM quota exceeded errors
"""

def get_optimized_meta_prompt(user_prompt: str, lab_name: str, institution: str) -> str:
    """Optimized meta-hypothesis prompt - reduced from ~400 to ~100 tokens"""
    return f"""Generate 5 distinct research directions for: "{user_prompt}"

Requirements:
- Different molecular mechanisms/approaches
- Vary scope: molecular to cellular levels  
- Feasible for {lab_name}'s lab at {institution}

Format: Numbered list 1-5 with brief, specific directions."""

def get_optimized_hypothesis_prompt(context_chunks: list, n: int, lab_name: str, institution: str, meta_hypothesis: str = None) -> str:
    """Enhanced hypothesis generation prompt with quality standards and clear formatting"""
    # Extract text from chunks while preserving metadata structure
    context_texts = []
    for chunk in context_chunks:
        if isinstance(chunk, dict):
            context_texts.append(chunk.get("document", str(chunk)))
        else:
            context_texts.append(str(chunk))
    context = "\n\n".join(context_texts)
    
    # Build the research focus based on meta-hypothesis or default
    if meta_hypothesis:
        research_focus = f"focusing specifically on: {meta_hypothesis}"
    else:
        research_focus = "conducting cutting-edge biomedical research"
    
    return f"""You are a senior research scientist at {lab_name}'s lab at {institution}, {research_focus}.

TASK: Generate exactly 1 high-quality scientific hypothesis based on the provided literature and the specified research direction.

REQUIRED FORMAT (follow exactly):
1. Hypothesis: [State a clear, testable hypothesis that directly addresses the research direction, citing specific sources from the literature]

2. Experimental Design: [Describe specific experiments to test the hypothesis, including methods, controls, and expected outcomes]

3. Rationale: [Explain the scientific reasoning behind the hypothesis, referencing specific findings from the literature]

QUALITY STANDARDS:
- Hypothesis must be novel, testable, and mechanistically specific
- Must directly address the specified research direction
- Must cite specific sources from the provided literature
- Experimental design must be feasible for a biomedical research lab
- Each section should be 2-4 sentences with specific details
- Focus on molecular mechanisms, cellular processes, or therapeutic approaches

LITERATURE CONTEXT:
{context}

Generate exactly 1 hypothesis following the format above. Ensure all three sections are present and well-developed."""

def get_optimized_critique_prompt(hypothesis: str, context_chunks: list, lab_name: str, critique_config: dict = None, meta_hypothesis: str = None) -> str:
    """Enhanced critique prompt with customizable evaluation criteria"""
    # Extract text from chunks while preserving metadata structure
    context_texts = []
    for chunk in context_chunks:
        if isinstance(chunk, dict):
            context_texts.append(chunk.get("document", str(chunk)))
        else:
            context_texts.append(str(chunk))
    context = "\n\n".join(context_texts)
    
    # Default critique configuration
    default_config = {
        "evaluation_criteria": [
            "Scientific rigor and testability",
            "Novelty and innovation",
            "Feasibility and experimental design",
            "Relevance to research focus",
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
    
    # Merge with custom config if provided
    if critique_config:
        default_config.update(critique_config)
    
    criteria_text = "\n".join([f"- {criterion}" for criterion in default_config["evaluation_criteria"]])
    focus_text = "\n".join([f"- {area}" for area in default_config["focus_areas"]])
    
    detailed_section = ""
    if default_config.get("detailed_feedback", True):
        detailed_section = """
Detailed Analysis:
- Strengths: [What works well]
- Weaknesses: [Areas for improvement] 
- Suggestions: [Specific recommendations]
- Literature gaps: [Missing context or citations]"""
    
    # Build research focus section
    research_focus_section = ""
    if meta_hypothesis:
        research_focus_section = f"""
Research Focus: {meta_hypothesis}

Evaluate how well this hypothesis addresses the specific research direction above."""
    
    return f"""Critically evaluate this scientific hypothesis for {lab_name}'s lab:{research_focus_section}

Hypothesis: {hypothesis}

Literature Context: {context}

Evaluation Criteria:
{criteria_text}

Focus Areas:
{focus_text}
{detailed_section}

IMPORTANT: Provide your evaluation in EXACTLY this format:

Critique: [Detailed critical analysis covering all criteria]

Novelty Score: [MUST be a single integer from 0-5, e.g., "Novelty Score: 4"]

Accuracy Score: [MUST be a single integer from 0-5, e.g., "Accuracy Score: 3"]

Relevancy Score: [MUST be a single integer from 0-5, e.g., "Relevancy Score: 5"]

SCORING GUIDELINES:
- Use ONLY integers 0, 1, 2, 3, 4, or 5
- Do NOT use fractions, decimals, or descriptive words in the score lines
- Do NOT use "out of 5" or "/5" format
- Example of CORRECT format: "Novelty Score: 4"
- Example of INCORRECT format: "Novelty Score: 4/5" or "Novelty Score: High"

NOVELTY SCORING CRITERIA (BE STRICT):
- 5: Groundbreaking, completely novel mechanism or approach never described before
- 4: Significant innovation with clear novelty in application or methodology
- 3: Moderate novelty with some innovative elements but builds on known mechanisms
- 2: Limited novelty, mostly incremental improvement on existing knowledge
- 1: Minimal novelty, largely derivative with minor modifications
- 0: No novelty, completely derivative or well-established mechanism

IMPORTANT: Distinguish between known biological mechanisms and novel applications. A hypothesis describing a well-known mechanism (even if not previously studied in a specific context) should score 2-3 for novelty, not 4-5.

Provide your evaluation now:"""
