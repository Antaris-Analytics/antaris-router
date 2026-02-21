"""
Task classification for Antaris Router.

Classifies prompts into complexity tiers (trivial, simple, moderate, complex, expert)
based on deterministic keyword matching, length, and structural analysis.
"""

import re
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

from .config import Config


@dataclass
class ClassificationResult:
    """Result of prompt classification."""
    tier: str
    confidence: float
    reasoning: List[str]
    signals: Dict[str, Any]


class TaskClassifier:
    """Classifies tasks into complexity tiers using deterministic rules."""
    
    def __init__(self, config: Config):
        """Initialize classifier with configuration.
        
        Args:
            config: Configuration instance with classification rules
        """
        self.config = config
        self.tiers = ['trivial', 'simple', 'moderate', 'complex', 'expert']
    
    def classify(self, prompt: str, context: Dict = None) -> ClassificationResult:
        """Classify a prompt into a complexity tier.
        
        Args:
            prompt: Text prompt to classify
            context: Optional context dict for additional signals
            
        Returns:
            ClassificationResult with tier, confidence, and reasoning
        """
        if not prompt or not prompt.strip():
            return ClassificationResult(
                tier="trivial",
                confidence=1.0,
                reasoning=["Empty or whitespace-only prompt"],
                signals={"empty": True}
            )
        
        prompt_lower = prompt.lower().strip()
        length = len(prompt)
        
        # Collect classification signals
        signals = self._analyze_signals(prompt, prompt_lower, length, context)
        
        # Score each tier based on signals
        tier_scores = self._score_tiers(signals, prompt_lower)
        
        # Select highest scoring tier
        best_tier = max(tier_scores.keys(), key=lambda t: tier_scores[t])
        confidence = tier_scores[best_tier]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(signals, best_tier)
        
        return ClassificationResult(
            tier=best_tier,
            confidence=confidence,
            reasoning=reasoning,
            signals=signals
        )
    
    def _analyze_signals(self, prompt: str, prompt_lower: str, length: int, context: Dict = None) -> Dict:
        """Analyze various signals in the prompt.
        
        Args:
            prompt: Original prompt text
            prompt_lower: Lowercase version of prompt
            length: Character length of prompt
            context: Optional context
            
        Returns:
            Dict of classification signals
        """
        signals = {
            'length': length,
            'word_count': len(prompt_lower.split()),
            'sentence_count': len([s for s in prompt.split('.') if s.strip()]),
            'has_code': self._has_code_blocks(prompt),
            'has_questions': '?' in prompt,
            'has_lists': self._has_list_structure(prompt),
            'keyword_matches': {},
            'structural_complexity': 0
        }
        
        # Keyword matching for each tier
        for tier in self.tiers:
            keywords = getattr(self.config, f'get_{tier}_keywords')()
            matches = sum(1 for keyword in keywords if keyword in prompt_lower)
            signals['keyword_matches'][tier] = matches
        
        # Code indicators
        code_indicators = self.config.get_code_indicators()
        signals['code_indicators'] = sum(1 for indicator in code_indicators if indicator in prompt)
        
        # Structural complexity
        signals['structural_complexity'] = self._calculate_structural_complexity(prompt)
        
        # Context signals
        if context:
            signals['context'] = context
        
        return signals
    
    def _has_code_blocks(self, prompt: str) -> bool:
        """Check if prompt contains code blocks."""
        return '```' in prompt or bool(re.search(r'\b(def |class |function|import|from |SELECT|CREATE)\b', prompt))
    
    def _has_list_structure(self, prompt: str) -> bool:
        """Check if prompt has list structure."""
        lines = prompt.strip().split('\n')
        list_lines = sum(1 for line in lines if re.match(r'^\s*[-*•\d+\.]\s', line.strip()))
        return list_lines >= 2
    
    def _calculate_structural_complexity(self, prompt: str) -> int:
        """Calculate structural complexity score."""
        score = 0
        
        # Multi-paragraph structure
        paragraphs = [p for p in prompt.split('\n\n') if p.strip()]
        if len(paragraphs) > 2:
            score += 2
        
        # Code blocks
        if self._has_code_blocks(prompt):
            score += 3
        
        # Lists
        if self._has_list_structure(prompt):
            score += 1
        
        # Questions
        score += prompt.count('?')
        
        # Technical terms (heuristic: words with underscores or camelCase)
        tech_terms = len(re.findall(r'\b[a-z]+[A-Z][a-zA-Z]*\b|\b\w+_\w+\b', prompt))
        score += min(tech_terms, 3)  # Cap at 3
        
        return score
    
    def _score_tiers(self, signals: Dict, prompt_lower: str) -> Dict[str, float]:
        """Score each tier based on signals.
        
        Args:
            signals: Classification signals
            
        Returns:
            Dict mapping tier names to confidence scores
        """
        scores = {}
        length = signals['length']
        thresholds = self.config.get_length_thresholds()
        
        for tier in self.tiers:
            score = 0.0
            
            # Base score from keyword matches
            keyword_score = signals['keyword_matches'].get(tier, 0)
            score += keyword_score * 0.3
            
            # Additional scoring for tier-specific patterns
            if tier == 'expert':
                # Check for system design indicators
                expert_patterns = ['distributed', 'microservices', 'scalability', 'architecture', 
                                 'fault tolerance', 'consistency', 'design decisions', 
                                 'trade-offs', 'bottlenecks', 'monitoring']
                pattern_matches = sum(1 for pattern in expert_patterns 
                                    if pattern in prompt_lower)
                score += pattern_matches * 0.15
                
            elif tier == 'complex':
                # Check for technical implementation indicators  
                complex_patterns = ['implement', 'algorithm', 'data structure', 'optimization',
                                  'thread-safe', 'performance', 'debugging']
                pattern_matches = sum(1 for pattern in complex_patterns
                                    if pattern in prompt_lower)
                score += pattern_matches * 0.1
            
            # Length-based scoring
            if tier == 'trivial':
                if length <= thresholds.get('trivial_max', 50):
                    score += 0.4
                elif length > thresholds.get('moderate_max', 1000):
                    score -= 0.3
            elif tier == 'simple':
                if thresholds.get('trivial_max', 50) < length <= thresholds.get('simple_max', 200):
                    score += 0.4
                elif length > thresholds.get('complex_max', 3000):
                    score -= 0.2
            elif tier == 'moderate':
                if thresholds.get('simple_max', 200) < length <= thresholds.get('moderate_max', 1000):
                    score += 0.3
            elif tier == 'complex':
                if thresholds.get('moderate_max', 1000) < length <= thresholds.get('complex_max', 3000):
                    score += 0.3
                elif length < thresholds.get('simple_max', 200):
                    score -= 0.2
            elif tier == 'expert':
                if length > thresholds.get('complex_max', 3000):
                    score += 0.4
                elif length > thresholds.get('moderate_max', 1000):
                    score += 0.2  # Medium length can still be expert
                elif length < thresholds.get('simple_max', 200):
                    score -= 0.3
            
            # Structural complexity bonus
            structural_score = signals['structural_complexity']
            if tier in ['complex', 'expert'] and structural_score > 3:
                score += 0.2
            elif tier in ['trivial', 'simple'] and structural_score == 0:
                score += 0.1
            
            # Code indicators
            if signals['code_indicators'] > 0:
                if tier in ['complex', 'expert']:
                    score += 0.3
                elif tier == 'trivial':
                    score -= 0.2
            
            # Question complexity
            if signals['has_questions']:
                if tier in ['simple', 'moderate']:
                    score += 0.1
            
            # Ensure score is in valid range
            score = max(0.0, min(1.0, score))
            scores[tier] = score
        
        # Don't normalize - keep absolute scores for better tier differentiation
        # Just ensure minimum scores
        for tier in self.tiers:
            if tier not in scores:
                scores[tier] = 0.0
        
        # If no clear winner, provide defaults based on length
        max_score = max(scores.values()) if scores else 0
        if max_score < 0.3:
            if length < thresholds.get('trivial_max', 50):
                scores['trivial'] = 0.6
            elif length < thresholds.get('simple_max', 200):
                scores['simple'] = 0.6
            elif length > thresholds.get('complex_max', 3000):
                scores['expert'] = 0.6
            else:
                scores['moderate'] = 0.6
        
        return scores
    
    def _generate_reasoning(self, signals: Dict, selected_tier: str) -> List[str]:
        """Generate human-readable reasoning for classification.
        
        Args:
            signals: Classification signals
            selected_tier: The selected complexity tier
            
        Returns:
            List of reasoning strings
        """
        reasoning = []
        
        # Length reasoning
        length = signals['length']
        thresholds = self.config.get_length_thresholds()
        
        if length <= thresholds.get('trivial_max', 50):
            reasoning.append(f"Very short prompt ({length} chars)")
        elif length > thresholds.get('complex_max', 3000):
            reasoning.append(f"Very long prompt ({length} chars)")
        
        # Keyword matches
        for tier in self.tiers:
            matches = signals['keyword_matches'].get(tier, 0)
            if matches > 0:
                reasoning.append(f"{matches} {tier}-tier keyword(s) found")
        
        # Structural features
        if signals['has_code']:
            reasoning.append("Contains code blocks or programming syntax")
        if signals['has_lists']:
            reasoning.append("Contains structured lists")
        if signals['structural_complexity'] > 3:
            reasoning.append("High structural complexity")
        
        # Code indicators
        if signals['code_indicators'] > 0:
            reasoning.append(f"{signals['code_indicators']} code indicator(s) found")
        
        # Final selection reason
        reasoning.append(f"Classified as '{selected_tier}' tier")
        
        return reasoning