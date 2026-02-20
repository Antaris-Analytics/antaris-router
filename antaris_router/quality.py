"""
Quality tracking and outcome learning for Antaris Router v2.0.

Tracks routing decisions → outcomes, learns which models perform
best for which types of tasks. All file-based, zero dependencies.
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

_log = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """A recorded routing decision with outcome."""
    prompt_hash: str
    tier: str
    model: str
    timestamp: float
    latency_ms: Optional[float] = None
    cost: Optional[float] = None
    quality_score: Optional[float] = None  # 0-1, user or auto-rated
    success: Optional[bool] = None
    escalated: bool = False  # Was this retried on a better model?
    context: Optional[Dict] = None


class QualityTracker:
    """Tracks model performance per task type and learns from outcomes.
    
    Records every routing decision and its outcome. Over time, builds
    a performance profile for each model on each task tier. Uses this
    to make smarter routing decisions.
    
    All data stored in plain JSON files.
    """

    def __init__(self, workspace: str = ".", max_age_days: int = 90):
        self.workspace = os.path.abspath(workspace)
        self._decisions_path = os.path.join(workspace, "routing_decisions.json")
        self._profiles_path = os.path.join(workspace, "model_profiles.json")
        self.max_age_days = max_age_days

        self.decisions: List[Dict] = []
        self.profiles: Dict[str, Dict] = {}  # model -> tier -> stats

        self._load()

    def _load(self):
        """Load existing tracking data."""
        if os.path.exists(self._decisions_path):
            try:
                with open(self._decisions_path) as f:
                    self.decisions = json.load(f)
            except Exception as exc:
                _log.warning(
                    "Could not load routing decisions from %s: %s — starting empty",
                    self._decisions_path, exc,
                )
                self.decisions = []
        if os.path.exists(self._profiles_path):
            try:
                with open(self._profiles_path) as f:
                    self.profiles = json.load(f)
            except Exception as exc:
                _log.warning(
                    "Could not load model profiles from %s: %s — starting empty",
                    self._profiles_path, exc,
                )
                self.profiles = {}

    def save(self):
        """Persist tracking data to disk."""
        os.makedirs(self.workspace, exist_ok=True)
        
        from .utils import atomic_write_json
        atomic_write_json(self._decisions_path, self.decisions)
        atomic_write_json(self._profiles_path, self.profiles)

    def record(self, decision: RoutingDecision):
        """Record a routing decision."""
        self.decisions.append(asdict(decision))
        self._update_profile(decision)
        
        # Auto-save every 100 decisions
        if len(self.decisions) % 100 == 0:
            self.save()

    def record_outcome(self, prompt_hash: str, quality_score: float = None,
                       success: bool = None, latency_ms: float = None):
        """Record the outcome of a previously recorded decision.
        
        Args:
            prompt_hash: Hash of the original prompt
            quality_score: 0.0-1.0 quality rating
            success: Whether the routing was successful
            latency_ms: Response latency in milliseconds
        """
        # Find the most recent decision with this hash
        for decision in reversed(self.decisions):
            if decision['prompt_hash'] == prompt_hash:
                if quality_score is not None:
                    decision['quality_score'] = quality_score
                if success is not None:
                    decision['success'] = success
                if latency_ms is not None:
                    decision['latency_ms'] = latency_ms
                
                # Update profile with new data
                self._update_profile(RoutingDecision(**{
                    k: v for k, v in decision.items()
                    if k in RoutingDecision.__dataclass_fields__
                }))
                break

    def _update_profile(self, decision: RoutingDecision):
        """Update model performance profile from a decision."""
        model = decision.model
        tier = decision.tier
        
        if model not in self.profiles:
            self.profiles[model] = {}
        if tier not in self.profiles[model]:
            self.profiles[model][tier] = {
                'total': 0,
                'successes': 0,
                'failures': 0,
                'escalations': 0,
                'avg_quality': 0.0,
                'avg_latency_ms': 0.0,
                'total_cost': 0.0,
                'quality_samples': 0,
                'latency_samples': 0,
            }
        
        stats = self.profiles[model][tier]
        stats['total'] += 1
        
        if decision.success is True:
            stats['successes'] += 1
        elif decision.success is False:
            stats['failures'] += 1
        
        if decision.escalated:
            stats['escalations'] += 1
        
        if decision.quality_score is not None:
            n = stats['quality_samples']
            stats['avg_quality'] = (stats['avg_quality'] * n + decision.quality_score) / (n + 1)
            stats['quality_samples'] = n + 1
        
        if decision.latency_ms is not None:
            n = stats['latency_samples']
            stats['avg_latency_ms'] = (stats['avg_latency_ms'] * n + decision.latency_ms) / (n + 1)
            stats['latency_samples'] = n + 1
        
        if decision.cost is not None:
            stats['total_cost'] += decision.cost

    def get_model_score(self, model: str, tier: str) -> float:
        """Get a composite quality score for a model on a tier.

        Returns 0.0-1.0 where higher is better. Factors in:
        - Success rate (40%)
        - Average quality (40%)
        - Escalation rate (20%, inverted — fewer escalations is better)

        Only decisions within the last ``max_age_days`` days are considered.
        Returns 0.5 (neutral) if no data available.
        """
        cutoff_ts = (
            datetime.now(timezone.utc) - timedelta(days=self.max_age_days)
        ).timestamp()

        # Build a fresh stats snapshot from time-windowed decisions
        total = 0
        successes = 0
        failures = 0
        escalations = 0
        quality_sum = 0.0
        quality_count = 0

        for d in self.decisions:
            if d.get('model') != model or d.get('tier') != tier:
                continue
            ts = d.get('timestamp', 0)
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts).timestamp()
                except Exception:
                    ts = 0.0
            if ts < cutoff_ts:
                continue
            total += 1
            if d.get('success') is True:
                successes += 1
            elif d.get('success') is False:
                failures += 1
            if d.get('escalated'):
                escalations += 1
            qs = d.get('quality_score')
            if qs is not None:
                quality_sum += float(qs)
                quality_count += 1

        if total == 0:
            # Fall back to profile aggregates if no windowed decisions exist
            if model not in self.profiles or tier not in self.profiles[model]:
                return 0.5
            stats = self.profiles[model][tier]
            if stats['total'] == 0:
                return 0.5
            success_rate = stats['successes'] / max(stats['total'], 1)
            quality = stats['avg_quality'] if stats['quality_samples'] > 0 else 0.5
            escalation_rate = 1.0 - (stats['escalations'] / max(stats['total'], 1))
            return 0.4 * success_rate + 0.4 * quality + 0.2 * escalation_rate

        success_rate = successes / total
        quality = (quality_sum / quality_count) if quality_count > 0 else 0.5
        escalation_rate = 1.0 - (escalations / total)

        return 0.4 * success_rate + 0.4 * quality + 0.2 * escalation_rate

    def recommend_model(self, tier: str, candidates: List[str],
                        optimize: str = "balanced") -> Tuple[str, float]:
        """Recommend the best model for a tier based on historical performance.
        
        Args:
            tier: Task complexity tier
            candidates: List of candidate model names
            optimize: Optimization goal - "quality", "cost", "speed", "balanced"
            
        Returns:
            Tuple of (recommended_model, confidence)
        """
        if not candidates:
            return ("", 0.0)
        
        scores = {}
        for model in candidates:
            base_score = self.get_model_score(model, tier)
            
            # Adjust based on optimization goal
            if optimize == "quality":
                scores[model] = base_score
            elif optimize == "cost":
                stats = self.profiles.get(model, {}).get(tier, {})
                cost = stats.get('total_cost', 0) / max(stats.get('total', 1), 1)
                # Lower cost = higher score (inverted)
                cost_factor = 1.0 / (1.0 + cost * 10) if cost > 0 else 0.5
                scores[model] = 0.3 * base_score + 0.7 * cost_factor
            elif optimize == "speed":
                stats = self.profiles.get(model, {}).get(tier, {})
                latency = stats.get('avg_latency_ms', 1000)
                speed_factor = 1.0 / (1.0 + latency / 1000)
                scores[model] = 0.3 * base_score + 0.7 * speed_factor
            else:  # balanced
                scores[model] = base_score
        
        best_model = max(scores, key=scores.get)
        confidence = scores[best_model]
        
        return (best_model, confidence)

    def should_escalate(self, model: str, tier: str, threshold: float = 0.3) -> bool:
        """Determine if a model has poor enough performance on a tier to warrant escalation.
        
        Args:
            model: Model name
            tier: Task tier
            threshold: Quality threshold below which to escalate
            
        Returns:
            True if the model should be skipped in favor of a better one
        """
        score = self.get_model_score(model, tier)
        return score < threshold

    def get_stats(self) -> Dict:
        """Get tracking statistics."""
        total = len(self.decisions)
        with_outcome = sum(1 for d in self.decisions if d.get('quality_score') is not None)
        escalated = sum(1 for d in self.decisions if d.get('escalated', False))
        
        return {
            'total_decisions': total,
            'with_outcomes': with_outcome,
            'escalated': escalated,
            'models_tracked': len(self.profiles),
            'outcome_rate': with_outcome / max(total, 1),
        }

    def trim(self, keep_last: int = 10000):
        """Trim old decisions to prevent unbounded growth.
        
        Keeps the most recent N decisions. Profiles are preserved
        (they're aggregated, not raw data).
        """
        if len(self.decisions) > keep_last:
            self.decisions = self.decisions[-keep_last:]
