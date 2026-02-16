"""
Cost tracking for Antaris Router.

Tracks routing decisions, actual token usage, and provides cost analysis
and savings estimates compared to always using premium models.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

from .registry import ModelInfo


@dataclass
class UsageRecord:
    """Record of model usage for cost tracking."""
    timestamp: str
    model_name: str
    tier: str
    input_tokens: int
    output_tokens: int
    actual_cost: float
    prompt_hash: Optional[str] = None
    routing_confidence: Optional[float] = None


class CostTracker:
    """Tracks model usage costs and provides analysis."""
    
    def __init__(self, storage_path: str = "./routing_costs.json"):
        """Initialize cost tracker.
        
        Args:
            storage_path: Path to store cost tracking data
        """
        self.storage_path = storage_path
        self.usage_history: List[UsageRecord] = []
        self.load()
    
    def log_usage(self, model: ModelInfo, tier: str, input_tokens: int, 
                  output_tokens: int, prompt_hash: str = None, 
                  confidence: float = None) -> float:
        """Log model usage for cost tracking.
        
        Args:
            model: ModelInfo that was used
            tier: Complexity tier that was routed to
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens generated
            prompt_hash: Optional hash of the prompt for deduplication
            confidence: Optional routing confidence score
            
        Returns:
            Actual cost of this usage
        """
        actual_cost = model.calculate_cost(input_tokens, output_tokens)
        
        record = UsageRecord(
            timestamp=datetime.utcnow().isoformat(),
            model_name=model.name,
            tier=tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            actual_cost=actual_cost,
            prompt_hash=prompt_hash,
            routing_confidence=confidence
        )
        
        self.usage_history.append(record)
        return actual_cost
    
    def report(self, period: str = "week", model_registry = None) -> Dict[str, Any]:
        """Generate cost report for a given period.
        
        Args:
            period: Time period ("day", "week", "month", "all")
            model_registry: Optional ModelRegistry for additional analysis
            
        Returns:
            Dictionary with cost breakdown and statistics
        """
        # Filter records by period
        if period != "all":
            cutoff_date = self._get_cutoff_date(period)
            filtered_records = [
                r for r in self.usage_history
                if datetime.fromisoformat(r.timestamp) >= cutoff_date
            ]
        else:
            filtered_records = self.usage_history
        
        if not filtered_records:
            return {
                "period": period,
                "total_cost": 0.0,
                "total_requests": 0,
                "breakdown": {}
            }
        
        # Calculate totals
        total_cost = sum(r.actual_cost for r in filtered_records)
        total_requests = len(filtered_records)
        total_input_tokens = sum(r.input_tokens for r in filtered_records)
        total_output_tokens = sum(r.output_tokens for r in filtered_records)
        
        # Breakdown by model
        model_breakdown = {}
        tier_breakdown = {}
        
        for record in filtered_records:
            # Model breakdown
            if record.model_name not in model_breakdown:
                model_breakdown[record.model_name] = {
                    "requests": 0,
                    "total_cost": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0
                }
            
            model_breakdown[record.model_name]["requests"] += 1
            model_breakdown[record.model_name]["total_cost"] += record.actual_cost
            model_breakdown[record.model_name]["input_tokens"] += record.input_tokens
            model_breakdown[record.model_name]["output_tokens"] += record.output_tokens
            
            # Tier breakdown
            if record.tier not in tier_breakdown:
                tier_breakdown[record.tier] = {
                    "requests": 0,
                    "total_cost": 0.0,
                    "avg_cost": 0.0
                }
            
            tier_breakdown[record.tier]["requests"] += 1
            tier_breakdown[record.tier]["total_cost"] += record.actual_cost
        
        # Calculate averages for tier breakdown
        for tier_data in tier_breakdown.values():
            tier_data["avg_cost"] = tier_data["total_cost"] / tier_data["requests"]
        
        return {
            "period": period,
            "start_date": filtered_records[0].timestamp if filtered_records else None,
            "end_date": filtered_records[-1].timestamp if filtered_records else None,
            "total_cost": round(total_cost, 4),
            "total_requests": total_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_cost_per_request": round(total_cost / total_requests, 4) if total_requests > 0 else 0,
            "model_breakdown": model_breakdown,
            "tier_breakdown": tier_breakdown
        }
    
    def savings_estimate(self, comparison_model: str = "gpt-4o", 
                        model_registry = None) -> Dict[str, Any]:
        """Calculate savings compared to always using an expensive model.
        
        Args:
            comparison_model: Model to compare costs against (default: gpt-4o)
            model_registry: ModelRegistry to get model cost information
            
        Returns:
            Dictionary with savings analysis
        """
        if not self.usage_history:
            return {
                "total_savings": 0.0,
                "percentage_saved": 0.0,
                "comparison_model": comparison_model,
                "records_analyzed": 0
            }
        
        if not model_registry:
            # Can't calculate without model registry
            return {
                "error": "Model registry required for savings calculation",
                "comparison_model": comparison_model
            }
        
        comparison_model_info = model_registry.get_model(comparison_model)
        if not comparison_model_info:
            return {
                "error": f"Comparison model '{comparison_model}' not found in registry",
                "comparison_model": comparison_model
            }
        
        actual_cost = sum(r.actual_cost for r in self.usage_history)
        comparison_cost = 0.0
        
        for record in self.usage_history:
            comparison_cost += comparison_model_info.calculate_cost(
                record.input_tokens, record.output_tokens
            )
        
        savings = comparison_cost - actual_cost
        percentage_saved = (savings / comparison_cost * 100) if comparison_cost > 0 else 0
        
        return {
            "actual_cost": round(actual_cost, 4),
            "comparison_cost": round(comparison_cost, 4),
            "total_savings": round(savings, 4),
            "percentage_saved": round(percentage_saved, 2),
            "comparison_model": comparison_model,
            "records_analyzed": len(self.usage_history)
        }
    
    def model_efficiency(self) -> Dict[str, Dict[str, float]]:
        """Calculate efficiency metrics per model.
        
        Returns:
            Dictionary with efficiency metrics per model
        """
        model_stats = {}
        
        for record in self.usage_history:
            model_name = record.model_name
            
            if model_name not in model_stats:
                model_stats[model_name] = {
                    "total_cost": 0.0,
                    "total_tokens": 0,
                    "requests": 0,
                    "confidence_scores": []
                }
            
            model_stats[model_name]["total_cost"] += record.actual_cost
            model_stats[model_name]["total_tokens"] += record.input_tokens + record.output_tokens
            model_stats[model_name]["requests"] += 1
            
            if record.routing_confidence is not None:
                model_stats[model_name]["confidence_scores"].append(record.routing_confidence)
        
        # Calculate efficiency metrics
        efficiency = {}
        for model_name, stats in model_stats.items():
            avg_confidence = (
                sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
                if stats["confidence_scores"] else 0.0
            )
            
            efficiency[model_name] = {
                "cost_per_request": stats["total_cost"] / stats["requests"],
                "cost_per_token": stats["total_cost"] / stats["total_tokens"] if stats["total_tokens"] > 0 else 0,
                "avg_routing_confidence": round(avg_confidence, 3),
                "total_requests": stats["requests"]
            }
        
        return efficiency
    
    def tier_performance(self) -> Dict[str, Dict[str, Any]]:
        """Analyze performance by complexity tier.
        
        Returns:
            Dictionary with performance metrics per tier
        """
        tier_stats = {}
        
        for record in self.usage_history:
            tier = record.tier
            
            if tier not in tier_stats:
                tier_stats[tier] = {
                    "total_cost": 0.0,
                    "requests": 0,
                    "models_used": set(),
                    "confidence_scores": []
                }
            
            tier_stats[tier]["total_cost"] += record.actual_cost
            tier_stats[tier]["requests"] += 1
            tier_stats[tier]["models_used"].add(record.model_name)
            
            if record.routing_confidence is not None:
                tier_stats[tier]["confidence_scores"].append(record.routing_confidence)
        
        # Calculate performance metrics
        performance = {}
        for tier, stats in tier_stats.items():
            avg_confidence = (
                sum(stats["confidence_scores"]) / len(stats["confidence_scores"])
                if stats["confidence_scores"] else 0.0
            )
            
            performance[tier] = {
                "avg_cost_per_request": stats["total_cost"] / stats["requests"],
                "total_requests": stats["requests"],
                "unique_models": len(stats["models_used"]),
                "models_used": sorted(list(stats["models_used"])),
                "avg_routing_confidence": round(avg_confidence, 3)
            }
        
        return performance
    
    def _get_cutoff_date(self, period: str) -> datetime:
        """Get cutoff date for filtering records.
        
        Args:
            period: Time period ("day", "week", "month")
            
        Returns:
            Cutoff datetime
        """
        now = datetime.utcnow()
        
        if period == "day":
            return now - timedelta(days=1)
        elif period == "week":
            return now - timedelta(weeks=1)
        elif period == "month":
            return now - timedelta(days=30)
        else:
            return datetime.min
    
    def save(self) -> None:
        """Save usage history to storage file."""
        data = {
            "version": "1.0.0",
            "saved_at": datetime.utcnow().isoformat(),
            "usage_history": [asdict(record) for record in self.usage_history]
        }
        
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        from .utils import atomic_write_json
        atomic_write_json(self.storage_path, data)
    
    def load(self) -> None:
        """Load usage history from storage file."""
        if not os.path.exists(self.storage_path):
            self.usage_history = []
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            self.usage_history = []
            for record_data in data.get("usage_history", []):
                record = UsageRecord(**record_data)
                self.usage_history.append(record)
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # Handle corrupted or invalid data
            print(f"Warning: Could not load usage history from {self.storage_path}: {e}")
            self.usage_history = []
    
    def clear_history(self, before_date: str = None) -> int:
        """Clear usage history before a given date.
        
        Args:
            before_date: ISO format date string. If None, clears all history.
            
        Returns:
            Number of records removed
        """
        original_count = len(self.usage_history)
        
        if before_date is None:
            self.usage_history = []
        else:
            cutoff = datetime.fromisoformat(before_date)
            self.usage_history = [
                r for r in self.usage_history
                if datetime.fromisoformat(r.timestamp) >= cutoff
            ]
        
        return original_count - len(self.usage_history)