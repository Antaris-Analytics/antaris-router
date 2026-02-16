"""
Semantic classifier for Antaris Router v2.0.

Uses TF-IDF vectors + cosine similarity for prompt classification
instead of naive keyword matching. Learns from labeled examples.
All file-based, zero external dependencies.
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SemanticResult:
    """Result of semantic classification."""
    tier: str
    confidence: float
    similar_examples: List[Tuple[str, float]]  # (example_text, similarity)
    signals: Dict[str, any] = field(default_factory=dict)


class TFIDFVectorizer:
    """Lightweight TF-IDF implementation using only stdlib.
    
    Builds term frequency-inverse document frequency vectors
    for text comparison without any ML dependencies.
    """

    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.vocab: set = set()
        self.doc_count: int = 0

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into normalized terms."""
        text = text.lower()
        # Split on non-alphanumeric, keep meaningful tokens
        tokens = re.findall(r'[a-z][a-z0-9_]*(?:\'[a-z]+)?', text)
        # Filter stopwords (minimal set to keep it lightweight)
        stops = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'shall',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'as', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'out', 'off', 'over', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
            'own', 'same', 'so', 'than', 'too', 'very', 'just', 'because',
            'but', 'and', 'or', 'if', 'while', 'this', 'that', 'these',
            'those', 'it', 'its', 'i', 'me', 'my', 'we', 'our', 'you',
            'your', 'he', 'him', 'his', 'she', 'her', 'they', 'them',
            'their', 'what', 'which', 'who', 'whom',
        }
        return [t for t in tokens if t not in stops and len(t) > 1]

    def fit(self, documents: List[str]):
        """Build IDF from a corpus of documents."""
        self.doc_count = len(documents)
        doc_freq: Dict[str, int] = defaultdict(int)

        for doc in documents:
            seen = set(self._tokenize(doc))
            for term in seen:
                doc_freq[term] += 1
                self.vocab.add(term)

        # IDF = log(N / (1 + df)) — smoothed to avoid division by zero
        for term, df in doc_freq.items():
            self.idf[term] = math.log((self.doc_count + 1) / (1 + df))

    def transform(self, text: str) -> Dict[str, float]:
        """Convert text to TF-IDF vector (sparse dict)."""
        tokens = self._tokenize(text)
        if not tokens:
            return {}
        tf = Counter(tokens)
        max_tf = max(tf.values())
        vector = {}
        for term, count in tf.items():
            # Augmented TF to prevent bias toward longer documents
            normalized_tf = 0.5 + 0.5 * (count / max_tf)
            idf = self.idf.get(term, math.log(self.doc_count + 1))
            vector[term] = normalized_tf * idf
        return vector

    def similarity(self, vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
        """Cosine similarity between two sparse vectors."""
        if not vec_a or not vec_b:
            return 0.0
        common = set(vec_a) & set(vec_b)
        if not common:
            return 0.0
        dot = sum(vec_a[t] * vec_b[t] for t in common)
        mag_a = math.sqrt(sum(v * v for v in vec_a.values()))
        mag_b = math.sqrt(sum(v * v for v in vec_b.values()))
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def to_dict(self) -> Dict:
        return {
            'idf': self.idf,
            'doc_count': self.doc_count,
            'vocab_size': len(self.vocab),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TFIDFVectorizer':
        v = cls()
        v.idf = data.get('idf', {})
        v.doc_count = data.get('doc_count', 0)
        v.vocab = set(v.idf.keys())
        return v


# ── Built-in training examples ──────────────────────────────────
# These are the "seed" examples the classifier ships with.
# Users can add more via learn().

SEED_EXAMPLES = {
    'trivial': [
        "What time is it?",
        "Hello",
        "Thanks!",
        "Yes",
        "No",
        "OK",
        "What's the weather like?",
        "Translate 'hello' to Spanish",
        "What is 2 + 2?",
        "Define photosynthesis",
        "Who wrote Hamlet?",
        "What's the capital of France?",
        "Summarize this in one sentence",
        "Fix this typo",
    ],
    'simple': [
        "Write a short email declining a meeting",
        "Explain the difference between TCP and UDP",
        "Convert this JSON to YAML",
        "Write a Python function to reverse a string",
        "List 5 benefits of exercise",
        "Rewrite this paragraph to be more concise",
        "What are the pros and cons of remote work?",
        "Write a regex to match email addresses",
        "Explain how DNS works",
        "Create a SQL query to find duplicate rows",
    ],
    'moderate': [
        "Implement a React component that displays a sortable data table with pagination",
        "Write a Python class that manages a connection pool with timeout and retry logic",
        "Design a REST API for a todo application with authentication",
        "Analyze this dataset and identify the top 3 trends",
        "Write unit tests for this authentication module",
        "Create a CI/CD pipeline configuration for a Node.js application",
        "Implement a caching layer with LRU eviction and TTL support",
        "Build a command-line tool that processes CSV files and generates reports",
        "Write a middleware that handles rate limiting with sliding window",
        "Implement a WebSocket server for real-time notifications",
    ],
    'complex': [
        "Design a microservices architecture for an e-commerce platform handling 10K concurrent users with event sourcing and CQRS",
        "Implement a distributed task queue with priority scheduling, retry logic, dead letter handling, and horizontal scaling",
        "Build a custom ORM that supports lazy loading, eager loading, migrations, and connection pooling across multiple database backends",
        "Create a real-time collaborative editor with operational transformation, conflict resolution, and offline support",
        "Design and implement a plugin system with sandboxed execution, dependency resolution, and hot reloading",
        "Build a compiler frontend for a domain-specific language with lexer, parser, AST, type checker, and code generation",
        "Implement a distributed consensus algorithm (Raft) with leader election, log replication, and cluster membership changes",
        "Design a multi-tenant SaaS platform with data isolation, custom domains, usage-based billing, and white-labeling",
    ],
    'expert': [
        "Design the complete architecture for a high-frequency trading platform with sub-millisecond latency, fault tolerance across data centers, regulatory compliance, and real-time risk management. Include technology selection rationale, deployment strategy, monitoring approach, and disaster recovery plan.",
        "Architect a globally distributed database system that provides tunable consistency levels, automatic sharding with range and hash partitioning, cross-region replication with conflict resolution, online schema migrations, and point-in-time recovery. Address CAP theorem tradeoffs explicitly.",
        "Design a machine learning platform that handles model training pipelines, feature stores, A/B testing infrastructure, model serving with canary deployments, drift detection, automated retraining triggers, and governance/audit trails. Must support both batch and real-time inference at scale.",
        "Create a comprehensive security architecture for a healthcare platform including HIPAA compliance, zero-trust networking, end-to-end encryption, key management, audit logging, intrusion detection, incident response automation, and penetration testing framework.",
    ],
}


class SemanticClassifier:
    """Semantic prompt classifier using TF-IDF + cosine similarity.
    
    Replaces naive keyword matching with actual semantic understanding.
    Ships with seed examples and learns from user feedback.
    """

    def __init__(self, workspace: str = "."):
        self.workspace = os.path.abspath(workspace)
        self._examples_path = os.path.join(workspace, "routing_examples.json")
        self._model_path = os.path.join(workspace, "routing_model.json")
        
        self.vectorizer = TFIDFVectorizer()
        self.examples: Dict[str, List[str]] = {}
        self.example_vectors: Dict[str, List[Dict[str, float]]] = {}
        
        self._load_or_init()

    def _load_or_init(self):
        """Load saved model or initialize from seed examples."""
        if os.path.exists(self._examples_path) and os.path.exists(self._model_path):
            self._load()
        else:
            self._init_from_seeds()

    def _init_from_seeds(self):
        """Initialize vectorizer from seed examples."""
        self.examples = {tier: list(exs) for tier, exs in SEED_EXAMPLES.items()}
        self._rebuild_model()

    def _rebuild_model(self):
        """Rebuild TF-IDF model from current examples."""
        all_docs = []
        for exs in self.examples.values():
            all_docs.extend(exs)
        self.vectorizer.fit(all_docs)
        
        # Pre-compute vectors for all examples
        self.example_vectors = {}
        for tier, exs in self.examples.items():
            self.example_vectors[tier] = [self.vectorizer.transform(e) for e in exs]

    def classify(self, prompt: str, top_k: int = 3) -> SemanticResult:
        """Classify a prompt using semantic similarity to known examples.
        
        Args:
            prompt: Text to classify
            top_k: Number of similar examples to return
            
        Returns:
            SemanticResult with tier, confidence, and similar examples
        """
        if not prompt or not prompt.strip():
            return SemanticResult(tier='trivial', confidence=1.0, similar_examples=[])
        
        query_vec = self.vectorizer.transform(prompt)
        
        # Calculate similarity to each tier's examples
        tier_scores: Dict[str, List[float]] = defaultdict(list)
        all_similarities: List[Tuple[str, str, float]] = []  # (tier, example, sim)
        
        for tier, vectors in self.example_vectors.items():
            for i, vec in enumerate(vectors):
                sim = self.vectorizer.similarity(query_vec, vec)
                tier_scores[tier].append(sim)
                all_similarities.append((tier, self.examples[tier][i], sim))
        
        # Score each tier by average of top-N similarities (not just max)
        final_scores = {}
        for tier, sims in tier_scores.items():
            if not sims:
                final_scores[tier] = 0.0
                continue
            # Use top-3 average for robustness (not just max which is noisy)
            top_sims = sorted(sims, reverse=True)[:3]
            final_scores[tier] = sum(top_sims) / len(top_sims)
        
        # Structural boost: long/complex prompts shouldn't route to trivial
        length = len(prompt)
        if length > 500:
            final_scores['trivial'] *= 0.3
            final_scores['simple'] *= 0.6
        if length > 2000:
            final_scores['trivial'] *= 0.1
            final_scores['simple'] *= 0.3
            final_scores['moderate'] *= 0.7
        
        # Code detection boost
        if re.search(r'```|def |class |function |import |SELECT |CREATE ', prompt):
            final_scores['trivial'] *= 0.2
            final_scores.setdefault('moderate', 0)
            final_scores['moderate'] = max(final_scores.get('moderate', 0), 0.3)
        
        # Pick best tier
        best_tier = max(final_scores, key=final_scores.get)
        best_score = final_scores[best_tier]
        
        # Confidence = how much the winner leads the runner-up
        sorted_scores = sorted(final_scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            margin = sorted_scores[0] - sorted_scores[1]
            confidence = min(1.0, 0.5 + margin * 2)
        else:
            confidence = 0.5
        
        # Top similar examples
        all_similarities.sort(key=lambda x: x[2], reverse=True)
        similar = [(text, sim) for _, text, sim in all_similarities[:top_k]]
        
        return SemanticResult(
            tier=best_tier,
            confidence=confidence,
            similar_examples=similar,
            signals={
                'tier_scores': final_scores,
                'prompt_length': length,
                'method': 'tfidf_cosine',
            }
        )

    def learn(self, prompt: str, correct_tier: str):
        """Learn from a labeled example. Updates the model.
        
        Args:
            prompt: The prompt text
            correct_tier: The correct tier for this prompt
        """
        if correct_tier not in self.examples:
            self.examples[correct_tier] = []
        
        # Avoid exact duplicates
        if prompt not in self.examples[correct_tier]:
            self.examples[correct_tier].append(prompt)
            self._rebuild_model()
            self.save()

    def save(self):
        """Persist examples and model to disk."""
        os.makedirs(self.workspace, exist_ok=True)
        
        with open(self._examples_path, 'w') as f:
            json.dump(self.examples, f, indent=2)
        
        with open(self._model_path, 'w') as f:
            json.dump(self.vectorizer.to_dict(), f, indent=2)

    def _load(self):
        """Load saved examples and model."""
        with open(self._examples_path) as f:
            self.examples = json.load(f)
        with open(self._model_path) as f:
            self.vectorizer = TFIDFVectorizer.from_dict(json.load(f))
        
        # Rebuild example vectors
        self.example_vectors = {}
        for tier, exs in self.examples.items():
            self.example_vectors[tier] = [self.vectorizer.transform(e) for e in exs]

    def get_stats(self) -> Dict:
        """Get classifier statistics."""
        return {
            'total_examples': sum(len(exs) for exs in self.examples.values()),
            'examples_per_tier': {t: len(exs) for t, exs in self.examples.items()},
            'vocab_size': len(self.vectorizer.vocab),
            'doc_count': self.vectorizer.doc_count,
        }
