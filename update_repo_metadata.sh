#!/bin/bash
# Update GitHub repo metadata

curl -X PATCH \
  -H "Accept: application/vnd.github.v3+json" \
  -H "Authorization: token ${GITHUB_TOKEN}" \
  https://api.github.com/repos/Antaris-Analytics/antaris-router \
  -d "{
    \"description\": \"File-based model router for 50-70% LLM cost reduction. Deterministic routing via keyword classification. Zero dependencies.\",
    \"topics\": [\"ai\", \"llm\", \"router\", \"cost-optimization\", \"deterministic\", \"file-based\", \"zero-dependencies\", \"python\", \"antaris-analytics\"]
  }"
