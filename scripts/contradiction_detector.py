"""LLM-as-Judge Contradiction Detection for hakua-memory.

Compares Ebbinghaus, RAG, and CoG (Semantic Graph) memories
to detect contradictions using an LLM judge.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hakua_memory import CompositeMemory


@dataclass
class ContradictionResult:
    """Result of contradiction detection."""
    has_contradiction: Optional[bool]
    description: str
    confidence: float
    sources: list[str]
    ebbinghaus_content: Optional[str] = None
    rag_content: Optional[str] = None
    cog_content: Optional[str] = None


class ContradictionDetector:
    """Detects contradictions across memory systems using LLM-as-Judge."""

    def __init__(self, memory: CompositeMemory):
        self.memory = memory
        self.ebbinghaus_store = memory.ebbinghaus
        self.semantic_graph_store = memory.semantic
        self.rag_store = memory.documents

    def extract_ebbinghaus_claims(self, query: str, top_k: int = 5) -> list[dict]:
        """Extract claims from Ebbinghaus memory."""
        results = self.ebbinghaus_store.recall(query, limit=top_k)
        claims = []
        for r in results:
            if isinstance(r, dict) and 'content' in r:
                claims.append({
                    'source': 'ebbinghaus',
                    'content': r['content'],
                    'salience': r.get('salience', 0.0),
                    'tags': r.get('tags', []),
                    'confidence': r.get('confidence', 0.0)
                })
        return claims

    def extract_rag_claims(self, query: str, top_k: int = 5) -> list[dict]:
        """Extract claims from RAG documents."""
        results = self.memory.search_documents(query, top_k=top_k)
        claims = []
        for r in results:
            claims.append({
                'source': 'rag',
                'content': r.get('content', ''),
                'document_title': r.get('document_title', ''),
                'page_number': r.get('page_number'),
                'score': r.get('score', 0.0)
            })
        return claims

    def extract_cog_claims(self, query: str, top_k: int = 5) -> list[dict]:
        """Extract claims from Semantic Graph (CoG)."""
        results = self.semantic_graph_store.search_nodes(query, top_k=top_k)
        claims = []
        for r in results:
            claims.append({
                'source': 'cog',
                'content': r.get('summary', ''),
                'label': r.get('label', ''),
                'node_type': r.get('node_type', ''),
                'confidence': r.get('confidence', 0.0),
                'status': r.get('status', '')
            })
        return claims

    def build_judge_prompt(self, query: str, ebbinghaus_claims: list[dict],
                           rag_claims: list[dict], cog_claims: list[dict]) -> str:
        """Build the prompt for LLM-as-Judge."""
        prompt = f"""以下の3つのメモリシステムから抽出された情報を比較し、矛盾があるか判定してください。

**検索クエリ**: {query}

---

### 1. Ebbinghaus記憶 (経験的・時間的記憶)
件数: {len(ebbinghaus_claims)}
"""
        for i, c in enumerate(ebbinghaus_claims):
            prompt += f"\n  {i+1}. {c['content'][:200]}... (salience: {c.get('salience', 0):.2f})"

        prompt += "\n\n### 2. RAG文書検索 (外部知識・引用ベース)"
        prompt += f"\n件数: {len(rag_claims)}"
        for i, c in enumerate(rag_claims):
            prompt += f"\n  {i+1}. {c['content'][:200]}... (doc: {c.get('document_title', 'N/A')})"

        prompt += "\n\n### 3. CoG/セマンティックグラフ (構造化知識・関係性)"
        prompt += f"\n件数: {len(cog_claims)}"
        for i, c in enumerate(cog_claims):
            prompt += f"\n  {i+1}. [{c.get('node_type', '')}] {c.get('label', '')}: {c['content'][:200]}... (confidence: {c.get('confidence', 0):.2f})"

        prompt += """

---

### 判定基準
以下の観点で矛盾を検出してください：
1. **事実の矛盾**: 同じ事象について異なる事実が述べられている
2. **数値の矛盾**: 同じ指標について異なる数値が示されている
3. **因果関係の矛盾**: 同じ現象について異なる原因・結果が述べられている
4. **時系列の矛盾**: 同じイベントについて異なる時期・順序が示されている
5. **確信度の乖離**: 高い確信度の情報同士で内容が食い違っている

### 出力形式 (JSON)
```json
{
  "has_contradiction": true/false,
  "description": "矛盾の具体的な説明",
  "confidence": 0.0-1.0,
  "contradiction_type": "factual/numerical/causal/temporal/confidence",
  "conflicting_sources": ["ebbinghaus", "rag", "cog"],
  "details": {
    "ebbinghaus": "該当内容",
    "rag": "該当内容",
    "cog": "該当内容"
  }
}
```

矛盾がない場合は has_contradiction: false とし、description に「矛盾なし」と記載してください。
"""
        return prompt

    def detect_contradictions(self, query: str, top_k: int = 5) -> ContradictionResult:
        """Detect contradictions for a given query across all three systems."""
        # Extract claims from all three systems
        ebbinghaus_claims = self.extract_ebbinghaus_claims(query, top_k)
        rag_claims = self.extract_rag_claims(query, top_k)
        cog_claims = self.extract_cog_claims(query, top_k)

        # If no claims from any system, return no contradiction
        if not ebbinghaus_claims and not rag_claims and not cog_claims:
            return ContradictionResult(
                has_contradiction=False,
                description="比較対象の情報がありません",
                confidence=1.0,
                sources=[]
            )

        # Build prompt for LLM judge
        _ = self.build_judge_prompt(query, ebbinghaus_claims, rag_claims, cog_claims)

        # In a real implementation, this would call an LLM
        # For now, return a structured result that can be used with an LLM
        return ContradictionResult(
            has_contradiction=None,  # To be determined by LLM
            description="LLM判定待ち",
            confidence=0.0,
            sources=["ebbinghaus", "rag", "cog"],
            ebbinghaus_content=json.dumps(ebbinghaus_claims, ensure_ascii=False),
            rag_content=json.dumps(rag_claims, ensure_ascii=False),
            cog_content=json.dumps(cog_claims, ensure_ascii=False)
        )

    def get_judge_prompt(self, query: str, top_k: int = 5) -> str:
        """Get the prompt to feed to an LLM for contradiction detection."""
        ebbinghaus_claims = self.extract_ebbinghaus_claims(query, top_k)
        rag_claims = self.extract_rag_claims(query, top_k)
        cog_claims = self.extract_cog_claims(query, top_k)
        return self.build_judge_prompt(query, ebbinghaus_claims, rag_claims, cog_claims)


def run_contradiction_check(memory: CompositeMemory, queries: list[str]) -> list[dict]:
    """Run contradiction detection on multiple queries."""
    detector = ContradictionDetector(memory)
    results = []

    for query in queries:
        prompt = detector.get_judge_prompt(query)
        result = {
            "query": query,
            "judge_prompt": prompt,
            "ebbinghaus_count": len(detector.extract_ebbinghaus_claims(query)),
            "rag_count": len(detector.extract_rag_claims(query)),
            "cog_count": len(detector.extract_cog_claims(query))
        }
        results.append(result)

    return results


if __name__ == "__main__":
    # Demo: Create a sample memory and test
    memory = CompositeMemory(Path(".memory_seed42"))

    test_queries = [
        "キャッシュフロー分析",
        "マイクロサービスアーキテクチャ",
        "医療DX推進",
        "スマートファクトリー導入",
        "オムニチャネル戦略"
    ]

    results = run_contradiction_check(memory, test_queries)

    # Save prompts for LLM evaluation
    output_path = Path("contradiction_judge_prompts.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(results)} judge prompts")
    print(f"Saved to: {output_path}")

    # Print first prompt as sample
    print("\n=== Sample Judge Prompt ===")
    print(results[0]["judge_prompt"][:1000] + "...")
