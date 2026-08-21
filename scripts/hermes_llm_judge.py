"""Hermes Agent LLM-as-Judge Integration for Contradiction Detection.

This module provides integration with Hermes Agent's LLM providers
to perform actual LLM-based contradiction detection.
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hakua_memory import CompositeMemory
from hakua_memory.ebbinghaus.store import EbbinghausMemoryStore
from hakua_memory.semantic_graph.store import SemanticGraphStore


class HermesLLMJudge:
    """Uses Hermes Agent's LLM providers for contradiction detection."""
    
    def __init__(self, memory: CompositeMemory, model: str = "gpt-5.6-luna", provider: str = "openai-codex"):
        self.memory = memory
        self.model = model
        self.provider = provider
        self.ebbinghaus_store = memory.ebbinghaus
        self.semantic_graph_store = memory.semantic
        self.rag_store = memory.documents
    
    def extract_claims(self, query: str, top_k: int = 5) -> dict:
        """Extract claims from all three memory systems."""
        # Ebbinghaus
        ebbinghaus_results = self.ebbinghaus_store.recall(query, limit=top_k)
        ebbinghaus_claims = []
        for r in ebbinghaus_results:
            if isinstance(r, dict) and 'content' in r:
                ebbinghaus_claims.append({
                    'content': r['content'],
                    'salience': r.get('salience', 0.0),
                    'confidence': r.get('confidence', 0.0),
                    'tags': r.get('tags', [])
                })
        
        # RAG
        rag_results = self.memory.search_documents(query, top_k=top_k)
        rag_claims = []
        for r in rag_results:
            rag_claims.append({
                'content': r.get('content', ''),
                'document_title': r.get('document_title', ''),
                'score': r.get('score', 0.0)
            })
        
        # CoG
        cog_results = self.semantic_graph_store.search_nodes(query, top_k=top_k)
        cog_claims = []
        for r in cog_results:
            cog_claims.append({
                'content': r.get('summary', ''),
                'label': r.get('label', ''),
                'node_type': r.get('node_type', ''),
                'confidence': r.get('confidence', 0.0)
            })
        
        return {
            'ebbinghaus': ebbinghaus_claims,
            'rag': rag_claims,
            'cog': cog_claims
        }
    
    def build_judge_prompt(self, query: str, claims: dict) -> str:
        """Build the prompt for LLM-as-Judge."""
        prompt = f"""以下の3つのメモリシステムから抽出された情報を比較し、矛盾があるか判定してください。

**検索クエリ**: {query}

---

### 1. Ebbinghaus記憶 (経験的・時間的記憶)
件数: {len(claims['ebbinghaus'])}
"""
        for i, c in enumerate(claims['ebbinghaus']):
            prompt += f"\n  {i+1}. {c['content'][:300]}... (salience: {c.get('salience', 0):.2f}, confidence: {c.get('confidence', 0):.2f})"
        
        prompt += f"\n\n### 2. RAG文書検索 (外部知識・引用ベース)\n件数: {len(claims['rag'])}"
        for i, c in enumerate(claims['rag']):
            prompt += f"\n  {i+1}. {c['content'][:300]}... (doc: {c.get('document_title', 'N/A')}, score: {c.get('score', 0):.2f})"
        
        prompt += f"\n\n### 3. CoG/セマンティックグラフ (構造化知識・関係性)\n件数: {len(claims['cog'])}"
        for i, c in enumerate(claims['cog']):
            prompt += f"\n  {i+1}. [{c.get('node_type', '')}] {c.get('label', '')}: {c['content'][:300]}... (confidence: {c.get('confidence', 0):.2f})"
        
        prompt += """

---

### 判定基準
以下の観点で矛盾を検出してください：
1. **事実の矛盾** (factual): 同じ事象について異なる事実が述べられている
2. **数値の矛盾** (numerical): 同じ指標について異なる数値が示されている
3. **因果関係の矛盾** (causal): 同じ現象について異なる原因・結果が述べられている
4. **時系列の矛盾** (temporal): 同じイベントについて異なる時期・順序が示されている
5. **確信度の乖離** (confidence): 高い確信度の情報同士で内容が食い違っている

### 出力形式 (JSONのみ)
```json
{
  "has_contradiction": true/false,
  "description": "矛盾の具体的な説明（日本語）",
  "confidence": 0.0-1.0,
  "contradiction_type": "factual/numerical/causal/temporal/confidence/none",
  "conflicting_sources": ["ebbinghaus", "rag", "cog"],
  "details": {
    "ebbinghaus": "Ebbinghausでの該当内容",
    "rag": "RAGでの該当内容",
    "cog": "CoGでの該当内容"
  }
}
```

**重要**: JSONのみを出力してください。説明文は含めないでください。
"""
        return prompt
    
    def judge_with_hermes(self, prompt: str) -> dict:
        """
        Call Hermes Agent's LLM for judgment.
        This would use the actual Hermes Agent chat interface.
        For now, returns a structured result for manual LLM call.
        """
        return {
            "prompt": prompt,
            "model": self.model,
            "provider": self.provider,
            "instruction": "上記プロンプトに基づき、JSONのみで矛盾判定結果を出力してください。"
        }
    
    def detect_contradictions(self, queries: list[str]) -> list[dict]:
        """Detect contradictions for multiple queries using Hermes LLM."""
        results = []
        
        for query in queries:
            claims = self.extract_claims(query)
            prompt = self.build_judge_prompt(query, claims)
            judge_request = self.judge_with_hermes(prompt)
            
            results.append({
                "query": query,
                "judge_request": judge_request,
                "claims_summary": {
                    "ebbinghaus_count": len(claims['ebbinghaus']),
                    "rag_count": len(claims['rag']),
                    "cog_count": len(claims['cog'])
                }
            })
        
        return results


def run_hermes_judge_evaluation(memory: CompositeMemory, queries: list[str]) -> list[dict]:
    """Run contradiction detection using Hermes Agent as LLM Judge."""
    judge = HermesLLMJudge(memory)
    return judge.detect_contradictions(queries)


if __name__ == "__main__":
    # Demo
    memory = CompositeMemory(Path(".memory_seed42"))
    
    test_queries = [
        "キャッシュフロー分析",
        "マイクロサービスアーキテクチャ",
        "医療DX推進",
        "スマートファクトリー導入",
        "オムニチャネル戦略"
    ]
    
    results = run_hermes_judge_evaluation(memory, test_queries)
    
    # Save for Hermes Agent to process
    output_path = Path("hermes_judge_requests.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Generated {len(results)} Hermes judge requests")
    print(f"Saved to: {output_path}")
    print("\nTo evaluate, run each prompt through Hermes Agent and save results to 'hermes_judge_results.json'")