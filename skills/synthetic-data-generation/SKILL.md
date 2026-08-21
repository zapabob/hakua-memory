--- 
name: synthetic-data-generation 
description: Generate synthetic business domain datasets for RAG and semantic graph training in restricted environments 
version: 0.1.0 
author: Hermes Agent 
platforms: [windows] 
tags: [synthetic, rag, semantic-graph, benchmark, japanese, english] 
--- 

# Synthetic Data Generation Skill

This skill covers the creation of synthetic business domain datasets for RAG and semantic graph training, with Japanese and English language support. Designed for environments with restricted external resource access.

## When to Use
- When you need to generate synthetic training data for RAG or semantic graph systems
- When external dataset access is restricted (e.g., workspace path blocks in Hermes desktop app)
- When working in isolated virtual environments with local project only
- When quick prototype data is needed for evaluation or comparison

## Prerequisites
- hakua-memory project (v0.2.3 or later)
- Python 3.11–3.13 with sys.path access to src/
- Access to hakua_memory.rag.chunking, hakua_memory.semantic_graph.store, hakua_memory.ebbinghaus.models
- Benchmark script: scripts/benchmark_cross_validation.py (created in v0.2.7)

## How to Run
```bash
# Generate synthetic business dataset (Japanese/English)
python -c "
import json, random
from pathlib import Path
jp_keywords = ['プロジェクト管理','予算策定','KPI管理','業務改善','効率化','DX推進']
ov_keywords = ['Project Management','Budget Planning','KPI Management','Process Improvement','Efficiency']
data = []
for i in range(20):
    tp_jp = random.choice(jp_keywords); data.append({'id':f'jp_{i}','language':'Japanese','topic':tp_jp,'content':f'{tp_jp}に関する検討事項です。'})
    tp_en = random.choice(ov_keywords); data.append({'id':f'ov_{i}','language':'English','topic':tp_en,'content':f'{tp_en} involves budget/schedule/resource allocation.'})
with open(Path('synthetic_business_dataset.json'),'w',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)
print(f'Generated {len(data)} entries')
"

# Run cross-validation benchmark
python scripts/benchmark_cross_validation.py
```

## Quick Reference
- **Synthetic dataset**: `synthetic_business_dataset.json` (40 entries: 20 Japanese, 20 English)
- **Benchmark script**: `scripts/benchmark_cross_validation.py`
- **RAG ingestion**: `CompositeMemory.ingest_document()` with temp files
- **Semantic graph construction**: `memory.add_node()` for node creation with `node_id`, `node_type`, `label`, `summary`, `status`, `authority`, `confidence`, `salience`, `evidence`
- **Constraint**: External folders (`C:\Users\downl\.hermes\attachments\`, `C:\Users\downl\Downloads\`) block direct access — use temp files or project-relative paths only

## Procedure
1. **Generate synthetic dataset**: Run inline Python to create `synthetic_business_dataset.json` with 20 Japanese + 20 English business domain entries
2. **Load dataset**: `json.load('synthetic_business_dataset.json')` to retrieve entries
3. **RAG ingestion**: For each entry, create temp `.md` file and call `memory.ingest_document(path, title, author, department)`
4. **Semantic graph construction**: Use `memory.add_node()` to create structured nodes with `node_id`, `node_type`, `label`, `summary`, `status`, `authority`, `confidence`, `salience`, `evidence`
5. **Benchmark comparison**: Execute `python scripts/benchmark_cross_validation.py` to get RAG/CoG/Ebbinghaus latency and stats
6. **Update documentation**: If needed, add performance comparison table to README using the benchmark output format

## Pitfalls
- **External folder access blocks**: `C:\Users\downl\.hermes\attachments\` and `C:\Users\downl\Downloads\` paths are blocked by workspace constraints — use temp files or project-relative paths only
- **RAG ingest requires files**: `ingest_document()` expects actual file paths; use `tempfile.NamedTemporaryFile()` for synthetic data
- **Japanese vocabulary must use Japanese concepts**: Per user preference, all concept labels must be Japanese (not English) — `JapaneseVocabulary` classes: `['AUDIT_LOG', 'HERMES_AGENT', 'MEMORY', 'OPENCLAW', 'POLICY_GUARD', 'VOICEVOX']`
- **Benchmark variance**: Latency depends on text length, chunk_size, overlap; always run multiple trials for stable stats
- **ruff compliance**: Always run `ruff check .` after adding new files; fix I001/E501/W292 before committing

## Verification
- `ruff check .` → **All checks passed!**
- `git status` → Verify expected changes present (new .json, new .py, modified .md)
- `benchmark_cross_validation.py` output format: JSON with `rag`, `cog`, `ebbinghaus`, `statistics` keys
- Japanese vocabulary check: `[c for c in dir(JapaneseVocabulary) if c.isupper() and not c.startswith('_')]` → 6 uppercase labels
- Git tag consistency: Ensure version increments match project milestones (v0.2.0 onwards)

## Supporting Reference Files (optional)
- `references/synthetic-data-format.md` — JSON schema for synthetic dataset entries
- `templates/synthetic_dataset_generator.py` — Starter generator script (boilerplate with keyword lists)
- `scripts/benchmark_cross_validation.py` — Already created; statically re-runnable benchmark action

## Version History
- **v0.1.0** (2026-08-22): Initial release — synthetic dataset generation + RAG training + semantic graph construction + benchmark comparison workflow