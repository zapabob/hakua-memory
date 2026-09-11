# 2026-09-11_sleep-memory-consolidation_Composer.md

## 概要

Ebbinghaus の睡眠サイクル（rehearse / forget-archive / dream-candidate）と dream preview/apply による意味記憶統合の試験を実装し、pytest と実行トライアルで検証した。

## 背景・要求

- ユーザー要求: 「睡眠による hakua-memory の記憶統合試験も実装実施」
- 既存実装には `sleep_cycle` / `dream_preview` / `dream_apply` があるが、専用回帰試験がほぼ無かった

## 前提・判断

- 実時間待機は使わず、可控 `time_fn` + `last_anchor_at` バックデートで減衰を再現
- sleep の recent/remote replay 予算は試験では 0 に固定し、本線の rehearse/forget/dream 分岐を決定論的に観測
- dream 統合の主経路は「高 salience + 減衰 + `max_sleep_rehearsals=0` → dream_candidate → preview/apply」
- Composite のデフォルト `remember(salience=0.65)` は keep 閾値 0.70 未満なので、統合試験では `ebbinghaus.remember(..., salience=0.9+)` を使用

## 変更対象ファイル

- `tests/test_sleep_consolidation.py`（新規）
- `scripts/run_sleep_consolidation_trial.py`（新規、tqdm 付きトライアル）
- `benchmarks/results/sleep_consolidation_trial.json`（実行証跡）
- 併せて継続作業: speculative-hybrid の主ゲートを `min_top_score` スイープへ修正（`scripts/benchmark_speculative_hybrid.py` / retrieval defaults）

## 実装詳細

1. pytest 7 本
   - 高 salience 減衰 → sleep rehearse
   - 低 salience 減衰 → forget + archive (`sleep-forget`)
   - experience functional forgetting → latent
   - rehearse cap 0 → dream_candidate
   - dream preview/apply → semantic 統合 + source archive (`dream-consolidated`) + idempotent re-apply
   - CompositeMemory.sleep ファサード経由の一連パイプライン
2. CLI トライアルで remember→age→sleep→dream を tqdm 可視化

## 実行コマンド

```powershell
py -3 -m pytest -p no:randomly tests/test_sleep_consolidation.py -q --tb=short
py -3 scripts/run_sleep_consolidation_trial.py --output benchmarks/results/sleep_consolidation_trial.json
```

## テスト・検証結果

- `tests/test_sleep_consolidation.py`: **7 passed**
- トライアル結果（要約）:
  - forgotten/archived: noise memory `3`
  - dream_candidates: `[2, 1]`
  - dream_apply: `semantic_memory_id=4`, sources `[2,1]` を `dream-consolidated` で archive
  - stats: `active_count=1`, `semantic_count=1`, `archived_count=3`

## 残留リスク

- sleep の recent/remote replay 予算 ON 時の分岐は本スイートでは未カバー
- CompositeMemory.remember が salience を露出していないため、ファサードだけでは dream 統合を起こしにくい
- speculative-hybrid の `min_top_score` 再ベンチは隔離 env 再実行が未完（初回 Friedman は完了済み）

## 次の推奨アクション

1. CompositeMemory.remember に `salience`/`valence` を透過
2. replay 予算 ON の sleep 回帰を追加
3. 隔離 env で `benchmark_speculative_hybrid.py` の min_top_score スイープ再実行
4. 外部多群ベンチ完走と README 反映、0.3.5 PyPI 公開
