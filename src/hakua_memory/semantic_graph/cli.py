"""Operator CLI: hermes semantic-graph ..."""

from __future__ import annotations

import argparse
import json
from typing import Any


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def register_cli(ctx: Any, runtime: Any) -> None:
    def setup_fn(parser: argparse.ArgumentParser) -> None:
        subs = parser.add_subparsers(dest="semantic_graph_command", required=True)

        subs.add_parser("status", help="Show DB path, schema, FTS, and counts")
        subs.add_parser(
            "embedding-status",
            help="Show configured embedding backend and namespace",
        )

        backfill = subs.add_parser(
            "embedding-backfill",
            help="Explicitly inspect or apply a bounded embedding backfill",
        )
        backfill.add_argument("--limit", type=_positive_int, required=True)
        mode = backfill.add_mutually_exclusive_group(required=True)
        mode.add_argument("--dry-run", action="store_true")
        mode.add_argument("--apply", action="store_true")

        subs.add_parser(
            "cognitive-status",
            help="Show Ebbinghaus bridge and projection status",
        )
        for name, help_text in (
            ("cognitive-sync", "Explicitly sync Ebbinghaus memories to the graph"),
            ("cognitive-repair", "Retry pending Ebbinghaus bridge events"),
        ):
            operation = subs.add_parser(name, help=help_text)
            operation.add_argument("--limit", type=_positive_int, required=True)
            operation_mode = operation.add_mutually_exclusive_group(required=True)
            operation_mode.add_argument("--dry-run", action="store_true")
            operation_mode.add_argument("--apply", action="store_true")

        search = subs.add_parser("search", help="Search graph nodes")
        search.add_argument("query")
        search.add_argument("--limit", type=int, default=8)

        show = subs.add_parser("show", help="Show a graph object")
        show.add_argument("type", choices=["run", "node", "edge", "artifact", "fragment", "evaluation"])
        show.add_argument("id")

        export = subs.add_parser("export", help="Export graph records")
        export.add_argument("--run-id", default="")
        export.add_argument("--format", choices=["json", "jsonl", "markdown"], default="json")
        export.add_argument("--include-artifacts", action="store_true")
        export.add_argument("--include-rejected", action="store_true")
        export.add_argument("--output", default="")

        purge = subs.add_parser(
            "purge",
            help="Delete rejected/superseded and old artifacts (operator only)",
        )
        purge.add_argument("--before", required=True, help="YYYY-MM-DD")
        purge.add_argument("--confirm", required=True, help="Must be exactly PURGE")

        vacuum = subs.add_parser("vacuum", help="VACUUM the SQLite database")
        vacuum.add_argument("--confirm", required=True, help="Must be exactly VACUUM")

    def handler_fn(args: argparse.Namespace) -> int:
        cmd = getattr(args, "semantic_graph_command", None)
        if cmd == "status":
            print(runtime.handle_status({}))
            return 0
        if cmd == "embedding-status":
            print(runtime.handle_embedding_status({}))
            return 0
        if cmd == "embedding-backfill":
            print(
                runtime.handle_embedding_backfill(
                    {
                        "limit": int(args.limit),
                        "dry_run": bool(args.dry_run),
                        "apply": bool(args.apply),
                    }
                )
            )
            return 0
        if cmd == "cognitive-status":
            print(runtime.handle_cognitive_status({}))
            return 0
        if cmd in {"cognitive-sync", "cognitive-repair"}:
            payload = {
                "limit": int(args.limit),
                "dry_run": bool(args.dry_run),
                "apply": bool(args.apply),
            }
            if cmd == "cognitive-sync":
                print(runtime.handle_cognitive_sync(payload))
            else:
                print(runtime.handle_cognitive_repair(payload))
            return 0
        if cmd == "search":
            print(
                runtime.handle_search(
                    {"query": args.query, "top_k": int(args.limit)}
                )
            )
            return 0
        if cmd == "show":
            print(
                runtime.handle_get(
                    {
                        "object_type": args.type,
                        "object_id": args.id,
                        "include_neighbors": True,
                        "include_evidence": True,
                    }
                )
            )
            return 0
        if cmd == "export":
            payload = {
                "format": args.format,
                "include_artifacts": bool(args.include_artifacts),
                "include_rejected": bool(args.include_rejected),
            }
            if args.run_id:
                payload["run_id"] = args.run_id
            if args.output:
                payload["output_path"] = args.output
            print(runtime.handle_export(payload))
            return 0
        if cmd == "purge":
            if args.confirm != "PURGE":
                print(json.dumps({"success": False, "error": "confirm must be PURGE"}))
                return 2
            result = runtime.store().purge_before(args.before)
            print(json.dumps({"success": True, "purged": result}, ensure_ascii=False))
            return 0
        if cmd == "vacuum":
            if args.confirm != "VACUUM":
                print(json.dumps({"success": False, "error": "confirm must be VACUUM"}))
                return 2
            runtime.store().vacuum()
            print(json.dumps({"success": True, "vacuum": True}))
            return 0
        print(json.dumps({"success": False, "error": f"unknown command: {cmd}"}))
        return 2

    ctx.register_cli_command(
        name="semantic-graph",
        help="Manage the semantic-graph memory store (operator tools include purge)",
        setup_fn=setup_fn,
        handler_fn=handler_fn,
    )
