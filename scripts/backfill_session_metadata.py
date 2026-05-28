#!/usr/bin/env python3
"""
Backfill session titles and summaries for existing sessions that lack them.

Leverages the existing backfill_session_metadata function from
agent.session_summarizer, processing sessions in batches.

Usage:
    python3 scripts/backfill_session_metadata.py [--limit N] [--batch-size N] [--force] [--dry-run]

Flags:
    --limit N        Maximum total sessions to process (default: 0 = all)
    --batch-size N   Sessions per batch (default: 20)
    --force          Re-generate titles/summaries even for sessions that already have them
    --dry-run        Only count and list sessions that need metadata, don't generate anything
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill session titles and summaries for sessions missing them."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum total sessions to process (0 = all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of sessions per batch (default: 20)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-generate metadata even for sessions that already have titles/summaries",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only count sessions needing metadata, don't generate anything",
    )
    args = parser.parse_args()

    # Set up import path so we can find hermes_state and agent modules
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from hermes_state import SessionDB
    from agent.session_summarizer import backfill_session_metadata

    db = SessionDB()

    # Count how many sessions need metadata
    all_missing = db.list_sessions_missing_summary(limit=100000, include_existing=args.force)
    total_missing = len(all_missing)

    if total_missing == 0:
        print("✓ All sessions already have titles and summaries. Nothing to do.")
        return 0

    print(f"Sessions needing metadata: {total_missing}")

    if args.dry_run:
        print("\n--dry-run: listing sessions missing metadata:\n")
        for row in all_missing[:50]:
            has_title = bool((row.get("title") or "").strip())
            has_summary = bool((row.get("summary") or "").strip())
            flags = []
            if not has_title:
                flags.append("no-title")
            if not has_summary:
                flags.append("no-summary")
            print(f"  {row['id'][:12]}…  msgs={row.get('message_count', 0):>4}  [{', '.join(flags)}]")
        if total_missing > 50:
            print(f"  ... and {total_missing - 50} more")
        return 0

    # Process in batches
    limit = args.limit if args.limit > 0 else total_missing
    batch_size = args.batch_size
    total_processed = 0
    total_updated = 0
    total_failed = 0
    batch_num = 0

    while total_processed < limit:
        remaining = limit - total_processed
        batch = min(batch_size, remaining)
        batch_num += 1

        print(f"\n--- Batch {batch_num} (up to {batch} sessions) ---")
        result = backfill_session_metadata(db, limit=batch, force=args.force)

        processed = result["processed"]
        updated = result["updated"]
        failed = result["failed"]

        total_processed += processed
        total_updated += updated
        total_failed += failed

        print(
            f"  Batch result: processed={processed}, updated={updated}, failed={failed}"
        )
        print(
            f"  Running total: processed={total_processed}, updated={total_updated}, failed={total_failed}"
        )

        if processed == 0:
            print("  No more sessions to process.")
            break

        # Small pause between batches to avoid rate-limits
        if total_processed < limit and processed > 0:
            time.sleep(1)

    print(f"\n{'='*50}")
    print(f"Backfill complete.")
    print(f"  Total processed: {total_processed}")
    print(f"  Total updated:   {total_updated}")
    print(f"  Total failed:    {total_failed}")

    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
