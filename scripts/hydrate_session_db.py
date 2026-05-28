#!/usr/bin/env python3

import json
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    from hermes_constants import get_hermes_home
    from hermes_state import SessionDB

    sessions_dir = get_hermes_home() / "sessions"
    files = sorted(sessions_dir.rglob("*.json"))
    db = SessionDB()

    imported = 0
    skipped = 0
    failed = 0

    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            result = db.import_session_dump(data)
            if result:
                if (
                    db.get_session(result)
                    and path.stem.replace("session_", "") != result
                ):
                    imported += 1
                else:
                    imported += 1
            else:
                skipped += 1
        except Exception as exc:
            failed += 1
            print(f"FAILED {path}: {exc}")

    total_sessions = db._conn.execute("select count(*) from sessions").fetchone()[0]
    total_messages = db._conn.execute("select count(*) from messages").fetchone()[0]

    print(
        json.dumps(
            {
                "files": len(files),
                "imported": imported,
                "skipped": skipped,
                "failed": failed,
                "db_sessions": total_sessions,
                "db_messages": total_messages,
            },
            indent=2,
        )
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
