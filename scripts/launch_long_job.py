#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch one detached, logged EventTune job")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--source-steps", type=int, default=100)
    parser.add_argument("--source-gradient-accumulation", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=448)
    parser.add_argument("--d4-views", type=int, default=4)
    parser.add_argument("--fixed-event-steps", type=int)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    split_dir = Path(args.split_dir).resolve()
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    git_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    command = [
        "bash",
        str(project / "scripts" / "run_event.sh"),
        str(split_dir),
        str(run_dir),
        str(args.source_steps),
    ]
    if args.fixed_event_steps is not None:
        command.append(str(args.fixed_event_steps))

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "HF_HUB_OFFLINE": "1",
            "PYTHON_BIN": str(project / ".venv" / "bin" / "python"),
            "SOURCE_GRADIENT_ACCUMULATION": str(args.source_gradient_accumulation),
            "CROP_SIZE": str(args.crop_size),
            "EVAL_D4_VIEWS": str(args.d4_views),
        }
    )
    log_path = run_dir / "launcher.log"
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    process = subprocess.Popen(
        command,
        cwd=project,
        env=environment,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    metadata = {
        "pid": process.pid,
        "gpu": args.gpu,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision,
        "command": command,
        "source_steps": args.source_steps,
        "source_gradient_accumulation": args.source_gradient_accumulation,
        "crop_size": args.crop_size,
        "d4_views": args.d4_views,
        "log": str(log_path),
    }
    (run_dir / "launcher.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
