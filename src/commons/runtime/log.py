"""Episode trace: one append-only JSONL file per episode.

Records, in order: one ``meta`` record; per seat decision, one ``decision``
record; per provider call, one ``call`` record (prompt digest, model settings,
attempt number, served provider and model, usage, and the raw response body
verbatim); per turn, one ``turn`` record with the applied actions and
rewards; and one ``result`` (or ``abandoned``) record. The final state in the
result record replays every intermediate state with ``replay_states``; call
records preserve what the model actually sampled.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class EpisodeLogger:
    def __init__(self, path: str | Path, meta: dict) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        self.write({"kind": "meta", **meta})

    def write(self, record: dict) -> None:
        record.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def call(self, *, turn: int, agent: str, digest: str, model: dict,
             attempt: int, purpose: str, result=None,
             action: str | None = None, error: str | None = None) -> None:
        rec = {"kind": "call", "turn": turn, "agent": agent, "digest": digest,
               "model": model, "attempt": attempt, "purpose": purpose,
               "action": action}
        if result is not None:
            rec.update(provider=result.provider, served_model=result.model,
                       usage=result.usage, cost_usd=result.cost_usd,
                       latency_s=result.latency_s,
                       transport_attempts=result.attempts,
                       finish_reason=result.finish_reason(), raw=result.raw)
        if error:
            rec["error"] = error
        self.write(rec)

    def result(self, r: dict) -> None:
        self.write({"kind": "result", **r})

    def abandon(self, reason: str) -> None:
        self.write({"kind": "abandoned", "reason": reason})

    def close(self) -> None:
        self._fh.close()


def read_records(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out
