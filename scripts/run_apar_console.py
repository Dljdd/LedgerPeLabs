#!/usr/bin/env python3
"""Build, verify, and serve the offline APAR competition console."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from socketserver import BaseServer
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from apar.demo.sentinel_v5_portable import run_portable_scenarios  # noqa: E402


def _canonical(document: object) -> bytes:
    return json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_embedded_hash(document: dict[str, Any], *, field: str, label: str) -> str:
    expected = document.get(field)
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} embedded hash is missing")
    payload = dict(document)
    payload.pop(field)
    actual = hashlib.sha256(_canonical(payload)).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} hash differs")
    return expected


def verify_console_assets(root: Path, *, trace_path: Path | None = None) -> dict[str, object]:
    root = root.resolve()
    evidence = _load_object(
        root / "web/public/data/console-evidence.json", label="console evidence"
    )
    fixed_trace = _load_object(
        trace_path or root / "web/public/data/verified-trace.json",
        label="fallback trace",
    )
    evidence_hash = _verify_embedded_hash(
        evidence, field="document_sha256", label="console evidence"
    )
    trace_hash = _verify_embedded_hash(
        fixed_trace, field="trace_sha256", label="fallback trace"
    )
    portable = evidence.get("portable")
    recovered = evidence.get("recovered")
    if not isinstance(portable, dict) or portable.get("arm") != "ensemble_with_graph":
        raise ValueError("console portable arm differs")
    if portable.get("authoritative") is not False or portable.get(
        "accepted_capacity_evidence"
    ) is not False:
        raise ValueError("console portable boundary differs")
    if not isinstance(recovered, dict) or recovered.get("qualifier") != (
        "Recovered diagnostic evidence — non-authoritative"
    ):
        raise ValueError("recovered evidence qualifier differs")
    if recovered.get("authoritative") is not False or recovered.get(
        "accepted_capacity_evidence"
    ) is not False:
        raise ValueError("recovered evidence boundary differs")
    if fixed_trace.get("replay_verified") is not True:
        raise ValueError("fallback trace is not replay verified")
    if fixed_trace.get("bundle_manifest_sha256") != portable.get(
        "bundle_manifest_sha256"
    ):
        raise ValueError("fallback trace bundle binding differs")
    traces = fixed_trace.get("traces")
    records = portable.get("records")
    if not isinstance(traces, list) or not isinstance(records, list) or len(traces) != len(records):
        raise ValueError("fallback trace event count differs")
    for index, (trace, record) in enumerate(zip(traces, records, strict=True), start=1):
        if not isinstance(trace, dict) or not isinstance(record, dict):
            raise ValueError(f"fallback trace record {index} is invalid")
        accepted = record.get("accepted_checkpoint_evidence")
        if not isinstance(accepted, dict) or (
            trace.get("event_id") != record.get("event_id")
            or trace.get("arm") != "ensemble_with_graph"
            or trace.get("calibrated_probability") != accepted.get("probability")
            or trace.get("final_action") != accepted.get("action")
        ):
            raise ValueError(f"fallback trace checkpoint evidence differs at event {index}")
    return {
        "status": "ready",
        "portable_arm": portable["arm"],
        "event_count": len(traces),
        "fallback_replay_verified": True,
        "fallback_trace_sha256": trace_hash,
        "console_evidence_sha256": evidence_hash,
        "recovered_authoritative": False,
        "accepted_capacity_evidence": False,
    }


def score_live_trace(root: Path, output_path: Path) -> dict[str, Any]:
    root = root.resolve()
    return run_portable_scenarios(
        bundle_root=root / "demo/sentinel-v5",
        scenario_path=root / "demo/sentinel-v5/scenarios.json",
        output_path=output_path.resolve(),
    )


def _node_major() -> int | None:
    node = shutil.which("node")
    if node is None:
        return None
    completed = subprocess.run(
        [node, "--version"], capture_output=True, check=False, text=True
    )
    if completed.returncode != 0:
        return None
    try:
        return int(completed.stdout.strip().lstrip("v").split(".", maxsplit=1)[0])
    except ValueError:
        return None


def _build_client(root: Path) -> None:
    major = _node_major()
    if major is None or major < 20:
        raise RuntimeError("Node.js 20.19 or newer is required to build the console")
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to build the console")
    completed = subprocess.run([npm, "run", "build"], cwd=root / "web", check=False)
    if completed.returncode != 0:
        raise RuntimeError("frontend build failed")


class ConsoleHandler(SimpleHTTPRequestHandler):
    server_version = "APARConsole/1"

    def __init__(
        self,
        request: socket.socket,
        client_address: tuple[str, int],
        server: BaseServer,
        *,
        directory: str,
        root: Path,
    ) -> None:
        self.root = root
        super().__init__(request, client_address, server, directory=directory)

    def _send_json(self, status: HTTPStatus, payload: object) -> None:
        body = _canonical(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self) -> None:
        policy = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        )
        self.send_header("Content-Security-Policy", policy)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/health":
            try:
                self._send_json(HTTPStatus.OK, verify_console_assets(self.root))
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    {"status": "degraded", "detail": str(exc)},
                )
            return
        requested = self.path.split("?", maxsplit=1)[0]
        local_path = Path(self.directory or "") / requested.lstrip("/")
        if requested != "/" and not local_path.is_file() and "." not in Path(requested).name:
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/score":
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "not_found"})
            return
        try:
            runtime = self.root / ".apar/console"
            runtime.mkdir(parents=True, exist_ok=True)
            report = score_live_trace(self.root, runtime / "live-trace.json")
            self._send_json(HTTPStatus.OK, report)
        except (OSError, RuntimeError, ValueError) as exc:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "scorer_unavailable", "fallback_available": True, "detail": str(exc)},
            )

    def log_message(self, message: str, *args: object) -> None:
        print(f"[apar-console] {message % args}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    start = subparsers.add_parser("start", help="verify, build, and serve the console")
    start.add_argument("--host", default="127.0.0.1")
    start.add_argument("--port", default=4173, type=int)
    start.add_argument("--skip-build", action="store_true", help=argparse.SUPPRESS)
    subparsers.add_parser("health", help="verify committed console assets")
    subparsers.add_parser("reset", help="remove only the generated live trace")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "health":
        print(json.dumps(verify_console_assets(root), indent=2, sort_keys=True))
        return 0
    if args.command == "reset":
        live_trace = root / ".apar/console/live-trace.json"
        if live_trace.is_file():
            live_trace.unlink()
        print("APAR console reset to the canonical committed trace.")
        return 0
    verify_console_assets(root)
    if not args.skip_build:
        _build_client(root)
    dist = root / "web/dist"
    if not (dist / "index.html").is_file():
        raise RuntimeError("web/dist is missing; run without --skip-build")
    class BoundConsoleHandler(ConsoleHandler):
        def __init__(
            self,
            request: socket.socket,
            client_address: tuple[str, int],
            server: BaseServer,
        ) -> None:
            super().__init__(
                request,
                client_address,
                server,
                directory=str(dist),
                root=root,
            )

    server = ThreadingHTTPServer((args.host, args.port), BoundConsoleHandler)
    print(f"APAR console ready at http://{args.host}:{args.port}/overview")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAPAR console stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
