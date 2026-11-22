#!/usr/bin/env python3
"""Agentic coding benchmark runner for lily (and other OpenAI-compatible servers).

Wraps Harbor (`harborframework.com`) to run SWE-bench Verified and
Terminal-Bench 2.0 presets against a local or LAN lily server while teeing
lily's per-request `timings` (decode tok/s, prefill tok/s, KV cache hit rate,
speculative draft acceptance) through a tiny stdlib HTTP passthrough proxy.

No Docker daemon is required: `--env podman` runs daemonless/rootless on
Linux, `--env apple-container` runs natively on macOS, and `--env modal` or
`--env daytona` offloads task sandboxes to cloud x86_64 containers while the
agent loop (`terminus-2`) still runs locally and calls your lily server.

Examples:
    # 1. Fetch the 50-task SWE-bench Verified Mini task list (once):
    python3 tools/bench/agent_bench.py fetch-swe-mini

    # 2. Dry-run to inspect the generated Harbor command(s):
    python3 tools/bench/agent_bench.py run --preset fast --label qwen38-q4xl --dry-run

    # 3. Run the <2h SWE-bench Verified subset with Podman (no Docker):
    python3 tools/bench/agent_bench.py run --preset swe-mini --env podman \\
        --url http://127.0.0.1:8000 --label qwen38-q4xl \\
        --out docs/bench/agent-qwen38-q4xl

    # 4. Run overnight Option A + full Terminal-Bench 2.0 (89 tasks):
    python3 tools/bench/agent_bench.py run --preset full --env podman \\
        --url http://127.0.0.1:8000 --label qwen38-q4xl \\
        --out docs/bench/agent-qwen38-q4xl-full

    # 5. Compare one or more completed runs:
    python3 tools/bench/agent_bench.py report docs/bench/agent-*
"""

from __future__ import annotations

import argparse
import http.client
import http.server
import json
import os
import platform
import shutil
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_SWE_MINI_LIST = REPO / "docs" / "bench" / "swe-bench-verified-mini.txt"

SWE_DATASET = "swe-bench/swe-bench-verified"
TB2_DATASET = "terminal-bench@2.0"

HF_SWE_MINI_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows"
    "?dataset=MariusHobbhahn%2Fswe-bench-verified-mini&config=default&split=test&offset=0&length=100"
)


@dataclass(frozen=True)
class SuiteSpec:
    name: str
    dataset: str
    default_limit: int | None
    default_timeout_multiplier: float | None = None
    use_swe_mini_list: bool = False


PRESETS: dict[str, list[SuiteSpec]] = {
    # Option A: ~1-2h on M5 Max (35 tasks from SWE-bench Verified, or the 50-task mini list).
    "swe-mini": [
        SuiteSpec(
            name="swe-verified-mini",
            dataset=SWE_DATASET,
            default_limit=35,
            use_swe_mini_list=True,
        ),
    ],
    # ~1-2h on M5 Max: 25 tasks from Terminal-Bench 2.0.
    "tb2-mini": [
        SuiteSpec(
            name="tb2-mini",
            dataset=TB2_DATASET,
            default_limit=25,
            default_timeout_multiplier=0.5,
        ),
    ],
    # ~2-5.5h on M5 Max: full 89-task Terminal-Bench 2.0.
    "tb2-full": [
        SuiteSpec(
            name="tb2-full",
            dataset=TB2_DATASET,
            default_limit=None,
        ),
    ],
    # ~1.5-2.5h hybrid: 25 SWE-bench Verified + 15 Terminal-Bench 2.0.
    "fast": [
        SuiteSpec(
            name="swe-verified-mini",
            dataset=SWE_DATASET,
            default_limit=25,
            use_swe_mini_list=True,
        ),
        SuiteSpec(
            name="tb2-mini",
            dataset=TB2_DATASET,
            default_limit=15,
            default_timeout_multiplier=0.5,
        ),
    ],
    # Overnight (~3-7h): Option A (50-task SWE-bench Verified Mini) + full TB 2.0 (89 tasks).
    "full": [
        SuiteSpec(
            name="swe-verified-mini",
            dataset=SWE_DATASET,
            default_limit=50,
            use_swe_mini_list=True,
        ),
        SuiteSpec(
            name="tb2-full",
            dataset=TB2_DATASET,
            default_limit=None,
        ),
    ],
}


# --- Timings normalisation (matches tools/bench/http_bench.py) ---------------


def normalise_timings(payload: dict[str, Any], wall_s: float) -> dict[str, Any] | None:
    """Extract lily or llama.cpp timings + OpenAI usage from a response object."""
    t = payload.get("timings")
    u = payload.get("usage")
    if not isinstance(t, dict) and not isinstance(u, dict):
        return None
    t = t if isinstance(t, dict) else {}
    u = u if isinstance(u, dict) else {}

    # Detect llama.cpp vs lily field names.
    if "predicted_n" in t or "prompt_n" in t:
        rec = {
            "engine": "llama",
            "prompt_tokens": u.get("prompt_tokens") or t.get("prompt_n"),
            "cached_tokens": t.get("cache_n"),
            "prefill_tokens": t.get("prompt_n"),
            "prefill_ms": t.get("prompt_ms"),
            "generated_tokens": t.get("predicted_n"),
            "decode_ms": t.get("predicted_ms"),
            "drafted_tokens": t.get("draft_n"),
            "accepted_tokens": t.get("draft_n_accepted"),
        }
    elif "generated_tokens" in t or "prefill_tokens" in t or "decode_ms" in t:
        rec = {
            "engine": "lily",
            "prompt_tokens": t.get("prompt_tokens") or u.get("prompt_tokens"),
            "cached_tokens": t.get("cached_tokens"),
            "prefill_tokens": t.get("prefill_tokens"),
            "prefill_ms": t.get("prefill_ms"),
            "generated_tokens": t.get("generated_tokens") or u.get("completion_tokens"),
            "decode_ms": t.get("decode_ms"),
            "drafted_tokens": t.get("drafted_tokens"),
            "accepted_tokens": t.get("accepted_tokens"),
        }
    else:
        # Generic OpenAI server without engine timings; still capture token usage.
        rec = {
            "engine": "openai",
            "prompt_tokens": u.get("prompt_tokens"),
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "prefill_tokens": None,
            "prefill_ms": None,
            "generated_tokens": u.get("completion_tokens"),
            "decode_ms": None,
            "drafted_tokens": None,
            "accepted_tokens": None,
        }

    rec["wall_s"] = round(wall_s, 4)
    rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    return rec


def extract_sse_timings(raw_body: bytes, wall_s: float) -> dict[str, Any] | None:
    """Scan an SSE response body from the end for the final chunk carrying timings/usage."""
    lines = raw_body.splitlines()
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line.startswith(b"data:"):
            continue
        data = line[5:].strip()
        if not data or data == b"[DONE]":
            continue
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (obj.get("timings") or obj.get("usage")):
            return normalise_timings(obj, wall_s)
    return None


# --- Lightweight HTTP passthrough proxy for capturing lily timings -----------


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class TimingsProxy:
    """Forwards OpenAI HTTP requests to lily while logging timings to JSONL."""

    def __init__(
        self,
        upstream_url: str,
        log_path: Path,
        bind_host: str = "127.0.0.1",
        bind_port: int = 0,
        suite_getter=None,
    ) -> None:
        parsed = urllib.parse.urlparse(upstream_url.rstrip("/"))
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"Unsupported upstream URL: {upstream_url!r}")
        self.scheme = parsed.scheme
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.upstream_prefix = parsed.path.rstrip("/")
        self.log_path = log_path
        self.suite_getter = suite_getter or (lambda: "unknown")
        self._lock = threading.Lock()
        self._server = _ThreadingHTTPServer((bind_host, bind_port), self._make_handler())
        self.host, self.port = self._server.server_address[:2]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _record(self, rec: dict[str, Any]) -> None:
        rec["suite"] = self.suite_getter()
        line = json.dumps(rec) + "\n"
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line)

    def _make_handler(self):
        proxy = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return  # Keep terminal output clean during Harbor runs.

            def _forward(self, method: str) -> None:
                t0 = time.monotonic()
                content_len = int(self.headers.get("Content-Length", "0") or "0")
                req_body = self.rfile.read(content_len) if content_len > 0 else None

                fwd_headers = {}
                for k, v in self.headers.items():
                    kl = k.lower()
                    if kl in ("host", "connection", "proxy-connection", "keep-alive", "transfer-encoding"):
                        continue
                    fwd_headers[k] = v

                conn_cls = (
                    http.client.HTTPSConnection
                    if proxy.scheme == "https"
                    else http.client.HTTPConnection
                )
                conn = conn_cls(proxy.upstream_host, proxy.upstream_port, timeout=3600)
                try:
                    target_path = proxy.upstream_prefix + self.path
                    conn.request(method, target_path, body=req_body, headers=fwd_headers)
                    resp = conn.getresponse()
                    is_sse = "text/event-stream" in (resp.getheader("Content-Type") or "").lower()

                    self.send_response(resp.status, resp.reason)
                    resp_headers = []
                    for k, v in resp.getheaders():
                        kl = k.lower()
                        if kl in ("connection", "keep-alive", "proxy-connection", "transfer-encoding"):
                            continue
                        if is_sse and kl == "content-length":
                            continue
                        resp_headers.append((k, v))

                    if is_sse:
                        self.send_header("Connection", "close")
                        for k, v in resp_headers:
                            self.send_header(k, v)
                        self.end_headers()
                        buf = bytearray()
                        while True:
                            chunk = resp.read(4096)
                            if not chunk:
                                break
                            buf.extend(chunk)
                            self.wfile.write(chunk)
                            self.wfile.flush()
                        self.close_connection = True
                        wall_s = time.monotonic() - t0
                        if resp.status == 200 and "/chat/completions" in self.path:
                            rec = extract_sse_timings(bytes(buf), wall_s)
                            if rec:
                                proxy._record(rec)
                    else:
                        body = resp.read()
                        for k, v in resp_headers:
                            if k.lower() == "content-length":
                                continue
                            self.send_header(k, v)
                        self.send_header("Content-Length", str(len(body)))
                        self.end_headers()
                        if body:
                            self.wfile.write(body)
                            self.wfile.flush()
                        wall_s = time.monotonic() - t0
                        if resp.status == 200 and "/chat/completions" in self.path:
                            try:
                                obj = json.loads(body)
                            except json.JSONDecodeError:
                                obj = None
                            if isinstance(obj, dict):
                                rec = normalise_timings(obj, wall_s)
                                if rec:
                                    proxy._record(rec)
                except Exception as exc:
                    err = json.dumps({"error": {"message": f"lily proxy error: {exc}"}}).encode()
                    self.send_response(502)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(err)))
                    self.end_headers()
                    self.wfile.write(err)
                finally:
                    conn.close()

            def do_GET(self) -> None:
                self._forward("GET")

            def do_POST(self) -> None:
                self._forward("POST")

            def do_OPTIONS(self) -> None:
                self._forward("OPTIONS")

        return Handler


# --- Harbor & environment helpers --------------------------------------------


def detect_env(requested: str) -> str:
    """Resolve `--env auto` preferring daemonless runtimes over Docker."""
    if requested != "auto":
        return requested
    if shutil.which("podman"):
        return "podman"
    if platform.system() == "Darwin" and shutil.which("container"):
        return "apple-container"
    if shutil.which("docker"):
        return "docker"
    # Fall back to podman in dry-run / instructions so we never push Docker by default.
    return "podman"


def detect_harbor_cmd() -> list[str]:
    if shutil.which("harbor"):
        return ["harbor"]
    if shutil.which("uvx"):
        return ["uvx", "--from", "harbor", "harbor"]
    return ["harbor"]


def discover_model_id(base_url: str, timeout: float = 5.0) -> str | None:
    """Query `<base_url>/v1/models` to auto-detect lily's served model name."""
    url = base_url.rstrip("/")
    if not url.endswith("/v1"):
        url += "/v1"
    req = urllib.request.Request(f"{url}/models", headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("data") or []
        if models and isinstance(models[0], dict) and models[0].get("id"):
            return str(models[0]["id"])
    except Exception:
        return None
    return None


def load_task_list(path: Path) -> list[str]:
    tasks: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    return tasks


def build_harbor_command(
    *,
    harbor_cmd: list[str],
    suite: SuiteSpec,
    env: str,
    agent: str,
    model_id: str,
    api_base: str,
    jobs_dir: Path,
    job_name: str,
    n_concurrent: int,
    n_attempts: int,
    limit: int | None,
    task_names: list[str] | None,
    timeout_multiplier: float | None,
    temperature: float | None,
    max_tokens: int | None,
    extra_agent_kwargs: list[str],
) -> list[str]:
    model_arg = model_id if "/" in model_id else f"openai/{model_id}"
    cmd = [
        *harbor_cmd,
        "run",
        "--dataset",
        suite.dataset,
        "--agent",
        agent,
        "--model",
        model_arg,
        "--env",
        env,
        "--n-concurrent",
        str(n_concurrent),
        "--n-attempts",
        str(n_attempts),
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        job_name,
        "--agent-kwarg",
        f"api_base={api_base}",
    ]
    if timeout_multiplier is not None:
        cmd.extend(["--timeout-multiplier", str(timeout_multiplier)])
    if temperature is not None:
        cmd.extend(["--agent-kwarg", f"temperature={temperature}"])
    if max_tokens is not None:
        cmd.extend(["--agent-kwarg", f"max_tokens={max_tokens}"])
    for kv in extra_agent_kwargs:
        cmd.extend(["--agent-kwarg", kv])

    if task_names:
        selected = task_names[:limit] if limit is not None else task_names
        for name in selected:
            cmd.extend(["--task-name", name])
    elif limit is not None:
        cmd.extend(["-l", str(limit)])

    return cmd


# --- Subcommands -------------------------------------------------------------


def cmd_fetch_swe_mini(args: argparse.Namespace) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(HF_SWE_MINI_ROWS_URL, headers={"User-Agent": "aurora-lily-bench/1"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        print(f"error: failed to fetch SWE-bench Verified Mini from Hugging Face: {exc}", file=sys.stderr)
        return 1

    ids = [row["row"]["instance_id"] for row in payload.get("rows", []) if "row" in row and "instance_id" in row["row"]]
    if not ids:
        print("error: no instance_ids returned from Hugging Face API", file=sys.stderr)
        return 1

    header = [
        "# SWE-bench Verified Mini (50 stratified tasks from MariusHobbhahn/swe-bench-verified-mini)",
        f"# Fetched {time.strftime('%Y-%m-%dT%H:%M:%S')}",
    ]
    out_path.write_text("\n".join(header + ids) + "\n", encoding="utf-8")
    print(f"Wrote {len(ids)} task IDs to {out_path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    suites = PRESETS[args.preset]
    env = detect_env(args.env)
    harbor_cmd = detect_harbor_cmd()

    model_id = args.model
    if not model_id and not args.dry_run:
        model_id = discover_model_id(args.url)
    if not model_id:
        model_id = args.label or "lily-model"

    label = args.label or model_id.replace("/", "_")
    out_dir = Path(args.out) if args.out else (REPO / "docs" / "bench" / f"{time.strftime('%Y-%m-%d')}-{label}-{args.preset}")
    jobs_dir = out_dir / "jobs"
    timings_path = out_dir / "timings.jsonl"

    swe_list_path = Path(args.swe_task_list) if args.swe_task_list else DEFAULT_SWE_MINI_LIST
    swe_tasks = load_task_list(swe_list_path) if swe_list_path.is_file() else None
    custom_tasks = load_task_list(Path(args.task_list)) if args.task_list else None

    if args.dry_run:
        api_base = f"http://{args.proxy_bind}:<ephemeral>/v1"
        print(f"# preset={args.preset} env={env} model={model_id} upstream={args.url}")
        if env == "podman" and not shutil.which("podman"):
            print("# note: podman not found on PATH; install podman (and enable podman.socket) or use --env modal / --env daytona")
        for suite in suites:
            limit = args.limit if args.limit is not None else suite.default_limit
            task_names = custom_tasks or (swe_tasks if suite.use_swe_mini_list else None)
            timeout_mult = args.timeout_multiplier if args.timeout_multiplier is not None else suite.default_timeout_multiplier
            cmd = build_harbor_command(
                harbor_cmd=harbor_cmd,
                suite=suite,
                env=env,
                agent=args.agent,
                model_id=model_id,
                api_base=api_base,
                jobs_dir=jobs_dir,
                job_name=suite.name,
                n_concurrent=args.n_concurrent,
                n_attempts=args.n_attempts,
                limit=limit,
                task_names=task_names,
                timeout_multiplier=timeout_mult,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_agent_kwargs=args.agent_kwarg,
            )
            print(" ".join(shlex_quote(c) for c in cmd))
        return 0

    if not shutil.which(harbor_cmd[0]):
        print(
            "error: neither `harbor` nor `uvx` is on PATH. Install Harbor with `uv tool install harbor`.",
            file=sys.stderr,
        )
        return 1
    if env in ("podman", "docker") and not shutil.which(env):
        print(
            f"error: `--env {env}` requested, but `{env}` is not on PATH. "
            "Install podman (recommended, daemonless) or run sandboxes in the cloud with `--env modal` / `--env daytona`.",
            file=sys.stderr,
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    current_suite = {"name": "init"}
    proxy = TimingsProxy(
        upstream_url=args.url,
        log_path=timings_path,
        bind_host=args.proxy_bind,
        suite_getter=lambda: current_suite["name"],
    )
    proxy.start()
    advertised_api_base = args.proxy_advertise or proxy.base_url

    meta = {
        "label": label,
        "preset": args.preset,
        "model": model_id,
        "agent": args.agent,
        "env": env,
        "upstream_url": args.url,
        "n_concurrent": args.n_concurrent,
        "n_attempts": args.n_attempts,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "suites": [s.name for s in suites],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    t_start = time.monotonic()
    rc = 0
    try:
        sub_env = os.environ.copy()
        sub_env["OPENAI_API_BASE"] = advertised_api_base
        sub_env.setdefault("OPENAI_API_KEY", "sk-lily-local")

        for suite in suites:
            current_suite["name"] = suite.name
            limit = args.limit if args.limit is not None else suite.default_limit
            task_names = custom_tasks or (swe_tasks if suite.use_swe_mini_list else None)
            timeout_mult = args.timeout_multiplier if args.timeout_multiplier is not None else suite.default_timeout_multiplier
            cmd = build_harbor_command(
                harbor_cmd=harbor_cmd,
                suite=suite,
                env=env,
                agent=args.agent,
                model_id=model_id,
                api_base=advertised_api_base,
                jobs_dir=jobs_dir,
                job_name=suite.name,
                n_concurrent=args.n_concurrent,
                n_attempts=args.n_attempts,
                limit=limit,
                task_names=task_names,
                timeout_multiplier=timeout_mult,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_agent_kwargs=args.agent_kwarg,
            )
            print(f"\n==> Running suite '{suite.name}' ({suite.dataset}) via {env}...", flush=True)
            proc = subprocess.run(cmd, env=sub_env, check=False)
            if proc.returncode != 0:
                print(f"warning: suite '{suite.name}' exited with code {proc.returncode}", file=sys.stderr)
                rc = proc.returncode
    finally:
        proxy.stop()
        meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        meta["wall_s"] = round(time.monotonic() - t_start, 2)
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print()
    print(format_report([out_dir]))
    return rc


def shlex_quote(s: str) -> str:
    import shlex

    return shlex.quote(s)


# --- Report aggregation ------------------------------------------------------


def _extract_reward(trial_result: dict[str, Any]) -> float | None:
    vr = trial_result.get("verifier_result")
    if isinstance(vr, dict):
        rewards = vr.get("rewards")
        if isinstance(rewards, dict) and rewards:
            for key in ("reward", "resolved", "pass", "score"):
                if key in rewards and isinstance(rewards[key], (int, float)):
                    return float(rewards[key])
            # Otherwise take the first numeric reward.
            for v in rewards.values():
                if isinstance(v, (int, float)):
                    return float(v)
    if isinstance(trial_result.get("reward"), (int, float)):
        return float(trial_result["reward"])
    return None


def _summarize_suite_trials(suite_dir: Path) -> dict[str, Any]:
    trials = 0
    passed = 0
    errors = 0
    for res_file in sorted(suite_dir.rglob("result.json")):
        # Skip job-level result.json directly in suite_dir if trial subdirs exist.
        if res_file.parent == suite_dir:
            continue
        try:
            data = json.loads(res_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        trials += 1
        reward = _extract_reward(data)
        if reward is not None and reward >= 1.0:
            passed += 1
        elif data.get("exception_info"):
            errors += 1
    return {"trials": trials, "passed": passed, "errors": errors}


def _summarize_timings(timings_path: Path, suite: str | None = None) -> dict[str, Any]:
    if not timings_path.is_file():
        return {}
    reqs = 0
    gen_tok = 0
    dec_ms = 0.0
    pre_tok = 0
    pre_ms = 0.0
    cached_tok = 0
    drafted = 0
    accepted = 0
    for line in timings_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if suite and r.get("suite") != suite:
            continue
        reqs += 1
        if r.get("generated_tokens"):
            gen_tok += int(r["generated_tokens"])
        if r.get("decode_ms"):
            dec_ms += float(r["decode_ms"])
        if r.get("prefill_tokens"):
            pre_tok += int(r["prefill_tokens"])
        if r.get("prefill_ms"):
            pre_ms += float(r["prefill_ms"])
        if r.get("cached_tokens"):
            cached_tok += int(r["cached_tokens"])
        if r.get("drafted_tokens"):
            drafted += int(r["drafted_tokens"])
        if r.get("accepted_tokens"):
            accepted += int(r["accepted_tokens"])
    return {
        "requests": reqs,
        "decode_tps": (gen_tok / (dec_ms / 1e3)) if gen_tok and dec_ms > 0 else None,
        "prefill_tps": (pre_tok / (pre_ms / 1e3)) if pre_tok and pre_ms > 0 else None,
        "cache_hit_pct": (100.0 * cached_tok / (cached_tok + pre_tok)) if (cached_tok + pre_tok) > 0 else None,
        "draft_accept_pct": (100.0 * accepted / drafted) if drafted > 0 else None,
        "generated_tokens": gen_tok,
    }


def _fmt_opt(val: float | None, fmt: str) -> str:
    return "–" if val is None else format(val, fmt)


def _fmt_wall(seconds: float | None) -> str:
    if seconds is None:
        return "–"
    mins = int(round(seconds / 60.0))
    if mins < 60:
        return f"{mins}m"
    return f"{mins // 60}h {mins % 60:02d}m"


def format_report(run_dirs: list[Path]) -> str:
    rows: list[list[str]] = []
    for run_dir in run_dirs:
        meta_path = run_dir / "meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        label = meta.get("label") or run_dir.name
        jobs_dir = run_dir / "jobs"
        timings_path = run_dir / "timings.jsonl"
        suite_names = meta.get("suites") or (
            sorted(p.name for p in jobs_dir.iterdir() if p.is_dir()) if jobs_dir.is_dir() else []
        )
        wall_s = meta.get("wall_s")
        for idx, sname in enumerate(suite_names):
            sdir = jobs_dir / sname
            tr = _summarize_suite_trials(sdir) if sdir.is_dir() else {"trials": 0, "passed": 0, "errors": 0}
            tm = _summarize_timings(timings_path, suite=sname)
            pct = (100.0 * tr["passed"] / tr["trials"]) if tr["trials"] else None
            avg_reqs = (tm["requests"] / tr["trials"]) if (tr["trials"] and tm.get("requests")) else None
            rows.append([
                label if idx == 0 else "",
                sname,
                f"{tr['passed']}/{tr['trials']} ({_fmt_opt(pct, '.1f')}%)" if tr["trials"] else "0/0",
                _fmt_opt(avg_reqs, ".1f"),
                _fmt_opt(tm.get("decode_tps"), ".1f"),
                _fmt_opt(tm.get("prefill_tps"), ".0f"),
                f"{_fmt_opt(tm.get('cache_hit_pct'), '.0f')}%" if tm.get("cache_hit_pct") is not None else "–",
                f"{_fmt_opt(tm.get('draft_accept_pct'), '.0f')}%" if tm.get("draft_accept_pct") is not None else "–",
                _fmt_wall(wall_s) if idx == 0 else "",
            ])

    headers = [
        "Model / Label",
        "Suite",
        "Resolved",
        "Turns/task",
        "Decode tok/s",
        "Prefill tok/s",
        "KV hit",
        "Draft acc",
        "Wall",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    dirs = [Path(p) for p in args.runs]
    print(format_report(dirs))
    return 0


# --- CLI ---------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Agentic coding benchmark runner for lily (via Harbor).")
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fetch-swe-mini", help="Download the 50-task SWE-bench Verified Mini ID list.")
    pf.add_argument("--out", default=str(DEFAULT_SWE_MINI_LIST), help="Output text file path.")
    pf.set_defaults(func=cmd_fetch_swe_mini)

    pr = sub.add_parser("run", help="Run a Harbor benchmark preset against lily.")
    pr.add_argument("--preset", choices=sorted(PRESETS), default="fast", help="Benchmark preset to run.")
    pr.add_argument("--url", default="http://127.0.0.1:8000", help="Upstream lily (or OpenAI-compatible) server URL.")
    pr.add_argument("--model", default=None, help="Model ID (auto-detected from /v1/models if omitted).")
    pr.add_argument("--label", default=None, help="Short label for reports and output directory.")
    pr.add_argument("--out", default=None, help="Output directory (default: docs/bench/<date>-<label>-<preset>).")
    pr.add_argument(
        "--env",
        default="auto",
        choices=["auto", "podman", "apple-container", "modal", "daytona", "e2b", "docker"],
        help="Harbor sandbox environment (default: auto, prefers podman over docker).",
    )
    pr.add_argument("--agent", default="terminus-2", help="Harbor agent adapter (default: terminus-2).")
    pr.add_argument("--n-concurrent", type=int, default=2, help="Concurrent task sandboxes (default: 2).")
    pr.add_argument("--n-attempts", type=int, default=1, help="Attempts per task (default: 1).")
    pr.add_argument("--limit", type=int, default=None, help="Override task count limit for each suite.")
    pr.add_argument("--timeout-multiplier", type=float, default=None, help="Override Harbor timeout multiplier.")
    pr.add_argument("--task-list", default=None, help="File of explicit task IDs to run (one per line).")
    pr.add_argument("--swe-task-list", default=None, help="Override SWE-bench Verified Mini task list path.")
    pr.add_argument("--temperature", type=float, default=None, help="Sampling temperature passed to agent.")
    pr.add_argument("--max-tokens", type=int, default=None, help="Max tokens per turn passed to agent.")
    pr.add_argument("--agent-kwarg", action="append", default=[], help="Extra Harbor --agent-kwarg key=value.")
    pr.add_argument("--proxy-bind", default="127.0.0.1", help="Bind address for the local timings proxy.")
    pr.add_argument("--proxy-advertise", default=None, help="Override API base URL handed to Harbor.")
    pr.add_argument("--dry-run", action="store_true", help="Print Harbor commands without running them.")
    pr.set_defaults(func=cmd_run)

    prep = sub.add_parser("report", help="Render a Markdown comparison table from one or more run dirs.")
    prep.add_argument("runs", nargs="+", help="One or more run output directories.")
    prep.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
