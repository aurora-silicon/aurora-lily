#!/usr/bin/env python3
"""Unit tests for tools/bench/agent_bench.py (no Docker/Podman/GPU/network required)."""

from __future__ import annotations

import http.server
import json
import socketserver
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

import agent_bench


class _MockServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True


class _MockLilyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002
        return

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "Qwen3.8-Flash-Next"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length).decode("utf-8"))
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            chunks = [
                b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n',
                (
                    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}],'
                    b'"timings":{"prompt_tokens":1000,"cached_tokens":800,"prefill_tokens":200,'
                    b'"prefill_ms":100.0,"generated_tokens":50,"decode_ms":500.0,'
                    b'"drafted_tokens":40,"accepted_tokens":32}}\n\n'
                ),
                b"data: [DONE]\n\n",
            ]
            for c in chunks:
                self.wfile.write(c)
                self.wfile.flush()
            self.close_connection = True
            return

        body = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "timings": {
                "prompt_tokens": 500,
                "cached_tokens": 250,
                "prefill_tokens": 250,
                "prefill_ms": 250.0,
                "generated_tokens": 100,
                "decode_ms": 1000.0,
                "drafted_tokens": 80,
                "accepted_tokens": 60,
            },
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class AgentBenchTests(unittest.TestCase):
    def test_proxy_captures_json_and_sse_timings(self):
        upstream = _MockServer(("127.0.0.1", 0), _MockLilyHandler)
        up_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
        up_thread.start()
        up_port = upstream.server_address[1]
        up_url = f"http://127.0.0.1:{up_port}"

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "timings.jsonl"
            proxy = agent_bench.TimingsProxy(up_url, log_path, suite_getter=lambda: "swe-verified-mini")
            proxy.start()
            try:
                self.assertEqual(agent_bench.discover_model_id(up_url), "Qwen3.8-Flash-Next")

                # Non-streaming request
                req = urllib.request.Request(
                    f"{proxy.base_url}/chat/completions",
                    data=json.dumps({"model": "Qwen3.8-Flash-Next", "messages": []}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["choices"][0]["message"]["content"], "ok")

                # Streaming SSE request
                req_sse = urllib.request.Request(
                    f"{proxy.base_url}/chat/completions",
                    data=json.dumps({"model": "Qwen3.8-Flash-Next", "stream": True, "messages": []}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req_sse, timeout=5) as resp:
                    raw_sse = resp.read()
                self.assertIn(b"data: [DONE]", raw_sse)
            finally:
                proxy.stop()
                upstream.shutdown()
                upstream.server_close()

            rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["engine"], "lily")
            self.assertEqual(rows[0]["generated_tokens"], 100)
            self.assertEqual(rows[0]["accepted_tokens"], 60)
            self.assertEqual(rows[1]["generated_tokens"], 50)
            self.assertEqual(rows[1]["cached_tokens"], 800)
            self.assertEqual(rows[1]["suite"], "swe-verified-mini")

    def test_build_harbor_command_with_task_list_and_limit(self):
        suite = agent_bench.PRESETS["swe-mini"][0]
        cmd = agent_bench.build_harbor_command(
            harbor_cmd=["harbor"],
            suite=suite,
            env="podman",
            agent="terminus-2",
            model_id="Qwen3.8-Flash-Next",
            api_base="http://127.0.0.1:12345/v1",
            jobs_dir=Path("/tmp/jobs"),
            job_name="swe-verified-mini",
            n_concurrent=2,
            n_attempts=1,
            limit=2,
            task_names=["django__django-1", "django__django-2", "django__django-3"],
            timeout_multiplier=0.5,
            temperature=0.0,
            max_tokens=4096,
            extra_agent_kwargs=[],
        )
        self.assertIn("--env", cmd)
        self.assertEqual(cmd[cmd.index("--env") + 1], "podman")
        self.assertEqual(cmd[cmd.index("--model") + 1], "openai/Qwen3.8-Flash-Next")
        self.assertEqual(cmd.count("--task-name"), 2)
        self.assertNotIn("-l", cmd)

    def test_report_aggregates_trials_and_timings(self):
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td) / "run1"
            (run_dir / "jobs" / "swe-verified-mini" / "trial_1").mkdir(parents=True)
            (run_dir / "jobs" / "swe-verified-mini" / "trial_2").mkdir(parents=True)
            (run_dir / "meta.json").write_text(
                json.dumps({"label": "qwen38-q4xl", "suites": ["swe-verified-mini"], "wall_s": 3660}),
                encoding="utf-8",
            )
            (run_dir / "jobs" / "swe-verified-mini" / "trial_1" / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": 1.0}}}),
                encoding="utf-8",
            )
            (run_dir / "jobs" / "swe-verified-mini" / "trial_2" / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": 0.0}}}),
                encoding="utf-8",
            )
            (run_dir / "timings.jsonl").write_text(
                "\n".join([
                    json.dumps({
                        "suite": "swe-verified-mini",
                        "generated_tokens": 100,
                        "decode_ms": 1000.0,
                        "prefill_tokens": 200,
                        "prefill_ms": 100.0,
                        "cached_tokens": 800,
                        "drafted_tokens": 100,
                        "accepted_tokens": 80,
                    }),
                    json.dumps({
                        "suite": "swe-verified-mini",
                        "generated_tokens": 100,
                        "decode_ms": 1000.0,
                        "prefill_tokens": 200,
                        "prefill_ms": 100.0,
                        "cached_tokens": 800,
                        "drafted_tokens": 100,
                        "accepted_tokens": 80,
                    }),
                ]) + "\n",
                encoding="utf-8",
            )
            md = agent_bench.format_report([run_dir])
            self.assertIn("qwen38-q4xl", md)
            self.assertIn("1/2 (50.0%)", md)
            self.assertIn("100.0", md)  # decode tok/s
            self.assertIn("2000", md)   # prefill tok/s
            self.assertIn("80%", md)    # KV hit & draft acc
            self.assertIn("1h 01m", md)


if __name__ == "__main__":
    unittest.main()
