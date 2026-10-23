#!/usr/bin/env python3
"""Real-model acceptance check in an isolated, silent service.

Requires built debug/release binaries, an installed preset model config and
an available audio device. Reads model assets without changing the live app.
"""
import argparse
import json
import socket
import subprocess
import tempfile
import time
import urllib.request
import wave
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, default=Path("target/debug"))
    args = parser.parse_args()
    config = args.config.resolve()
    binary = (args.bin_dir / "sayit-service").resolve()
    with tempfile.TemporaryDirectory(prefix="sayit-stream-test-") as directory:
        root = Path(directory)
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        with (root / "service.log").open("w+") as log:
            process = subprocess.Popen([str(binary), "--port", str(port), "--data-dir", str(root), "--config", str(config)],
                                       stdout=log, stderr=subprocess.STDOUT)
            try:
                base = f"http://127.0.0.1:{port}/v1/"
                deadline = time.monotonic() + 15
                while not (root / "api-token").exists():
                    assert process.poll() is None, "service exited during startup"
                    assert time.monotonic() < deadline, "service startup timed out"
                    time.sleep(0.05)
                token = (root / "api-token").read_text().strip()

                def api(resource, body=None):
                    request = urllib.request.Request(base + resource,
                        data=None if body is None else json.dumps(body).encode(),
                        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=15) as response:
                        data = response.read()
                        return json.loads(data) if data else None

                def until(check, timeout=45, interval=0.15):
                    deadline = time.monotonic() + timeout
                    while time.monotonic() < deadline:
                        value = check()
                        if value:
                            return value
                        assert process.poll() is None, "service exited"
                        time.sleep(interval)
                    raise AssertionError("acceptance check timed out")

                until(lambda: api("health"))
                api("settings", {"volume": 0, "compute_device": "cpu", "playback_rate": 1.25})
                text = "The quiet morning brings a clear new sentence. Streaming speech should begin before the final sentence is generated. " * 12
                job = api("jobs", {"text": text, "queue_policy": "append"})
                started = time.monotonic()
                observed = []
                def generated():
                    playback = api("state")["playback"]
                    observed.append(playback)
                    return playback if playback["job_id"] == job["id"] and playback["generation_complete"] else None
                complete = until(generated, timeout=90, interval=0.15)
                generation_seconds = time.monotonic() - started
                assert next(j for j in api("jobs") if j["id"] == job["id"])["state"] == "playing"
                progressive = [p for p in observed if p["job_id"] == job["id"] and p["duration_seconds"] > 0 and not p["generation_complete"]]
                assert progressive, "no audio delivered before generation completed"
                assert any(p["position_seconds"] > 0 for p in progressive), "device never played during synthesis"
                first_audio = next(i for i, p in enumerate(observed) if p in progressive) * 0.15
                api("playback/pause", {})
                paused = until(lambda: (p if (p := api("state")["playback"])["state"] == "paused" else None))
                time.sleep(0.25)
                assert abs(api("state")["playback"]["position_seconds"] - paused["position_seconds"]) < 0.03
                api("playback/rate", {"rate": 1.5})
                api("playback/seek", {"seconds": complete["duration_seconds"] - 0.75})
                api("playback/resume", {})
                history = until(lambda: next((h for h in api("history") if h["job"]["id"] == job["id"]), None))
                assert history["job"]["state"] == "completed", history["job"].get("error")
                with wave.open(history["audio_path"]) as audio:
                    assert audio.getnframes() > 0
                    assert abs(audio.getnframes() / audio.getframerate() - history["duration_seconds"]) < 0.01
                # A second long request must stop immediately and stay stopped.
                cancelled = api("jobs", {"text": text * 2, "queue_policy": "append"})
                until(lambda: api("state")["playback"]["job_id"] == cancelled["id"])
                api("playback/stop", {})
                until(lambda: next(j for j in api("jobs") if j["id"] == cancelled["id"])["state"] == "cancelled")
                time.sleep(0.5)
                assert api("state")["playback"]["job_id"] != cancelled["id"], "stopped stream restarted"
                assert not (root / "audio" / (cancelled["id"] + ".wav")).exists()
                replaced = api("jobs", {"text": text * 2, "queue_policy": "append"})
                until(lambda: api("state")["playback"]["job_id"] == replaced["id"])
                api("history/" + job["id"] + "/replay", {})
                until(lambda: next(j for j in api("jobs") if j["id"] == replaced["id"])["state"] == "cancelled")
                replay = until(lambda: (p if (p := api("state")["playback"])["job_id"] == job["id"] else None))
                api("playback/seek", {"seconds": replay["duration_seconds"] - 0.5})
                until(lambda: api("state")["playback"]["job_id"] is None)
                print(json.dumps({"result": "passed", "progressive_snapshots": len(progressive),
                    "first_audio_poll_seconds": round(first_audio, 2), "generation_seconds": round(generation_seconds, 2),
                    "acceptance_seconds": round(time.monotonic() - started, 2),
                    "audio_seconds": round(history["duration_seconds"], 2),
                    "checks": ["real model", "audio before completion", "device playback", "pause", "seek", "rate change", "archive", "stop during synthesis", "replay replacing synthesis"]}))
            except Exception:
                log.flush()
                log.seek(0)
                print(log.read()[-6000:])
                raise
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


if __name__ == "__main__":
    main()
