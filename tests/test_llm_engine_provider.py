import dataclasses
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core import llm


class _Engine:
    """A stand-in for `sally-inference serve` speaking its SSE dialect."""

    def __init__(self, words=("Hel", "lo"), status=200, error_body=None, slow=False):
        self.requests = []
        self.disconnected = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append((self.path, json.loads(self.rfile.read(length))))
                if status != 200:
                    body = json.dumps(error_body).encode()
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                try:
                    for word in words:
                        chunk = {"choices": [{"index": 0, "delta": {"content": word}}]}
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        if slow:
                            threading.Event().wait(0.2)
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    outer.disconnected.set()

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def engine(monkeypatch):
    made = []

    def make(**kw):
        e = _Engine(**kw)
        made.append(e)
        cfg = dataclasses.replace(llm.settings.llm, provider="engine", engine_url=e.url, engine_timeout_s=5.0)
        monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, llm=cfg))
        return e

    yield make
    for e in made:
        e.close()


def test_engine_provider_streams_deltas_and_never_loads_llama_cpp(engine):
    e = engine()
    assert list(llm.stream_chat([{"role": "user", "content": "hi"}], max_tokens=7, temperature=0.1)) == ["Hel", "lo"]
    path, body = e.requests[0]
    assert path == "/v1/chat/completions"
    assert body["stream"] is True and body["max_tokens"] == 7 and body["temperature"] == 0.1
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    # conftest blocks get_llm: reaching here proves the in-process model was not used.


def test_chat_joins_engine_stream(engine):
    engine(words=(" a", "b "))
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ab"


def test_json_mode_fails_loudly_instead_of_returning_unconstrained_text(engine):
    e = engine()
    with pytest.raises(llm.EngineError, match="json_mode"):
        llm.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert e.requests == []


def test_engine_refusal_surfaces_its_message(engine):
    engine(status=400, error_body={"error": {"message": "prompt too long", "code": "resource_exhausted"}})
    with pytest.raises(llm.EngineError, match="HTTP 400.*prompt too long"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_unreachable_engine_says_how_to_start_it(monkeypatch):
    cfg = dataclasses.replace(llm.settings.llm, provider="engine", engine_url="http://127.0.0.1:1/v1")
    monkeypatch.setattr(llm, "settings", dataclasses.replace(llm.settings, llm=cfg))
    with pytest.raises(llm.EngineError, match="sally-inference serve"):
        llm.chat([{"role": "user", "content": "hi"}])


def test_cancel_closes_the_connection_and_releases_the_lock(engine):
    e = engine(words=tuple("abcdefgh"), slow=True)
    cancel = threading.Event()
    seen = []
    for token in llm.stream_chat([], cancel=cancel):
        seen.append(token)
        cancel.set()
    assert seen == ["a"]
    assert e.disconnected.wait(3), "the engine never saw the client leave"
    assert llm._GENERATION_LOCK.acquire(blocking=False)
    llm._GENERATION_LOCK.release()


def test_default_provider_is_in_process_llama_cpp():
    assert llm.settings.llm.provider == "llama_cpp"
