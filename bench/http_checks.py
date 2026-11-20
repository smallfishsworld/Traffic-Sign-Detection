"""Raw-HTTP checks of a running ``tsd serve`` (used for the Docker smoke test).

    python bench/http_checks.py --url http://127.0.0.1:8799 --image some.jpg [--expect-429]

Each check writes the request by hand on a socket, so it controls exactly which headers are sent
and how much of the body:

* ``411``: a chunked upload without ``Content-Length``; only the headers are sent.
* ``413``: a ``Content-Length`` above the upload limit; only the headers are sent.
* ``429`` (``--expect-429``, for a server started with ``TSD_MAX_INFLIGHT=1``): one upload sends
  its headers and half of its body and stays in flight; a second request must get 429 with
  ``Retry-After``; after the first upload is completed (200) the next request is admitted again.

A response to a request whose body was never sent shows that the check happened before the
upload was read. Loopback targets only (``bench/load_test.py``'s rule; ``--allow-remote``
overrides). Prints one line per check and exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import http.client
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_test import check_target

BOUNDARY = "tsd-http-check"


def multipart(data: bytes) -> bytes:
    head = (
        f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="image"; filename="x.jpg"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    return head + data + f"\r\n--{BOUNDARY}--\r\n".encode()


def headers(host: str, extra: dict[str, str]) -> bytes:
    lines = [
        "POST /v1/detect HTTP/1.1",
        f"Host: {host}",
        f"Content-Type: multipart/form-data; boundary={BOUNDARY}",
        *(f"{k}: {v}" for k, v in extra.items()),
        "Connection: close",
    ]
    return ("\r\n".join(lines) + "\r\n\r\n").encode()


def response(sock: socket.socket) -> tuple[int, dict[str, str], bytes]:
    resp = http.client.HTTPResponse(sock)
    resp.begin()
    return resp.status, {k.lower(): v for k, v in resp.getheaders()}, resp.read()


def connect(host: str, port: int) -> socket.socket:
    return socket.create_connection((host, port), timeout=30)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True, help="A JPEG for the uploads.")
    parser.add_argument("--max-upload-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--expect-429", action="store_true")
    parser.add_argument("--allow-remote", action="store_true")
    args = parser.parse_args()
    check_target(args.url, args.allow_remote)
    parts = urlsplit(args.url)
    host, port = parts.hostname or "127.0.0.1", parts.port or 80
    body = multipart(args.image.read_bytes())
    failures = 0

    def report(name: str, ok: bool, detail: str) -> None:
        nonlocal failures
        failures += not ok
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}", flush=True)

    with connect(host, port) as s:  # 411: headers only, chunked, no Content-Length
        s.sendall(headers(host, {"Transfer-Encoding": "chunked"}))
        status, _, text = response(s)
        report("411 without Content-Length", status == 411, f"{status} {text.decode()[:80]}")
    with connect(host, port) as s:  # 413: headers only, announcing more than the limit
        too_big = args.max_upload_bytes + 1024 * 1024
        s.sendall(headers(host, {"Content-Length": str(too_big)}))
        status, _, text = response(s)
        detail = f"{status} {text.decode()[:80]} (Content-Length {too_big}, no body sent)"
        report("413 above the upload limit", status == 413, detail)
    if not args.expect_429:
        sys.exit(1 if failures else 0)
    with connect(host, port) as slow:  # in flight: headers and half of the body
        slow.sendall(headers(host, {"Content-Length": str(len(body))}) + body[: len(body) // 2])
        time.sleep(1.0)  # admitted; the server is waiting for the rest of the body
        with connect(host, port) as s:
            s.sendall(headers(host, {"Content-Length": str(len(body))}) + body)
            status, hdrs, text = response(s)
            retry = hdrs.get("retry-after")
            detail = f"{status} Retry-After={retry} {text.decode()[:60]}"
            report("429 at the admission limit", status == 429 and retry == "1", detail)
        slow.sendall(body[len(body) // 2 :])
        status, _, _ = response(slow)
        report("the in-flight upload completes", status == 200, str(status))
    with connect(host, port) as s:
        s.sendall(headers(host, {"Content-Length": str(len(body))}) + body)
        status, _, _ = response(s)
        report("admitted again after the slot is free", status == 200, str(status))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
