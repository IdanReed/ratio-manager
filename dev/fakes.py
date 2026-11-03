#!/usr/bin/env python3
"""Local stand-ins for everything the manager talks to except qBittorrent
(which runs for real, see dev/run.sh): TorrentLeech's site and tracker,
Prowlarr, autobrr and ntfy. Stdlib only.

  :18090  TL: /rss/download/<id>/<key>/<id>.torrent, /a/<pk>/announce,
          /a/<pk>/scrape (several info_hash at once), /rss/<key>
  :18091  Prowlarr: /api/v1/indexer, /api/v1/search
  :18092  autobrr: auth, indexer, filters, downloaders
  :18093  ntfy: logs every POST

Torrents are real: each id's content is deterministic bytes, so writing the
file into a qBittorrent save path (`python3 dev/fakes.py content <id> <dir>`)
makes that box seed it after a recheck. Ids ending in 7 are NOT private (the
guard must refuse them). The tracker answers "Unregistered torrent" for
REG_DELAY seconds after a hash's first announce.
"""

import hashlib
import json
import random
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PIECE = 1 << 20
REG_DELAY = 8
PASSKEY = "0123456789abcdef0123"
first_seen = {}
log_lock = threading.Lock()


def log(*a):
    with log_lock:
        print("fakes:", *a, flush=True)


def benc(x):
    if isinstance(x, int):
        return b"i%de" % x
    if isinstance(x, str):
        x = x.encode()
    if isinstance(x, bytes):
        return b"%d:%s" % (len(x), x)
    if isinstance(x, list):
        return b"l" + b"".join(benc(i) for i in x) + b"e"
    if isinstance(x, dict):
        return b"d" + b"".join(benc(k) + benc(x[k]) for k in sorted(x, key=lambda k: k.encode() if isinstance(k, str) else k)) + b"e"
    raise TypeError(x)


def size_of(tid):
    return (20 + int(tid) % 7 * 10) * PIECE  # 20..80 MiB


def content(tid):
    rnd = random.Random(int(tid))
    return rnd.randbytes(size_of(tid))


_cache = {}


def torrent(tid):
    if tid in _cache:
        return _cache[tid]
    data = content(tid)
    pieces = b"".join(hashlib.sha1(data[i:i + PIECE]).digest() for i in range(0, len(data), PIECE))
    info = {"name": f"Fake.Release.{tid}.2026.1080p.WEB.mkv", "length": len(data), "piece length": PIECE, "pieces": pieces}
    if not tid.endswith("7"):
        info["private"] = 1
    raw = benc({"announce": f"http://127.0.0.1:18090/a/{PASSKEY}/announce", "info": info})
    h = hashlib.sha1(benc(info)).hexdigest()
    _cache[tid] = (raw, h)
    return raw, h


class TL(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, body, ctype="application/x-bittorrent", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        parts = u.path.strip("/").split("/")
        if parts[:2] == ["rss", "download"]:
            tid = parts[2]
            log("TL .torrent", tid)
            return self.reply(torrent(tid)[0])
        if parts[0] == "rss":
            items = "".join(
                f"<item><title>Fake.Release.{i}.2026.1080p.WEB</title><link>http://127.0.0.1:18090/rss/download/{i}/KEY/x.torrent</link>"
                f"<category>Movies :: HD</category><pubDate>Mon, 02 Nov 2026 20:00:00 +0000</pubDate></item>" for i in range(7001, 7004))
            return self.reply(f"<rss><channel>{items}</channel></rss>".encode(), "application/rss+xml")
        if parts[0] == "a" and parts[-1] in ("announce", "scrape"):
            qs = urllib.parse.parse_qs(u.query, keep_blank_values=True, encoding="latin-1")
            hashes = [x.encode("latin-1").hex() for x in qs.get("info_hash", [])]
            if parts[-1] == "announce":
                h = hashes[0] if hashes else ""
                first_seen.setdefault(h, time.time())
                if time.time() - first_seen[h] < REG_DELAY:
                    return self.reply(benc({"failure reason": "Unregistered torrent"}), "text/plain")
                return self.reply(benc({"interval": 1800, "peers": b""}), "text/plain")
            files = {}
            for h in hashes:
                r = random.Random(h + str(int(time.time() // 60)))
                files[bytes.fromhex(h)] = {"complete": r.randint(1, 20), "incomplete": r.randint(0, 60), "downloaded": r.randint(0, 200)}
            log("TL scrape", len(hashes), "hashes")
            return self.reply(benc({"files": files}), "text/plain")
        self.reply(b"no", "text/plain", 404)


class Prowlarr(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def js(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/v1/indexer"):
            return self.js([{"id": 7, "name": "TorrentLeech (freeleech)", "enable": True}])
        if self.path.startswith("/api/v1/search"):
            rows = []
            for i in range(9001, 9006):
                rows.append({"guid": f"tl-{i}", "title": f"Fake.Release.{i}.2026.1080p.WEB", "size": size_of(str(i)),
                             "seeders": 3, "leechers": 30, "indexerFlags": ["freeleech"],
                             "infoUrl": f"https://www.torrentleech.org/torrent/{i}", "publishDate": "2026-11-02T20:00:00Z",
                             "categories": [{"id": 2000, "name": "Movies/HD"}]})
            return self.js(rows)
        self.js({"error": "no"}, 404)


autobrr = {"filters": {11: {"id": 11, "name": "TorrentLeech freeleech race", "enabled": True, "actions": []}},
           "downloaders": {3: {"id": 3, "name": "seedbox"}, 4: {"id": 4, "name": "pulsed"}}, "next": 20}


class Autobrr(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def js(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if self.path.endswith("/auth/login"):
            self.send_header("Set-Cookie", "user_session=x; Path=/")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"null")

    def do_POST(self):
        b = self.body()
        if self.path == "/api/auth/login":
            return self.js({})
        if self.path == "/api/filters":
            i = autobrr["next"]
            autobrr["next"] += 1
            autobrr["filters"][i] = dict(b, id=i, actions=[])
            log("autobrr: created filter", b.get("name"))
            return self.js(autobrr["filters"][i], 201)
        self.js({}, 404)

    def do_GET(self):
        if self.path == "/api/indexer":
            return self.js([{"id": 1, "name": "TorrentLeech", "identifier": "torrentleech"}])
        if self.path == "/api/filters":
            return self.js(list(autobrr["filters"].values()))
        if self.path.startswith("/api/filters/"):
            return self.js(autobrr["filters"][int(self.path.rsplit("/", 1)[1])])
        if self.path == "/api/downloaders":
            return self.js(list(autobrr["downloaders"].values()))
        self.js({}, 404)

    def do_PUT(self):
        b = self.body()
        if self.path.startswith("/api/filters/"):
            i = int(self.path.rsplit("/", 1)[1])
            autobrr["filters"][i] = b
            log("autobrr: filter", b.get("name"), "actions", [a.get("type") + " " + a.get("webhook_host", "") for a in b.get("actions", [])])
            return self.js({})
        self.js({}, 404)

    def do_DELETE(self):
        kind, i = self.path.split("/")[2], int(self.path.rsplit("/", 1)[1])
        log("autobrr: delete", kind, i)
        (autobrr["filters"] if kind == "filters" else autobrr["downloaders"]).pop(i, None)
        self.js({}, 204)


class Ntfy(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        log("NTFY", self.headers.get("Title"), "|", self.rfile.read(n).decode(errors="replace")[:200])
        self.send_response(200)
        self.end_headers()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "content":
        tid, d = sys.argv[2], sys.argv[3]
        import os
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"Fake.Release.{tid}.2026.1080p.WEB.mkv"), "wb") as f:
            f.write(content(tid))
        print(torrent(tid)[1])
        return
    if len(sys.argv) > 1 and sys.argv[1] == "torrent":
        sys.stdout.buffer.write(torrent(sys.argv[2])[0])
        return
    for port, h in ((18090, TL), (18091, Prowlarr), (18092, Autobrr), (18093, Ntfy)):
        s = ThreadingHTTPServer(("127.0.0.1", port), h)
        threading.Thread(target=s.serve_forever, daemon=True).start()
    log("listening on 18090-18093")
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
