"""智能质检 · RT-DETR 前端服务。

零额外依赖（只用 Python 标准库 + 项目已有 torch/PIL）：
    cd 桌面/智能质检/实验/webapp
    python app.py                 # 默认 http://127.0.0.1:8765
    python app.py --port 9000 --open

接口：
    GET  /                     单页前端
    GET  /api/health           设备 / 权重 / 引擎状态
    GET  /api/samples          内置示例图
    GET  /api/sample/<id>      示例图（?w= 控制宽度）
    POST /api/detect           {image: dataURL | sample:"id", conf, iou, engine}
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import sys
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

import engine

BASE = Path(__file__).resolve().parent
STATIC = BASE / "static"
PROJ = engine.PROJ_DIR                        # .../智能质检（权重与示例图所在项目根）
MAX_BODY = 48 * 1024 * 1024
INFER_LOCK = threading.Lock()

SAMPLE_SOURCES = [
    (PROJ / "质检项目用图", 8),
    (PROJ / "ultralytics-main" / "ultralytics" / "assets", 4),
]
SAMPLES: list[dict] = []


def _sample_key(p: Path):
    """工业场景图优先，其余按文件名，保证"载入示例图"总是给一张像样的质检图。"""
    keywords = ("生产线", "电路板", "微处理器", "新能源", "工业", "车间", "芯片")
    rank = next((i for i, kw in enumerate(keywords) if kw in p.name), len(keywords))
    return (rank, p.name)


def _build_samples():
    idx = 0
    for folder, limit in SAMPLE_SOURCES:
        if not folder.is_dir():
            continue
        files = [p for p in folder.iterdir()
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        for p in sorted(files, key=_sample_key)[:limit]:
            SAMPLES.append({"id": f"s{idx}", "name": p.name, "path": p,
                            "source": folder.name})
            idx += 1


def _read_image_bytes(payload: dict) -> bytes:
    sample = payload.get("sample")
    if sample:
        for s in SAMPLES:
            if s["id"] == sample:
                return s["path"].read_bytes()
        raise ValueError("示例图不存在")
    data = payload.get("image") or ""
    if "," in data:
        data = data.split(",", 1)[1]
    if not data:
        raise ValueError("缺少图片数据")
    raw = base64.b64decode(data, validate=False)
    if len(raw) > MAX_BODY:
        raise ValueError("图片过大（上限 48MB）")
    return raw


class Handler(BaseHTTPRequestHandler):
    server_version = "QualityInspect/1.0"

    def log_message(self, fmt, *args):
        pass

    # ---------- helpers ----------
    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, data: bytes, ctype: str, max_age: int = 0):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", f"public, max-age={max_age}" if max_age else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path: Path):
        if not path.is_file():
            self._json({"ok": False, "error": "not found"}, 404)
            return
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._bytes(path.read_bytes(), ctype)

    # ---------- routes ----------
    def do_GET(self):
        url = urlparse(self.path)
        path, qs = url.path, parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                self._file(STATIC / "index.html")
            elif path.startswith("/static/"):
                target = (STATIC / path[len("/static/"):]).resolve()
                if STATIC.resolve() not in target.parents and target != STATIC.resolve():
                    self._json({"ok": False, "error": "forbidden"}, 403)
                else:
                    self._file(target)
            elif path == "/api/health":
                info = engine.engine_info()
                info["ok"] = True
                self._json(info)
            elif path == "/api/samples":
                self._json({"samples": [
                    {"id": s["id"], "name": s["name"], "source": s["source"],
                     "url": f"/api/sample/{s['id']}?w=360"}
                    for s in SAMPLES
                ]})
            elif path.startswith("/api/sample/"):
                sid = path.rsplit("/", 1)[-1]
                item = next((s for s in SAMPLES if s["id"] == sid), None)
                if not item:
                    self._json({"ok": False, "error": "not found"}, 404)
                    return
                width = int((qs.get("w") or ["0"])[0] or 0)
                img = Image.open(item["path"]).convert("RGB")
                if width and width < img.width:
                    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=88)
                self._bytes(buf.getvalue(), "image/jpeg", max_age=3600)
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"ok": False, "error": str(exc)}, 500)

    def do_POST(self):
        if urlparse(self.path).path != "/api/detect":
            self._json({"ok": False, "error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY:
                self._json({"ok": False, "error": "请求体为空或过大"}, 413)
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            raw = _read_image_bytes(payload)
            with INFER_LOCK:
                result = engine.detect(
                    raw,
                    conf=float(payload.get("conf", 0.25)),
                    iou=float(payload.get("iou", 0.55)),
                    engine=str(payload.get("engine") or "official"),
                )
            result["ok"] = True
            self._json(result)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._json({"ok": False, "error": str(exc)}, 500)


def main():
    ap = argparse.ArgumentParser(description="智能质检 RT-DETR 前端")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    _build_samples()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print("=" * 62)
    print("  智能质检 · RT-DETR 视觉检测前端")
    print(f"  地址   {url}")
    print(f"  设备   {engine._device()} {engine._cuda_name() or ''}")
    print(f"  权重   {engine.WEIGHTS['detector'].name} / {engine.WEIGHTS['dualmap'].name}")
    print(f"  示例图 {len(SAMPLES)} 张")
    print("  模型在第一次检测时加载，请稍候几秒")
    print("=" * 62)
    if args.open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")


if __name__ == "__main__":
    main()
