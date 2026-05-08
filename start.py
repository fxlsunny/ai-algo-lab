#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
start.py — AI_Algo_Lab 一键启动脚本

功能：
  1. 启动本地 HTTP 服务器托管整个 AI_Algo_Lab
  2. 内置 LLM API 代理（/api/llm/ 路径）→ 同源请求，完全绕过 CORS
  3. 自动打开浏览器

用法：
  python start.py                    # 默认 8766 端口
  python start.py --port 8888        # 自定义端口
  python start.py --no-browser       # 不自动打开浏览器

一次运行，彻底解决 CORS 问题 + 零外部依赖（仅 Python 3.6+ 标准库）
"""
import argparse
import json
import os
import sys
import webbrowser
import urllib.request
import urllib.error
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

# Windows 控制台 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))

# 允许代理的 LLM 域名（安全白名单）
ALLOWED_HOSTS = {
    "hunyuan.cloud.tencent.com", "api.hunyuan.cloud.tencent.com",
    "api.openai.com", "api.anthropic.com", "api.deepseek.com",
    "dashscope.aliyuncs.com", "open.bigmodel.cn", "api.moonshot.cn",
    "api.groq.com", "api.siliconflow.cn", "api.baichuan-ai.com",
    "api.minimax.chat", "ark.cn-beijing.volces.com",
    "generativelanguage.googleapis.com",
    "localhost", "127.0.0.1",
}

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "*",
    "Access-Control-Max-Age": "86400",
}


class Handler(SimpleHTTPRequestHandler):
    """把 SimpleHTTPRequestHandler 扩展成静态文件 + LLM 代理二合一"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT, **kwargs)

    # 静默日志
    def log_message(self, format, *args):
        try:
            msg = format % args
            if " 200 " in msg or " 304 " in msg:
                return  # 只打印非正常响应
            sys.stdout.write("  %s %s\n" % (self.log_date_time_string(), msg))
        except Exception:
            pass

    def do_OPTIONS(self):
        self.send_response(204)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()

    def do_POST(self):
        if self.path.startswith("/api/llm/"):
            self._proxy_llm()
        elif self.path.startswith("/api/config/"):
            self._handle_config()
        elif self.path.startswith("/api/history"):
            self._handle_history()
        else:
            self._not_found()

    def do_PUT(self):
        if self.path.startswith("/api/llm/"):
            self._proxy_llm()
        elif self.path.startswith("/api/config/"):
            self._handle_config()
        elif self.path.startswith("/api/history"):
            self._handle_history()
        else:
            self._not_found()

    def do_DELETE(self):
        if self.path.startswith("/api/llm/"):
            self._proxy_llm()
        elif self.path.startswith("/api/history"):
            self._handle_history()
        else:
            self._not_found()

    def do_GET(self):
        if self.path.startswith("/api/llm/"):
            self._proxy_llm()
        elif self.path == "/api/health":
            self._send_json(200, {"ok": True, "proxy": True,
                                  "service": "AI_Algo_Lab unified server",
                                  "features": ["llm_proxy", "config", "history"]})
        elif self.path.startswith("/api/config/"):
            self._handle_config()
        elif self.path.startswith("/api/history"):
            self._handle_history()
        else:
            # 静态文件由父类处理
            super().do_GET()
            return

    # ================================================================
    # API: /api/config/<name>  — 读写 data/configs/<name>.json
    # ================================================================
    def _handle_config(self):
        """
        GET    /api/config/<name>        → 读取配置
        POST   /api/config/<name>        → 保存配置（整体覆盖）
        DELETE /api/config/<name>        → 删除配置
        """
        name = self.path[len("/api/config/"):].strip("/")
        if not name or "/" in name or ".." in name:
            self._send_json(400, {"error": "Invalid config name"}); return
        if not name.endswith(".json"):
            name += ".json"

        cfg_dir = os.path.join(ROOT, "data", "configs")
        os.makedirs(cfg_dir, exist_ok=True)
        path = os.path.join(cfg_dir, name)

        if self.command == "GET":
            if not os.path.exists(path):
                self._send_json(200, {"exists": False, "data": None}); return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._send_json(200, {"exists": True, "data": data})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.command in ("POST", "PUT"):
            try:
                cl = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(cl) if cl > 0 else b"{}"
                obj = json.loads(raw.decode("utf-8"))
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)
                self._send_json(200, {"ok": True, "path": f"data/configs/{name}"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        elif self.command == "DELETE":
            try:
                if os.path.exists(path): os.remove(path)
                self._send_json(200, {"ok": True})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(405, {"error": "Method not allowed"})

    # ================================================================
    # API: /api/history       — 列出/批量保存历史
    # API: /api/history/<id>  — 单条读/删
    # ================================================================
    def _handle_history(self):
        """
        GET    /api/history                → 列出所有记录（仅元信息：id/title/ts）
        GET    /api/history?full=1         → 列出所有记录完整数据
        GET    /api/history/<id>           → 读取单条
        POST   /api/history                → 批量保存数组 [{id, ...}]
        POST   /api/history/<id>           → 保存单条
        DELETE /api/history/<id>           → 删除单条
        DELETE /api/history                → 清空所有
        """
        hist_dir = os.path.join(ROOT, "data", "history")
        os.makedirs(hist_dir, exist_ok=True)

        # 解析路径和 query
        p = self.path
        q = ""
        if "?" in p:
            p, q = p.split("?", 1)

        after = p[len("/api/history"):].strip("/")
        # after 可能是 "" 或 "<id>"

        if self.command == "GET":
            if not after:
                # 列表
                full = "full=1" in q
                items = []
                for fn in sorted(os.listdir(hist_dir), reverse=True):
                    if not fn.endswith(".json"): continue
                    fp = os.path.join(hist_dir, fn)
                    try:
                        with open(fp, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if full:
                            items.append(data)
                        else:
                            items.append({
                                "id": data.get("id") or fn[:-5],
                                "title": data.get("title") or data.get("scenario", "")[:60],
                                "ts": data.get("ts") or data.get("timestamp"),
                                "file": fn,
                            })
                    except Exception:
                        continue
                self._send_json(200, {"count": len(items), "items": items})
            else:
                # 单条
                safe_id = "".join(c for c in after if c.isalnum() or c in "-_")
                fp = os.path.join(hist_dir, safe_id + ".json")
                if not os.path.exists(fp):
                    self._send_json(404, {"error": "Not found"}); return
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._send_json(200, {"data": data})
                except Exception as e:
                    self._send_json(500, {"error": str(e)})

        elif self.command in ("POST", "PUT"):
            try:
                cl = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(cl) if cl > 0 else b"{}"
                obj = json.loads(raw.decode("utf-8"))

                if not after:
                    # 批量保存：obj 可以是数组或 {items: [...]}
                    items = obj if isinstance(obj, list) else obj.get("items", [])
                    saved = 0
                    for it in items:
                        rid = str(it.get("id") or "").strip()
                        if not rid: continue
                        safe_id = "".join(c for c in rid if c.isalnum() or c in "-_")
                        if not safe_id: continue
                        fp = os.path.join(hist_dir, safe_id + ".json")
                        with open(fp, "w", encoding="utf-8") as f:
                            json.dump(it, f, ensure_ascii=False, indent=2)
                        saved += 1
                    self._send_json(200, {"ok": True, "saved": saved})
                else:
                    # 单条保存
                    safe_id = "".join(c for c in after if c.isalnum() or c in "-_")
                    fp = os.path.join(hist_dir, safe_id + ".json")
                    with open(fp, "w", encoding="utf-8") as f:
                        json.dump(obj, f, ensure_ascii=False, indent=2)
                    self._send_json(200, {"ok": True, "path": f"data/history/{safe_id}.json"})
            except Exception as e:
                self._send_json(500, {"error": str(e)})

        elif self.command == "DELETE":
            try:
                if not after:
                    # 清空（保留 example 文件）
                    for fn in os.listdir(hist_dir):
                        if fn.endswith(".json") and not fn.startswith("example"):
                            os.remove(os.path.join(hist_dir, fn))
                    self._send_json(200, {"ok": True})
                else:
                    safe_id = "".join(c for c in after if c.isalnum() or c in "-_")
                    fp = os.path.join(hist_dir, safe_id + ".json")
                    if os.path.exists(fp): os.remove(fp)
                    self._send_json(200, {"ok": True})
            except Exception as e:
                self._send_json(500, {"error": str(e)})
        else:
            self._send_json(405, {"error": "Method not allowed"})

    def _proxy_llm(self):
        """代理路径格式：/api/llm/<target_encoded_url>
           目标 URL 可以是完整 URL 或仅 host+path"""
        raw = self.path[len("/api/llm/"):]
        target = unquote(raw)
        if not (target.startswith("http://") or target.startswith("https://")):
            target = "https://" + target

        # 白名单
        host = urlparse(target).hostname or ""
        if host not in ALLOWED_HOSTS and not host.endswith(".local"):
            self._send_json(403, {"error": f"Host not whitelisted: {host}",
                                  "allowed": sorted(ALLOWED_HOSTS)})
            return

        # 读请求体
        body = None
        try:
            cl = int(self.headers.get("Content-Length", 0))
            if cl > 0:
                body = self.rfile.read(cl)
        except Exception:
            body = None

        # 转发 headers（去掉 Host/Origin/Referer/Content-Length）
        fwd_headers = {}
        for k in self.headers.keys():
            if k.lower() in {"host", "origin", "referer", "content-length"}:
                continue
            fwd_headers[k] = self.headers[k]

        try:
            req = urllib.request.Request(
                target, data=body, headers=fwd_headers, method=self.command)
            with urllib.request.urlopen(req, timeout=120) as r:
                status = r.status
                resp_headers = dict(r.headers.items())
                data = r.read()
        except urllib.error.HTTPError as e:
            status = e.code
            resp_headers = dict(e.headers.items()) if e.headers else {}
            data = e.read() or b""
        except Exception as e:
            self._send_json(502, {"error": "Proxy error", "detail": str(e),
                                  "target": target})
            return

        self.send_response(status)
        for k, v in resp_headers.items():
            if k.lower() in {"access-control-allow-origin", "access-control-allow-methods",
                             "access-control-allow-headers", "transfer-encoding",
                             "content-encoding"}:
                continue
            self.send_header(k, v)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _not_found(self):
        self._send_json(404, {"error": "Not found", "path": self.path})

    def _send_json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        # 给所有静态资源也带上 CORS 头（不强制但稳健）
        for k, v in CORS_HEADERS.items():
            self.send_header(k, v)
        super().end_headers()


def get_lan_ips():
    """获取本机所有 LAN IP（非 127.x），用于在启动 banner 中提示用户。"""
    import socket
    ips = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None):
            ip = info[4][0]
            if "." in ip and not ip.startswith("127."):
                ips.add(ip)
    except Exception:
        pass
    # 备用方案：通过 UDP 连接探测出口 IP
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(ips)


def print_banner(port, host):
    print("\n" + "=" * 70)
    print(f"  AI_Algo_Lab 统一服务器（HTML 托管 + LLM 代理 二合一）")
    print("=" * 70)
    # 本机访问
    print(f"  ▶ 本机访问:  http://localhost:{port}/index.html")
    print(f"              http://127.0.0.1:{port}/index.html")
    # LAN 访问（手机/同局域网其它设备）
    if host in ("0.0.0.0", "::", ""):
        ips = get_lan_ips()
        if ips:
            print(f"  ▶ 手机访问:  （同 WiFi 局域网下用手机浏览器打开）")
            for ip in ips:
                print(f"              http://{ip}:{port}/index.html")
        else:
            print(f"  ▶ 手机访问:  （未获取到 LAN IP，请用 ipconfig / ifconfig 查）")
    else:
        print(f"  ⚠ 仅限本机:  当前绑定 {host}，外部设备无法访问")
        print(f"              如需 LAN 访问，改用: python start.py --host 0.0.0.0")
    print(f"  ▶ LLM 代理:  /api/llm/<target_url>（同源，无 CORS）")
    print(f"  ▶ 持久化:    /api/config/  /api/history/")
    print("")
    print('  📌 手机访问需开放防火墙端口（首次会弹提示，选"允许"）')
    print(f"  📌 停止服务:  按 Ctrl+C")
    print("=" * 70 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8766)
    # 默认绑定 0.0.0.0 以支持 LAN 访问（手机/其它设备）
    ap.add_argument("--host", default="0.0.0.0",
                    help="绑定地址，默认 0.0.0.0（所有网卡）。仅本机用 127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    os.chdir(ROOT)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print_banner(args.port, args.host)

    # 不再主动打开浏览器；由服务管理器负责或用户手动访问。
    # （保留 --no-browser 入参以维持兼容；本服务已不再调用 webbrowser.open）
    _ = args.no_browser

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  👋 服务器已停止")
        srv.server_close()


if __name__ == "__main__":
    main()
