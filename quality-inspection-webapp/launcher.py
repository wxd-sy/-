"""智能质检台 · 桌面启动器（纯 Win32，无 PIL / pystray 依赖）

双击即用：自动启动 Web 后端（无黑窗）、打开浏览器，并在系统托盘常驻。
托盘右键：打开质检台 / 重启服务 / 查看日志 / 退出（退出时自动关闭后端）。

命令行参数：
    --test      启动后端、自检、写日志后退出（用于打包后验证）
    --no-tray   启动后端并打开浏览器，不驻留托盘
    --stop      结束已在运行的后端服务
"""
from __future__ import annotations

import ctypes
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from ctypes import wintypes
from pathlib import Path

APP_DIR = (Path(sys.executable).parent if getattr(sys, "frozen", False)
           else Path(__file__).resolve().parent)
PYTHON = r"C:\ProgramData\anaconda3\envs\yolov8_310\python.exe"
PORT = 8765
URL = f"http://127.0.0.1:{PORT}"
LOG_DIR = APP_DIR / "logs"
APP_NAME = "智能质检台"
MUTEX = "Global\\QualityInspectLauncher_8765"

_proc: subprocess.Popen | None = None
_proc_lock = threading.Lock()


# ============================================================ 服务管理
def health_ok(timeout: float = 1.5) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(URL + "/api/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def port_busy() -> bool:
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def _log_path() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"server-{time.strftime('%Y%m%d')}.log"


def start_server(wait: float = 75.0) -> bool:
    global _proc
    if health_ok():
        return True
    if port_busy():
        return False
    if not APP_DIR.is_dir():
        return False

    python = PYTHON if Path(PYTHON).exists() else "python"
    with _proc_lock:
        log = open(_log_path(), "ab", buffering=0)
        try:
            log.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] launcher starting\n".encode())
        except Exception:
            pass
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        _proc = subprocess.Popen(
            [python, "app.py", "--port", str(PORT)],
            cwd=str(APP_DIR), stdout=log, stderr=log, stdin=subprocess.DEVNULL,
            creationflags=flags,
        )

    deadline = time.time() + wait
    while time.time() < deadline:
        if health_ok():
            return True
        if _proc.poll() is not None:
            return False
        time.sleep(0.4)
    return False


def stop_server() -> None:
    global _proc
    with _proc_lock:
        if _proc and _proc.poll() is None:
            try:
                _proc.terminate()
                _proc.wait(timeout=6)
            except Exception:
                try:
                    _proc.kill()
                except Exception:
                    pass
        _proc = None


def kill_by_port() -> None:
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "TCP"], creationflags=subprocess.CREATE_NO_WINDOW
        ).decode("utf-8", "ignore")
    except Exception:
        return
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[1].endswith(f":{PORT}") and parts[3] == "LISTENING":
            pid = parts[4]
            if pid.isdigit() and int(pid) != os.getpid():
                subprocess.run(["taskkill", "/F", "/PID", pid],
                               creationflags=subprocess.CREATE_NO_WINDOW,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ============================================================ Win32 托盘
user32 = ctypes.WinDLL("user32", use_last_error=True)
shell32 = ctypes.WinDLL("shell32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WM_DESTROY = 0x0002
WM_COMMAND = 0x0111
WM_APP = 0x8000
WM_TRAY = WM_APP + 1
NIM_ADD, NIM_MODIFY, NIM_DELETE = 0, 1, 2
NIF_MESSAGE, NIF_ICON, NIF_TIP = 0x01, 0x02, 0x04
WM_LBUTTONUP, WM_RBUTTONUP, WM_LBUTTONDBLCLK = 0x0202, 0x0205, 0x0203
MF_STRING, MF_SEPARATOR, MF_GRAYED = 0x0000, 0x0800, 0x0001
TPM_RIGHTBUTTON, TPM_RETURNCMD, TPM_NONOTIFY = 0x0002, 0x0100, 0x0080
IMAGE_ICON, LR_LOADFROMFILE, LR_DEFAULTSIZE = 1, 0x0010, 0x0040
IDI_APPLICATION = 32512
CS_VREDRAW, CS_HREDRAW = 0x0001, 0x0002

LRESULT = ctypes.c_ssize_t
WNDPROC = ctypes.WINFUNCTYPE(LRESULT, wintypes.HWND, wintypes.UINT,
                            wintypes.WPARAM, wintypes.LPARAM)


class GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD), ("Data4", ctypes.c_ubyte * 8)]


class NOTIFYICONDATAW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD), ("hWnd", wintypes.HWND), ("uID", wintypes.UINT),
        ("uFlags", wintypes.UINT), ("uCallbackMessage", wintypes.UINT),
        ("hIcon", wintypes.HICON), ("szTip", wintypes.WCHAR * 128),
        ("dwState", wintypes.DWORD), ("dwStateMask", wintypes.DWORD),
        ("szInfo", wintypes.WCHAR * 256), ("uVersion", wintypes.UINT),
        ("szInfoTitle", wintypes.WCHAR * 64), ("dwInfoFlags", wintypes.DWORD),
        ("guidItem", GUID), ("hBalloonIcon", wintypes.HICON),
    ]


class WNDCLASSW(ctypes.Structure):
    _fields_ = [
        ("style", wintypes.UINT), ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int), ("cbWndExtra", ctypes.c_int),
        ("hInstance", wintypes.HINSTANCE), ("hIcon", wintypes.HICON),
        ("hCursor", wintypes.HANDLE), ("hbrBackground", wintypes.HBRUSH),
        ("lpszMenuName", wintypes.LPCWSTR), ("lpszClassName", wintypes.LPCWSTR),
    ]


def _setup_win32():
    """显式声明签名，避免 64 位下句柄被截断成 32 位。"""
    P = ctypes.POINTER
    user32.RegisterClassW.argtypes = [P(WNDCLASSW)]
    user32.RegisterClassW.restype = wintypes.WORD
    user32.CreateWindowExW.argtypes = [
        wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        wintypes.HWND, wintypes.HMENU, wintypes.HINSTANCE, ctypes.c_void_p,
    ]
    user32.CreateWindowExW.restype = wintypes.HWND
    user32.DefWindowProcW.argtypes = [wintypes.HWND, wintypes.UINT,
                                      wintypes.WPARAM, wintypes.LPARAM]
    user32.DefWindowProcW.restype = LRESULT
    user32.GetMessageW.argtypes = [P(wintypes.MSG), wintypes.HWND,
                                   wintypes.UINT, wintypes.UINT]
    user32.GetMessageW.restype = ctypes.c_int
    user32.TranslateMessage.argtypes = [P(wintypes.MSG)]
    user32.DispatchMessageW.argtypes = [P(wintypes.MSG)]
    user32.DispatchMessageW.restype = LRESULT
    user32.CreatePopupMenu.restype = wintypes.HMENU
    user32.AppendMenuW.argtypes = [wintypes.HMENU, wintypes.UINT,
                                   ctypes.c_size_t, wintypes.LPCWSTR]
    user32.TrackPopupMenu.argtypes = [wintypes.HMENU, wintypes.UINT, ctypes.c_int,
                                      ctypes.c_int, ctypes.c_int, wintypes.HWND,
                                      ctypes.c_void_p]
    user32.TrackPopupMenu.restype = ctypes.c_int
    user32.GetCursorPos.argtypes = [P(wintypes.POINT)]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                    wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT,
                                    wintypes.WPARAM, wintypes.LPARAM]
    user32.SendMessageW.restype = LRESULT
    user32.PostQuitMessage.argtypes = [ctypes.c_int]
    user32.DestroyWindow.argtypes = [wintypes.HWND]
    user32.DestroyMenu.argtypes = [wintypes.HMENU]
    user32.LoadImageW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR, wintypes.UINT,
                                  ctypes.c_int, ctypes.c_int, wintypes.UINT]
    user32.LoadImageW.restype = wintypes.HICON
    user32.LoadIconW.argtypes = [wintypes.HINSTANCE, wintypes.LPCWSTR]
    user32.LoadIconW.restype = wintypes.HICON
    shell32.Shell_NotifyIconW.argtypes = [wintypes.DWORD, P(NOTIFYICONDATAW)]
    shell32.Shell_NotifyIconW.restype = wintypes.BOOL
    kernel32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetModuleHandleW.restype = wintypes.HINSTANCE
    kernel32.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel32.CreateMutexW.restype = wintypes.HANDLE


_setup_win32()


def resource_path(rel: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, rel)


class TrayApp:
    ID_OPEN, ID_RESTART, ID_LOGS, ID_QUIT = 1001, 1002, 1003, 1004
    CLASS_NAME = "QualityInspectTrayWnd"

    def __init__(self):
        self.hwnd = None
        self.hicon = None
        self._wndproc = WNDPROC(self._on_message)   # 必须保持引用，否则被 GC
        self._exit_code = 0

    # ---------- 消息处理 ----------
    def _on_message(self, hwnd, msg, wparam, lparam):
        if msg == WM_TRAY:
            low = lparam & 0xFFFF
            if low in (WM_LBUTTONUP, WM_LBUTTONDBLCLK):
                webbrowser.open(URL)
            elif low == WM_RBUTTONUP:
                self._show_menu()
            return 0
        if msg == WM_COMMAND:
            cmd = wparam & 0xFFFF
            if cmd == self.ID_OPEN:
                webbrowser.open(URL)
            elif cmd == self.ID_RESTART:
                self.set_tip("智能质检台 · 正在重启…")
                threading.Thread(target=self._restart, daemon=True).start()
            elif cmd == self.ID_LOGS:
                LOG_DIR.mkdir(parents=True, exist_ok=True)
                try:
                    os.startfile(str(LOG_DIR))
                except Exception:
                    pass
            elif cmd == self.ID_QUIT:
                self._exit_code = 0
                user32.PostQuitMessage(0)
            return 0
        if msg == WM_DESTROY:
            user32.PostQuitMessage(0)
            return 0
        return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

    def _restart(self):
        stop_server()
        kill_by_port()
        ok = start_server()
        self.set_tip("智能质检台 · 运行中" if ok else "智能质检台 · 启动失败")
        if ok:
            webbrowser.open(URL)

    def _show_menu(self):
        menu = user32.CreatePopupMenu()
        user32.AppendMenuW(menu, MF_STRING, self.ID_OPEN, "打开质检台")
        user32.AppendMenuW(menu, MF_STRING, self.ID_RESTART, "重启服务")
        user32.AppendMenuW(menu, MF_STRING, self.ID_LOGS, "打开日志文件夹")
        user32.AppendMenuW(menu, MF_SEPARATOR, 0, None)
        user32.AppendMenuW(menu, MF_STRING, self.ID_QUIT, "退出（同时关闭服务）")

        pt = wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        # 让菜单能因点击别处而消失
        user32.SetForegroundWindow(self.hwnd)
        cmd = user32.TrackPopupMenu(
            menu, TPM_RIGHTBUTTON | TPM_RETURNCMD | TPM_NONOTIFY,
            pt.x, pt.y, 0, self.hwnd, None,
        )
        user32.PostMessageW(self.hwnd, 0x0000, 0, 0)   # WM_NULL
        user32.DestroyMenu(menu)
        if cmd:
            user32.SendMessageW(self.hwnd, WM_COMMAND, cmd, 0)

    # ---------- 托盘图标 ----------
    def _load_icon(self):
        ico = resource_path(os.path.join("assets", "app.ico"))
        if os.path.isfile(ico):
            h = user32.LoadImageW(None, ico, IMAGE_ICON, 0, 0,
                                  LR_LOADFROMFILE | LR_DEFAULTSIZE)
            if h:
                return h
        return user32.LoadIconW(
            None, ctypes.cast(ctypes.c_void_p(IDI_APPLICATION), wintypes.LPCWSTR)
        )

    def set_tip(self, text: str):
        if not self.hwnd:
            return
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = self.hwnd
        nid.uID = 1
        nid.uFlags = NIF_TIP
        nid.szTip = text[:127]
        shell32.Shell_NotifyIconW(NIM_MODIFY, ctypes.byref(nid))

    def _add_icon(self):
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = self.hwnd
        nid.uID = 1
        nid.uFlags = NIF_MESSAGE | NIF_ICON | NIF_TIP
        nid.uCallbackMessage = WM_TRAY
        nid.hIcon = self.hicon
        nid.szTip = "智能质检台 · 正在启动…"
        shell32.Shell_NotifyIconW(NIM_ADD, ctypes.byref(nid))

    def _remove_icon(self):
        if not self.hwnd:
            return
        nid = NOTIFYICONDATAW()
        nid.cbSize = ctypes.sizeof(NOTIFYICONDATAW)
        nid.hWnd = self.hwnd
        nid.uID = 1
        shell32.Shell_NotifyIconW(NIM_DELETE, ctypes.byref(nid))

    # ---------- 主循环 ----------
    def run(self):
        hinst = kernel32.GetModuleHandleW(None)
        wc = WNDCLASSW()
        wc.style = CS_VREDRAW | CS_HREDRAW
        wc.lpfnWndProc = self._wndproc
        wc.hInstance = hinst
        wc.hIcon = self._load_icon()
        wc.lpszClassName = self.CLASS_NAME
        if not user32.RegisterClassW(ctypes.byref(wc)):
            err = ctypes.get_last_error()
            if err != 1410:     # ERROR_CLASS_ALREADY_EXISTS
                raise ctypes.WinError(err)

        self.hwnd = user32.CreateWindowExW(
            0, self.CLASS_NAME, APP_NAME, 0, 0, 0, 0, 0, None, None, hinst, None
        )
        if not self.hwnd:
            raise ctypes.WinError(ctypes.get_last_error())

        self.hicon = self._load_icon()
        self._add_icon()

        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        self._remove_icon()
        user32.DestroyWindow(self.hwnd)
        return self._exit_code


# ============================================================ 入口
def run_tray(open_browser: bool = True) -> None:
    app = TrayApp()

    def boot():
        ok = start_server()
        app.set_tip("智能质检台 · 运行中" if ok else "智能质检台 · 启动失败（见日志）")
        if ok and open_browser:
            webbrowser.open(URL)

    threading.Thread(target=boot, daemon=True).start()
    try:
        app.run()
    finally:
        stop_server()


def main() -> None:
    args = sys.argv[1:]

    if "--stop" in args:
        kill_by_port()
        stop_server()
        return

    if "--test" in args:
        ok = start_server()
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "launcher_test.log").write_text(
            f"health={ok}\nport_busy={port_busy()}\nproc={_proc.pid if _proc else None}\n",
            encoding="utf-8",
        )
        stop_server()
        return

    if "--no-tray" in args:
        if start_server():
            webbrowser.open(URL)
        return

    handle = kernel32.CreateMutexW(None, False, MUTEX)
    if kernel32.GetLastError() == 183:      # ERROR_ALREADY_EXISTS
        webbrowser.open(URL)
        return
    try:
        run_tray()
    finally:
        stop_server()
        try:
            kernel32.CloseHandle(handle)
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        (LOG_DIR / "launcher_error.log").write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        raise
