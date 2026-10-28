"""生成智能质检台的 .ico / .png 图标（无需外部素材）。"""
from pathlib import Path
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

CYAN = (34, 211, 238, 255)
CYAN_SOFT = (34, 211, 238, 120)
BG = (11, 20, 27, 255)
BG2 = (16, 30, 40, 255)


def draw_icon(size=256):
    # 4 倍超采样后缩小，边缘更干净
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # 圆角底板
    pad = int(s * 0.04)
    d.rounded_rectangle([pad, pad, s - pad, s - pad], radius=int(s * 0.22),
                        fill=BG, outline=CYAN_SOFT, width=max(2, int(s * 0.012)))

    # 内层微光渐变（用两个同心圆角矩形模拟）
    inset = int(s * 0.10)
    d.rounded_rectangle([inset, inset, s - inset, s - inset], radius=int(s * 0.16),
                        fill=BG2, outline=(34, 211, 238, 45), width=max(1, int(s * 0.006)))

    # 扫描框四角
    m = int(s * 0.24)          # 边距
    arm = int(s * 0.14)        # 角长
    w = max(3, int(s * 0.035))  # 线宽
    for x1, y1, x2, y2, x3, y3 in [
        (m, m + arm, m, m, m + arm, m),                                  # 左上
        (s - m - arm, m, s - m, m, s - m, m + arm),                      # 右上
        (s - m, s - m - arm, s - m, s - m, s - m - arm, s - m),          # 右下
        (m + arm, s - m, m, s - m, m, s - m - arm),                      # 左下
    ]:
        d.line([(x1, y1), (x2, y2), (x3, y3)], fill=CYAN, width=w, joint="curve")

    # 中心准星圆
    cx = cy = s // 2
    r = int(s * 0.085)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=CYAN, width=w)

    # 十字线
    g = int(s * 0.045)
    d.line([(cx, cy - r - g), (cx, cy - r)], fill=CYAN, width=w)
    d.line([(cx, cy + r), (cx, cy + r + g)], fill=CYAN, width=w)
    d.line([(cx - r - g, cy), (cx - r, cy)], fill=CYAN, width=w)
    d.line([(cx + r, cy), (cx + r + g, cy)], fill=CYAN, width=w)

    # 中心点
    dot = int(s * 0.022)
    d.ellipse([cx - dot, cy - dot, cx + dot, cy + dot], fill=CYAN)

    return img.resize((size, size), Image.LANCZOS)


base = draw_icon(256)
png_path = OUT / "app.png"
ico_path = OUT / "app.ico"
base.save(png_path)
base.save(ico_path, format="ICO",
          sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print("saved:", png_path)
print("saved:", ico_path)
