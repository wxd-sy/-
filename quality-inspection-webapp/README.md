# Quality Inspection Web UI / 智能质检台前端

配套本仓库「双映射网络目标检测模型压缩」的**可视化质检前端**：上传图片即可看到检测框、类别、
置信度、面积占比与耗时统计，支持在官方 RT-DETR 与 Dual-Mapping 压缩模型之间切换对比。

后端只用 Python 标准库 `http.server`（无 FastAPI/Flask 依赖），前端是单文件 HTML（无构建、无 CDN），
桌面启动器为纯 Win32 托盘程序（不依赖 PIL / pystray）。

---

## 1. 启动

### 方式 A：命令行（跨平台）

```bash
cd quality-inspection-webapp
python app.py --port 8765 --open
```

浏览器访问 <http://127.0.0.1:8765>。

> 需要与 `app.py` 同目录的 `vendor/`（推理依赖已随仓库提供）。
> 权重文件不入库（见第 3 节），需放在 `../测试文件/` 或修改 `engine.py` 的 `WEIGHTS`。

### 方式 B：Windows 一键启动

| 文件 | 作用 |
|---|---|
| `start.vbs` | 静默启动后端（`pythonw`）并打开浏览器，无黑窗、无托盘 |
| `stop.bat` | 停止后端 |

### 方式 C：打包成 exe（Windows 桌面图标 / 托盘）

启动器 `launcher.py` 支持托盘常驻：打开质检台 / 重启服务 / 打开日志 / 退出（自动关服务）。

```bat
python make_icon.py
python -m PyInstaller --noconfirm --clean --onefile --windowed ^
  --icon assets/app.ico --add-data "assets/app.ico;assets" ^
  --exclude-module PIL --exclude-module pystray --exclude-module numpy ^
  --exclude-module scipy --exclude-module matplotlib --exclude-module pandas ^
  --exclude-module torch --exclude-module torchvision --exclude-module cv2 ^
  --exclude-module PyQt5 --exclude-module tkinter ^
  --name "quality-inspection" launcher.py
```

产物约 9MB，`launcher.py` 参数：`--test` / `--no-tray` / `--stop`。

---

## 2. 目录结构

```
quality-inspection-webapp/
├── app.py                  # HTTP 服务 + REST API（标准库）
├── engine.py               # 权重加载、letterbox、推理、后处理、COCO 中文名
├── launcher.py             # Win32 托盘启动器（ctypes）
├── make_icon.py            # 生成 assets/app.ico
├── start.vbs / stop.bat    # Windows 静默启动 / 停止
├── static/index.html       # 单页前端
├── assets/                 # 图标
├── vendor/                 # 自带推理依赖（约 600KB）
│   ├── config.py / config_rtdetr.py
│   ├── bn_fuse.py / mapping_network.py / target_cnn_spec.py
│   ├── rtdetr/             # mapping_backbone、encoder_decoder、spec…
│   └── RT-DETR-main/.../src   # 官方模型与骨干网络代码
└── logs/                   # 运行时日志（已 gitignore）
```

---

## 3. 权重

不随仓库提供（体积大），路径定义在 `engine.py` 的 `WEIGHTS`：

```python
PROJ_DIR = Path(__file__).resolve().parent.parent      # 仓库根
WEIGHTS = {
    "backbone": PROJ_DIR / "测试文件" / "rtdetr_r18.pth",
    "detector": PROJ_DIR / "测试文件" / "rtdetr_r18vd_coco.pth",
    "dualmap":  PROJ_DIR / "测试文件" / "expD_results" / "expD_dualmap_best.pt",
}
```

| 键 | 来源 | 用途 |
|---|---|---|
| `rtdetr_r18vd_coco.pth` | 官方 RT-DETR-R18-vd COCO 权重 | 官方引擎（精度优先） |
| `rtdetr_r18.pth` | PResNet-18 backbone | Dual-Mapping 的基准卷积 |
| `expD_dualmap_best.pt` | 本仓库实验 D 训练产物 | 6,144 参数压缩模型（研究对照） |

---

## 4. 两个引擎

| 引擎 | 说明 | 实测 |
|---|---|---|
| `official` | 官方 RT-DETR-R18-vd，COCO 80 类 | bus.jpg：公交车 96.2%、人 94.7%，约 50ms |
| `dualmap` | 实验 D：6,144 参数、压缩 1,843× | VOC mAP@50≈7.4%，阈值需降到 0.05，仅供对照 |

---

## 5. API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 设备、torch 版本、权重存在性、引擎列表、80 类名 |
| GET | `/api/samples` | 内置示例图列表 |
| GET | `/api/sample/<id>?w=480` | 示例图（按宽度缩放） |
| POST | `/api/detect` | `{image: dataURL}` 或 `{sample: "s0"}`，加 `conf` `iou` `engine` |

```json
{
  "ok": true,
  "engine": {"id": "official", "label": "RT-DETR-R18 官方"},
  "detections": [
    {"cls_id": 5, "cls": "bus", "cls_zh": "公交车", "conf": 0.9615,
     "xyxy": [29.0, 231.0, 806.0, 736.0], "area_ratio": 45.0}
  ],
  "summary": {"total": 11, "classes": 5, "avg_conf": 0.5758, "infer_ms": 52.3, "total_ms": 66.6}
}
```

---

## 6. 界面功能

- 拖拽 / 点击 / `Ctrl+V` 粘贴，多图队列切换
- 画布：检测框 + 类别 + 置信度，滚轮缩放、拖动平移、框与表格联动高亮
- 参数：置信度阈值、按类别分组的 NMS IoU，改动自动重检
- 统计：检出数 / 类别数 / 平均置信度 / 推理耗时
- 明细表：中英类别、置信度条、面积占比、xyxy 坐标；类别分布条形图
- 导出：检测 JSON、原图分辨率标注 PNG；`Ctrl+Enter` 检测，`Ctrl+S` 导图
- 深链：`/?sample=s9` 直接打开示例并检测

---

## 7. 环境

- Python 3.10+，`torch` + `torchvision`（CUDA 可选，CPU 也能跑但慢 10–20 倍）
- 参考环境：RTX 5070 Ti Laptop，torch 2.12+cu128
- 首次检测加载 81MB 权重需 5–10 秒，之后单张 40–60ms（GPU）

```bash
pip install torch torchvision pillow
```
