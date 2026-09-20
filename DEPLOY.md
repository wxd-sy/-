# 新服务器部署指南

## 一、本地准备（在 Windows 上执行一次）

### 1.1 打包代码

```powershell
cd D:\Amy_files\Lab\AI\lab01
tar czf dual_mapping_code.tar.gz --exclude='__pycache__' dual_mapping_deliverable
```

### 1.2 确认文件大小

```powershell
ls -lh dual_mapping_code.tar.gz
# 应该约 0.2MB
```

---

## 二、上传到 AutodL 或其他 Linux 服务器

以下假设服务器为 `root@your-server-ip`，端口为 `12345`。请替换为你的实际地址。

### 2.1 上传代码包

```powershell
scp -P 12345 dual_mapping_code.tar.gz root@your-server-ip:/root/
```

### 2.2 上传权重文件 (134MB)

```powershell
scp -P 12345 "mappingNetwork\data\weights\rtdetr_r18.pth" root@your-server-ip:/root/
scp -P 12345 "mappingNetwork\data\weights\rtdetr_r18vd_coco.pth" root@your-server-ip:/root/
```

### 2.3 上传 VOC 数据集 (439MB)

```powershell
scp -P 12345 "mappingNetwork\data\voc\VOCtrainval.tar" root@your-server-ip:/root/
```

---

## 三、服务器端环境搭建

SSH 登录服务器后执行：

### 3.1 解压代码

```bash
cd /root
tar xzf dual_mapping_code.tar.gz
cd dual_mapping_deliverable
ls               # 确认看到 rtdetr/, config_rtdetr.py 等
```

### 3.2 放置权重文件

```bash
mkdir -p data/weights
mv /root/rtdetr_r18.pth data/weights/
mv /root/rtdetr_r18vd_coco.pth data/weights/
ls -lh data/weights/
# 应看到:
# rtdetr_r18.pth         (43MB)
# rtdetr_r18vd_coco.pth  (91MB)
```

### 3.3 解压 VOC 数据

```bash
mkdir -p data
mv /root/VOCtrainval.tar data/
tar xf data/VOCtrainval.tar -C data/
ls data/VOCdevkit/VOC2007/JPEGImages/ | wc -l
# 应输出 5011
```

### 3.4 安装 Python 依赖

```bash
pip install torch torchvision tqdm numpy matplotlib scipy peft
```

如果 pip 不可用（AutodL 通常已预装 torch）：
```bash
# 检查 PyTorch 是否可用
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# 如果是 AutodL，应看到类似: 2.8.0 True

# 只装缺失的包
pip install tqdm scipy peft
```

---

## 四、验证环境

```bash
cd /root/dual_mapping_deliverable

# 4.1 代码语法检查
python -c "
import ast
for f in ['bn_fuse.py','config_rtdetr.py','mapping_network.py',
          'rtdetr/mapping_backbone.py','rtdetr/trainer.py','rtdetr/train.py']:
    ast.parse(open(f).read())
    print(f'OK: {f}')
"

# 4.2 快速干跑（dummy 数据，无需数据集）
python rtdetr/train.py --pretrained_ckpt data/weights/rtdetr_r18.pth \
    --dummy --dummy_samples 4 --batch_size 2 --epochs 1 --use_query_mapping

# 期望看到: Compression ratio: 1843.1x
```

---

## 五、运行实验

### 5.1 一键运行所有实验

```bash
cd /root/dual_mapping_deliverable
bash run_experiments.sh
```

总时间约 **15-30 小时**（取决于 GPU），包含：

| 实验 | 预计时间 |
|------|---------|
| A: Frozen Baseline | 2 分钟 |
| B: Full Fine-tune | 5-8 小时 |
| C: LoRA r=16 | 3-5 小时 |
| D: Dual-Mapping | 2-4 小时 |
| E: Backbone Only | 2-4 小时 |
| F: z_scaling × 4 | 8-16 小时 |

### 5.2 逐项运行（需要调试时）

完整命令见 `RUN.md` 第五章。

---

## 六、一键脚本：本地到服务器的完整自动化

如果经常需要部署到新服务器，保存以下脚本为 `deploy.ps1`：

```powershell
# deploy.ps1 - 完整部署到新服务器
param(
    [string]$Server = "root@your-server-ip",
    [int]$Port = 12345
)

$LocalCode = "D:\Amy_files\Lab\AI\lab01"
$Deliverable = "D:\Amy_files\Lab\AI\lab01\dual_mapping_deliverable"

Write-Host "1. 打包代码..."
Set-Location $LocalCode
tar czf dual_mapping_code.tar.gz --exclude='__pycache__' dual_mapping_deliverable

Write-Host "2. 上传..."
scp -P $Port dual_mapping_code.tar.gz ${Server}:/root/
scp -P $Port "$Deliverable\..\mappingNetwork\data\weights\rtdetr_r18.pth" ${Server}:/root/
scp -P $Port "$Deliverable\..\mappingNetwork\data\weights\rtdetr_r18vd_coco.pth" ${Server}:/root/
scp -P $Port "$Deliverable\..\mappingNetwork\data\voc\VOCtrainval.tar" ${Server}:/root/

Write-Host "3. 服务器解压..."
ssh -p $Port $Server @"
cd /root && tar xzf dual_mapping_code.tar.gz && rm dual_mapping_code.tar.gz
cd dual_mapping_deliverable && mkdir -p data/weights
mv /root/rtdetr_r18.pth data/weights/
mv /root/rtdetr_r18vd_coco.pth data/weights/
mv /root/VOCtrainval.tar data/ && tar xf data/VOCtrainval.tar -C data/
pip install tqdm scipy peft -q
echo 'Deploy complete. Run: cd /root/dual_mapping_deliverable && bash run_experiments.sh'
"@

Write-Host "Done!"
```

使用方式：
```powershell
.\deploy.ps1 -Server "root@connect.westc.seetacloud.com" -Port 24759
```

---

## 七、常见问题

| 问题 | 解决 |
|------|------|
| `Permission denied (publickey)` | SSH key 未配置，用密码登录或添加 key |
| `Connection refused` | 检查 IP 和端口是否正确 |
| `No such file or directory` | 确认 `cd /root/dual_mapping_deliverable` 正确 |
| `No module named 'peft'` | `pip install peft` |
| `CUDA out of memory` | 减小 batch size: `--batch_size 4` |
| `ModuleNotFoundError: No module named 'src'` | 确认 `RT-DETR-main/` 目录存在 |
| 权重文件上传中断 | 用 `rsync -avz -e "ssh -p PORT"` 替代 scp（支持断点续传） |
