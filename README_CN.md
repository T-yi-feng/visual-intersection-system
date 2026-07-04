# 路口车辆检测与拥堵分析系统（OpenCV + YOLO）
# 需要注意的是：由于代码文件太大，将训练集-OBB删除了，可以后续添加其他的YOLO模型，运行的时候是可以切换模型的。
本项目用于道路交通视频的智能分析，支持从视频输入到可视化输出的一体化流程，包含：
- 车辆检测与跟踪
- 透视标定与 BEV（俯视）映射
- 车辆朝向/轨迹/交织关系分析
- 实时综合拥堵指数时间线
- 峰值事件与分级消融分析
- Web 端实时监看（Streamlit）

适用对象：课程项目、创新创业比赛、科研实验、交通场景演示。

## 1. 主要功能

### 1.1 核心算法功能
- 多类别车辆检测与跟踪（小型汽车、大型客车、货车等）
- 类别自适应置信度阈值过滤（按类别分别设置阈值）
- 基于单应矩阵的像素坐标到世界平面映射
- 三窗口联动展示（原视频窗口、变换窗口、综合分析窗口）
- 实时输出综合拥堵指数（Phi）
- 支持事件峰值定位与消融实验导出（图表、CSV、JSON）

### 1.2 工程与交付功能
- 支持按路口管理配置（`configs/intersections.json`）
- 支持交互式路口与视频选择（`test0.py`）
- 支持无窗口模式批处理（`--show-windows 0`）
- 支持 Web 可视化控制台（`app/smart_console.py`）
- 统一输出目录结构，方便复核与报告引用

## 2. 目录说明

关键目录如下：
- `src/`：核心算法脚本（检测、BEV、风险分析）
- `app/`：Web 控制台（Streamlit）
- `configs/`：路口配置、标定文件、风险参数
- `scripts/`：启动与辅助脚本（重启 Web、点选标定等）
- `mp4/`、`mp4_huangshanlu/`、`mp4_yongdukou/`：视频输入目录
- `outputs/`：所有分析结果输出目录
- `test0.py`：主启动器（命令行/交互式）
- `StartApp.bat`：桌面化一键启动算法
- `run_software.bat`：一键启动 Web 控制台，或标定工具

## 3. 运行环境要求

### 3.1 操作系统
- 推荐：Windows 10/11（项目内 `.bat` 与 PowerShell 脚本已适配）
- 其他系统：Linux/macOS 可用 Python 命令运行核心脚本，但批处理脚本需要自行替换

### 3.2 Python 版本
- 推荐：Python 3.10 或 3.11

### 3.3 硬件建议
- CPU：4 核及以上
- 内存：16GB 及以上
- GPU（可选，强烈建议）：NVIDIA 显卡，安装对应 CUDA 的 PyTorch 可明显提速

### 3.4 依赖库
项目依赖见 `requirements.txt`，主要包括：
- opencv-python
- ultralytics
- numpy
- shapely
- scipy
- matplotlib
- pillow
- pandas
- streamlit
- streamlit-autorefresh

## 4. 环境配置步骤（给其他人复现）

以下步骤可直接提供给任何新同学使用。

### 4.1 创建虚拟环境

PowerShell 示例：

```powershell
cd 项目根目录
python -m venv .venv_project
.\.venv_project\Scripts\Activate.ps1
```

如果提示执行策略限制，可先执行：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

### 4.2 安装依赖

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4.3 GPU 环境（可选）

如需启用 GPU，请确保：
- 安装 NVIDIA 驱动
- 安装与驱动匹配的 CUDA 版本
- 安装 CUDA 版 PyTorch（不是 CPU-only 版）

安装后可用以下命令检查：

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 5. 如何运行软件

本项目支持三种运行方式。

### 5.1 方式 A：一键启动算法（最简单）

双击：
- `StartApp.bat`

功能：自动寻找 Python 运行时并启动 `test0.py`。

### 5.2 方式 B：命令行启动主程序（推荐开发/调试）

```powershell
python test0.py
```

运行后会进入交互选择：
- 选择路口
- 选择视频

也支持直接指定：

```powershell
python test0.py 路径\to\video.mp4 default
```

或：

```powershell
python test0.py default
```

### 5.3 方式 C：启动 Web 控制台

双击：
- `run_software.bat`

功能：
- 默认启动 Smart Console（Streamlit，默认地址 `http://127.0.0.1:8501`）
- 自动重启旧进程并打开浏览器

若需要快速进入点选标定工具：

```powershell
run_software.bat calib
```

## 6. 输入数据与配置要求

### 6.1 视频输入
- 将视频放入对应路口的 `video_dir`（在 `configs/intersections.json` 中配置）
- 支持后缀：`.mp4`、`.avi`、`.mov`、`.mkv`、`.wmv`

### 6.2 路口配置
文件：`configs/intersections.json`

每个路口包含：
- `display_name`：显示名称
- `video_dir`：视频目录
- `calibration_image`：标定底图
- `homography`：单应矩阵配置路径
- `risk_params`：风险参数配置路径
- `runtime`：该路口默认运行参数

### 6.3 单应矩阵配置
文件格式必须包含：
- `image_points`
- `world_points_m`

点顺序建议固定为：
- 左上
- 右上
- 右下
- 左下

可用点选工具生成：

```powershell
python scripts\calibrate_homography_click.py
```

## 7. 输出结果说明

每次运行会在统一目录下生成结果（默认在 `outputs/unified/...`），主要结构：
- `videos/`：输出视频（如开启保存）
- `events/`：峰值事件分析、消融图、CSV、JSON
- `charts/`：Phi 时间线图等
- `preview_frames/`：阈值快照等预览帧
- `live/`：Web 实时预览帧与实时指标文件

其中 `live/` 常见文件：
- `live_metrics.json`
- `live_phi_timeline.csv`

## 8. 常用参数（命令行）

主参数由 `test0.py` 传递到 `src/detect_and_bev.py`。常用项：
- `--model`：模型路径，如 `yolo11m.pt`
- `--imgsz`：推理尺寸
- `--conf` / `--iou`：检测阈值
- `--conf-car` / `--conf-bus` / `--conf-truck` / `--conf-van`：类别阈值
- `--frame-stride`：跳帧步长（越大越省算力）
- `--target-process-fps`：目标处理帧率
- `--third-panel-mode`：第三窗口刷新模式（`quality` / `balanced` / `speed`）
- `--show-windows`：1 显示窗口，0 无界面运行
- `--ablation-enable`：是否导出消融结果

示例：

```powershell
python test0.py default --third-panel-mode balanced --output-root outputs\demo_run
```

## 9. 新路口接入步骤

1. 新建视频目录并放入样例视频
2. 准备该路口标定底图
3. 使用 `scripts/calibrate_homography_click.py` 生成单应配置
4. 在 `configs/intersections.json` 新增一个 site 节点
5. 配置该 site 的 `risk_params` 与 `runtime`
6. 运行 `python test0.py` 选择新路口验证

## 10. 常见问题排查

### 10.1 启动时报缺少模块
- 现象：`missing required modules`
- 处理：激活正确虚拟环境并执行 `pip install -r requirements.txt`

### 10.2 打不开 Web 页面
- 检查 `http://127.0.0.1:8501`
- 查看日志：`outputs/web_preview/streamlit_stdout.log` 与 `streamlit_stderr.log`
- 先执行 `run_software.bat` 重启服务

### 10.3 视频无法读取
- 确认视频路径存在且后缀受支持
- 确认 `configs/intersections.json` 中 `video_dir` 配置正确

### 10.4 标定配置无效
- 确认 `homography` 文件存在
- 确认含有 `image_points` 与 `world_points_m`
- 确认点数量一致且不少于 4

### 10.5 速度慢
- 使用 GPU
- 增大 `frame-stride`
- 减小 `imgsz`
- 设置 `--third-panel-mode speed`

## 11. 对外分发建议（给其他同学/评委）

分发时建议包含：
- 项目源码
- `requirements.txt`
- 至少一个可运行视频
- 已校准好的 `homography` 配置
- 对应路口风险参数文件
- 本 README

首次上手者可按本 README 的 4 到 6 节完成部署并运行。