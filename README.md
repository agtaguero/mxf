# MotrixSim 社区版上的 LIBERO π0.5 评测

项目将 LIBERO Spatial、Object、Goal 和 Long（`libero_10`）的 40 项任务移植到 MotrixSim 0.10.1。环境使用原始碰撞几何、原生任务谓词、π0.5 的外部与腕部原始 RGB 输入和 7 维动作。模型、分词器和归一化数据复用相邻 sim-platform 的冻结安装。MuJoCo 2.3.7 和 robosuite 1.4.0 用于原生导出和对照。

正式发布目录为 `results/libero-pi05/`。主报告 `report.md` 包含逐套件和逐任务成绩；`episodes.jsonl` 包含逐回合结果、配对初态和策略输入、动作摘要；`results.json` 包含协议和策略调用计数。修复与留出验证见[一致性记录](results/libero-pi05/validation.md)，早期迭代见[历史记录](results/libero-pi05/validation-history.md)。

## 复现完整评测

需要已准备好标准 LIBERO 参考与 π0.5 的相邻 sim-platform 工作区，以及支持 Vulkan 的渲染设备和 CUDA 策略设备。所有命令从本仓库根目录运行。工作目录使用新的空路径，各套件独立初始化随机数。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-motrix.txt
```

依次导出四个原生套件，保留官方编号与种子对应的家具位姿、OSC 复位信息和原生编译参数。导出不需要运行策略或渲染。

```bash
work=.cache/libero-reproduction
for suite in libero_spatial libero_object libero_goal libero_10; do
    ../sim-platform/.runtimes/envs/libero/bin/python -m mxf.eval --engine mujoco \
        --suite "$suite" --tasks {0..9} --episodes 50 --export-only \
        --output "$work/scenes/$suite"
done
```

共享工作站上可先顺序运行套件。独立套件并行时应先测量主机可用内存；错峰加载模型可减少启动峰值。单 GPU 机器可把四个套件顺序传给同一次调用。每个套件内严格按任务、初态、动作顺序使用同一条策略随机流。默认每 3 个回合退出一次仿真与图形进程，使操作系统回收原生分配；`MXF_EPISODES_PER_PROCESS` 可调整批大小。π0.5 进程、推理队列和受序号校验的策略连接贯穿整个套件。关闭录像仍会真实渲染策略需要的图像。

```bash
bash scripts/libero-campaign.sh "$work/scenes" "$work/scored" 1 libero_object libero_spatial libero_goal libero_10
```

在四套件都结束后，将 2,000 个结果逐例与原生参考核对，向新的目录发布正式结果。

```bash
reference=../sim-platform/.local-data/validation/libero-vla-default-20261118/pi05-full
.venv/bin/python -m mxf.full_report "$work/scored" "$work/scenes" "$reference" "$work/published"
```

## 单项诊断

用 `mxf.eval --engine mujoco` 的非精简模式生成完整原生动作与场景，再调用 `mxf.eval --engine motrixsim --replay --no-video` 可以验证同一初态的固定动作。使用 `mxf.render_compare` 可在同状态、无物理推进时比较两台相机。以下示例用官方初态 11 检查方向色卡。

```bash
scene="$work/scenes/libero_spatial/task-00"
MUJOCO_GL=egl ../sim-platform/.runtimes/envs/libero/bin/python -m mxf.render_compare \
    native "$scene" "$work/uv-native" --state 11 --chart
WGPU_BACKEND=vulkan .venv/bin/python -m mxf.render_compare \
    motrix "$scene" "$work/uv-motrix" --state 11 --chart
```

适配器根据模型属性处理惯量组、自由关节阻尼、介质力、碰撞材质与纹理映射。早期针对灶台的几何实验仍可从诊断代码复现；正式入口默认使用原始几何。逐物理步轨迹、动作回放、标定图和模型缓存均在运行工作目录生成，发布结果无需这些大文件。
