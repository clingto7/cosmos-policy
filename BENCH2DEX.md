# Bench2Dex 中的 Cosmos Policy

将本仓库源码放在 Bench2Dex 的 `policy/cosmos_policy/` 后，可以从 Bench2Dex 根目录调用 `bridge.py`。训练、数据读取、检查点恢复和策略推理服务都在本目录；闭环测评调用同一 Bench2Dex 仓库的 `run_policy.py`，并使用独立的仿真 Python 环境。数据、基础模型和检查点由命令行路径传入，不放入源码目录。

复制 `policy/cosmos_policy/configs/local.example.json` 为 `configs/local.json`，填入 Cosmos 和 Isaac Sim 环境的 Python 路径。该文件已被 Git 忽略。训练只需要 `cosmos_python`，测评还需要 `sim_python`。

在 Bench2Dex 根目录运行以下命令。默认训练命令只打印计划，加入 `--execute` 才启动或恢复训练；`--print-command` 则只显示桥接器准备执行的命令。

```sh
python3 policy/cosmos_policy/bridge.py --config policy/cosmos_policy/configs/local.json \
  train --manifest /path/to/embodiment-bundle.json \
  --robot-map robots/active_dof_maps.yml \
  --base-model /path/to/model-480p-16fps.pt --tokenizer /path/to/tokenizer.pth \
  --output /path/to/private/runs --run-name example --expected-tasks 3
```

```sh
python3 policy/cosmos_policy/bridge.py --config policy/cosmos_policy/configs/local.json \
  eval --task-id 26 --checkpoint-id example-32k \
  --manifest /path/to/task26-manifest.json --embedding /path/to/task26-prompt.npz \
  --embodiment-manifest /path/to/embodiment-bundle.json \
  --checkpoint /path/to/checkpoints/iter_000032000 \
  --base-model /path/to/model-480p-16fps.pt --tokenizer /path/to/tokenizer.pth \
  --scene scenes/26_canned_food_tray_line_arrangement.yaml \
  --anchor-dir /path/to/task26/replay-generalization \
  --robot-map robots/active_dof_maps.yml --output /path/to/private/evaluations/task26
```

实际场景文件名和路径以任务数据为准。测评前会核对场景描述与数据审计记录；未登记的关节不会被静默补零。`--render-gpu N` 可在多卡实例上指定物理 Vulkan 渲染 GPU，仿真物理仍使用 CPU。

源码可以作为没有独立 `.git` 的目录放进 Bench2Dex。训练身份记录 `cosmos_policy/**/*.py` 的 SHA256；完整测评协议另记录测评脚本和场景的 SHA256。只有本目录自身是 Git 工作树时才记录 Cosmos 的 Git 提交，避免误记外层 Bench2Dex 的提交。已有运行以各自冻结的身份与检查点为准，不受新目录部署影响。

准备数据、训练预算和四通道测评的细节分别见 [按本体训练流程](docs/experiments/per-embodiment-training.md)与[最终测评流程](docs/experiments/bench2dex-final-evaluation.md)。
