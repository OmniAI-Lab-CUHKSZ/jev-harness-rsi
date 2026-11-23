# Jev Harness RSI

Sol 从 AppWorld 的 Train 轨迹中改进一个公共 harness，使固定的 27B 模型在新场景上尽可能用 Jev 选择完成任务，只在需要时生成代码。程序根据当前状态、可执行片段和接口生成问题与候选。Jev 也参与菜单组装、动作选择和参数绑定。实验先比较整题成功数，成功数相同时比较在线生成 token。

This repository contains the Phase 1 AppWorld runner, the evolving shared harness, the Sol optimization workbench, and the exact starting harness and task IDs for the official adaptive 27B experiment.

## 快速开始

在仓库根目录运行以下命令。主控使用 Python 3.10 或更新版本。预览和单元测试可以在普通 CPU 机器上运行。

```sh
python -m pip install -e ".[test]"
jev-appworld plan
python -m pytest -q
```

`plan` 输出全部任务 ID、执行与优化预算、完整 H0。也可以直接运行 `python -m jev_harness plan`。固定配置与输入可以单独查看。[实验配置](jev_harness/resources/experiment.json) · [H0](jev_harness/resources/h00.json) · [逐题划分](jev_harness/resources/task_split.json)

## 主实验

| 项目 | 正式设置 |
|---|---|
| 在线执行 | 同一 BF16 AutoTrust-Decider-Pro-27B。决策使用发布方的适配器与 `/v1/decide`，开放生成使用同一底模 |
| 低置信推理 | 对“下一步选哪个片段”，初选最高概率低于 0.60 时，最多思考 256 个 token，再读取每个候选的概率。按初选与推理后分布各占一半选取最终动作。窄菜单初选 NONE 时先展开 |
| 离线优化 | GPT-6.2 Sol，推理强度 max。每批 3 个场景、9 题，至多 2 轮。每次补丁编辑 1–3 个命名对象，范围涵盖执行适配、片段、状态、菜单组装、决策和生成交接 |
| Train / Dev / Test-N | 90 / 57 / 90 题，以场景隔离。Train 共 10 批。Test-N 是预选的 30 个 Normal 场景，每个场景 3 题 |
| 选版 | 先比成功数，再比生成 token，两项相同保留候选。H0 在本次配置下实测 Dev，后续复用已测父版的逐题记录 |
| 最终对照 | 同一 Test-N 题在同一张卡、各自独立的 AppWorld 初始状态下，配对比较普通生成 agent 与最终公共 harness |
| 成本 | 整题最多生成 8,192 个 token。主成本计入开放生成、低置信思考及已记录重试的输出。决策读分 token、服务请求、输入量和耗时另记 |

固定的 15 个 H0 片段带有历史 Train 来源，清单列出涉及的 30 道原始 Train 题。正式优化从 H0 开始。具体组件接口、Sol 的轨迹访问范围、试补丁步骤和 Train 示例见[实验设计](docs/experiment.md)。

## 准备真实运行环境

**模型端。** 实际运行使用 Linux、NVIDIA A800 80 GB、Python 3.10、vLLM 0.31.0、PyTorch 2.13.0、Transformers 5.17.0。每张卡加载一个完整的模型副本，各卡并行跑不同题。准备兼容的 NVIDIA 驱动、CUDA 编译工具和对应 Python 的开发头文件，然后在模型端执行下列 Bash 命令。完整模型约 54 GB，下载目录同时提供服务脚本、校准数据与适配器。

```sh
python3.10 -m venv .venv-model
.venv-model/bin/python -m pip install "vllm==0.31.0" "transformers==5.17.0" "huggingface_hub==1.33.0"
export JEV_MODEL_PYTHON="$PWD/.venv-model/bin/python"
export JEV_MODEL_DIR="$PWD/models/jev-27b"
.venv-model/bin/hf download autotrust/AutoTrust-Decider-Pro-27B \
  --revision 14b425269ed24452e0a5ad5a8f3e51d139647a88 --local-dir "$JEV_MODEL_DIR"
```

**AppWorld 端。** 正式环境使用 Python 3.12、AppWorld 0.2.0.dev0、数据 0.2.0、SQLAlchemy 2.1.6 和 SQLModel 0.0.48。独立安装模拟器可以保留它和模型端各自的依赖版本。仓库中的评分适配器保留了这个版本的 SQLite 内存库识别与评分异常处理。

```sh
python3.12 -m venv .venv-appworld
export APPWORLD_PYTHON="$PWD/.venv-appworld/bin/python"
export APPWORLD_ROOT="$PWD/appworld-data"
"$APPWORLD_PYTHON" -m pip install \
  "appworld @ git+https://github.com/stonybrooknlp/appworld.git@42b5bcf3cd334fee33f0c37c02070a9f5807add5" \
  "sqlalchemy==2.1.6" "sqlmodel==0.0.48"
"$APPWORLD_PYTHON" -m appworld.cli install
"$APPWORLD_PYTHON" -m appworld.cli download data --version 0.2.0 --mode minimal
```

模型与基准数据从发布方获取，各自的许可和使用说明见上游页面。[AutoTrust 发布版本](https://huggingface.co/autotrust/AutoTrust-Decider-Pro-27B/tree/14b425269ed24452e0a5ad5a8f3e51d139647a88) · [AppWorld 对应源码](https://github.com/StonyBrookNLP/appworld/tree/42b5bcf3cd334fee33f0c37c02070a9f5807add5)

**Sol 端。** 正式演化使用主控机已登录的 Codex CLI，账号需能调用 `gpt-6.2-sol` 并使用 max 推理强度。工作台调用 Codex 的 `app-server`，将本轮允许查看的 Train 信息发给 Sol，再由本地程序执行返回的修改。默认使用当前账号配置的提供方。如果使用与正式实验相同的已配置提供方别名，可设置 `JEV_SOL_PROVIDER=openai_http`。启动器检查实际返回的模型及推理强度。单题试跑只使用执行模型和 AppWorld。

## 试跑、正式启动与续跑

在同一台 Linux 机器部署时，沿用前面导出的四个环境变量。`PACKAGE_DIR` 由主控定位到安装的包目录。

```sh
jev-appworld smoke --connections configs/connections.local.example.json \
  --root runs/smoke --gpu local:0 --task 6ea6792_2

jev-appworld run --connections configs/connections.local.example.json \
  --root runs/main --gpus local:0,local:1,local:2,local:3
```

首次试跑会启动真实的官方决策服务，读取题目、执行操作并请求 AppWorld 判分。`runs/smoke/smoke_result.json` 中的 `success` 表示整题是否做对。正式运行从第一批九题开始。同一命令和输出目录可续跑，控制器读取已完成的逐题记录、补丁和批次选择。单个正式输出目录同时使用一个主控进程。

多节点部署时，将仓库放在模型和模拟器机器上，使用 SSH 模板。模板中的 Python、仓库、模型及数据路径均指相应远端机器，SSH 密钥路径指主控机。可在 `model_nodes` 下增加节点，或设置独立的 SSH 端口。自动分卡每 60 秒有机会重新扫描，默认最多 4 张 A800，候选需至少 58,000 MiB 可用显存且利用率不高于 30%。指定 `--gpus` 则使用给定设备。自动分配其他机型时可指定 `--gpu-name`。[多节点配置模板](configs/connections.ssh.example.json)

```sh
jev-appworld run --connections configs/connections.ssh.example.json \
  --root runs/main --max-gpus 4
```

## 去哪里看代码和结果

| 内容 | 位置 |
|---|---|
| 逐题实验流程、父子版比较、Dev 选版、Test-N 配对 | [phase1_run.py](phase1_run.py)、[formal_run.py](formal_run.py) |
| 在线程序组装、Jev 选择、开放生成与重入 | [semantic_assembly.py](jev_harness/semantic_assembly.py)、[runtime.py](jev_harness/runtime.py)、[formal_runtime.py](jev_harness/formal_runtime.py) |
| 六类可修改对象、补丁安装和可执行逻辑 | [semantic_library.py](jev_harness/semantic_library.py)、[card_library.py](jev_harness/card_library.py)、[policy_programs.py](jev_harness/policy_programs.py) |
| Sol 提示、完整轨迹索引、局部试验与历史影响预览 | [phase1_optimizer.py](jev_harness/phase1_optimizer.py)、[semantic_workbench.py](jev_harness/semantic_workbench.py)、[fragment_trials.py](jev_harness/fragment_trials.py)、[history_workbench.py](jev_harness/history_workbench.py) |
| 模型端、模拟器端、设备分配 | [official_model_worker.py](jev_harness/official_model_worker.py)、[env_worker.py](jev_harness/env_worker.py)、[allocation.py](jev_harness/allocation.py) |
| 任务记录、已安装版本与选版结果 | 运行目录中的 `runs/<stage>/<task>/record.json`、`versions/`、`checkpoints.json` |
| Sol 实际收到的提示、返回和调用用量 | `batches/<batch>/<revision>/sol*/turn_*/` |
| 最终两臂成绩与配对区间 | `test_report.json`，执行机制统计在 `test_mechanisms.json` |

源码包提供完整 H0、正式协议和测试。运行目录中的轨迹可能包含模拟账号的会话值，Git 默认忽略运行目录、模型、数据和本机连接配置。
