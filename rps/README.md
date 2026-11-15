# OpenBCI-RPS-sEMG

这是整合工程中的原三分类子目录。下文直接调用01—08的命令应在`rps/`内运行；也可以在工程根目录使用`python run.py rps --help`查看统一入口。当前整体分发的验证记录与哈希清单见根目录`docs/VALIDATION.md`及`SOURCE_MANIFEST.json`。

基于 OpenBCI 前四通道表面肌电的石头、剪刀、布识别实验。包含采集提示与 UDP Marker、离线预处理、SVM 基础训练、当天校准、跨 Session 评估、连续回放、LSL 实时识别及识别后的人工标注。

本仓库保存程序与示例配置。原始受试者记录、特征、训练模型和实验输出保留在本地，由 `.gitignore` 排除。首次使用需要自己的采集文件；下载源码不会同时获得可直接推理的训练模型。

## 程序入口

| 文件 | 作用 |
|---|---|
| `collection/openbci_rps_collection.py` | PsychoPy 提示、平衡随机动作顺序及发送 UDP Marker；原始采样由 OpenBCI GUI 保存 |
| `collection/config_rps_emg.json` | 当前 4 Run 正式采集示例配置 |
| `01_preprocess_openbci_rps_4ch.py` | Marker 分段、因果滤波、窗口、质检与 20 维特征 |
| `02_train_svm_openbci_rps_4ch.py` | 历史数据训练 StandardScaler＋SVM 基础模型 |
| `03_calibrate_svm_openbci_rps_4ch.py` | 历史数据与当天 Run1+2 校准，Run3+4 独立测试 |
| `04_evaluate_svm_cross_session_4ch.py` | 冻结第一份模型，评估第二份 Session |
| `05_replay_svm_openbci_rps_4ch.py` | 因果窗口连续回放、活动门控、单次触发与事后评估 |
| `06_live_svm_openbci_rps_4ch.py` | 准备匹配门控、基础 LSL 实时接收和显示 |
| `07_live_svm_rps_label_4ch.py` | 实时识别后点击实际动作；记录正确率、错误和手动漏检；可选稳健活动门控 |
| `08_today_only_svm_rps_4ch.py` | 仅用当天三分类数据训练、独立测试、准备门控和实时启动的可选流程 |
| `run_daily_calibration.py` | 自动串联 01→03→06，打印匹配当天模型的 07 启动命令 |
| `tools/check_repository.py` | 检查源码、来源哈希、采集配置和各入口帮助命令 |
| `tests/test_rest_gate.py` | 不依赖受试者数据、模型或设备的合成信号/界面回调测试 |

01—07 保留当前项目已有的脚本名称与算法。源码快照校验见 `SOURCE_SNAPSHOT.json`。

## 实验约定

- 250 Hz；仅前四个 EXG 通道；输入单位默认微伏。
- 正式采集 4 Run × 18 Trial＝72 Trial；每个 Run 三类各 6 次。
- Marker 11＝石头，12＝剪刀，13＝布，20＝动作结束，90＝作废。
- Run 开始 101—104，结束 111—114，中止 121—124。
- 原采集文件正式配置为 5 Run，保留在 `collection/config_rps_emg.original.json` 供追溯。当前示例只将 `protocol.formal.runs` 改为 4；采集脚本本身未修改。运行时须显式指定 `--mode formal`，默认 pilot 仍为 2 Run。
- 采集动作持续 4 秒；预处理默认取动作后 0.5—3.5 秒为核心段。
- 陷波 50 Hz、带通 20—100 Hz；使用有状态单向 SOS 滤波，实时与离线遵守同一参数约定。
- 窗长 200 ms，步长 100 ms；每通道 MAV、RMS、WL、ZC、SSC，共 20 维。
- 作废、异常与质量标记保留在清单中；是否可用于拟合由 usable/QC 标记与各阶段规则决定。

## 环境

分析和实时脚本语法支持 Python 3.9 及以上。新环境示例：

```powershell
conda create -n rps python=3.11
conda activate rps
python -m pip install -r requirements.txt
python tools/check_repository.py
python -m unittest discover -s tests -v
```

已有能正常使用的 `bci` 环境可以继续使用。已有 joblib 模型优先在与训练时相同的 numpy、scikit-learn、joblib 环境中运行，不应为了上传代码而升级依赖。

开发时的数值测试环境为 numpy 2.3.5、pandas 2.2.3、scipy 1.17.0、scikit-learn 1.8.0、joblib 1.5.3；原生 LSL 接口测试使用 pylsl 1.18.6。以上是已检查的环境记录，不代表其他环境或真实板卡已经验证。

PsychoPy 只用于采集提示，不是分析/实时识别的必装依赖。优先保留已有的采集环境，或使用 [PsychoPy 官方安装说明](https://psychopy.org/download.html) 中适合所选版本的独立环境。`requirements-collection.txt` 仅列出可选依赖。

## 采集

先在 OpenBCI GUI 中启动板卡和原始数据记录，并配置接收 UDP Marker 的端口，与配置文件的 `network` 一致。

不连接设备的配置自检：

```powershell
python collection/openbci_rps_collection.py --self-test
```

正式采集示例：

```powershell
python collection/openbci_rps_collection.py --config collection/config_rps_emg.json --mode formal --subject USER01 --session DAY01 --output-root outputs/collection
```

提示程序只记录事件和发送 Marker，不代替 OpenBCI GUI 保存脑电板采样。开始正式实验前，应先确认原始 TXT 中确实收到动作和 Run Marker。

## 历史数据与基础模型

将同一个受试者的历史原始 TXT 放到本地 `data/history`。若历史覆盖多个日期，可按日期划分：

```powershell
python 01_preprocess_openbci_rps_4ch.py --input data/history --subject USER01 --split-mode by-date --output outputs/HISTORY
```

终端会打印实际生成的预处理子目录。在下一步使用该目录：

```powershell
$historyData = Read-Host "粘贴刚生成的历史预处理子目录完整路径"
python 02_train_svm_openbci_rps_4ch.py --data "$historyData" --output outputs/BASE
```

保留基础模型及其配套配置和预处理数据。后续校准需要核对来源、哈希、日期与预处理参数，不能只随意移动一个模型文件后删除它依赖的历史数据。

## 当天校准

准备当天一份完整的 4 Run、72 Trial TXT。输入当前文件和原基础模型的位置：

```powershell
$todayTxt = Read-Host "粘贴当天 OpenBCI-RAW TXT 完整路径"
$baseModel = Read-Host "粘贴原 base_model.joblib 完整路径"
python run_daily_calibration.py --input "$todayTxt" --base-model "$baseModel" --subject USER01 --activity-mode legacy
```

该命令新建输出目录，依次完成：

1. 01 预处理当天数据，`--split-mode none` 保留后续按 Run 使用的划分信息。
2. 03 用历史数据和 Run1 拟合候选，Run2 选择当天权重；冻结选择后，使用历史数据与 Run1+2 拟合当天模型。
3. Run3+4 只用于独立测试，不用于选权重、训练或改参数。
4. 06 从匹配的源 Run1+2 准备连续检测门控。
5. 打印校准报告路径和使用同一模型、门控组合的实时识别命令；不会自动启动实时接收。

这条流程是“历史数据＋当天校准”，不是仅用当天数据重训。03 不支持直接省略历史基础模型。

## 连续回放与第二份数据

04 评估切段分类在另一份完整预处理数据上的效果。05 进一步按时间顺序回放整段记录，检查第一次输出是否正确、漏检、重复触发及释放情况。

```powershell
$dailyModel = Read-Host "粘贴已冻结的 daily_model.joblib 完整路径"
$targetData = Read-Host "粘贴另一份完整预处理子目录完整路径"
python 04_evaluate_svm_cross_session_4ch.py --model "$dailyModel" --target "$targetData" --output outputs/CROSS_SESSION
python 05_replay_svm_openbci_rps_4ch.py --model "$dailyModel" --target "$targetData" --evaluation-purpose new-data-check --output outputs/REPLAY
```

同一目标反复查看、调参后的结果应记为调试结果，不能继续称作全新独立测试。没有人工确认的静息真值时，程序不自动报告真实静息误触发率。

## 实时识别与事后标注

OpenBCI GUI Networking 设置 LSL、TimeSeriesRaw、流名 `rps_raw`。GUI 发送原始微伏数据，Python 自行按模型参数滤波；不要把已经滤波的流当作原始数据重复处理。

```powershell
python 07_live_svm_rps_label_4ch.py --list-streams
```

校准流程最后打印的命令包含模型与门控的完整路径，复制后启动即可。两者必须匹配，脚本会核对哈希和参数。

自由完成一次动作、出现识别输出后，点击“实际是石头／剪刀／布”。待标注事件按编号排队，新事件不会覆盖旧事件。没有输出时，可在无待标注事件的情况下手动记录一次漏检。每次动作后充分放松，等待释放后再做下一次。

标注仅用于事后评分，不参与模型训练或实时推理。已标注动作正确率＝正确／（正确＋错误＋手动漏检）；未标注输出不计入该分母，并单独显示。没有明确动作起止时间和全部静息真值，不能从按钮标注推导精确动作延迟或完整误触发率。

## 07 的两种活动门控

| 参数 | 行为 |
|---|---|
| `--activity-mode legacy` | 默认；沿用源 Run1+2 选择的固定活动门控，与旧流程对照 |
| `--activity-mode robust --rest-seconds 8 --min-active-channels 2` | 预热后测当前静息，每路用中位数与 MAD 估计尺度；默认至少两路持续增强才启动 |

稳健模式下先放松，测量通过且显示“已放松，等待动作”后再做手势。测量时不输出；参考在测量后冻结，不在动作期间自动更新。需要时先放松，再点“重新测静息”。该按钮不会训练 SVM，也不会清除已有人工标注。

稳健模式取第 k 强通道的标准化 RMS 偏离作为活动分数（默认 k＝2）；每路尺度为 `max(1.4826×MAD, 0.15×静息RMS中位数, 0.01 μV)`。默认启动/释放分数为 6/3，都是待实测比较的起始设置，并非已经在独立测试集上找到的最优参数。若某个手势只在一路明显增强，两路规则也可能增加漏检。

两种模式的活动分数刻度不同。活动条表示相对门限的位置，不是分类正确概率或识别完成进度。

## 日志与连接

每次实时运行新建子目录，保存 `raw_samples.csv`、`window_trace.csv`、`events.csv`、`action_labels.csv`、`label_accuracy_summary.json`、`label_confusion_matrix.csv`、配置和会话摘要。稳健模式另保存 `rest_calibration.json`，其中保留重测历史、每路参考和微伏门限。

CSV/JSON 快照被 Excel/WPS 等暂时占用时，程序保留待写内容、继续接收并重试；停止时必要时另存带 recovery 标记的文件。其他磁盘写入错误仍明确报错。

“重新同步LSL”只重建 Python 接收订阅、清理本地接收缓存并重新预热，不重连板卡，不能清空 GUI 或板卡上游队列。LSL 数据年龄和接收速率用于排查传输；不等于从板卡采样到屏幕显示的端到端延迟。改变门控算法不能修复上游数据积压。

## 已观察结果及边界

2026-11-03 的单人单次记录中，当天 Run3+4 共 36 次动作、每类 12 次：原基础模型 Trial BA 为 66.67%，历史数据＋当天校准模型为 91.67%（33/36），提升 25 个百分点。该数值来自该次离线切段分类测试，不代表连续实时识别率，也不是新 robust 门控的实测成绩。

开发环境对既有标注、文件锁、LSL 订阅和新门控共执行过 37 项检查，其中包括原生 pylsl 订阅测试；部分回归依赖本地合成夹具，未作为受试者数据上传。本仓库附带 7 项可以独立运行的合成/界面回调测试及入口检查。真实板卡、电极、跨日泛化和稳健门控的最终收益仍需独立实时验证。

## 上传 GitHub

具体步骤见 [docs/GITHUB_UPLOAD.md](docs/GITHUB_UPLOAD.md)。建议从这个干净目录初始化仓库；不要直接把包含全部采集数据和模型的实验工作目录整体上传。

当前未替维护者选择开源许可证；仓库可见性和代码复用许可由维护者决定。依赖项目遵守各自许可证。

