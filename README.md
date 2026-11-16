# OpenBCI 4CH sEMG · 石头/布与石头/剪刀/布识别

基于OpenBCI前四通道表面肌电的手势识别项目。当前主线为**石头/布二分类**，保留此前完整的三分类SVM流程与实时试验版本。覆盖采集提示与UDP Marker、原始TXT预处理、特征提取、分组验证、当天校准、连续回放、LSL实时识别和预测后的人工标注。

默认实时引擎为`classic`：先确认放松，自由做动作，由固定SVM判断，输出后再点击实际动作。`extended`保留固定4秒按钮识别等对照能力。

源码可以直接提交GitHub。数据、训练模型和运行结果放在本地`data/`、`models/`、`outputs/`，由`.gitignore`排除。本仓库没有预装受试者模型；需要自己的记录或已有配套模型目录。

## 快速开始

在项目根目录打开PowerShell，沿用已有`bci`环境：

```powershell
conda activate bci
python -m pip install -r requirements.txt
python run.py check --help-checks
```

PsychoPy仅用于采集提示，有现成可用的采集环境可继续使用。其安装说明见`requirements-collection.txt`与[PsychoPy官方文档](https://psychopy.org/download.html)。分析/实时入口的代码语法支持Python 3.9及以上；实际数值测试环境记录在`requirements-tested.txt`与`docs/VALIDATION.md`。已有joblib模型应在与训练兼容的数值库环境中使用。

## 每天怎样运行

### 1．采集一组当天数据

```powershell
python run.py collect --mode formal --subject USER01 --session DAY01 --output-root outputs/collection
```

正式协议：4 Run，每Run石头6次、布6次，共48次。原始采样由OpenBCI GUI保存为TXT；提示程序的事件CSV不代替原始TXT。配置为`configs/rp_collection.json`，也可以显式传入`--config configs/rp_collection.json`。首次开采前核对GUI Marker输入端口与配置一致。

### 2．首次建立二分类基础模型并校准

历史输入可以包含旧三分类TXT：保留石头11与布13，剪刀12单独审计，不充当静息。当天TXT必须晚于历史训练日期。

```powershell
python run.py init --history-input "data/history/day01.txt" "data/history/day02.txt" --today-input "data/today/day03.txt" --subject USER01 --output outputs/RP_BINARY_NEW_DAY
```

也支持`--history-data`读取旧三分类/二分类预处理目录；`--legacy-base`只从旧三分类模型旁的`training_config.json`定位历史数据，不使用旧三分类SVM权重。

### 3．以后复用二分类基础模型

```powershell
python run.py day --base-model "models/base_model.joblib" --today-input "data/today/new_day.txt" --subject USER01 --output outputs/RP_BINARY_NEW_DAY
```

路径为示例，请改成自己的真实路径。基础模型旁的`training_config.json`以及它引用的预处理目录也必须保留。`init/day`会串联预处理、校准、源数据门控准备、Run3+4回放，并在终端打印完整输出目录。

当天分工：Run1初步拟合，Run2选校准与门控参数；Run1+2拟合最终当天模型；Run3+4用于独立测试。不能看到Run3+4结果后反复改参数，再把同一段成绩作为新的独立验证。

### 4．实时自动识别

在GUI Networking中启用LSL，Data Type=`TimeSeriesRaw`，Name=`rps_raw`；流可以有8通道，程序默认读取前4通道，250 Hz，输入单位μV。

```powershell
$runDir = Read-Host "粘贴刚完成的输出目录（里面有pipeline_state.json）"
python run.py live --run "$runDir" --engine classic --activity-mode legacy --gate-profile source --output outputs/LIVE_CLASSIC
```

启动后先放松，等程序允许下一次动作；做一次手势，输出后点击实际动作。没有输出的动作可手动记录漏检。同一次动作不要重复计数。标签仅用于统计，不更新SVM，也不重置门控。

### 5．诊断信号与等待原因

先停止并保存实时记录：

```powershell
python run.py diagnose --logs outputs
```

会从原始输入独立复算RMS与legacy活动分数，检查分析窗口年龄，生成`diagnose_report.json`、`window_rms_check.csv`和HTML报告。LSL年龄不是肌肉动作到输出的完整延迟；GUI上游等待和真实静息误触发需要额外时间/状态标注。

## 两个实时引擎

| 引擎 | 入口文件 | 用途 |
|---|---|---|
| `classic`（默认） | `binary/07_live_rp_binary_label_4ch.py`，1.0.1-rp-classic-tune | 当前主线；原06自动门控流程的二分类适配，支持配套门控与显式试验覆盖 |
| `extended` | `binary/07_live_rp_binary_extended_4ch.py`，1.4.0-rp-button | 当前静息门控、响应配置及固定时长按钮识别等可选对照 |

按钮对照示例：

```powershell
python run.py live --run "$runDir" --engine extended --trigger-mode manual --manual-duration-seconds 4 --output outputs/LIVE_BUTTON
```

点击“开始”后再做动作，固定采样结束后分类，最后标注真实动作。未提前选择石头或布。

`classic --gate-profile trial`包含ON=7.2、OFF=7.0、等待0.2秒、5窗4票、5窗释放。这只是根据少量历史截图设置的临时试验值，不能视为个体通用阈值。新日期校准后优先使用`source`。活动值决定何时允许输出，SVM决定类别；活动高低不直接映射成石头/布。详见`docs/GATING.md`。

## 更新记录

这三个二分类实时文件的历次更新见[CHANGELOG.md](CHANGELOG.md)和[详细版本历史](docs/BINARY_VERSION_HISTORY.md)，包括源码快照、逐行差异、SHA256及验证范围。

## 项目文件

| 位置 | 内容 |
|---|---|
| `run.py` | 统一入口，默认二分类 |
| `binary/00_*.py` | 二分类PsychoPy采集与Marker |
| `binary/01_*.py` | 预处理、因果滤波、窗口、QC、20维特征 |
| `binary/02_*.py` | 按日期/Session/Run分组训练基础模型 |
| `binary/03_*.py` | 当天校准与独立测试 |
| `binary/07_*.py` | 两个明确区分的实时引擎，均支持事后标注 |
| `binary/run_rp_binary.py` | 自动串联基础训练/当天校准、回放、按引擎转发实时参数 |
| `rps/` | 原三分类采集、01—08、当天流水线、原有测试和说明 |
| `tools/` | RMS诊断、版本比较、源码检查 |
| `archive/binary/` | 1.0.0—1.4.0、classic1.0.1和整合入口的历史快照；详见版本记录 |
| `configs/` | 二分类采集JSON |
| `tests/` | 入口兼容、门控、标签隔离、合成数据端到端测试 |
| `docs/` | 协议、每日流程、门控、数据流、验证、GitHub上传说明 |

## 三分类原流程

```powershell
python run.py rps --help
python run.py rps collect --mode formal --subject USER01 --session RPS01
python run.py rps preprocess --input "data/history/rps_day01.txt" --subject USER01 --output outputs/RPS_PREPROCESS
python run.py rps today-only --help
```

三分类正式协议为4 Run×18 Trial=72次；08为三分类当天数据单独训练的可选实验。二分类与三分类模型/schema不能混用。原三分类逐步命令见`rps/README.md`。

## 结果与局限

本项目仍处于实时稳定性验证阶段，不能因为某条旧记录离线/回放达到较高成绩，就宣称持续实时准确率相同。历史汇总与评估口径见`docs/RESULTS.md`。CI和合成信号测试只验证软件行为，不替代真实受试者/板卡验证。

## 检查与上传

```powershell
python -m unittest discover -s tests -v
python -m unittest discover -s rps/tests -v
python run.py check --verify-snapshot
```

源码清单与SHA见`SOURCE_MANIFEST.json`，来源快照见`SOURCE_PROVENANCE.json`。源码修改后，初始分发哈希校验自然会变化；日常开发以Git记录为准。上传步骤见`docs/GITHUB_UPLOAD.md`。

许可证由仓库维护者选择，本次整理未代替作者授予新的开源许可；第三方库遵守其自身许可。
