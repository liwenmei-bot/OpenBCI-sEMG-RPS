# 二分类逐步运行

以下命令均在仓库根目录PowerShell中运行。输入路径为示例，应换成真实文件；含空格和中文时保留双引号。已有可用的`bci`环境可继续使用，不必为了提交源码升级训练依赖。

```powershell
conda activate bci
python run.py check --help-checks
```

## 首次建立二分类主线

先准备历史数据及一份晚于历史训练日期的当天记录。历史可以是旧三分类，程序会审计排除剪刀，重新拟合真正的二分类SVM。

```powershell
$history1 = Read-Host "历史原始TXT 1完整路径"
$history2 = Read-Host "历史原始TXT 2完整路径"
$todayTxt = Read-Host "当天完整4 Run原始TXT路径"
python run.py init --history-input "$history1" "$history2" --today-input "$todayTxt" --subject USER01 --output outputs/RP_BINARY_NEW_DAY
```

只有一个历史TXT时只传一个值。若已完成历史预处理，可用`--history-data "实际预处理子目录"`代替`--history-input`。若使用`--legacy-base`，入口只读取旧三分类模型旁配置来定位历史特征；不会把三分类SVM改名当成二分类SVM。

入口会完成历史转换、基础训练、当天预处理、当天校准、配套门控准备和Run3+4连续回放，并写入`pipeline_state.json`。保留整个输出目录，记录终端打印的实际路径。

## 以后每天运行

先按相同协议采集一份当天记录，再复用首次建立的二分类基础模型：

```powershell
$baseModel = Read-Host "二分类base_model.joblib完整路径"
$todayTxt = Read-Host "新一天完整4 Run原始TXT路径"
python run.py day --base-model "$baseModel" --today-input "$todayTxt" --subject USER01 --output outputs/RP_BINARY_NEW_DAY
```

正常重复使用时只换`$todayTxt`；`$baseModel`继续指向同一套基础模型。若有意更换历史训练集，应重新执行`init`并重新检查独立验证结果。

日期、来源哈希、通道、预处理配置会被核对。`base_model.joblib`旁的`training_config.json`及其引用的预处理目录不能丢弃。生成的状态文件保留实际路径，移动整个实验目录到另一台电脑后，需要重新生成状态或明确修复路径；它不是可随意搬动的单文件模型包。

## 开始实时识别

OpenBCI GUI启动板卡与LSL，选择`TimeSeriesRaw`，流名`rps_raw`。检查可发现的流：

```powershell
python run.py streams
$runDir = Read-Host "本次输出目录，里面应有pipeline_state.json"
python run.py live --run "$runDir" --engine classic --gate-profile source --stream-name rps_raw --output outputs/LIVE_CLASSIC
```

先放松，观察程序进入可接收动作状态，再做一次手势。输出后点击实际动作；没有输出的动作手动记漏检。人工标注不改模型，不替代放松释放。待标注事件按顺序处理，避免把较早事件标成刚刚的动作。

只想比较固定4秒采样时，显式使用可选引擎：

```powershell
python run.py live --run "$runDir" --engine extended --trigger-mode manual --manual-duration-seconds 4 --output outputs/LIVE_BUTTON
```

此模式点击开始后再做动作，结束后才判断并标注；没有预先输入动作类别。它与自动门控的评估方式不同，分开记录结果。

## 回放、诊断和对照

```powershell
python run.py replay --run "$runDir"
$logDir = Read-Host "停止并保存后的某次实时记录目录"
python run.py diagnose --logs "$logDir"
python run.py compare --code-dir binary --list
python run.py compare --code-dir binary --run "$runDir" --help
```

版本比较工具列出入口与实时脚本，并在运行前检查参数兼容性。不同代入口不一定能与任意实时脚本交叉配对。完整快照在`archive/binary/`各版本子目录，需成对检查，不能只凭文件修改时间认版本。

诊断报告可复算已保存的原始输入与窗口RMS、活动分数和时间戳。它不能从没有真值的日志自动恢复真实手势、完整准确率或物理动作延迟。
