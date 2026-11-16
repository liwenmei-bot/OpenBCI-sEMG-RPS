# 二分类实时文件完整更新记录

补录日期：2026-11-15。本文针对当前binary目录的两个实时文件和run_rp_binary.py，依据能够核对的源码快照、函数差异和已保存测试记录整理。

过去的开发日期未统一保留，未知日期在JSON中为null。下文按分支和源码演变排列；这次补录提交使用实际提交时间，没有把过去研发过程伪造成旧Git提交。文件名后缀_1/_2/_3不是可靠的版本号，应核对LIVE_VERSION和SHA256。

## 现在这三个文件分别是什么

| 当前文件 | 对应版本 | 用途 |
|---|---|---|
| binary/07_live_rp_binary_label_4ch.py | classic 1.0.1 | 默认旧流程自动门控，动作后判断与标注 |
| binary/07_live_rp_binary_extended_4ch.py | 扩展分支1.4.0 | 自动静息门控及显式按钮3—4秒采集 |
| binary/run_rp_binary.py | 2026-11-15整合入口 | 串联训练/校准/回放，并选择classic或extended |

classic是从原三分类06/07另行适配的对照分支，1.0.1与扩展分支1.4.0的数字不能用来比较新旧或准确率。扩展分支历史文件多数也叫07_live_rp_binary_label_4ch.py；整合后才通过extended文件名区分。

## 各次更新

### 1.0.0-rp-binary

建立石头/布二分类链路。

- Marker11/13映射为class0/1；历史剪刀记录被排除，不充当静息。
- 默认legacy沿用配套离线活动参考与启动/释放、投票参数。
- 实时输出后人工标注；标注用于统计，冻结SVM和门控。

范围：缺少真实静息和动作起始真值时，不能从候选次数计算准确率或物理延迟。

源码快照：[07_live_rp_binary_label_4ch.py](../archive/binary/v1_0_0/07_live_rp_binary_label_4ch.py), [run_rp_binary.py](../archive/binary/v1_0_0/run_rp_binary.py)。

### 1.1.0-rp-stable

处理放松确认不稳定和重复触发。

- 默认活动模式由legacy改为robust，启动时测量约8秒当前静息。
- stable配置至少等待0.5秒，至少5窗/80%票，末两窗一致，释放至少5窗；保留replay对照。
- 增加低活动后的候选重启、确认进度与原始峰峰值等日志/提示。

范围：更严格的确认可能增加等待时间；软件测试不等于真实板卡更准确。

源码快照：[07_live_rp_binary_label_4ch.py](../archive/binary/v1_1_0/07_live_rp_binary_label_4ch.py), [run_rp_binary.py](../archive/binary/v1_1_0/run_rp_binary.py)。

逐行改动：[查看差异](../docs/version_diffs/rp-1_1_0.patch)。

验证依据：[已有记录](../docs/history_evidence/rp-1_1_0.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

### 1.2.0-rp-guided

尝试用当前静息和短动作参考修正活动门控。

- 增加GuidedActivityGate及测石头/测布参考按钮；默认guided。
- 扩展阻塞原因说明、界面布局和流选择核对。
- SVM保持冻结，但门控使用带已知动作名称的参考。

范围：这个历史版本需要提前采集命名动作参考，不符合后来要求的完全自由动作交互，随后被rest-only替代。

源码快照：[07_live_rp_binary_label_4ch.py](../archive/binary/v1_2_0/07_live_rp_binary_label_4ch.py), [run_rp_binary.py](../archive/binary/v1_2_0/run_rp_binary.py)。

逐行改动：[查看差异](../docs/version_diffs/rp-1_2_0.patch)。

验证依据：[已有记录](../docs/history_evidence/rp-1_2_0.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

### 1.3.0-rp-rest

取消提前选择动作参考，恢复先动作后判断。

- 删除GuidedActivityGate，增加仅测当前静息的RestOnlyActivityGate，默认rest。
- 采用当前静息的稳健RMS偏离；保留legacy和robust对照。
- 预测后实际动作只用于评分；相反事后标签的模拟测试得到相同轨迹与事件。

范围：新日期、电极和幅度变化仍需验证；没有引入新的分类器训练。

源码快照：[07_live_rp_binary_label_4ch.py](../archive/binary/v1_3_0/07_live_rp_binary_label_4ch.py), [run_rp_binary.py](../archive/binary/v1_3_0/run_rp_binary.py)。

逐行改动：[查看差异](../docs/version_diffs/rp-1_3_0.patch)。

验证依据：[已有记录](../docs/history_evidence/rp-1_3_0.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

### 1.3.1-rp-rest

把活动不足与类别投票不稳分开解释。

- 界面明确显示当前活动相对ON/OFF、启动进度及释放原因。
- run入口开放activity-on-z和activity-off-z。
- 默认仍为ON=6/OFF=3；4/3只是待比较设置，未静默替换默认值。

范围：特征、滤波、分类器及主要门控类与1.3.0保持一致；这次主要完善参数入口和状态解释。

源码快照：[07_live_rp_binary_label_4ch.py](../archive/binary/v1_3_1/07_live_rp_binary_label_4ch.py), [run_rp_binary.py](../archive/binary/v1_3_1/run_rp_binary.py)。

逐行改动：[查看差异](../docs/version_diffs/rp-1_3_1.patch)。

验证依据：[已有记录](../docs/history_evidence/rp-1_3_1.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

### 1.3.2-rp-rest

排查保持动作十几秒才输出。

- 新增可选responsive配置；默认仍为stable。
- 对短暂且有效的活动回落做去抖，并限制投票历史的最大年龄。
- 记录候选重启次数、活动回落进度和累计确认时间；无效/QC/过时数据仍拦截。

范围：固定预测的合成回落序列曾由stable12.9秒变为responsive1.0秒；这不是受试者或板卡实测延迟。

源码快照：[07_live_rp_binary_label_4ch.py](../archive/binary/v1_3_2/07_live_rp_binary_label_4ch.py), [run_rp_binary.py](../archive/binary/v1_3_2/run_rp_binary.py)。

逐行改动：[查看差异](../docs/version_diffs/rp-1_3_2.patch)。

验证依据：[已有记录](../docs/history_evidence/rp-1_3_2.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

### 1.4.0-rp-button

改为点击开始后做动作，约3—4秒结束再判断。

- 新增ButtonGestureDetector及trigger-mode manual；版本默认仍为auto。
- 按钮模式默认4秒，前1秒作为过渡，其后有效窗口汇总分类，结束前不输出。
- 按钮模式绕过活动门限，保留质量/时效检查及有效窗比例要求；结果后才标注实际类别。
- 该源文件后来在binary目录明确命名为07_live_rp_binary_extended_4ch.py。

范围：支持3—4秒采集，默认4秒；没有预先输入真实类别，没有重训SVM。

源码快照：[07_live_rp_binary_label_4ch.py](../archive/binary/v1_4_0/07_live_rp_binary_label_4ch.py), [run_rp_binary.py](../archive/binary/v1_4_0/run_rp_binary.py)。

逐行改动：[查看差异](../docs/version_diffs/rp-1_4_0.patch)。

验证依据：[已有记录](../docs/history_evidence/rp-1_4_0.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

### 1.0.0-rp-classic

从原三分类06/07建立旧流程二分类对照分支。

- 保留原06因果滤波、窗口、活动分数与自动状态机，适配两类输出及11/13映射。
- 沿用三分类07的预测后标注界面，增加--run路径解析与--check-only。

范围：只找到包含该分支起点的构建记录，未找到可独立校验的1.0.0完整原始快照；未用重建代码冒充原件。

源码状态：没有独立完整原始快照；构建记录所证明的分支起点单独列出。

验证依据：[已有记录](../docs/history_evidence/classic.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

### 1.0.1-rp-classic-tune

在旧流程分支增加显式门控试验，并保持默认source。

- 默认source原样使用门控JSON；增加gate-profile trial和六项显式覆盖。
- trial共享ON7.2/OFF7.0、等待0.2秒、5窗4票、5窗释放；写入日志并标明未独立验证。
- 拒绝legacy阈值与robust分数混用，并在创建界面前核对参数。
- 二分类有符号间隔及零边界与SVC.predict语义对齐；所有实际动作标签仍仅用于事后统计。

范围：trial来自少量截图，不能宣称适用于所有日期或已提升真实准确率。

源码快照：[07_live_rp_binary_label_4ch.py](../archive/binary/classic_v1_0_1/07_live_rp_binary_label_4ch.py)。

与三分类07的适配差异：[查看对比](../docs/version_diffs/rps07_to_classic_1_0_1.patch)；该对比不是缺失classic1.0.0快照的替代品。

验证依据：[已有记录](../docs/history_evidence/classic.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

### repository-integration-20261115

把classic和extended作为独立引擎接入统一入口。

- 根目录run.py调用binary/run_rp_binary.py；默认classic，按钮对照显式extended。
- 按引擎过滤和转发参数，避免将按钮参数传入classic；默认source不自动套用截图试验阈值。
- 训练、校准与回放帮助逻辑保留；活跃算法文件仅明确命名和组织。

范围：版本比较工具的“run入口+extended”组合另有兼容问题，本次历史补录未修改该工具。

源码快照：[run_rp_binary.py](../archive/binary/repository_integration_20261115/run_rp_binary.py)。

逐行改动：[查看差异](../docs/version_diffs/repository-integration-20261115.patch)。

验证依据：[已有记录](../docs/history_evidence/repository_integration.validation.json)。这些是此前保留的结果，不表示本次重跑了全部旧版本。

## 验证口径

| 扩展版本 | 已保留的软件测试记录 | 原生LSL回环会话 |
|---|---|---|
| 1.1.0 | 13项稳定性＋10项流水线回归 | 2次 |
| 1.2.0 | 17项guided＋13项兼容＋10项流水线 | 2次 |
| 1.3.0 | 14项静息/标签隔离＋13项兼容＋10项流水线 | 2次 |
| 1.3.1 | 7项阈值＋14项静息/标签隔离＋13项兼容 | 未在该版清单单独列出 |
| 1.3.2 | 11项响应＋14项静息/标签隔离＋13项兼容＋7项阈值 | 未在该版清单单独列出 |
| 1.4.0 | 15项按钮＋45项自动模式回归 | 1次 |

LSL回环不是用户Cyton板卡会话；这些检查不证明跨日准确率、真实静息误触发率或肌肉起始延迟。旧数据95.83%等分类/回放成绩的口径另见RESULTS.md，不归给任何新实时阈值。

## 怎样查具体改动

机器可读记录在binary_version_history.json，包含每个快照的SHA256、文件大小、命令行默认值、版本父节点和逐行diff路径。history_evidence保存去除本机绝对路径后的历史验证清单，version_diffs保存8组可读差异。

旧run快照引用当时同目录的00/01/02/03/07文件；archive保留的是历史来源，不是每版都复制一整套可独立执行的工程。当前运行仍从根目录run.py进入。

## 后续怎样记录

修改活跃文件时，在同一次提交更新CHANGELOG.md，写出触发问题、实际改动、测试范围和结果。原版本快照保持可追溯，命名和版本不靠文件复制后缀判断。需要发布时使用真实Git标签与实际提交时间。
