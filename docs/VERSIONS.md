# 版本与入口

当前仓库将角色放入不同目录，避免把相同文件名的不同版本互相覆盖。

| 位置 | 版本/角色 | 默认入口是否使用 |
|---|---|---|
| `binary/07_live_rp_binary_label_4ch.py` | 1.0.1-rp-classic-tune；原06式自动门控、二分类、事后标注、source/trial配置 | 是，`--engine classic` |
| `binary/07_live_rp_binary_extended_4ch.py` | 1.4.0-rp-button；静息门控及按钮固定时长识别等对照 | 仅显式`--engine extended` |
| `binary/run_rp_binary.py` | 本次适配的二分类流水线，按引擎过滤并转发兼容参数 | 是，由根目录`run.py`调用 |
| `archive/binary/v1_0_0`至`v1_3_2` | 六套已有历史实时/入口源文件快照 | 否 |
| `rps/` | 原三分类采集、01—08与当天自动流程 | 仅`python run.py rps ...` |

历史快照包含legacy、静息测量、曾要求动作参考的guided等不同实验。它们仅供追溯和显式比较，不代表每一个都符合当前默认交互。分类模型本身必须使用相同类别/schema，不能将三分类模型改文件名后用于二分类。

本次集成的关键改动是统一入口、明确classic/extended文件、拒绝不兼容参数组合，避免把`--trigger-mode manual`之类按钮版参数传给旧版parser。另将版本比较工具中个人电脑的默认路径改为显式输入或项目相对路径。

其余采集、预处理、基础训练、当天校准与实时算法均保留所找到的源文件版本；没有为“统一风格”重写信号算法。各文件来源与集成差异见`SOURCE_PROVENANCE.json`，本次分发的最终文件哈希见`SOURCE_MANIFEST.json`。

三分类目录中的`SOURCE_SNAPSHOT.json`追溯原先三分类分发，不涵盖此处全部新增文件；检查当前整体分发用根目录`python run.py check --verify-snapshot`。原三分类说明中的既往测试数为历史记录，本次实际执行的检查见`docs/VALIDATION.md`。
