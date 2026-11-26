OpenBCI 四通道 sEMG 石头 vs 布二分类系统 v2.1
=================================================

一、实验设计
- OpenBCI GUI通过LSL输出原始时序数据。
- 使用前4通道，配置采样率250 Hz。
- 4 Run × 每Run 20 Trial；每Run石头10次、布10次，共80 Trial。
- 单Trial：静息3秒、准备1秒、动作4秒、恢复2.5秒。
- 动作有效片段：动作开始后0.5至3.5秒。
- 200 ms窗口、100 ms步长；每Trial应得到29个窗口。
- 标签：0=Rock（石头），1=Paper（布）。

二、安装依赖
conda activate bci
pip install numpy scipy pandas scikit-learn joblib pylsl matplotlib

三、运行顺序（在项目根目录）
1. 检查LSL与四通道：
   python 01_check_lsl_4ch.py
2. 采集80 Trial：
   python 02_collect_rp_4ch.py --out data
3. 预处理并生成特征：
   python 03_preprocess_rp_4ch.py --data data
4. 训练、分组交叉验证及Run4独立验证：
   python 04_train_rp_4ch.py --data data --out models
5. 离线回放（替换成实际模型文件名）：
   python 05_replay_rp_4ch.py --data data --model "models\rp_binary_model_时间戳.joblib"
6. 手动触发实时识别：
   python 06_live_rp_manual_4ch.py --model "models\rp_binary_model_时间戳.joblib"

四、训练与验证逻辑
- 首先严格检查80 Trial、每Run 20 Trial、每类10 Trial、Trial ID唯一。
- WARNING（轻度时间戳异常）保留并标记；FAIL（严重异常）必须重采。Run4报告同时给出PASS子集准确率（仅是子集描述，并非重新训练的敏感性分析）。
- 03生成window_features.csv，并保存到本Session目录。
- 04直接读取同一份window_features.csv，不重复计算特征。
- 候选模型：Linear SVM、RBF SVM、LDA、Random Forest。
- 特征方案：20维基础时域特征、25维（增加相对RMS及屈伸肌比值）。
- 模型选择：仅使用Run1+Run2，按Trial ID分组的StratifiedGroupKFold四折交叉验证；一个Trial的窗口不会跨训练折与验证折。
- 最终拟合：Run1+Run2+Run3。
- 最终独立测试：Run4。Run4不参与算法/特征选择，也不参与最终拟合。
- 评价单位：Trial级多数投票，不把重叠窗口当成独立受试样本。

五、数据隔离
- 每次采集自动建立独立Session目录：data/RP_时间戳/。
- data/LATEST_SESSION.txt指向最近一次Session。
- 训练和预处理默认只读取最近Session，不会把以前日期的NPZ混进来。
- 正式采集前不要手动把不同日期的NPZ复制到同一个Session。

六、重要说明
- config_rp.json是真正读取的配置文件；采样率、滤波、窗口、采集时序和特征设置均从这里读取。
- 默认LSL名称rps_raw、类型EMG；如GUI流名称不同，使用--stream-name和--stream-type修改。
- 20–100 Hz和50 Hz陷波是本实验初始设置，不是所有肌电设备通用的最优参数。若本地工频为60 Hz，应修改配置。
- 当前手动实时模式是按Enter后采集完整动作片段，再批量分类，不是每100 ms连续显示结果。
- 自动门控模式暂时禁用：上一版动作起点和分类片段没有严格对齐，完成在线因果滤波与状态机验证前不用于正式实验。
- 代码完成语法和合成数据流程测试，不代表已在你的OpenBCI硬件、GUI及LSL环境完成实测。正式采集前先运行01。

七、v2.1 质控与诊断
- 00_diagnose_lsl_4ch.py：60秒静息、石头、布诊断，支持--self-test。
- WARNING不代表硬件丢包，也不代表无丢包；应对照GUI Sample Index。
- FAIL阈值为初始工程规则：时间戳缺口>100ms、非递增比例>10%、时间戳跨度估计误差>10%、接收暂停>250ms、采集时长偏差>0.5秒、点数不足、非有限值、平直或幅值异常。不能作为硬件丢包确证。
- 保留原始时间戳和每次尝试的质控详情；不排序、不插值。
- 默认05只回放Run4（不是新增独立实验）；--all-runs含训练数据，不可用于泛化评价。
- 06保存实时原始片段及人工标注；投票比例不是概率。
- 采集与手动预测均在每阶段清除旧LSL缓存，降低动作错位风险；仍需硬件实测。
- 不要把RS模型或数据混入新项目；只加载可信joblib模型。
- 建议按实际工频设置notch_hz，采集后保持配置一致。
