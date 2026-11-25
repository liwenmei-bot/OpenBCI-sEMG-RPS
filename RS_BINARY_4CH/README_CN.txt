OpenBCI 四通道 sEMG 石头/剪刀二分类

依赖安装：
pip install numpy scipy pandas scikit-learn joblib pylsl

请先启动 OpenBCI GUI，连接 Cyton，并开启 LSL 输出。
默认 LSL name=rps_raw、type=EMG、采样率250Hz。若实际名称不同，请在命令中传 --stream-name 和 --stream-type。

推荐顺序：
1) python 01_check_lsl_4ch.py
2) python 02_collect_rs_4ch.py --out data
3) python 03_preprocess_rs_4ch.py --data data --out outputs/preprocess
4) python 04_train_rs_4ch.py --data data --out models
5) python 05_replay_rs_4ch.py --data data --model "models\\rs_binary_model_实际文件名.joblib"
6) python 06_live_rs_manual_4ch.py --model "models\\rs_binary_model_实际文件名.joblib"
7) 手动模式稳定后，再试 python 07_live_rs_auto_4ch.py --model "模型路径"

08_today_only_rs_4ch.py 提供检查、采集、训练的逐步工作流入口。

数据说明：每个Run随机排列10个Rock和10个Scissors，共4Run/80Trial。训练脚本用Run1-2训练、Run3选型，再用Run1-3重训并在Run4独立验证。窗口不是独立样本，评价以Trial为单位。

安全与实验注意：保持电池供电；不要在人体连接状态下给采集板充电。滤波设置和电极位置需要先通过实际波形确认。该代码尚需在你的具体OpenBCI/LSL版本和真实数据上验证，不能预先保证准确率。
