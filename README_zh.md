# MASTER-EXT25 与 CoSTER

本项目对应最终版论文《面向数据驱动投资管理的中国 A 股市场新基准：MASTER-EXT25 与 CoSTER 股票排序模型》，包含数据集制备、最终算法、对照实验和投资绩效评估。

## 目录

- `CoSTER.py`：最终 CoSTER 的独立实现、分阶段训练与推断。
- `data_tools/`：CSMAR 文件读取、成分股重建、Alpha158/Market63 特征、训练期归一化及校验。
- `experiments/baselines/`：Ridge、Random Forest、XGBoost、LightGBM、LSTM、GRU、Transformer、StockMamba、ACT、PRISM-VQ。
- `evaluation/`：三种模型的预测、IC/RankIC 配对区间、Top-k 与权重敏感性、成员口径敏感性和计算量统计。
- `investment/`：多空与纯多头投资、交易成本、CH-3 因子回归及增量 alpha。
