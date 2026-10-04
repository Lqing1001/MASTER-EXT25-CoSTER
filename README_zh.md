# MASTER-EXT25 与 CoSTER

本项目对应最终版论文《面向数据驱动投资管理的中国 A 股市场新基准：MASTER-EXT25 与 CoSTER 股票排序模型》，包含数据集制备、最终算法、对照实验和投资绩效评估。

## 目录

- `CoSTER.py`：最终 CoSTER 的独立实现、分阶段训练与推断。
- `data_tools/`：CSMAR 文件读取、成分股重建、Alpha158/Market63 特征、训练期归一化及校验。
- `experiments/baselines/`：Ridge、Random Forest、XGBoost、LightGBM、LSTM、GRU、Transformer、StockMamba、ACT、PRISM-VQ。
- `evaluation/`：三种模型的预测、IC/RankIC 配对区间、Top-k 与权重敏感性、成员口径敏感性和计算量统计。
- `investment/`：多空与纯多头投资、交易成本、CH-3 因子回归及增量 alpha。
- `reference/`：最终论文表格与数值结果；`paper/`：最终中英文 PDF。

## 使用顺序

1. 安装 `requirements.txt`，运行 `python -m unittest tests.test_release`。
2. 按 [数据制备说明](docs/DATA_PREPARATION.md) 自行下载 CSMAR 相关文件，完成成分重建、特征计算和严格训练期归一化。
3. 按 [实验说明](docs/EXPERIMENTS.md) 训练模型或加载最终检查点。
4. 按 [投资实验说明](docs/INVESTMENT.md) 生成收益、交易成本、因子归因和配对检验结果。

快速检查：

```bash
python CoSTER.py demo
python -m experiments.train --dataset-root datasets/master_ext_strict_20191224_v1 --output-root runs/coster --include-baselines --dry-run
```

第二条命令只显示复现任务。去掉 `--dry-run` 才开始训练。最终检查点另存于 `CoSTER_final_checkpoints.zip`，解压后得到 `checkpoints/`，可直接推断；训练时增加 `--checkpoint-root checkpoints` 可复用论文中的冻结编码器。

CoSTER 使用 8 个可学习时间位置向量，Top-k 候选为 1、2、4、8。沪深 300、种子 0 的完整 CoSTER 验证集 V=(IC+RankIC)/2 支持选择 k=1，两个股票池及三个种子统一使用该设置。连续分数权重为 0.5。

训练期为 2010—2019，验证期为 2020—2021，测试期为 2022—2025，具体边界见 `configs/final_protocol.json`。对照模型的实现口径见 [基线说明](docs/BASELINES.md)。

## GitHub 发布

将本目录作为源码仓库上传；将 `CoSTER_final_checkpoints.zip` 放入 GitHub Release 附件。原始 CSMAR 文件、特征数组和逐股票预测不在发布包内，相关目录已加入 `.gitignore`。

代码许可证与第三方声明已保留；中英文论文 PDF 作为文稿资料提供。作者和引用信息见 `CITATION.cff`。验证范围见 [检查记录](docs/VERIFICATION.md)。
