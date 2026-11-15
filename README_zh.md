# MASTER-EXT25 与 CoSTER

论文 **A New Chinese A-Share Market Benchmark for Data-Driven Investment Management: MASTER-EXT25 and the CoSTER Stock-Ranking Model** 的代码与实验结果。

[English](README.md) · [数据准备](docs/DATA_PREPARATION.md) · [模型实验](docs/EXPERIMENTS.md) · [投资评价](docs/INVESTMENT.md)

MASTER-EXT25将中国A股股票排序评价扩展至2025年。CoSTER将市场引导的MASTER与连续时序分支相结合，通过每日得分标准化与等权融合生成股票排序。

## 项目内容

- `CoSTER.py`：模型、分阶段训练与推断。
- `data_tools/`：数据处理、特征构建与验证。
- `experiments/`：基线、消融与对照实验。
- `evaluation/`、`investment/`：预测评价与投资组合分析。
- `reference/`、`figures/`：实验结果与图表。
- `paper/`：论文及LaTeX源码。

## 快速开始

```bash
pip install -r requirements.txt
python CoSTER.py demo
python scripts/reproduce_results.py
```

演示使用随机权重；复现脚本使用项目内的汇总结果。完整训练与评价需自行获取CSMAR数据，具体步骤见上述文档。

## 许可

项目代码采用MIT许可证，第三方组件遵循各自许可证。引用信息见`CITATION.cff`。
