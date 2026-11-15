# References / 参考文献

This project bibliography covers the final manuscript and the implemented methods. [BibTeX](references.bib) contains all entries; [paper/references.bib](paper/references.bib) contains the bibliography used by the supplied PDFs. The five additional software-method references cover Ridge, Random Forests, XGBoost, Mamba-2/SSD and moving-block bootstrap.

项目参考文献涵盖最终论文和代码所用方法。下方的对应关系说明引用用途；基线适配口径见 [BASELINES.md](docs/BASELINES.md)，第三方代码归属见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

## Citation map / 引用对应关系

| Component | References |
|---|---|
| MASTER in CoSTER | [li2024master](#li2024master) |
| Alpha158 / Qlib feature conventions | [yang2020qlib](#yang2020qlib) |
| Ridge / Random Forest / XGBoost / LightGBM | [hoerl1970ridge](#hoerl1970ridge), [breiman2001random](#breiman2001random), [chen2016xgboost](#chen2016xgboost), [ke2017lightgbm](#ke2017lightgbm) |
| LSTM / GRU / Transformer | [hochreiter1997lstm](#hochreiter1997lstm), [cho2014learning](#cho2014learning), [vaswani2017attention](#vaswani2017attention) |
| StockMamba and Mamba-2 SSD | [zhang2026stockmamba](#zhang2026stockmamba), [dao2024ssm](#dao2024ssm) |
| ACT / PRISM-VQ | [li2026act](#li2026act), [kim2026prism](#kim2026prism) |
| Newey–West HAC / moving-block bootstrap | [newey1987simple](#newey1987simple), [kunsch1989bootstrap](#kunsch1989bootstrap) |
| Chinese three-factor attribution | [liu2019size](#liu2019size) |

## Core model and benchmark / 核心方法与基准

<a id="li2024master"></a>

Li, T.; Liu, Z.; Shen, Y.; Wang, X.; Chen, H.; Huang, S. (2024). **MASTER: Market-guided stock transformer for stock price forecasting**. Proceedings of the AAAI Conference on Artificial Intelligence, 38, 162–170. [Source](https://doi.org/10.1609/aaai.v38i1.27767).

<a id="yang2020qlib"></a>

Yang, X.; Liu, W.; Zhou, D.; Bian, J.; Liu, T.-Y. (2020). **Qlib: An AI-oriented quantitative investment platform**. Preprint, arXiv:2009.11189. [Source](https://doi.org/10.48550/arXiv.2009.11189).


## Comparison models / 对照模型

<a id="hoerl1970ridge"></a>

Hoerl, Arthur E.; Kennard, Robert W. (1970). **Ridge Regression: Biased Estimation for Nonorthogonal Problems**. Technometrics, 12(1), 55–67. [Source](https://doi.org/10.1080/00401706.1970.10488634).

<a id="breiman2001random"></a>

Breiman, Leo (2001). **Random Forests**. Machine Learning, 45, 5–32. [Source](https://doi.org/10.1023/A:1010933404324).

<a id="chen2016xgboost"></a>

Chen, Tianqi; Guestrin, Carlos (2016). **XGBoost: A Scalable Tree Boosting System**. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. [Source](https://doi.org/10.1145/2939672.2939785).

<a id="ke2017lightgbm"></a>

Ke, Guolin; Meng, Qi; Finley, Thomas; Wang, Taifeng; Chen, Wei; Ma, Weidong; Ye, Qiwei; Liu, Tie-Yan (2017). **LightGBM: A Highly Efficient Gradient Boosting Decision Tree**. Advances in Neural Information Processing Systems, 30, 3146–3154.

<a id="hochreiter1997lstm"></a>

Hochreiter, Sepp; Schmidhuber, Jürgen (1997). **Long Short-Term Memory**. Neural Computation, 9(8), 1735–1780. [Source](https://doi.org/10.1162/neco.1997.9.8.1735).

<a id="cho2014learning"></a>

Cho, Kyunghyun; van Merriënboer, Bart; Gulcehre, Caglar; Bahdanau, Dzmitry; Bougares, Fethi; Schwenk, Holger; Bengio, Yoshua (2014). **Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation**. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), 1724–1734. [Source](https://doi.org/10.3115/v1/D14-1179).

<a id="vaswani2017attention"></a>

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; Polosukhin, I. (2017). **Attention is all you need**. Advances in Neural Information Processing Systems, 30, 5998–6008. [Source](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html).

<a id="zhang2026stockmamba"></a>

Zhang, P. (2026). **StockMamba: State-space gated stock transformer with rank-aware optimization**. Mathematics, 14, 1859. [Source](https://doi.org/10.3390/math14111859).

<a id="dao2024ssm"></a>

Dao, Tri; Gu, Albert (2024). **Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality**. Proceedings of the 41st International Conference on Machine Learning, 235, 10041–10071. [Source](https://proceedings.mlr.press/v235/dao24a.html).

<a id="li2026act"></a>

Li, J.; Zhang, L. (2026). **ACT: Anti-crosstalk learning for cross-sectional stock ranking via temporal disentanglement and structural purification**. Preprint, arXiv:2604.20204. [Source](https://doi.org/10.48550/arXiv.2604.20204).

<a id="kim2026prism"></a>

Kim, N.; Song, J. W. (2026). **Vector-Quantized Discrete Latent Factors Meet Financial Priors: Dynamic Cross-Sectional Stock Ranking Prediction for Portfolio Construction**. Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence, 4421–4429. [Source](https://doi.org/10.24963/ijcai.2026/492).


## Investment evaluation and statistical inference / 投资评价与统计推断

<a id="newey1987simple"></a>

Newey, W. K.; West, K. D. (1987). **A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix**. Econometrica, 55, 703–708. [Source](https://doi.org/10.2307/1913610).

<a id="kunsch1989bootstrap"></a>

Künsch, Hans R. (1989). **The Jackknife and the Bootstrap for General Stationary Observations**. The Annals of Statistics, 17(3), 1217–1241. [Source](https://doi.org/10.1214/aos/1176347265).

<a id="liu2019size"></a>

Liu, J.; Stambaugh, R. F.; Yuan, Y. (2019). **Size and value in China**. Journal of Financial Economics, 134, 48–69. [Source](https://doi.org/10.1016/j.jfineco.2019.03.008).

<a id="jegadeesh1993returns"></a>

Jegadeesh, N.; Titman, S. (1993). **Returns to buying winners and selling losers: Implications for stock market efficiency**. Journal of Finance, 48, 65–91. [Source](https://doi.org/10.1111/j.1540-6261.1993.tb04702.x).

<a id="brown1992survivorship"></a>

Brown, S. J.; Goetzmann, W.; Ibbotson, R. G.; Ross, S. A. (1992). **Survivorship bias in performance studies**. Review of Financial Studies, 5, 553–580. [Source](https://doi.org/10.1093/rfs/5.4.553).

<a id="white2000reality"></a>

White, H. (2000). **A reality check for data snooping**. Econometrica, 68, 1097–1126. [Source](https://doi.org/10.1111/1468-0262.00152).

<a id="bailey2017probability"></a>

Bailey, D. H.; Borwein, J. M.; López de Prado, M.; Zhu, Q. J. (2017). **The probability of backtest overfitting**. Journal of Computational Finance, 20, 39–69. [Source](https://doi.org/10.21314/JCF.2016.322).


## Further manuscript references / 论文其他参考文献

<a id="kelly2019characteristics"></a>

Kelly, B. T.; Pruitt, S.; Su, Y. (2019). **Characteristics are covariances: A unified model of risk and return**. Journal of Financial Economics, 134, 501–524. [Source](https://doi.org/10.1016/j.jfineco.2019.05.001).

<a id="freyberger2020dissecting"></a>

Freyberger, J.; Neuhierl, A.; Weber, M. (2020). **Dissecting characteristics nonparametrically**. Review of Financial Studies, 33, 2326–2377. [Source](https://doi.org/10.1093/rfs/hhz123).

<a id="kozak2020shrinking"></a>

Kozak, S.; Nagel, S.; Santosh, S. (2020). **Shrinking the cross-section**. Journal of Financial Economics, 135, 271–292. [Source](https://doi.org/10.1016/j.jfineco.2019.06.008).

<a id="gu2020empirical"></a>

Gu, S.; Kelly, B.; Xiu, D. (2020). **Empirical asset pricing via machine learning**. Review of Financial Studies, 33, 2223–2273. [Source](https://doi.org/10.1093/rfs/hhaa009).

<a id="chen2024deep"></a>

Chen, L.; Pelger, M.; Zhu, J. (2024). **Deep learning in asset pricing**. Management Science, 70, 714–750. [Source](https://doi.org/10.1287/mnsc.2023.4695).

<a id="leippold2022machine"></a>

Leippold, M.; Wang, Q.; Zhou, W. (2022). **Machine learning in the Chinese stock market**. Journal of Financial Economics, 145, 64–82. [Source](https://doi.org/10.1016/j.jfineco.2021.08.017).

<a id="mclean2016does"></a>

McLean, R. D.; Pontiff, J. (2016). **Does academic research destroy stock return predictability?**. Journal of Finance, 71, 5–32. [Source](https://doi.org/10.1111/jofi.12365).

<a id="hou2020replicating"></a>

Hou, K.; Xue, C.; Zhang, L. (2020). **Replicating anomalies**. Review of Financial Studies, 33, 2019–2133. [Source](https://doi.org/10.1093/rfs/hhy131).

<a id="jensen2023replication"></a>

Jensen, T. I.; Kelly, B. T.; Pedersen, L. H. (2023). **Is there a replication crisis in finance?**. Journal of Finance, 78, 2465–2518. [Source](https://doi.org/10.1111/jofi.13249).

<a id="chen2022open"></a>

Chen, A. Y.; Zimmermann, T. (2022). **Open source cross-sectional asset pricing**. Critical Finance Review, 11, 207–264. [Source](https://doi.org/10.1561/104.00000112).

<a id="feng2019temporal"></a>

Feng, F.; He, X.; Wang, X.; Luo, C.; Liu, Y.; Chua, T.-S. (2019). **Temporal relational ranking for stock prediction**. ACM Transactions on Information Systems, 37, 27:1–27:30. [Source](https://doi.org/10.1145/3309547).

<a id="xu2021hist"></a>

Xu, W.; Liu, W.; Wang, L.; Xia, Y.; Bian, J.; Yin, J.; Liu, T.-Y. (2021). **HIST: A graph-based framework for stock trend forecasting via mining concept-oriented shared information**. Preprint, arXiv:2110.13716. [Source](https://doi.org/10.48550/arXiv.2110.13716).

<a id="zhao2023doubleadapt"></a>

Zhao, L.; Kong, S.; Shen, Y. (2023). **DoubleAdapt: A meta-learning approach to incremental learning for stock trend forecasting**. Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 3492–3503. [Source](https://doi.org/10.1145/3580305.3599315).

<a id="tian2024fintech"></a>

Tian, Maoxi; El Khoury, Rim; Nasrallah, Nohade; Alshater, Muneer M. (2024). **Assessing systemic risk spillovers from FinTech to China's financial system**. The European Journal of Finance, 30(8), 803–826. [Source](https://doi.org/10.1080/1351847X.2023.2244008).

<a id="elkhoury2023spillover"></a>

El Khoury, Rim; Nasrallah, Nohade; Hussainey, Khaled; Assaf, Rima (2023). **Spillover analysis across FinTech, ESG, and renewable energy indices before and during the Russia–Ukraine war: International evidence**. Journal of International Financial Management & Accounting, 34(2), 279–317. [Source](https://doi.org/10.1111/jifm.12179).

<a id="elkhoury2024digital"></a>

El Khoury, Rim; Nasrallah, Nohade; Harb, Etienne G. (2024). **Did the intensity of countries' digital transformation affect IT companies' performance during covid-19?**. Journal of Decision Systems, 33(2), 199–219. [Source](https://doi.org/10.1080/12460125.2022.2094528).

## Recent introduction references / 引言近期文献

<a id="li2025portfolio"></a>

Li, J.; Luo, L. (2025). **Constructing stock portfolio with transformer**. International Journal of Computational Science and Engineering, 28(4), 446–457. [Source](https://doi.org/10.1504/IJCSE.2025.147607).

<a id="chen2025dhmoe"></a>

Chen, W.; Wang, Y. (2025). **DHMoE: Diffusion Generated Hierarchical Multi-Granular Expertise for Stock Prediction**. Proceedings of the AAAI Conference on Artificial Intelligence, 39(11), 11490–11499. [Source](https://doi.org/10.1609/aaai.v39i11.33250).

<a id="detthamrong2026esg"></a>

Detthamrong, U.; Klangbunrueang, R.; Chansanam, W.; Dasri, R. (2026). **Deep Learning for Sustainable Finance: Robust ESG Index Forecasting in an Emerging Market Context**. Sustainability, 18(1), 110. [Source](https://doi.org/10.3390/su18010110).

<a id="serban2025rebalancing"></a>

Șerban, F. (2025). **Sustainable Portfolio Rebalancing Under Uncertainty: A Multi-Objective Framework with Interval Analysis and Behavioral Strategies**. Sustainability, 17(13), 5886. [Source](https://doi.org/10.3390/su17135886).

## Capital allocation and stock-market foundations / 资本配置与股票市场基础

<a id="wurgler2000allocation"></a>

Wurgler, J. (2000). **Financial markets and the allocation of capital**. Journal of Financial Economics, 58(1–2), 187–214. [Source](https://doi.org/10.1016/S0304-405X%2800%2900070-2).

<a id="levine1991stock"></a>

Levine, R. (1991). **Stock Markets, Growth, and Tax Policy**. Journal of Finance, 46(4), 1445–1465. [Source](https://doi.org/10.1111/j.1540-6261.1991.tb04625.x).

<a id="carpenter2021real"></a>

Carpenter, J. N.; Lu, F.; Whitelaw, R. F. (2021). **The real value of China's stock market**. Journal of Financial Economics, 139(3), 679–696. [Source](https://doi.org/10.1016/j.jfineco.2020.08.012).


## Portfolio constraints and implementation costs / 投资约束与实施成本

<a id="clarke2002constraints"></a>

Clarke, R.; de Silva, H.; Thorley, S. (2002). **Portfolio Constraints and the Fundamental Law of Active Management**. Financial Analysts Journal, 58(5), 48–66. [Source](https://doi.org/10.2469/faj.v58.n5.2468).

<a id="novymarx2016costs"></a>

Novy-Marx, R.; Velikov, M. (2016). **A Taxonomy of Anomalies and Their Trading Costs**. Review of Financial Studies, 29(1), 104–147. [Source](https://doi.org/10.1093/rfs/hhv063).
