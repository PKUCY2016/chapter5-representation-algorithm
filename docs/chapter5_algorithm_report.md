# Chapter 5 文献转算法验收报告

版本：2026-09-30

## 1. 交付范围

本目录把 Yi Ma 等人的 *Deep Representation Learning Book*, Chapter 5 中与 CRATE、MSSA/ISTA、TSSA/ToST 和 causal CRATE 相关的明确公式，转成 NumPy/float64 reference。它是可审计的算法原型，不是训练好的 CRATE 模型，也不是随机分析或全局收敛证明。

随机分析主线 `01_geometry`、`02_spectral`、`03_generative` 没有被修改。

## 2. 原始文献与逐项对应

主要来源：

1. Chapter 5 在线正文：https://ma-lab-berkeley.github.io/deep-representation-learning-book/Ch5.html
2. Chapter 5 TeX：https://raw.githubusercontent.com/Ma-Lab-Berkeley/deep-representation-learning-book/main/chapters/chapter5/deep-networks.tex
3. 官方 CRATE 仓库：https://github.com/Ma-Lab-Berkeley/CRATE
4. White-Box Transformers：https://arxiv.org/abs/2311.13110

对应关系：

| 文献对象 | 公式/思想 | 当前实现 | 验收边界 |
|---|---|---|---|
| coding rate | 式 (5.2.2)、(5.2.3) | `coding_rate`、`subspace_coding_rate` | 只验证数值计算 |
| exact gradient | 式 (5.2.9) | `exact_coding_rate_gradient` | 不声称可扩展训练 |
| Neumann | 式 (5.2.10)-(5.2.12) | `neumann_gradient` 与余项诊断 | 条件为 `||A||<1` |
| SSA/MSSA | 式 (5.2.14)-(5.2.15) | `mssa` | softmax kernel 是近似 attention，不等同于 Neumann 截断 |
| CRATE layer | 式 (5.2.20)-(5.2.22) | `crate_layer` | architecture-level `Z + MSSA`；精确梯度步另有缩放形式 |
| ISTA | 式 (5.2.21) | `ista` | `nonnegative=True` 才是文献 ReLU/nonnegative LASSO 形式 |
| TSSA | 式 (5.3.16)-(5.3.18) | `tssa` | low-rank reference；未做 ToST 训练 |
| causal CRATE | 式 (5.3.19)-(5.3.30) | causal mask、prefix 和 projected cache | 是 reference，不是生产 KV runtime |
| Example 5.4 | looped transformer 研究方向 | `looped_crate` | 没有补写不存在的目标函数或收敛定理 |

## 3. 核心状态与接口

状态空间固定为：`SOURCE_READ`、`SOURCE_SPECIFIED`、`OPERATOR_IMPL`、`SYNTHETIC_CONDITIONAL`、`CONDITIONAL_APPROX`、`DERIVED_PROTOTYPE`、`PASS`、`FAIL_CLOSED`、`UPSTREAM_BLOCKED`、`UNKNOWN`、`NOT_RUN`、`NOT_EVALUATED`、`STALE`。

接口：

```text
mssa(Z, U, epsilon, kappa, mask=None)
ista(H, D, eta, lambda_, nonnegative=False)
crate_layer(Z, layer_config, mask=None)
looped_crate(X, run_spec)
```

`looped_crate` 的 receipt 包含 run_id、source/config hash、seed、task、R、dtype/device、预算、状态、停止原因、coding-rate、sparsity、SNR、范数、operator residual、wall time、memory、FLOPs、compile time 和 diagnostics。

## 4. R0-R3 验证

### R0 来源与命题

Chapter 5 HTML/TeX、CRATE、AoT、TSSA/ToST、causal CRATE 和 loop 参考已登记在 `source_manifest.json`。Example 5.4 保留为研究方向，不被写成已有优化目标。

官方 CRATE 远程 commit 当前因本机网络代理不可达而无法 pin；该事实记录为 `REMOTE_COMMIT_UNAVAILABLE_NETWORK_BLOCKED`，没有伪造 hash。

### R1 公式级 reference

使用 float64，`rtol=1e-8`、`atol=1e-10`。测试覆盖直接 slogdet 对照、exact gradient、Neumann 条件与余项、MSSA softmax 行和、ISTA 手算步、CRATE layer、TSSA/dense control、causal cache、非法输入和 receipt schema。

当前测试命令：

```bash
cd /tmp
PYTHONPATH=/abs/path/research_loop/04_representation_algorithm \
  python -m unittest discover \
  -s /abs/path/research_loop/04_representation_algorithm/tests -v
```

最近一次结果：14 项通过。

### R2 Theorem 5.1 条件实验

冻结配置：`d=128, K=4, p=32, N=128, group_size=32, tau=0.75, eta=0.1`，`delta={0.05,0.10,0.20}`，开发 seed `{0,1,2}`，审计 seed `{10,11,12}`，float64。

数据是低秩子空间联合空间中的 Gaussian toy，并保留非正交子空间、重尾噪声和错误 `U` 负对照。结果只能标记 `SYNTHETIC_CONDITIONAL`，不能替代 Theorem 5.1 的高概率证明。

### R3 causal、复杂度和 loop

- causal full-mask 与 projected-cache incremental reference 逐元素一致；
- complexity 脚本在 `N={128,256,512,1024,2048}` 测量 MSSA、TSSA、dense attention 的 wall time 和 tracemalloc 峰值，并给出 analytic multiply-add FLOPs estimate；
- loop 实验使用 6 个 seed、`R={1,2,4,8}`，并包含 parameter-matched no-loop 与 equal-FLOPs untied baseline；
- audit split 是合成 held-out，只能作为 toy evidence；
- toy 结果中平均输出 SNR 低于输入，未观察到 loop 经验收益。

观测斜率约为：MSSA `2.30`、TSSA `0.63`、dense attention `1.97`。这些是本机 CPU/float64 reference 的观测，不是渐近复杂度定理。

## 5. 代码审计重点

1. `mssa` 的 softmax 是行归一化后转置聚合，必须与列向量记号一起阅读；测试验证了行和为 1。
2. `ista(nonnegative=True)` 对应文献中的 ReLU/nonnegative LASSO；默认 `False` 是为了暴露一般 soft-threshold 选项，不能把默认值误称为 CRATE 原始形式。
3. `causal_mssa_cached` 是 prefix consistency reference；`causal_mssa_incremental` 才使用 projected cache，但仍是 NumPy correctness reference，不是生产 KV cache。
4. `tssa` 不构造 N×N token similarity 矩阵；这支持复杂度实验中的线性 token 方向，但不代表完成了 ToST 训练或真实任务评估。
5. looped CRATE 是 Example 5.4 的 derived prototype。coding-rate、ISTA surrogate 或 SNR 的局部变化都不等于任务误差下降、全局收敛或随机分析定理。

## 6. 验收结论

代码和证据可交给算法专家复核，当前交付状态为：

- 算子实现：`OPERATOR_IMPL`；
- 定理 toy：`SYNTHETIC_CONDITIONAL`；
- causal/复杂度：`CONDITIONAL_APPROX`；
- loop：`DERIVED_PROTOTYPE`；
- 真实任务收益、生产级 KV cache、远程 source commit pin：`NOT_EVALUATED` 或 `UPSTREAM_BLOCKED`。

任何新的原始论文、公式版本或同事指出的差异，都应先更新 `source_manifest.json` 和本表，再修改代码、测试、hash 和 round evidence。
