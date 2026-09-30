# Chapter 5 Representation Algorithm

> A source-traceable NumPy/float64 reference implementation of the representation-learning operators in Chapter 5 of *Deep Representation Learning*.

[![Tests](https://img.shields.io/badge/tests-14%20passed-2ea44f)](#验证与证据) [![Python](https://img.shields.io/badge/python-3.10%2B-3776ab)](#快速开始) [![License](https://img.shields.io/badge/license-see%20upstream-lightgrey)](#来源与引用)

本项目把文献中的 coding rate、MSSA/ISTA、CRATE layer、TSSA/ToST 相关公式，以及 causal prefix/cache reference，整理成可运行、可审计的算法原型。它的目标是让算法专家能够从原始公式追到代码、测试、实验配置和每轮证据；它不是训练好的 CRATE 模型，也不把 toy 实验包装成任务收益、全局收敛或随机分析定理。

## 目录

- [项目回答的问题](#项目回答的问题)
- [从文献到算法](#从文献到算法)
- [快速开始](#快速开始)
- [核心 API](#核心-api)
- [验证与证据](#验证与证据)
- [状态与边界](#状态与边界)
- [目录结构](#目录结构)
- [来源与引用](#来源与引用)

## 项目回答的问题

Chapter 5 的主线可以概括为：把表示学习写成可解释的 coding-rate reduction，再用近似梯度和稀疏编码步骤构造 Transformer-like layer。

```text
输入表示 Z
   │
   ├─ coding rate / subspace coding rate
   │        │
   │        └─ exact gradient → Neumann approximation
   │                              │
   │                              └─ MSSA (attention-like mixing)
   │
   └─ ISTA / nonnegative sparse coding
                    │
                    └─ CRATE layer: Z + MSSA + ISTA
                                      │
                                      ├─ TSSA low-rank variant
                                      ├─ causal mask + prefix reference
                                      └─ looped CRATE prototype
```

## 从文献到算法

实现过程遵循“来源 → 冻结规格 → 算子 → 证据”的顺序：

1. **固定来源与命题边界。** `source_manifest.json` 登记 Chapter 5 HTML/TeX、官方 CRATE 仓库和相关论文；Example 5.4 保留为研究方向，不凭空补写目标函数或收敛定理。
2. **冻结可复现配置。** `specification.json` 固定 `float64`、误差容限、Theorem 5.1 toy 配置、loop 深度和 receipt 字段。
3. **先实现局部算子。** `reference.py` 以 NumPy 实现 coding-rate、梯度、Neumann、MSSA、ISTA、TSSA、dense-attention control 和 causal cache；每个算子都保留诊断信息。
4. **再组合成层和循环。** `crate_layer` 组合 `Z + MSSA + ISTA`；`looped_crate` 用固定的 `run_spec` 运行多轮原型并输出带 hash、seed、预算和停止原因的 receipt。
5. **最后分层验收。** 公式级测试、Theorem 5.1 条件 toy、causal 一致性、复杂度观测和 loop audit 分开记录，保留 `NOT_EVALUATED`、`UNKNOWN`、`UPSTREAM_BLOCKED` 等状态。

### 文献对象与实现对象

| 文献对象 | 代码入口 | 说明 |
| --- | --- | --- |
| Coding rate | `coding_rate`, `subspace_coding_rate` | 对应 Chapter 5 §5.2 的 log-det 目标 |
| Exact gradient | `exact_coding_rate_gradient` | 直接矩阵梯度 reference |
| Neumann approximation | `neumann_gradient` | 返回谱范数条件和余项诊断 |
| MSSA | `mssa` | 行归一化 softmax kernel 的 attention-like 近似 |
| ISTA | `ista` | 支持文献对应的 nonnegative/ReLU 形式 |
| CRATE layer | `crate_layer` | `Z + MSSA` 与稀疏编码步骤的组合 |
| TSSA | `tssa` | 不构造 token-token 的 `N × N` 相似度矩阵 |
| Causal CRATE | `causal_mask`, `causal_mssa_cached`, `causal_mssa_incremental` | prefix consistency 与 projected-cache correctness reference |
| Looped prototype | `looped_crate` | Example 5.4 的 derived prototype，不是已证明的训练方案 |

## 快速开始

```bash
git clone https://github.com/PKUCY2016/chapter5-representation-algorithm.git
cd chapter5-representation-algorithm
python -m venv .venv
source .venv/bin/activate
pip install numpy pytest
pytest -q
```

当前 reference 只依赖 NumPy；报告构建脚本可能需要本机已有的文档工具。运行最小算子示例：

```python
import numpy as np
from reference import crate_layer, orthogonal_subspaces

rng = np.random.default_rng(0)
Z = rng.normal(size=(16, 32))
U = orthogonal_subspaces(d=32, K=2, p=8, rng=rng)

Y, diagnostics = crate_layer(
    Z,
    {"U": U, "epsilon": 1.0, "eta": 0.1, "lambda": 0.05,
     "nonnegative": True},
)
print(Y.shape, diagnostics.keys())
```

## 核心 API

- `coding_rate(Z, epsilon)`：整体表示的 log-det coding rate。
- `subspace_coding_rate(Z, U, epsilon)`：给定子空间字典的 coding rate。
- `exact_coding_rate_gradient(...)` / `neumann_gradient(...)`：精确梯度与截断近似，并报告条件诊断。
- `mssa(Z, U, epsilon, kappa, mask=None)`：MSSA mixing；`mask` 支持 causal prefix。
- `ista(H, D, eta, lambda_, nonnegative=False)`：一次 ISTA 更新；`nonnegative=True` 对应 ReLU/nonnegative LASSO 形式。
- `crate_layer(...)`：组合层。
- `tssa(...)` / `dense_attention(...)`：低秩 TSSA 与 dense control。
- `causal_mssa_cached(...)` / `causal_mssa_incremental(...)`：完整 prefix 与 projected cache reference。
- `looped_crate(X, run_spec)`：输出结构化 receipt，包含 `run_id`、source/config hash、seed、预算、状态、停止原因、coding-rate、SNR、范数和诊断字段。

## 验证与证据

在当前提交中：

- **R1 公式级 reference：** 14 项测试通过，覆盖 log-det 对照、梯度、Neumann 条件与余项、MSSA 行和、ISTA、CRATE layer、TSSA/dense control、causal cache、非法输入和 receipt schema。
- **R2 Theorem 5.1 条件 toy：** 冻结 `d=128, K=4, p=32, N=128` 以及开发/审计 seeds；结果标记为 `SYNTHETIC_CONDITIONAL`。
- **R3 causal/复杂度/loop：** causal full-mask 与 projected-cache 逐元素一致；复杂度脚本记录 wall time、峰值内存和 analytic multiply-add estimate；loop 使用 `R={1,2,4,8}`、6 个 seed，并保留 parameter-matched 与 equal-FLOPs 对照。
- **负结果也保留：** toy loop 中平均输出 SNR 低于输入，未观察到 loop 经验收益；这不是被隐藏的失败结果。

复核入口：

- [算法报告（Markdown）](docs/chapter5_algorithm_report.md)
- [算法报告（PDF）](docs/chapter5_algorithm_report.pdf)
- [Review protocol](docs/REVIEW_PROTOCOL.md)
- [Round evidence](rounds/)
- [Frozen specification](specification.json)
- [Source manifest](source_manifest.json)

## 状态与边界

| 能力 | 当前状态 | 可以怎样解读 |
| --- | --- | --- |
| 公式级 NumPy 算子 | `OPERATOR_IMPL` / `PASS` | 代码路径和局部数值性质已测试 |
| Theorem 5.1 toy | `SYNTHETIC_CONDITIONAL` | 只在冻结合成分布与条件下观察 |
| Causal/cache reference | `CONDITIONAL_APPROX` | correctness reference，不是生产 KV runtime |
| 复杂度观测 | `CONDITIONAL_APPROX` | 本机 CPU/float64 观测，不是 FLOPs 定理 |
| Looped CRATE | `DERIVED_PROTOTYPE` | 研究原型，不等于任务收益或收敛证明 |
| 真实任务训练/评估 | `NOT_EVALUATED` | 本仓库没有声称完成 |
| 官方远程 commit pin | `UPSTREAM_BLOCKED` | 当前网络无法访问远程 commit，未伪造 hash |

本仓库明确不宣称：随机分析主线的证明、任意数据上的泛化、全局收敛、真实 LLM 训练、生产级 KV cache、ToST 训练完成或真实任务性能提升。

## 目录结构

```text
reference.py                 # 公式级 NumPy/float64 reference
specification.json            # 冻结配置、状态枚举、receipt schema
source_manifest.json          # 文献来源、命题和边界
benchmarks/                   # 复杂度与 loop 实验脚本
tests/                        # 14 项局部测试
docs/chapter5_algorithm_report.md
docs/chapter5_algorithm_report.pdf
rounds/                       # R0-R3 claim/review/evidence/decision
```

## 来源与引用

主要来源：

1. Yi Ma et al., [*Deep Representation Learning Book*, Chapter 5](https://ma-lab-berkeley.github.io/deep-representation-learning-book/Ch5.html)
2. Chapter 5 [source TeX](https://raw.githubusercontent.com/Ma-Lab-Berkeley/deep-representation-learning-book/main/chapters/chapter5/deep-networks.tex)
3. [Official CRATE repository](https://github.com/Ma-Lab-Berkeley/CRATE)
4. [White-Box Transformers](https://arxiv.org/abs/2311.13110)
5. [Looped Transformers](https://proceedings.mlr.press/v202/giannou23a.html) and related [arXiv reference](https://arxiv.org/abs/2311.12424)

若你使用本仓库，请同时引用原始文献；本仓库代码是面向复核的 reference implementation，不替代原论文、官方实现或任务级 benchmark。
