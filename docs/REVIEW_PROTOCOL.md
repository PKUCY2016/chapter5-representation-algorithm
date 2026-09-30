# 同事复核协议

收到新的原始文献、版本或公式差异后，按以下顺序复核：

1. 记录文献 URL、版本/commit、章节和公式编号，更新 `source_manifest.json`。
2. 建立“文献公式 - reference 函数 - 测试 - 不能推出的结论”四列差异表。
3. 先做维度、归一化、参数量词和 mask/cache 方向检查，再做数值对照。
4. 若公式发生变化，先新增回归测试，再修改实现；不得修改旧 evidence 来掩盖旧结果。
5. 重跑源码目录外测试、复杂度脚本和 loop receipt；更新源文件 hash、`claim.md`、`review.md`、`evidence.json`、`decision.md`。
6. 任何无法确认的内容标记 `UNKNOWN`、`NOT_EVALUATED` 或 `UPSTREAM_BLOCKED`，不补写成收敛或任务收益结论。

当前最关键的人工复核点：

- MSSA 的 softmax 聚合方向和 `p/(N epsilon^2)` 缩放；
- `ISTA(nonnegative=True)` 与式 (5.2.21) 的对应；
- TSSA 的 assignment temperature、二阶统计量和 `D` 对角项；
- causal incremental cache 与全量 causal mask 的 prefix 一致性；
- loop 是否仍只是 Example 5.4 的 derived prototype。
