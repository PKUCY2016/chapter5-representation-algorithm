# R3 causal、复杂度与 loop

状态：`CONDITIONAL_APPROX` / `DERIVED_PROTOTYPE`。完成 causal prefix 与增量 projected-cache reference 一致性；完成 N={128,256,512,1024,2048} 上 MSSA、TSSA 和 dense attention 的 wall-time/tracemalloc 对照；完成 R={1,2,4,8} 的 6 seed loop receipt、参数匹配 no-loop 和等 FLOPs untied baseline。结果仍是合成原型，真实任务收益未评价。
