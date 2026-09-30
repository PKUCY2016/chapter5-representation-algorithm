# R1 审阅

独立审阅发现并已修正：reference 现在包含 exact coding-rate gradient、真实一阶 Neumann 梯度、逐头截断余项及条件界；MSSA 的 softmax attention 仍单独标为论文定义的近似算子，不把它等同于 Neumann 截断。11 项测试通过。该轮仍只证明算子实现，不证明优化收敛或科学有效。
