# 历史调度源码

这三份文件逐字保存原研究的训练调度，供核对筛选、恢复、候选保留、冻结底座和微调逻辑。它们依赖旧项目布局及辅助模块，不作为可移植脚本直接运行。

- train_original.py：标准化正式训练与验证候选复选。
- optimize_original.py：第一阶段追加训练与控制。
- train_sampler_original.py：实际采样输出约束训练。

新的单次运行入口为上一级run.py，差异和复现边界见docs/REPRODUCIBILITY.md。
