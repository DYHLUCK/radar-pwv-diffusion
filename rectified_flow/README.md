# 条件整流流匹配 · RADAR＋PWV短临雷达外推

学习噪声到未来雷达场的直线路径速度。这是已完成实验的**轻量研究实现**，不是某篇论文完整官方复现；没有显式物理演变模块或GAN。

## 模型与任务

- 过去10帧雷达＋10帧PWV作为条件，联合预测未来20帧雷达；6分钟间隔，提前量6—120分钟，网格66×70。
- 雷达/PWV分别堆叠为时间通道，经卷积条件编码后融合；不是GRU、Transformer或自回归滚动。
- 轻量卷积U形组件；参数量约1,152,204。生成固定16步；目标标准化仅拟合训练期。
- R1：2023年5—8月训练，2025年5—6月验证、7—8月历史测试；R2交换年份。
- 当前调优改善了本模型，但点预报CSI/RMSE仍未超过直接预测基线。见[结果摘要](docs/RESULTS.md)。

## 目录

```text
run.py                     本模型命令入口
src/model.py               本模型构建与预测接口
src/models.py              与实验逐字一致的网络和采样器
src/objectives.py          原始三种损失
src/optimization_losses.py  重建/平衡损失
src/sampler_objective.py    实际采样输出辅助约束
src/dataset.py             可移植数据读取与PNG缓存构建
src/runtime.py             可移植单运行训练、恢复、评价
configs/                   R1/R2的原始训练及四种微调配置
provenance/                核心源码哈希、历史协议、权重哈希索引
reference/                 历史训练调度源码，仅作逻辑对照
examples/frames.csv        原始PNG配对清单格式示例
tests/                     CPU合成数据检查
```

## 安装与检查

建议Python3.10及以上；本次在Python3.12/PyTorch2.8.0环境中做CPU检查。先按自己的设备安装PyTorch，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python run.py --help
```

`requirements-tested.txt`记录本机检查所用版本。正式训练默认CUDA，GPU版全数据训练未在本次整理中重新执行；小型运行检查使用CPU合成数据，不是新的实验结果。

## 数据

见[数据约定](docs/DATA.md)。可通过`--data-root`直接指向已有缓存，不必复制数据。数据、训练权重和完整预测场均不随仓库分发。

若从PNG重新制作缓存，准备完整CSV，每行包含同一时间的雷达和PWV路径：

```bash
python run.py prepare --frames-csv /path/to/frames.csv --output data/cache
```

程序严格检查灰度L模式、70×66尺寸、30帧连续性及划分边界，不插值。`examples/frames.csv`只有格式示例，不是可训练数据。

## 训练与微调

### 从头训练

```bash
python run.py train --config configs/R1_native.json --data-root data/cache --output outputs/R1_native
```

`configs/R2_native.json`对应反向年份。三个历史训练种子为20261019/20/21；每次复制配置、修改`seed`并使用不同输出目录，单次命令不会自动启动整组研究。

### 保持主体的调优

将对应原选中权重放到`checkpoints/original_selected.pt`，从它初始化：

```bash
python run.py train --config configs/R1_sampled_skill.json --data-root data/cache --output outputs/R1_sampled_skill --init checkpoints/original_selected.pt
```

可选配置：`continue`沿用原损失；`reconstruction`加强重建；`balanced`加入软CSI/FSS/时间差/背景约束；`sampled_skill`沿用原配方并约束实际16步采样图。微调每项16轮，初始化为原实验权重，不是接在第一阶段平衡损失权重之后。

实际采样辅助项系数0.5，每4批取最多16个样本，反传最后4个**采样迭代**，监督全部20个未来时刻。它增加计算量；不声称与控制完全同算力。

恢复中断运行：在完全相同命令后加`--resume`。输出目录非空时默认拒绝覆盖，恢复会核对配置、代码、索引及初始化权重。默认输出预算4GiB、磁盘至少20GiB余量，不保存全数据预测场。

## 评价与预测图

```bash
python run.py evaluate --checkpoint outputs/R1_sampled_skill/selected.pt --data-root data/cache --round R1 --split valid --output outputs/R1_validation --members 4
```

评估历史测试时将`--split`改为`test`并指定新输出目录；先写权重/索引锁，再评分，不做测试选模。CSV包含全部20提前量，20/30/40dBZ的CSI/POD/FAR、FSS1/3/5，以及RMSE/MAE。先各成员评分，再平均其技巧，既不是best-of-N，也不是集合平均图的分数。

单次预测输入为NPZ：`radar_dbz`和`pwv_mm`各为`(10,66,70)`，按过去到现在排序，单位分别为dBZ和mm：

```bash
python run.py predict --checkpoint outputs/R1_sampled_skill/selected.pt --input example_input.npz --output outputs/prediction.npz --members 4
python run.py plot --prediction outputs/prediction.npz --output outputs/prediction.png
```

如有真值NPZ（字段`truth_dbz`、形状`(20,66,70)`），绘图时加`--truth truth.npz`。统一0—50dBZ色标。用`--device cpu`可以在CPU上检查；对齐历史样本噪声时，`--issue-id`须用该缓存的窗口起始索引。

## 与历史实验的关系

五份核心源码逐字保留，哈希在[源码清单](provenance/source_manifest.json)；新写的可移植入口不替代历史证据。[复现边界](docs/REPRODUCIBILITY.md)明确区分单次训练与原24项筛选、三种子/四成员联合选模和64项调优的完整调度。

数据是雷达反射率目标，不直接证明极端地面降水能力；本轮未验证PWV独立贡献或地形增益。历史测试已被查看，不能称全新外部测试。概率校准尚未评估。

授权：仓库作者尚未指定开源许可证，可自行补充。
