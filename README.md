# 生成模型代码上传包

三个目录均为独立代码包。每个包有独立README、依赖、配置、训练/评价/预测入口、测试及来源记录。

| 目录 | 对应模型 |
|---|---|
| conditional_diffusion/ | 直接条件扩散 |
| residual_diffusion/ | 条件残差扩散，含直接预测底座与训练入口 |
| rectified_flow/ | 条件整流流匹配（不属于扩散模型） |

归档目录archives/将提供三个分别打包的ZIP及合并ZIP。GitHub上传时使用对应ZIP解压后的代码目录，不上传整个归档目录。

各包的核心网络、损失、采样源码保持原样；可移植入口新补齐，历史调度单列reference/。数据、checkpoint、完整预测、临时日志、Python安装目录和认证设置不在代码包内。
