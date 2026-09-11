# 上游来源与修改说明

本仓库基于 PathPlanning 团队公开的
[ManipulationPlanning-SI-RRT](https://github.com/PathPlanning/ManipulationPlanning-SI-RRT)
开发，并保留其 MIT 许可证和原版权声明。

上游 SI-RRT 的论文为：

Nuraddin Kerimov, Aleksandr Onegin, and Konstantin Yakovlev. “Safe Interval Randomized
Path Planning For Manipulators.” Proceedings of the International Conference on Automated
Planning and Scheduling, 35(1), 213–217, 2025.

论文页面：<https://ojs.aaai.org/index.php/ICAPS/article/view/36120>

本精简版中的 CAIR-SI-RRT 修改主要涉及动态障碍物预测更新下的受影响区域定位、树节点与边
失效传播、保留有效搜索结构的增量修复、双树重新连接、预测区域生成及相应测试和实验工具。
这些修改不改变上游代码的许可证要求，也不应被理解为本仓库作者创作了全部基础规划框架。

