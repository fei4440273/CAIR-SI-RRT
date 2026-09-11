# CAIR-SI-RRT

CAIR-SI-RRT 是面向机械臂动态环境规划的增量修复方法。它以 SI-RRT 的安全时间区间搜索为
基础，在障碍物预测持续更新时定位受影响的树结构，传播失效状态，并尽量保留仍然有效的
搜索结果。预测区域使用保形校准提供统计覆盖意义，但本项目的主要算法改动集中在受影响
区域检索、增量树修复和双树重新连接，而不是提出一种新的 RRT*。

本仓库是 Candidate V8 的公开精简源码版，候选标识为
`candidate-v8-reachable-ellipsoid-axis-v1`。

## 目录

- `MSIRRT/`：CAIR-SI-RRT/SI-RRT 规划器、在线更新和树修复实现。
- `STRRT_Planner/`：规划器编译所需的碰撞检测与配置读写模块。
- `scripts/uncertainty/`：动态场景、预测区域、校准和在线试验工具。
- `tests/uncertainty/`：与公开 Python 工具对应的测试。
- `examples/`：1 个和 20 个动态球的最小测试场景。
- `protocols/`：Candidate V8 的冻结实验协议。
- `VERSION.json`：版本身份与发布树摘要。

## Docker 构建与测试

需要 Docker 和 Docker Compose。首次构建依赖镜像会下载并编译第三方依赖：

```bash
docker compose build cair
docker compose run --rm cair ./scripts/build_and_test.sh
```

构建脚本使用 Release 配置生成 `build/MSIRRT/MSIRRT_Planner` 和
`build/MSIRRT/MSIRRT_RepairSequence`，随后执行全部 C++ 测试。

运行最小动态预测与在线规划检查：

```bash
docker compose run --rm cair ./scripts/run_smoke_test.sh
```

冒烟结果写到容器内的 `/tmp/cair-si-rrt-smoke`，脚本会检查结果 JSON 的结构。它用于确认
代码链路可以运行，不是论文性能实验。

## 本机构建

如果系统已经安装 CMake、Eigen3、RapidJSON、URDF、Orocos KDL、FCL、Coal、Assimp 和
项目所需的 ROS 2 运行库，可以直接执行：

```bash
./scripts/build_and_test.sh
python3 -m pip install -r requirements-dev.txt
PYTHONPATH=. python3 -m pytest -q tests/uncertainty
```

`requirements-dev.txt` 只包含运行公开 Python 测试所需的 pytest；核心场景与预测脚本只使用
Python 标准库。


## Candidate V8 配置

论文试验使用的核心在线配置包括：

```text
MSIRRT_ADAPTIVE_REPAIR_POLICY=optimized
MSIRRT_CONDITIONAL_SAMPLING=1
MSIRRT_REACHABLE_ELLIPSOID_SAMPLING=1
MSIRRT_FEASIBILITY_AWARE_HORIZON=1
MSIRRT_REUSE_STORED_PREVIOUS_INTERVALS=1
MSIRRT_REUSE_FIRST_PREDICTION_SCENE=1
MSIRRT_REUSE_PARSED_PREDICTION_METADATA=1
MSIRRT_COOPERATIVE_DEADLINE_CHECKS=1
MSIRRT_MAX_PLANNING_TIME=0.1
```

完整参数见 `protocols/candidate_v8_protocol.json`。

## 已验证结果与边界

在原工程的冻结协议、冻结场景、相同种子和每次 100 ms 预算下，本地重新运行得到：

- 主实验完成 `455/500`，五类场景分别为 `100/85/85/100/85`。
- 强制预测失效实验完成 `91/100`。
- 两组实验中记录的八类安全与截止期事件均为 0。

这些数字说明恢复后的 V8 在对应仿真协议下复现了冻结结论，不代表形式化安全保证，也不能
外推到任意机器人、障碍物或硬件系统。本精简仓库不包含数百 GB 的原始结果、论文草稿和
历史失败版本，因此不能单独从本仓库端到端重建论文全部图表。

## 版本完整性

`VERSION.json` 记录原工程的运行源码摘要、已验证重建二进制哈希、冻结协议哈希和本发布树
摘要。二进制没有提交到仓库；使用不同编译器、路径或链接器重新构建时，二进制 SHA-256
可能不同。

## 上游项目与许可证

本项目基于 [ManipulationPlanning-SI-RRT](https://github.com/PathPlanning/ManipulationPlanning-SI-RRT)
修改。SI-RRT 对应论文为 *Safe Interval Randomized Path Planning For Manipulators*，发表于
ICAPS 2025。具体署名、链接和修改边界见 `NOTICE.md`。

代码沿用上游 MIT 许可证，详见 `LICENSE`。使用或引用本项目时，应同时尊重上游项目和相关
论文的署名要求。
