# PhysX Cloth Twist

用 `PxDeformableSurface` 复现 Newton 的
`newton/examples/cloth/example_cloth_twist_xpbd.py` 场景。
模拟在 PhysX C++/CUDA 中运行，窗口使用仓库已有的 OpenGL/FreeGLUT 渲染器。
示例默认开启原生 FEM/TGS 内部的实验性 **Planar-DAT 自碰撞保护**。
Newton 仅在导出和离线审计时使用，不参与 PhysX 运行时求解。

## 本机直接运行

```bash
cd /home/oem/code/repos/PhysX/physx/bin/linux.x86_64/checked
DISPLAY=:1 ./SnippetClothTwist_64 --paused
# 同参数关闭 DAT，对照原生接触路径：
DISPLAY=:1 ./SnippetClothTwist_64 --paused --no-dat
```

按 **空格**开始或暂停，**R** 重置并重播，**N** 切换线框，
**W/A/S/D** 或方向键移动相机，**鼠标左键拖动**转动视角，**Esc** 退出。
默认运行 600 帧（10 秒模拟时间），结束后保留窗口供观察。
窗口标题显示模拟时间和状态；模拟时间不一定等于真实耗时。
橙红色、绿色点分别表示两端受驱动的顶点。

## 重新导出网格和编译

以下命令在 PhysX 仓库根目录执行。Python 仅用于一次性导出参考网格，
使用本机 Newton 的虚拟环境；运行 C++ 示例无需 Python 或 USD 库。

```bash
/home/oem/code/repos/newton/.venv/bin/python \
  physx/snippets/snippetclothtwist/export_mesh.py \
  physx/bin/linux.x86_64/checked/square_cloth.mesh

cmake -S physx/compiler/public -B physx/compiler/linux-gcc-checked
cmake --build physx/compiler/linux-gcc-checked --target SnippetClothTwist -j 8
```

构建命令复用本机已配置的 GCC checked 构建目录和 CUDA 依赖。
新环境需先按 `physx/documentation/platformreadme/linux/README_LINUX.md`
配置启用 GPU 和 Snippets 的构建。
生成的网格、程序和库放在已被本机 Git 排除的 `physx/bin/` 下。

## 场景对应关系

- 使用 Warp 的同一份 `square_cloth.usd`，经 Newton 的相同 builder 变换和
  质量计算导出：2,500 个顶点、4,802 个三角形，尺寸为 1.5 m × 1 m。
- 保持 Newton 的模拟坐标：初始网格在 YZ 平面，零重力。
  相机位于 `(2.25, 0, 0)`；显示时做 Z-up 到渲染器 Y-up 的转换。
- 按 Y 坐标的最大/最小值选择两端，共 100 个顶点。
  两端绕各自中心的 Y 轴反向转动，每端速度 60°/s，10 秒后停止旋转。
  保留参考程序的 1/600 秒目标时间滞后。
- 60 Hz 帧步长，每帧默认 10 次 `simulate/fetchResults`。
  固定端逆质量设为零，每子步写入到达下一目标位置所需的速度，
  让 PhysX 在 TGS 内部时间切片中推进固定端。
- 面密度使用 Newton 导出的顶点质量（0.2 kg/m²）。
  PhysX 使用 1 mm 材料厚度、2.5 MPa 杨氏模量、0.25 泊松比、0.2 摩擦系数。
  此时厚度积分后的 Lamé 参数 μh=λh=1000，与 Newton TGS 分支输入一致。
  默认启用自碰撞，rest offset 为 2 mm，contact offset 为 3.5 mm。
- 默认采用 12 次 PhysX TGS 位置迭代；弹性/弯曲阻尼为 600，
  对应参考脚本中 TGS 分支的阻尼选择。

这是相同场景的 **PhysX FEM/TGS + Planar-DAT 实现**，不是 Newton global XPBD
求解器的移植。弯曲刚度按 PhysX 的
`B * thickness³ * edgeLength / dualLength` 换算，以全网格平均 dualLength
近似匹配 Newton 的 `1e-3 * edgeLength`，不会逐边完全一致。
接触算法及接触参数含义也不同，因此不能保证相同褶皱或无穿透。

## DAT 如何接入原生 TGS

API 开关（在 `scene.addActor()` 之前设置）：

```cpp
cloth->setDeformableSurfaceFlag(
    PxDeformableSurfaceFlag::eENABLE_SELF_COLLISION_DAT, true);
```

开关默认关闭，只在这个示例中默认开启。演员加入场景后不允许切换；
实验范围为 TGS 中隔离的三角布料自碰撞，不支持 PGS，也没有验证刚体、
多块布料或附件耦合的反力一致性。DAT 的保护只覆盖本布料内部，
不能将它当成这些耦合场景的防穿透保证。

接入位置位于 `PxgFEMClothCore.cpp`，每次外层 simulate 的第一轮建立
检测基准和 VT/EE 候选，每个 TGS slice 执行：

```text
原生位置预测 → DAT 提交
原生 VT/EE 接触位移 → 每次 applyExternalDelta 后 DAT 提交
原生膜/弯曲求解及 copy/remap 平均 → DAT 提交
其余外部位移 → DAT 提交
原生速度阻尼
```

求解内核暂时写出的候选位置，经 DAT 接受后才暴露给下一个接触阶段。
`base` 保存本次检测基准，`accepted` 保存上一阶段已接受的位置。
对四顶点 VT/EE 对计算分离平面及逐顶点截断比例（γ=0.85），然后用
`0.425 * contactOffset` 位移预算保证基准附近候选的覆盖范围。
固定端也参与截断；不会通过强制覆盖固定点来绕过保护。
提交时用实际位置修正同步速度和累计位移，再进入下一阶段。
核心 CUDA 实现见 `source/gpusimulationcontroller/src/CUDA/ClothDat.cuh`，
由 Newton `fem_contacts.py` 改写，保留其 Apache-2.0 许可及出处；
许可副本见同目录 `ClothDat.LICENSE`。

本版优先验证接入语义：候选通过 GPU 穷举 AABB 筛选生成，使用全部唯一边，
不沿用原生接触的拓扑/距离过滤。窄阶段使用双精度最近特征和分离平面。
候选查询按顶点/边分行分配 GPU 工作，复用查询端数据，跳过 EE 下三角，
消除原先每对索引的 64 位除法。大网格枚举成本仍是二次量级。
候选容量为每顶点 128 条。溢出、非分离初态、非有限候选位置会设置
sticky 错误、拒绝位置提交，并通过 PhysX 错误回调报告；错误后需重建演员。

## 验证与选项

```bash
cd /home/oem/code/repos/PhysX/physx/bin/linux.x86_64/checked
./SnippetClothTwist_64 --headless --frames 600
./SnippetClothTwist_64 --headless --frames 600 --no-self-contact
# 导出完整轨迹（程序运行成功后再审计）：
./SnippetClothTwist_64 --headless --dump /tmp/physx-dat.bin
/home/oem/code/repos/newton/.venv/bin/python \
  /home/oem/code/repos/PhysX/physx/snippets/snippetclothtwist/audit_trajectory.py \
  /tmp/physx-dat.bin --report /tmp/physx-dat-audit.json
```

每帧检查位置和速度为有限值、坐标绝对值小于 2 m、固定端位置误差
小于 2e-5 m。每 60 帧打印检查结果，失败时退出码为 1；
实时检查不包含独立几何交叉审计。`audit_trajectory.py` 对每帧重新建立几何
查询，调用 Newton 的双精度非 incident 边—三角形交叉审计内核，
不使用 DAT 候选、平面或求解器，并用 NumPy 独立检查驱动轨迹。
报告包括完整帧数、有限状态、边界误差、每帧交叉数量和实例。

2026-09-14，本机 RTX 5090 D v2，checked 构建，600 帧对照结果：

| 模式 | 有交叉的帧数 | 第一处检出交叉的帧 | 最大固定端误差 |
|---|---:|---:|---:|
| FEM/TGS + DAT | 0 / 600 | 无 | 8.19e-7 m |
| 同参数 `--no-dat` | 486 / 600 | 115 | 8.19e-7 m |

两条轨迹均完成 10 秒、保持有限值和场景范围；上述是单次本机场景测试，
不构成任意场景、任意时间步或最小厚度的无穿透保证。
另用初始完全重合的非 incident 三角形验证了 status=2 的拒绝路径。
100 层近邻平行三角形触发 status=1，验证候选溢出不会丢掉候选后继续提交。
两项可复现回归（在仓库根目录运行，无需 Newton Python 依赖）：

```bash
python3 physx/snippets/snippetclothtwist/test_dat_failures.py -v
```

最终优化版完整 600 帧独立审计仍为 0 交叉，最大固定端误差为 8.19e-7 m；
优化查询与原穷举查询经过 600 帧、6,000 次检测的完整候选集合核对。
两项错误拒绝回归通过；CUDA Compute Sanitizer memcheck 与 synccheck
各运行 3 帧，均为 0 errors。

## 性能复测

```bash
# 在上述二进制目录中执行；相同网格、10 子步、12 次迭代：
./SnippetClothTwist_64 --benchmark
# 调试用：逐次核对优化查询和原穷举查询的完整候选集合：
PHYSX_DAT_AUDIT_CANDIDATES=1 ./SnippetClothTwist_64 --headless
# 调试用：改用原穷举查询，其余优化仍开启：
PHYSX_DAT_EXHAUSTIVE=1 ./SnippetClothTwist_64 --benchmark
```

`--benchmark` 自动使用无窗口模式，前 10 帧作为预热，默认统计其余 590 帧，
输出平均值、中位数和 P95。计时涵盖驱动更新、模拟、同步、GPU 回读和逐帧
检查，不含初始化及渲染；性能对照时不要同时使用 `--dump`。
环境开关在进程启动时读取；不需要时请取消设置。候选核对模式会运行两遍
查询并同步回读数据，只用于正确性检查，不能用于性能测量。

除候选查询外，DAT 将有限值检查并入截断内核，将下一阶段的比例初始化
并入提交内核，省去独立的 prepare 启动；无需截断时跳过 atomicMin，
错误状态改为每个 warp 原子读取一次并广播，位移超出预算时才计算双精度
平方根。没有布料附件时跳过空的附件位移阶段。
这些优化保持原有子步数、迭代数、双精度窄阶段和 DAT 位移预算。

2026-09-14，同机 RTX 5090 D v2 / checked 构建，60 帧 Nsight CUDA 剖析：

| 指标 | 优化前 | 最终优化后 | 降幅 |
|---|---:|---:|---:|
| DAT 内核累计 GPU 耗时 | 538.28 ms | 382.21 ms | 29.0% |
| 全部内核累计 GPU 耗时 | 1240.90 ms | 1056.60 ms | 14.9% |
| 内核启动次数 | 456,015 | 391,215 | 14.2% |

上述 GPU 累计耗时不等于端到端帧耗时或渲染 FPS。两次剖析都包含相同的
60 帧及初始化；原版记录包含 43,200 次 `cloth_datPrepare`，最终版没有该内核。
对照必须在各自程序和 GPU 库目录启动，并用相同的绝对 `--mesh` 路径。
PhysX 的 GPU 库选择受工作目录影响，仅切换可执行文件路径不足以隔离版本；
本次最终性能对照还核对了 `/proc/<pid>/maps` 中实际加载的 GPU 库路径。

三轮 600 帧端到端计时（前 10 帧预热，均核对实际 GPU 库路径并通过实时检查）：

| 轮次 | 优化前平均帧耗时 | 最终优化后平均帧耗时 |
|---|---:|---:|
| 1 | 36.43 ms | 33.25 ms |
| 2 | 57.84 ms | 33.65 ms |
| 3 | 35.75 ms | 32.59 ms |

三次平均帧耗时的中位数为 **36.43 → 33.25 ms，下降约 8.7%**。
第二轮基线波动明显（P95 为 114.52 ms），因此不采用三轮均值夸大收益。
这是无窗口、完整 10 子步 / 12 迭代的测量；桌面负载和 GPU 状态会影响结果。

固定端参与 DAT 截断，复杂褶皱下可能受阻；性能试验中出现过固定端误差超限
而暂停的情况。成功轨迹不能保证每次运行都完成 600 帧。程序保留失败停机和
非零退出码，失败运行不纳入完整 600 帧耗时对照。这些测量针对本机场景，
不能证明任意 GPU 或网格上的全局最优。

其他参数：`--mesh PATH`、`--substeps N`、`--iterations N`（1–255）。
`--frames` 大于 600 时，两端会完成旋转后保持位置，布料继续模拟。
