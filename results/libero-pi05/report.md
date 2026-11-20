# MotrixSim 社区版 π0.5 LIBERO 40×50 评测

MotrixSim 完成 2,000 个闭环回合，成功 1839/2,000（91.95%）。平台归档的同权重 MuJoCo 参考为 1927/2,000（96.35%）。

| 套件 | 每回合动作上限 | MotrixSim | MuJoCo 参考 | 成功率差 |
|---|---:|---:|---:|---:|
| libero_spatial | 220 | 460/500（92.0%） | 490/500（98.0%） | -6.0 个百分点 |
| libero_object | 280 | 483/500（96.6%） | 492/500（98.4%） | -1.8 个百分点 |
| libero_goal | 300 | 458/500（91.6%） | 484/500（96.8%） | -5.2 个百分点 |
| libero_10 | 520 | 438/500（87.6%） | 461/500（92.2%） | -4.6 个百分点 |

四套件独立初始化，套件内依次执行任务 0–9 和初态 0–49。策略种子 7，环境种子 0。每回合先开爪预热 10 步，随后每 5 步至多调用一次 π0.5。两路 256×256 RGB 输入由 MotrixSim 渲染，动作控制为 20 Hz，物理步长为 2 ms。每个控制步检查原生任务谓词，成功立即终止。

参考来自 sim-platform 已封存的 2,000 回合。逐回合配对核对初态，逐套件核对模型清单、指令和调用计数。两引擎各自闭环运行，其动作与后续轨迹会分化。成绩适用于本次 MotrixSim 移植、模型和平台约定的评测协议。

40 个任务各有 50 个有效回合，2,000 个初态哈希逐例匹配。正式成绩包含 161 个 MotrixSim 达到动作上限的回合，执行异常为 0。MotrixSim 比 MuJoCo 少成功 88 次，成功率低 4.40 个百分点。差距最大的三项为 Goal 05 推盘到灶台前（19 对 47）、Spatial 05 从烤盅取碗（33 对 46）及 Spatial 07 从灶台取碗（37 对 50），每项比较相同的 50 个初态。

## 相同初态的结果分布

| 套件 | 两端均成功 | 仅 MotrixSim 成功 | 仅 MuJoCo 成功 | 两端均超时 |
|---|---:|---:|---:|---:|
| libero_spatial | 455 | 5 | 35 | 5 |
| libero_object | 475 | 8 | 17 | 0 |
| libero_goal | 448 | 10 | 36 | 6 |
| libero_10 | 417 | 21 | 44 | 18 |

## 逐任务成绩

| 套件 | 任务 | MotrixSim | MuJoCo 参考 |
|---|---|---:|---:|
| libero_spatial | 00 · pick up the black bowl between the plate and the ramekin and place it on the plate | 50/50 | 50/50 |
| libero_spatial | 01 · pick up the black bowl next to the ramekin and place it on the plate | 50/50 | 50/50 |
| libero_spatial | 02 · pick up the black bowl from table center and place it on the plate | 50/50 | 50/50 |
| libero_spatial | 03 · pick up the black bowl on the cookie box and place it on the plate | 47/50 | 49/50 |
| libero_spatial | 04 · pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 46/50 | 48/50 |
| libero_spatial | 05 · pick up the black bowl on the ramekin and place it on the plate | 33/50 | 46/50 |
| libero_spatial | 06 · pick up the black bowl next to the cookie box and place it on the plate | 50/50 | 50/50 |
| libero_spatial | 07 · pick up the black bowl on the stove and place it on the plate | 37/50 | 50/50 |
| libero_spatial | 08 · pick up the black bowl next to the plate and place it on the plate | 49/50 | 49/50 |
| libero_spatial | 09 · pick up the black bowl on the wooden cabinet and place it on the plate | 48/50 | 48/50 |
| libero_object | 00 · pick up the alphabet soup and place it in the basket | 44/50 | 49/50 |
| libero_object | 01 · pick up the cream cheese and place it in the basket | 47/50 | 49/50 |
| libero_object | 02 · pick up the salad dressing and place it in the basket | 49/50 | 50/50 |
| libero_object | 03 · pick up the bbq sauce and place it in the basket | 50/50 | 49/50 |
| libero_object | 04 · pick up the ketchup and place it in the basket | 49/50 | 47/50 |
| libero_object | 05 · pick up the tomato sauce and place it in the basket | 48/50 | 50/50 |
| libero_object | 06 · pick up the butter and place it in the basket | 49/50 | 50/50 |
| libero_object | 07 · pick up the milk and place it in the basket | 49/50 | 50/50 |
| libero_object | 08 · pick up the chocolate pudding and place it in the basket | 48/50 | 50/50 |
| libero_object | 09 · pick up the orange juice and place it in the basket | 50/50 | 48/50 |
| libero_goal | 00 · open the middle drawer of the cabinet | 50/50 | 48/50 |
| libero_goal | 01 · put the bowl on the stove | 50/50 | 50/50 |
| libero_goal | 02 · put the wine bottle on top of the cabinet | 50/50 | 48/50 |
| libero_goal | 03 · open the top drawer and put the bowl inside | 42/50 | 45/50 |
| libero_goal | 04 · put the bowl on top of the cabinet | 50/50 | 48/50 |
| libero_goal | 05 · push the plate to the front of the stove | 19/50 | 47/50 |
| libero_goal | 06 · put the cream cheese in the bowl | 50/50 | 50/50 |
| libero_goal | 07 · turn on the stove | 50/50 | 50/50 |
| libero_goal | 08 · put the bowl on the plate | 50/50 | 50/50 |
| libero_goal | 09 · put the wine bottle on the rack | 47/50 | 48/50 |
| libero_10 | 00 · put both the alphabet soup and the tomato sauce in the basket | 43/50 | 48/50 |
| libero_10 | 01 · put both the cream cheese box and the butter in the basket | 48/50 | 50/50 |
| libero_10 | 02 · turn on the stove and put the moka pot on it | 49/50 | 50/50 |
| libero_10 | 03 · put the black bowl in the bottom drawer of the cabinet and close it | 44/50 | 50/50 |
| libero_10 | 04 · put the white mug on the left plate and put the yellow and white mug on the right plate | 48/50 | 50/50 |
| libero_10 | 05 · pick up the book and place it in the back compartment of the caddy | 48/50 | 46/50 |
| libero_10 | 06 · put the white mug on the plate and put the chocolate pudding to the right of the plate | 44/50 | 45/50 |
| libero_10 | 07 · put both the alphabet soup and the cream cheese box in the basket | 46/50 | 50/50 |
| libero_10 | 08 · put both moka pots on the stove | 24/50 | 26/50 |
| libero_10 | 09 · put the yellow and white mug in the microwave and close it | 44/50 | 46/50 |

[逐回合结果](episodes.jsonl) · [结构化汇总](results.json) · [物理和渲染验证记录](validation.md)

## 仿真批次退出后的内存

680 个仿真批次中捕获 677 次退出，其中 677 次采样时未发现其他仿真进程。在这 677 次中，677 次退出后的作业匿名内存低于批内采样峰值。各套件首末各十个独立退出样本的匿名内存中位数，变化的绝对值最大为 1.007 GiB。逐批计入退出时的交换占用后，各套件中位数的变化范围为 -0.060 至 +0.001 GiB。作业交换空间的每秒采样峰值为 1.000 GiB。 [逐批观测](memory-report.md)
