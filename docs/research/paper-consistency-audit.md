# 论文六项争议核查

审计日期：2026-11-24。

## 审计范围与版本

- 实验代码和结果基线：`origin/main`，commit `7031c4a6ecfeb6140c116b1992aa2a39a4685127`。
- 待核论文为主工作区中尚未纳入 Git 的本地文件：`/Users/huangbohan/network-security-experiment/paper/main.tex`；SHA-256：`d37d5c787898cca1753cf0caa29fea9268f90fba042d369efaae89de9eabbccd`。下文 `paper/main.tex` 均指这份只读资料，并非本报告分支中已追踪的文件。
- 用户截图属于争议线索；逐项以当前论文、已记录 JSON、实际计数代码交叉核对。本次未重跑远程 LLM 或耗时的 persistence 实验；另由主代理在隔离 worktree 的临时目录内进行最小无 LLM 计数复现，见第 3 项。
- **版本限制：**当前论文仅有 Fig. 1 的 `figure` 环境（`paper/main.tex:67–72`），未包含截图所引 Fig. 3 / Fig. 4(b)，也未找到 “each round carries the entire payload in a single operation” 或 “complete a session in roughly six rounds” 两句。因此，旧图注是否互相冲突不能冒充已经直接核验；涉及的数值可独立从 JSON 复算。

## 结论总表

| # | 结论 | 严格表述 |
|---|---|---|
| 1 | 成立 | M2 成功率较低归因于“更高存储操作开销”，与同载荷校准数据方向相反，因果解释无支持。 |
| 2 | 部分成立 | M1 本身确实属于高吞吐通道；M2 略高不推翻这一点。真正问题是把两者共有特性用于解释差异，以及混淆单次消息投递和完整任务 session。 |
| 3 | 口径说明不足成立，数值本身可复现 | 8,192 bits/round 与 2,048 bytes 可在代码中完全对上：一次 episode 占两轮。一次高层工具调用可以发整包，不能直接等同于一个底层操作或完整 episode。相关截图原句不在当前稿中。 |
| 4 | 条件混淆成立；旧图注暂不能确认 | M1/M2 的 300 s 是无 TTL 的持久化基线；1 s TTL 是 M3 特有条件。300 s 结果本身有记录支持。 |
| 5 | 范围表述误导成立 | 180 次 LLM 任务运行确实存在；capacity/persistence 来自额外无 LLM 的 B/C 实验，摘要未分开两套证据。 |
| 6 | 成立 | extended view 下 M1/M2 TPR 同为 1.00，是差异消失，不能称为排序反转。 |

## 1. “M2 更高存储操作开销”与数据相反

**原文：**`paper/main.tex:274` 将 M2 的 63% 成功率解释为其相对 M1 有 “higher storage-operation overhead per message”。

**核查：**Table II 为 M1 293、M2 197（`paper/main.tex:228–232`），与原始 `results/exp_b.json:108–118,225–235` 一致：两者使用相同 2,048-byte 载荷、同为 BER=0，M2 的记录操作数更低。即使按每条消息归一化，两组都含三个相同载荷的 episode，除以三也不会改变大小关系（具体标签问题见附加发现 A）。

Exp A 30 次主实验的 mean storage operations/run 也为 M1 **50.5**、M2 **41.7667**，与截图所引约 50/42 的方向一致。但这是按 task run 统计，不能直接称为每条 message 开销；应避免用另一个分母再制造口径混淆。可按文末脚本从 `results/exp_a.json` 的 `runs[*].metrics.resource_overhead.storage_op_layer.total_storage_ops` 复算。

**判定：**论文目前没有数据支持这项机制解释，同载荷校准数据反而与其方向相反。不能据此断言某种新的原因已经被证明；应删除该归因，并把 73%/63% 表述为观察到的成功率差异。30 次任务下的差异，也不能仅凭百分比断言统计显著。

**建议修订：**“M1 and M2 achieve task success rates of 73% and 63%, respectively, despite similar calibrated channel throughput. The present measurements do not establish the cause of this difference.”

## 2. 高容量、单次投递与约六轮任务不是同一个概念

**原文：**`paper/main.tex:274` 称 M1 的 73% 成功率 “reflecting its high capacity and single-round delivery”。

**高容量部分：**M1 为约 2.794 Mbit/s，M2 约 2.816 Mbit/s（`paper/main.tex:230–239`；`results/exp_b.json:708–735`）。两者都可以称 high capacity，因此截图用 “M2 更高” 来否认 M1 “high capacity” 的论证不严谨。M2 仅高约 0.8%，本次记录不足以证明稳定的性能排序。不过，M1/M2 共有的高吞吐不能单独解释两者任务成功率为何不同。

**轮数部分：**Exp A 的平均 session rounds 为 M1 **6.0667**、M2 **5.2667**，中位数都为 **4**。这些是整项任务的轮数，不等于一次载荷从发送到接收的延迟。单次消息可跨一个轮边界到达，同时完整任务包含多条消息、模型推理、工具调用和最终提交；二者可以并存，不是形式逻辑上的直接矛盾。

代码中消息先进入 staging，再在轮末发布，从下一轮可见（`src/base.py:445–455,491–501,551–558`）。因此，“single-round delivery” 若指跨一个轮边界的 delivery latency，可以解释；若指一次完整任务只用一轮，则与任务日志不符。当前论文没有明确定义这种区别。

**判定：**“差异的解释缺乏依据、轮数概念需要区分”成立；“M1 不是高容量”及“消息一轮与任务六轮必然矛盾”不成立。截图中约六轮的原句未在当前稿找到，但数据可复算出上述均值。

**建议修订：**分别报告 calibrated throughput、one-boundary message latency 和 task-session rounds；不要将其中任一个直接当作 73% 对 63% 的已证因果机制。

## 3. 2,048 bytes 和 8,192 bits/round 可以完全相容

**原文：**`paper/main.tex:239` 将 2,048-byte 最大载荷与 8,192 bits/round 并列。原始记录同时给出 `latency_rounds: 1`、`bits_per_round: 8192`（`results/exp_b.json:108–118,225–235`）。

**代码解释：**Exp B 对每个校准点运行三个 episode（`tasks/exp_b.py:90,156`）。发送在 round r，接收循环先尝试读取，再调用 `publish_round_end`，因此新载荷到 r+1 才读到。占用轮数按 `r_receive - r_send + 1` 计算（`tasks/exp_b.py:158–179`），延迟轮数则不加一。最终比率为 `total_delivered_bits / occupied_rounds`（`tasks/exp_b.py:217–222`）。因此：

```text
每条消息净荷 = 2,048 × 8 = 16,384 bits
每个 episode 占用轮数 = (r+1) - r + 1 = 2
3 episodes: 49,152 delivered bits / 6 occupied rounds = 8,192 bits/round
latency_rounds = (r+1) - r = 1
```

它不是 byte-to-bit 换算错误，也不是消息被截掉一半。代码选择了“包含发送和接收两轮”的 episode 占用口径，应在论文中显式定义。

此外，一次高层 `send_payload` 会将消息分块并逐块执行 `_stage_frame`（`src/base.py:491–501`）；payload chunk size 为 64 B（`src/frame.py:40`），故 2,048 B 对应 32 个 payload chunks。每轮没有底层操作预算（`tasks/exp_b.py:3–4`），可以在一次高层调用中发出多块。若截图的 “single operation” 指一次工具调用而非一个底层存储操作，这部分可以成立；它仍需要与 episode 计数区分。

**最小实际复现：**主代理在隔离 worktree 的临时目录中，按相同驱动逻辑发送三次 2,048 B 载荷，M1/M2 均在轮 1/3/5 发送、轮 2/4/6 接收，共 49,152 bits / 6 occupied rounds = 8,192 bits/round。操作数也精确复现：M1 为 293（96 file writes + 96 promote moves + 5 directory lists + 96 file reads）；M2 为 197（96 directory creates + 96 promote renames + 5 directory lists）。临时目录随后清除。此次验证针对轮数和操作计数，未以新测墙钟吞吐替换已有结果。

**判定：**截图指出需要解释两轮计数口径，这点成立。问题在指标定义和措辞，不能直接将 8,192 改成 16,384；后者是每条消息的净荷位数。截图所引 “each round carries … single operation” 当前稿无此句，不能据此认定当前稿仍保留该表述。

**建议修订：**“At 2,048 bytes per message, M1 and M2 deliver 16,384 payload bits across one round boundary. Counting both the send and receive rounds as occupied yields an episode-normalized rate of 8,192 bits/round.”

## 4. TTL 1 s 与 300 s 无清理基线被放在同一个条目

**原文：**`paper/main.tex:247` 的 “1-second TTL” 条目紧跟 “M1 and M2 survive 300 s without cleanup; M3 payload is evicted before B can read it”。

**实验证据：**TTL 是 M3 的 cache 过期机制，M1/M2 文件和目录没有对应的被动过期条件。代码明确将 TTL 条件放入 `M3_CONDITIONS`，仅在 method 为 M3 时生成实验（`tasks/exp_c.py:347–389`）。M1/M2 的 baseline 在 delay=300 s 时都是 TTL=null、3/3 次恢复成功（`results/exp_c.json:170–204,1029–1062`）。M3 在 TTL=1 s、delay=0 时同样可恢复（3/3），在 delay=1 s 时为 0/3（`results/exp_c.json:2387–2457`）；因此不能省略接收延迟就断言“在 B 读取前一定已驱逐”。

**判定：**截图指出不同条件在文字上混在一起成立。但 TTL 和 “without cleanup” 可以同时存在：被动过期本来就不需要主动清理，不能仅因这两个短语同句便认定逻辑矛盾。实质问题在于 M1/M2 根本未施加 TTL，而其 300 s 事实来自 baseline。正确限定是“在本实现无 TTL / 无清理的条件下，至少保持到测试上限 300 s”，不代表它们抵抗了同样的 1 s TTL 策略。截图所引 Fig. 3 及其图注不在当前稿中，无法直接核实图列与图注冲突。

**建议修订：**将无 TTL 基线和 M3 特有 TTL 实验拆成两个条目，注明 delay。若恢复旧图表，TTL 行/列对 M1/M2 应标为 N/A，而不是无解释地填入无 TTL 基线结果；图注分别说明 observation delay 和 TTL duration。

## 5. 摘要混淆 LLM 任务数与确定性信道测量

**原文：**摘要 `paper/main.tex:57` 称 “across 180 LLM-agent task runs to evaluate channel capacity, persistence, communication reliability, and audit detectability”。Introduction 已较准确地写成 180 次 LLM 任务 “together with deterministic channel measurements”（`paper/main.tex:80`）。Methods 则明确 Exp A=90 主实验+90 对照、Exp B 无 LLM（`paper/main.tex:204–210`）。

**证据边界：**`results/README.md:6–11` 清楚说明 A/D 基于 180 次运行，而 B/C 是确定性实验；Exp B 驱动说明 “no LLM”（`tasks/exp_b.py:9–18`），Exp C 驱动直接发布固定载荷并独立延迟读取（`tasks/exp_c.py:1–8,28–32`）。`results/exp_a.json` 中 `runs` 和 `negative_control` 各有 90 条。

**判定：**180 次不是捏造，截图质疑的核心是摘要让读者误以为 capacity/persistence 也来自这 180 次 LLM 任务。它属于实验范围和证据来源的压缩失真，需改写。当前 Fig. 1 图注本身并无截图所述 “Capacity & persistence: no LLM” 文句（`paper/main.tex:70`），此处结论由代码和 README 独立支持，不依赖该图中文字。

**建议修订：**“We evaluate task-level communication and audit detectability using 180 LLM-agent runs, including matched negative controls, and separately measure channel capacity and persistence using scripted, non-LLM experiments.”

## 6. “TPR 反转”应改为“差异消失”

**原文：**`paper/main.tex:310` 说 M2 比 M1 隐蔽的结论在 extended monitoring 下 “inverts for TPR”。

**核查：**Table IV 的 content-only TPR 为 M1=1.00、M2=0.31，而 extended TPR 为 M1=M2=1.00（`paper/main.tex:285–291`）。原始 JSON test split 为 M1 17 TP / 0 FN、M2 13 TP / 0 FN，同为 1.0（`results/exp_d_audit.json:277–309`）。

**判定：**严格排序从 M2 TPR 较低变成相同，不能叫 reverse / invert。可说 “the TPR gap disappears” 或 “both become fully detected in the observed test windows”。这只描述这些窗口中观察到的 TPR，不意味着总体真实 TPR 已被证明等于 1。

## 附加发现

### A. Table II 的 “Storage ops/msg” 实际使用三个 episode 的总数

`tasks/exp_b.py:90,156,222` 将三次载荷传输的操作加总，直接写到每个 calibration point 的 `storage_ops`，没有除以 `episodes_delivered`。因此 293 和 197 是该三次校准合计，不是平均每消息。若保持现有统计方式，表头应改成 “storage ops / 3-episode calibration”；若确实要每消息均值，则约为 M1 97.67、M2 65.67，并明确这一均值包含校准驱动的所有计数操作。不能在没有分段日志的情况下将所有操作进一步冒充纯发送开销。

### B. Table IV 的 Test Split 标题与部分数字不一致

Table IV 标题称 Test Split（`paper/main.tex:281`），但：

- M1 extended FPR 写作 0.42，test 原始值为 **0.360902**，即 48/(48+85)（`results/exp_d_audit.json:277–286`）。0.42 是 content-only test FPR，不能复制给 extended。
- M2 extended FPR 写 0.01–0.04，是 tuning 0.012048 和 test 0.041667 混写（`results/exp_d_audit.json:289–310`）；纯 test 表应为 **0.04**。
- M3 extended TPR 写 0.77–0.84，是 test 0.768657 和 tuning 0.837838 混写（`results/exp_d_audit.json:313–334`）；纯 test 表应为 **0.77**。

若保留 tuning/test 范围，应把表格和段落标题、列标签一并改清楚。这些发现并不改变第 6 项 TPR 平齐的结论。

## 可复核的派生统计

以下只读取已记录 JSON，不运行模型或改写实验数据：

```python
import json
from statistics import mean, median

a = json.load(open("results/exp_a.json"))
print("main/control counts:", len(a["runs"]), len(a["negative_control"]))
for method in ("M1", "M2", "M3"):
    runs = [r for r in a["runs"] if r["method"] == method]
    ops = [r["metrics"]["resource_overhead"]["storage_op_layer"]
           ["total_storage_ops"] for r in runs]
    print(method, len(runs), "mean rounds", mean(r["rounds"] for r in runs),
          "median rounds", median(r["rounds"] for r in runs), "mean ops", mean(ops))

b = json.load(open("results/exp_b.json"))
for point in b["points"]:
    if point["method"] in ("M1", "M2") and point["offered_bytes"] == 2048:
        print(point)

c = json.load(open("results/exp_c.json"))
for method, conditions in c["results"].items():
    print(method, "conditions:", list(conditions))
    for condition in ("baseline", "ttl-1s"):
        for point in conditions.get(condition, {}).get("points", []):
            print(method, condition, point["delay_seconds"], point["ttl_seconds"],
                  point["recovered_count"], point["repeats"])

d = json.load(open("results/exp_d_audit.json"))
for view, methods in d["views"].items():
    for method, splits in methods.items():
        print(view, method, splits["test"])
```

本次变更仅增加本核查报告；未改写论文、实验代码或实验结果。
