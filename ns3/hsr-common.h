/*
 * hsr-common.h — Python(RL) <-> NS-3(LENA) 二进制协议（Stage B）
 *
 * Python 侧镜像定义见 src/envs/ns3_env.py 的 HSR_STATE_DTYPE / HSR_ACTION_DTYPE。
 * 两侧字节布局必须逐一对齐（#pragma pack(1) + static_assert 校验尺寸）。
 *
 * 传输：子进程 stdin/stdout 二进制流（stdio 管道），锁步协议：
 *   C++ 启动 → 写 4 字节 magic（HSR1=56B / HSR2=72B / HSR3=84B / HSR4=99B）
 *   每 dt(100ms):  C++ 写 HsrState → 阻塞读 HsrAction(8B) → 应用动作
 *   结束: C++ 写 done=1 的完整 HsrState（尽力而为，管道已关则忽略）
 *
 * magic 必须与结构体尺寸同步升级：本项目已多次栽在「旧 Python 按旧尺寸解析
 * 新二进制而静默错位」上，故每次扩字段都同步改 magic，握手失败即显式报错。
 *
 * RSRP 换算：LENA 上报原始值 v ∈ [0,97]，dBm = v - 140。
 */

#pragma pack(push, 1)

typedef struct
{
    uint32_t step;             /* 决策步序号（0 起） */
    uint32_t time_ms;          /* 仿真时刻 ms */
    uint16_t serving_cell;     /* 服务小区 cellId */
    int16_t serving_rsrp_dbm;  /* 服务小区 RSRP dBm（无测量时 -200） */
    int16_t nbr_cell[3];       /* top-3 邻区 cellId（按 RSRP 降序） */
    int16_t nbr_rsrp_dbm[3];   /* top-3 邻区 RSRP dBm */
    uint16_t n_nbr;            /* 实际邻区数（0-3） */
    float speed_mps;           /* UE 速度 */
    float heading_cos;         /* 运动方向与"指向服务站"方向的 cos ∈ [-1,1] */
    float dist_to_serving_m;   /* UE 到服务站 2D 距离 */
    float ue_x;                /* UE x 坐标 m */
    float throughput_mbps;     /* 上一间隔实测下行吞吐 Mbps（FlowMonitor） */
    uint16_t ho_interrupt_ms;  /* RRC Start->EndOk duration ms, NOT business packet gap */
    uint8_t ho_event;          /* 上一间隔内完成切换 0/1 */
    uint8_t rlf;               /* 无线链路失败 0/1 */
    uint8_t done;              /* episode 结束 0/1 */
    /* ---- 负载/隧道可观测（占用原 reserved[5]，**协议尺寸不变**）----
     * 动机（设计文档 P1-6）：旧观测里没有任何负载维度，只有 buffer 作代理，
     * 于是「切向空闲次强小区做负载分流」这一目标在观测上**不可实现** —— RL
     * 看不到「空闲」。这里给出每小区背景 UE 数作为负载代理（NS-3 侧已维护
     * 每个背景 UE 的 servingCell，统计成本 O(n_bg)）。
     * load_* 为该小区附着的背景 UE 数（截断到 255）；in_tunnel 为 UE 是否位于
     * 隧道段（洞口/洞内策略本质不同，需显式特征）。 */
    uint8_t load_serving;      /* 服务小区附着的背景 UE 数 */
    uint8_t load_nbr[3];       /* top-3 邻区各自附着的背景 UE 数 */
    uint8_t in_tunnel;         /* UE 是否在隧道段 0/1 */

    /* ---- 容量/业务量指标（100~1000 用户规模验证用）----
     * 高铁运营关心的是「多少人能同时上网、每人多少带宽、时延多少、什么时候到
     * 极限」。旧协议只有控车单用户的吞吐，无法回答这些问题。
     * HSR3 修正 HSR2 的三处口径缺陷（2026-09-22，P0）：
     *   1) tp_agg_mbps 是**全网**聚合，而 n_ue_cell 是**单小区**附着数 —— 用
     *      它算「人均速率」会高估约 n_cells 倍。故新增 tp_agg_cell_mbps
     *      （服务小区聚合），两者口径各自显式命名。
     *   2) n_ue_cell 的计数器只统计背景 UE，控车未计入 → 与注释不符。现由
     *      C++ 侧 +1 补齐（控车必然附着于 g_servingCell）。
     *   3) tp_min_ue_mbps 取「本 100ms 内有流量用户的最小值」，1000 用户时
     *      大量用户某间隔为 0（调度/突发 Off）→ 噪声极大且趋 0。改为**截至
     *      当前的 per-UE 平均吞吐分位数** tp_ue_p05 / tp_ue_p50，并补 Jain
     *      公平性指数（三者都是「边缘用户速率/公平性」的正确口径）。
     * 协议 72B → 84B（MAGIC 同步到 HSR3）。 */
    uint16_t n_ue_cell;        /* 服务小区附着的 UE 总数（背景 + 控车） */
    uint16_t n_bg_total;       /* 场景中背景 UE 总数 */
    float tp_agg_mbps;         /* **全网**聚合下行吞吐 Mbps（全部 UE 之和） */
    float ctrl_delay_ms;       /* 控车流本间隔平均单向时延 ms（FlowMonitor） */
    float tp_agg_cell_mbps;    /* **服务小区**聚合下行吞吐 Mbps（人均速率分子） */
    float tp_ue_p05_mbps;      /* per-UE 平均吞吐的 5 分位（边缘用户速率） */
    float tp_ue_p50_mbps;      /* per-UE 平均吞吐的中位数 */
    float jain;                /* 公平性指数 (Σx)²/(n·Σx²)，x = per-UE 平均吞吐 */
    /* HSR4: request result 0=none,1=accepted (dispatched, not RRC success),
       2=no neighbor,3=same target,4=busy,5=cooldown,6=unknown cell,7=unsupported. */
    uint8_t ho_request_result;
    uint8_t ho_in_progress;
    uint8_t ho_failed;          /* interval failure event; not a recovery command */
    float interval_ms;
    float no_rx_interval_ms;   /* interval with zero ctrl RX packets, not packet gap/p95 */
    float no_rx_run_ms;        /* consecutive zero-RX intervals, including censored tail */
} HsrState;

typedef struct
{
    uint8_t action;            /* 0..6，语义与 Python env 一致：
                                  0 stay; 1/2/5 HO->nbr1/2/3;
                                  3/4 dual(+nbr1/2); 6 triple。
                                  仅 1/2/5 触发真实 NS-3 切换，
                                  dual/triple 由 Python 侧近似（LENA 无 EN-DC） */
    uint8_t reserved[7];
} HsrAction;

#pragma pack(pop)

static_assert(sizeof(HsrState) == 99, "HsrState byte layout mismatch (expect 99)");
static_assert(sizeof(HsrAction) == 8, "HsrAction byte layout mismatch (expect 8)");
