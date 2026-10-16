/*
 * hsr-rl.cc — 高铁 LTE LENA 切换 + RL 锁步 IPC（Stage B/C）
 *
 * 场景：沿轨 N 个 eNB（固定间距 spacing，垂直偏移 offset，35m 高），
 * 单 UE 300km/h 匀速通过，RL（Python 侧）每 dt=100ms 决策一次切换。
 * NoOpHandoverAlgorithm 关闭自动切换 —— 所有切换由 RL 通过
 * LteHelper::HandoverRequest 显式触发（与 Python 侧 A3 兜底配合）。
 *
 * IPC（二进制协议见 hsr-common.h）：
 *   启动 → stdout 写 magic "HSR1"
 *   每 dt: stdout 写 HsrState(56B) → 阻塞读 stdin HsrAction(8B) → 应用
 *   Python 关闭 stdin（EOF）→ 提前退出；仿真到 dur → 写 done=1 退出
 *
 * 观测来源：
 *   - RSRP：eNB 侧 LteEnbRrc::RecvMeasurementReport trace（周期 120ms 上报，
 *     缓存最近一次；ns-3.43 的 MeasResults 用 measResultPCell/measResultListEutra）
 *   - 吞吐：FlowMonitor 每间隔 rxBytes 增量（切换中断自然体现为吞吐塌陷）
 *   - HO 事件/中断：eNB HandoverStart/HandoverEndOk trace 计时
 *
 * 构建（ns-3.43，云端）：
 *   cp hsr-common.h hsr-rl.cc /workspace/ns3/scratch/
 *   cd /workspace/ns3 && ./ns3 build hsr-rl
 *
 * 无 IPC 自检模式（验证场景与切换机制）：
 *   ./ns3 run "hsr-rl --test"        # 文本输出 + 内置 A3 策略
 */

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/lte-module.h"
#include "ns3/lte-ue-net-device.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/spectrum-propagation-loss-model.h"
#include "ns3/friis-spectrum-propagation-loss.h"
#include "ns3/eps-bearer.h"
#include "ns3/epc-tft.h"

#include "hsr-common.h"

#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <limits>
#include <cstdint>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("HsrRl");

// ---------------- 隧道路损模型（v2：链路感知 + 洞内微站） ----------------

// 隧道参数（TunnelLossModel 在全局区之前定义，故于此声明并初始化；
// main 解析命令行后覆盖）
static double g_tunnelLossDb = 25.0;
static std::vector<std::pair<double, double>> g_tunnels; // 隧道纵向区间 (x1,x2) m
// v2：洞内微站（漏缆近似）——微站与洞内 UE 同段链路只加小耦合衰减，
// 宏站→洞内 UE 仍加 g_tunnelLossDb（穿洞）。
static double g_tunnelMicroLossDb = 3.0;   // 洞内微站链路额外衰减 dB
static double g_tunnelEnbSpacing = 250.0;  // 洞内微站间距 m；0 = 不部署（退回 v1 行为）
static double g_tunnelEnbTxDbm = 30.0;     // 洞内微站 TxPower dBm（宏站 43）

// ---- 高速快衰落（Jakes）----
// 设计文档（建模与训练完整方案 §信道）明确要求「每链路独立快衰落 + LOS 空间
// 相关 —— 全部保留，这是基线可信的物理前提」；Python 侧 channel_3gpp.py 有
// 完整 Jakes(N=34)。但 NS-3 侧只装了自定义 TunnelLossModel、LENA 默认不带快
// 衰落 ⇒ Stage2 的物理世界比 Stage1 干净得多：RSRP 不会因多普勒而抖动，
// 「最强邻区频繁翻转」这一现象根本不出现，RL 学不到真实的切换时机鲁棒性。
// 这里按文档/Python 同构补上：h(t)=(1/√N)Σ exp(j·2π·f_d·cos(2πn/N)·t + φ_n)，
// f_d = v·f_c/c，逐链路独立随机相位（不是共模分量）。
static bool g_fastFading = false;
static double g_fcHz = 2.12e9;             // 载波（earfcn 100 ≈ 2120 MHz）
static uint32_t g_jakesN = 34;
static uint64_t g_fadingSeed = 20240917ULL;

struct JakesState
{
    bool init = false;
    std::vector<double> phi;
};
static std::map<std::pair<const MobilityModel*, const MobilityModel*>, JakesState>&
JakesMap()
{
    static std::map<std::pair<const MobilityModel*, const MobilityModel*>, JakesState> m;
    return m;
}

// 返回瞬时**功率**增益（dB，均值 0）
static double
JakesGainDb(const MobilityModel* a, const MobilityModel* b, double t)
{
    JakesState& st = JakesMap()[std::make_pair(a, b)];
    if (!st.init)
    {
        uint64_t h = g_fadingSeed * 1000003ULL
                     + (uint64_t)(uintptr_t)a * 7919ULL
                     + (uint64_t)(uintptr_t)b * 104729ULL;
        std::mt19937_64 rng(h);
        std::uniform_real_distribution<double> u(0.0, 2.0 * M_PI);
        st.phi.resize(g_jakesN);
        for (uint32_t n = 0; n < g_jakesN; ++n)
        {
            st.phi[n] = u(rng);
        }
        st.init = true;
    }
    Vector va = a->GetVelocity();
    Vector vb = b->GetVelocity();
    double sa = std::sqrt(va.x * va.x + va.y * va.y + va.z * va.z);
    double sb = std::sqrt(vb.x * vb.x + vb.y * vb.y + vb.z * vb.z);
    double v = std::max(sa, sb);           // 基站静止，取运动端速度
    double fd = v * g_fcHz / 3.0e8;        // 多普勒：83 m/s @2.12GHz ≈ 590 Hz
    if (fd <= 0.0)
    {
        return 0.0;
    }
    double re = 0.0, im = 0.0;
    const double inv = 1.0 / std::sqrt((double)g_jakesN);
    for (uint32_t n = 0; n < g_jakesN; ++n)
    {
        double ang = 2.0 * M_PI * fd * std::cos(2.0 * M_PI * (double)n / (double)g_jakesN)
                     * t + st.phi[n];
        re += std::cos(ang);
        im += std::sin(ang);
    }
    re *= inv;
    im *= inv;
    double p = re * re + im * im;          // 瞬时功率增益（均值 1）
    p = std::max(p, 1e-4);
    return 10.0 * std::log10(p);
}

// v2 链路感知：洞内 UE 经洞内微站（漏缆近似）只加小耦合衰减；
// 宏站→洞内 UE 仍加 tunnelLossDb（穿洞）。v1 对所有链路统一 25dB、
// 洞内又无基站 → 进洞必 RLF（8/8 实测，场景建模缺陷）。
// 自包含：内嵌 Friis 基底模型，不依赖 LENA 默认传播模型的装配顺序。
class TunnelLossModel : public SpectrumPropagationLossModel
{
  public:
    static TypeId GetTypeId()
    {
        static TypeId tid =
            TypeId("ns3::TunnelLossModel")
                .SetParent<SpectrumPropagationLossModel>()
                .SetGroupName("Lte")
                .AddConstructor<TunnelLossModel>();
        return tid;
    }
    TunnelLossModel()
    {
        m_base = CreateObject<FriisSpectrumPropagationLossModel>();
    }

  protected:
    Ptr<SpectrumValue> DoCalcRxPowerSpectralDensity(Ptr<const SpectrumSignalParameters> params,
                                                    Ptr<const MobilityModel> a,
                                                    Ptr<const MobilityModel> b) const override
    {
        Ptr<SpectrumValue> rx = m_base->CalcRxPowerSpectralDensity(params, a, b);
        Vector pa = a->GetPosition();
        Vector pb = b->GetPosition();
        // 轨道 UE：y≈0、低高度；宏 eNB：y=offset(≥35)、高 35m；洞内微站：y=3、高 5m。
        bool aIsUe = pa.y < pb.y;
        double ueX = aIsUe ? pa.x : pb.x;
        double enbX = aIsUe ? pb.x : pa.x;
        for (const auto& t : g_tunnels)
        {
            bool ueIn = (ueX >= t.first && ueX <= t.second);
            bool enbIn = (enbX >= t.first && enbX <= t.second);
            if (ueIn && enbIn)
            {
                // 洞内微站 ↔ 洞内 UE（漏缆近似）：只加小耦合衰减，保持洞内可覆盖
                *rx *= std::pow(10.0, -g_tunnelMicroLossDb / 10.0);
                break;
            }
            if (ueIn)
            {
                // 宏站 ↔ 洞内 UE：穿洞衰减（v1 行为）
                *rx *= std::pow(10.0, -g_tunnelLossDb / 10.0);
                break;
            }
        }
        // 高速快衰落（Jakes）：对**所有**链路生效（宏站与洞内微站一视同仁），
        // 使 RSRP 随多普勒抖动 —— 「最强邻区频繁翻转」才会在观测里出现。
        if (g_fastFading)
        {
            double db = JakesGainDb(PeekPointer(a), PeekPointer(b),
                                    Simulator::Now().GetSeconds());
            *rx *= std::pow(10.0, db / 10.0);
        }
        return rx;
    }
    // SpectrumPropagationLossModel 纯虚：本模型无随机流（Friis 基底也无），直接返回 0
    int64_t DoAssignStreams(int64_t stream) override
    {
        (void)stream;
        return 0;
    }

  private:
    Ptr<SpectrumPropagationLossModel> m_base;
};

// ---------------- 全局仿真状态 ----------------

static uint16_t g_servingCell = 0;          // 由最近测量上报更新
static int g_servingRsrpRaw = -1;           // 原始值 (dBm = raw - 140)
static std::vector<std::pair<uint16_t, int>> g_nbr; // (cellId, rawRsrp) 按 RSRP 降序
static bool g_measValid = false;

static bool g_inHo = false;                 // 切换进行中
static double g_hoStartTime = 0.0;
static double g_lastHoTime = -100.0;        // 上次切换完成时刻（200ms 保护间隔）
static uint32_t g_hoCount = 0;

static uint32_t g_step = 0;
static bool g_hoEvent = false;              // 本间隔内完成切换
static uint16_t g_hoInterruptMs = 0;        // 本间隔内完成切换的中断时长
static bool g_rlf = false;
static uint8_t g_requestResult = 0;
static bool g_hoFailed = false;
static double g_lastSampleTime = 0.1; // 应用启动，首拍累计量覆盖 0.1s 起的流量
static double g_noRxRunMs = 0.0;
static double g_noRxIntervalMs = 0.0;

static bool g_ioFailed = false;
static uint64_t g_lastRxBytes = 0;

// ---- 容量/业务量测量（100~1000 用户）----
// 每小区附着 UE 数用**计数器**维护（附着/切换完成/释放时增减），不做逐 UE 扫描：
// 1000 用户 × 4 候选 × 每 100ms 的 O(n) 扫描会成为瓶颈。
static std::map<uint16_t, uint32_t> g_cellUeCount;
// UE 索引 → IP（0 = 控车，1..N = 背景 UE），用于把 FlowMonitor 的流按用户聚合
static std::vector<Ipv4Address> g_ueIps;
static std::map<uint32_t, uint32_t> g_ipToIdx;   // IP(主机序) → UE 索引
// 上一间隔的累计值（FlowMonitor 的统计是累计量，需要差分）
static std::vector<uint64_t> g_prevRxBytes;
static std::vector<uint64_t> g_prevRxPkts;
static std::vector<double> g_prevDelaySum;       // 秒
// 本间隔结果
static double g_tpAggMbps = 0.0;
static double g_ctrlDelayMs = 0.0;

// ---- HSR3：服务小区聚合 / per-UE 分位数 / 公平性（P0 口径修复）----
// g_ueCumBytes 是**累计**接收字节（不是逐间隔增量）：per-UE 平均吞吐必须用
// 累计量除以已运行时长，否则「某 100ms 没被调度到」会被当成 0 Mbps 用户，
// 1000 用户规模下最小/分位数会被这些瞬时 0 主导（HSR2 的 tp_min 就是这样
// 失去意义的）。
static std::vector<uint64_t> g_ueCumBytes;   // UE 索引 → 累计 rxBytes
static double g_tpAggCellMbps = 0.0;         // 服务小区聚合吞吐 Mbps
static double g_tpUeP05Mbps = 0.0;           // per-UE 平均吞吐 5 分位（边缘用户）
static double g_tpUeP50Mbps = 0.0;           // per-UE 平均吞吐中位数
static double g_jain = 0.0;                  // 公平性指数
// 100~1000 用户压测的两个开关（默认关闭 = 保持既有标定完全一致）：
static bool g_skipLostPkts = false;   // 跳过 FlowMonitor::CheckForLostPackets（省 CPU）
static double g_attachStaggerS = 0.0; // 背景 UE 接入错峰窗口 s（0 = 全部 t=0 同时接入）

static Ptr<LteHelper> g_lte;
static Ptr<NetDevice> g_ueDev;
static Ptr<MobilityModel> g_ueMob;
static std::map<uint16_t, Vector> g_enbPos;         // cellId -> eNB 位置
static std::map<uint16_t, Ptr<NetDevice>> g_enbByCell; // cellId -> eNB 设备
static uint32_t g_totalHo = 0;

// ---- 背景多用户 / 隧道（高铁真实场景）----
static uint16_t g_ctrlImsi = 1;                 // 控车 IMSI（第一个安装的 UE）
static Ipv4Address g_ctrlAddr;                  // 控车 IP（FlowMonitor 过滤用）
static uint32_t g_nBgUe = 0;
static double g_bgSpeedMin = 83.0, g_bgSpeedMax = 138.0;
static double g_bgSpawnS = 6000.0;              // 背景 UE 初始纵向散布范围 m
static double g_voipFrac = 0.2, g_videoFrac = 0.4;
static double g_bgClusterFrac = 0.0;  // 聚集慢行 UE 比例（扎堆偶数号小区，制造非均匀负载）
static double g_coldBoostDb = 0.0;    // 冷（奇号）宏小区 Tx 功率提升 dB（S2 冷热对拓扑）
// g_tunnelLossDb / g_tunnels 定义在文件顶部（TunnelLossModel 类依赖）

struct BgUe
{
    Ptr<NetDevice> dev;
    uint64_t imsi = 0;
    uint16_t servingCell; // 仅在 ConnectionEstablished / HandoverEndOk 更新
    double speed;         // m/s（带符号，负=对向）
    bool attached = false;
    bool inHo = false;
};
static std::vector<BgUe> g_bg;

static BgUe*
FindBgByImsi(uint64_t imsi)
{
    for (auto& b : g_bg)
    {
        if (b.imsi == imsi)
        {
            return &b;
        }
    }
    return nullptr;
}

// ---------------- 二进制 I/O ----------------

static bool
WriteAll(const void* buf, size_t n)
{
    const char* p = static_cast<const char*>(buf);
    while (n > 0)
    {
        ssize_t w = write(1, p, n);
        if (w <= 0)
        {
            return false; // EPIPE：Python 侧已关闭
        }
        p += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

static bool
ReadAll(void* buf, size_t n)
{
    char* p = static_cast<char*>(buf);
    while (n > 0)
    {
        ssize_t r = read(0, p, n);
        if (r <= 0)
        {
            return false; // EOF：Python 侧结束 episode
        }
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

// ---------------- trace 回调 ----------------

// eNB 侧：LteEnbRrc::RecvMeasurementReport (imsi, cellId, rnti, report)
static void
RecvMeasurementReport(uint64_t imsi, uint16_t cellId, uint16_t rnti,
                      LteRrcSap::MeasurementReport report)
{
    (void)cellId;
    (void)rnti;
    if (imsi != g_ctrlImsi)
    {
        return; // 背景 UE 的测量上报不覆盖控车状态
    }
    const LteRrcSap::MeasResults& r = report.measResults;
    // MeasResultPCell 在 ns-3.43 无 physCellId 成员；服务小区 ID 由回调参数 cellId 提供
    g_servingCell = cellId;
    g_servingRsrpRaw = r.measResultPCell.rsrpResult;
    g_nbr.clear();
    for (const auto& m : r.measResultListEutra)
    {
        if (m.haveRsrpResult)
        {
            g_nbr.emplace_back(m.physCellId, static_cast<int>(m.rsrpResult));
        }
    }
    std::sort(g_nbr.begin(), g_nbr.end(),
              [](const std::pair<uint16_t, int>& a, const std::pair<uint16_t, int>& b) {
                  return a.second > b.second;
              });
    g_measValid = true;
}

// eNB 侧：ConnectionEstablished (imsi, cellId, rnti) —— 初始接入完成
static void
ConnEstablished(uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    (void)rnti;
    if (BgUe* b = FindBgByImsi(imsi))
    {
        if (!(b->attached) || b->servingCell != cellId)
        {
            g_cellUeCount[cellId]++;
        }
        b->attached = true;
        b->inHo = false;
        b->servingCell = cellId;
    }
}

// eNB 侧：HandoverStart (imsi, cellId, rnti, targetCellId)
static void
HoStart(uint64_t imsi, uint16_t cellId, uint16_t rnti, uint16_t targetCellId)
{
    (void)cellId;
    (void)rnti;
    (void)targetCellId;
    if (BgUe* b = FindBgByImsi(imsi))
    {
        b->inHo = true;
        return; // 背景 UE 切换不计入控车
    }
    if (imsi != g_ctrlImsi)
    {
        return;
    }
    if (!g_inHo)
    {
        g_inHo = true;
        g_hoStartTime = Simulator::Now().GetSeconds();
    }
}

// eNB 侧：HandoverEndOk (imsi, cellId, rnti)
static void
HoEndOk(uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    (void)rnti;
    if (BgUe* b = FindBgByImsi(imsi))
    {
        if (b->attached && b->servingCell != cellId)
        {
            if (g_cellUeCount[b->servingCell] > 0)
            {
                g_cellUeCount[b->servingCell]--;
            }
            g_cellUeCount[cellId]++;
        }
        b->inHo = false;
        b->attached = true;
        b->servingCell = cellId;
        return; // 背景 UE 切换不计入控车
    }
    if (imsi != g_ctrlImsi)
    {
        return;
    }
    if (g_inHo)
    {
        g_hoInterruptMs = static_cast<uint16_t>(
            (Simulator::Now().GetSeconds() - g_hoStartTime) * 1000.0);
        g_hoEvent = true;
        g_inHo = false;
        g_lastHoTime = Simulator::Now().GetSeconds();
        g_hoCount++;
        g_totalHo++;
    }
    else
    {
        // 兜底：没有配对的 Start（如安全网触发的切换）
        g_hoEvent = true;
        g_lastHoTime = Simulator::Now().GetSeconds();
        g_totalHo++;
    }
    (void)cellId;
}

// UE 侧：HandoverEndError (imsi, cellId, rnti) —— 切换失败，解除占用
static void
HoEndError(uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    (void)cellId;
    (void)rnti;
    if (BgUe* b = FindBgByImsi(imsi))
    {
        b->inHo = false;
        return;
    }
    if (imsi != g_ctrlImsi)
    {
        return;
    }
    g_hoFailed = true;
    g_inHo = false;
}

// UE 侧：RadioLinkFailure (imsi, cellId, rnti)
static void
RadioLinkFailure(uint64_t imsi, uint16_t cellId, uint16_t rnti)
{
    (void)cellId;
    (void)rnti;
    if (BgUe* b = FindBgByImsi(imsi))
    {
        if (b->attached && g_cellUeCount[b->servingCell] > 0)
        {
            g_cellUeCount[b->servingCell]--;
        }
        b->attached = false;
        b->inHo = false;
        return;
    }
    if (imsi != g_ctrlImsi)
    {
        return;
    }
    g_rlf = true;
    if (g_inHo) { g_hoFailed = true; }
}

// ---------------- 动作应用 ----------------

static void
ApplyAction(uint8_t action, uint16_t servingCell, const uint16_t* nbrCell, uint16_t nNbr)
{
    g_requestResult = 0;
    // 仅 1/2/5 是真实切换；0/3/4/6 在 NS-3 侧无操作（dual 由 Python 近似）
    if (action != 1 && action != 2 && action != 5)
    {
        if (action != 0) { g_requestResult = 7; }
        return;
    }
    int idx = (action == 1) ? 0 : (action == 2) ? 1 : 2;
    if (idx >= nNbr)
    {
        g_requestResult = 2;
        return; // 目标邻区不存在
    }
    uint16_t target = nbrCell[idx];
    if (target == servingCell)
    {
        g_requestResult = 3;
        return;
    }
    if (g_inHo)
    {
        g_requestResult = 4;
        return; // 切换进行中，拒绝并发请求
    }
    if (Simulator::Now().GetSeconds() - g_lastHoTime < 0.2)
    {
        g_requestResult = 5;
        return; // 200ms 切换保护间隔（防乒乓轰炸）
    }
    auto srcIt = g_enbByCell.find(servingCell);
    if (srcIt == g_enbByCell.end())
    {
        g_requestResult = 6;
        return; // 未知服务小区（测量尚未就绪）
    }
    if (g_enbByCell.find(target) == g_enbByCell.end())
    {
        g_requestResult = 6;
        return;
    }
    g_requestResult = 1; // accepted for dispatch, RRC completion reported separately
    // ns-3.43 签名 HandoverRequest(Time, ueDev, sourceEnbDev, targetCellId)
    g_lte->HandoverRequest(Seconds(0), g_ueDev, srcIt->second, target);
}

// ---------------- 每 dt 的锁步回调 ----------------

static void
StepCallback(Ptr<FlowMonitor> monitor, Ptr<Ipv4FlowClassifier> classifier,
             double dt, bool testMode, bool finalState)
{
    dt = std::max(1e-9, Simulator::Now().GetSeconds() - g_lastSampleTime);
    g_lastSampleTime = Simulator::Now().GetSeconds();
    // 1) 实测吞吐：**逐 UE** 聚合（控车 + 背景）。FlowStats 是累计量，按 UE
    // 索引差分。100~1000 用户时这是唯一的容量真相来源；只统计控车会完全看不
    // 到「小区被多少用户瓜分」。
    // 大规模压测时可跳过丢包统计：吞吐/时延只用 rxBytes 与 delaySum，
    // CheckForLostPackets 在 1000+ 流时是可观的纯 CPU 开销。
    if (!g_skipLostPkts)
    {
        monitor->CheckForLostPackets();
    }
    const size_t nUe = g_ueIps.size();
    std::vector<uint64_t> rxNow(nUe, 0);
    std::vector<uint64_t> pktNow(nUe, 0);
    std::vector<double> delayNow(nUe, 0.0);
    if (g_prevRxBytes.size() != nUe)
    {
        g_prevRxBytes.assign(nUe, 0);
        g_prevRxPkts.assign(nUe, 0);
        g_prevDelaySum.assign(nUe, 0.0);
        g_ueCumBytes.assign(nUe, 0);
    }
    // UE → 服务小区（控车 = g_servingCell；背景 UE 用各自已确认的 servingCell）
    std::vector<uint16_t> ueCell(nUe, 0);
    if (nUe > 0)
    {
        ueCell[0] = g_servingCell;
    }
    for (size_t i = 1; i < nUe; ++i)
    {
        ueCell[i] = (i - 1 < g_bg.size()) ? g_bg[i - 1].servingCell : 0;
    }
    for (const auto& it : monitor->GetFlowStats())
    {
        Ipv4FlowClassifier::FiveTuple ft = classifier->FindFlow(it.first);
        auto mit = g_ipToIdx.find(ft.destinationAddress.Get());
        if (mit == g_ipToIdx.end())
        {
            continue; // 非 UE 流（如回程）
        }
        uint32_t idx = mit->second;
        if (idx >= nUe)
        {
            continue;
        }
        rxNow[idx] += it.second.rxBytes;
        pktNow[idx] += it.second.rxPackets;
        delayNow[idx] += it.second.delaySum.GetSeconds();
    }
    double aggBytes = 0.0;
    double aggCellBytes = 0.0;
    for (size_t i = 0; i < nUe; ++i)
    {
        uint64_t dBytes = (rxNow[i] > g_prevRxBytes[i]) ? (rxNow[i] - g_prevRxBytes[i]) : 0;
        uint64_t dPkts = (pktNow[i] > g_prevRxPkts[i]) ? (pktNow[i] - g_prevRxPkts[i]) : 0;
        double dDelay = (delayNow[i] > g_prevDelaySum[i]) ? (delayNow[i] - g_prevDelaySum[i]) : 0.0;
        aggBytes += static_cast<double>(dBytes);
        if (ueCell[i] == g_servingCell)
        {
            aggCellBytes += static_cast<double>(dBytes);
        }
        g_ueCumBytes[i] += dBytes;
        if (i == 0)
        {
            g_noRxIntervalMs = dPkts == 0 ? dt * 1000.0 : 0.0;
            g_noRxRunMs = dPkts == 0 ? g_noRxRunMs + dt * 1000.0 : 0.0;
            // 控车单向时延：ΔdelaySum / Δpackets
            g_ctrlDelayMs = (dPkts > 0) ? (dDelay / static_cast<double>(dPkts) * 1000.0) : 0.0;
        }
        g_prevRxBytes[i] = rxNow[i];
        g_prevRxPkts[i] = pktNow[i];
        g_prevDelaySum[i] = delayNow[i];
    }
    g_tpAggMbps = aggBytes * 8.0 / dt / 1e6;
    g_tpAggCellMbps = aggCellBytes * 8.0 / dt / 1e6;

    // ---- per-UE 平均吞吐的分位数与公平性（HSR3）----
    // 用**累计字节 / 已运行时长**而非本间隔增量：100~1000 用户时大量 UE 在某
    // 个 100ms 内没被调度到（或突发业务处于 Off），逐间隔最小值会被这些瞬时
    // 0 主导（HSR2 的 tp_min_ue_mbps 即因此无意义）。
    const double span = std::max(Simulator::Now().GetSeconds() - 0.2, dt);
    std::vector<double> ueTp;
    ueTp.reserve(nUe);
    double sum = 0.0, sumSq = 0.0;
    for (size_t i = 0; i < nUe; ++i)
    {
        if (g_ueCumBytes[i] <= 0)
        {
            continue; // 尚未收到任何字节（未接入/无业务）
        }
        double v = static_cast<double>(g_ueCumBytes[i]) * 8.0 / span / 1e6;
        ueTp.push_back(v);
        sum += v;
        sumSq += v * v;
    }
    if (!ueTp.empty())
    {
        std::sort(ueTp.begin(), ueTp.end());
        auto quantile = [&ueTp](double p) {
            size_t k = (size_t)(p * (double)(ueTp.size() - 1) + 0.5);
            return ueTp[std::min(k, ueTp.size() - 1)];
        };
        g_tpUeP05Mbps = quantile(0.05);
        g_tpUeP50Mbps = quantile(0.50);
        double n = static_cast<double>(ueTp.size());
        g_jain = (sumSq > 0.0) ? (sum * sum) / (n * sumSq) : 0.0;
    }
    // 控车单用户吞吐沿用旧字段（奖励/观测口径不变）
    uint64_t ctrlRx = rxNow.empty() ? 0 : rxNow[0];
    double tpMbps = static_cast<double>(ctrlRx - g_lastRxBytes) * 8.0 / dt / 1e6;
    g_lastRxBytes = ctrlRx;

    // 2) 位置/几何
    Vector ue = g_ueMob->GetPosition();
    double dx = 0.0, dy = 0.0, dist = 0.0, cosDir = 1.0;
    auto sit = g_enbPos.find(g_servingCell);
    if (sit != g_enbPos.end())
    {
        dx = sit->second.x - ue.x;
        dy = sit->second.y - ue.y;
        dist = std::sqrt(dx * dx + dy * dy);
        cosDir = (dist > 1e-6) ? dx / dist : 1.0;
    }

    // 3) 打包状态
    HsrState st{};
    st.step = g_step;
    st.time_ms = static_cast<uint32_t>(Simulator::Now().GetMilliSeconds());
    st.serving_cell = g_measValid ? g_servingCell : 0;
    st.serving_rsrp_dbm = g_measValid ? static_cast<int16_t>(g_servingRsrpRaw - 140)
                                      : static_cast<int16_t>(-200);
    uint16_t nbrCells[3] = {0, 0, 0};
    int k = 0;
    for (const auto& n : g_nbr)
    {
        if (k >= 3)
        {
            break;
        }
        if (n.first == g_servingCell)
        {
            continue; // 排除服务小区自身
        }
        nbrCells[k] = n.first;
        st.nbr_cell[k] = static_cast<int16_t>(n.first);
        st.nbr_rsrp_dbm[k] = static_cast<int16_t>(n.second - 140);
        k++;
    }
    st.n_nbr = static_cast<uint16_t>(k);

    // 隧道判定提到这里（测试模式打印与负载观测共用；原先只在 testMode 分支内算）
    bool inTunnel = false;
    for (const auto& t : g_tunnels)
    {
        if (ue.x >= t.first && ue.x <= t.second)
        {
            inTunnel = true;
            break;
        }
    }

    // ---- 负载代理：各候选小区附着的背景 UE 数 ----
    // 设计文档 P1-6 指出旧观测无任何负载维度，「切向空闲小区做分流」不可实现。
    // NS-3 侧已维护每个背景 UE 的 servingCell（ConnectionEstablished /
    // HandoverEndOk 更新），统计成本 O(n_bg)（n_bg ≤ 32，每 100 ms 一次）。
    // 判据用 servingCell 而非 attached：servingCell 在创建时即初始化为几何
    // 最近小区，并在 ConnectionEstablished / HandoverEndOk 更新；而 attached
    // 依赖 trace 回调置位，实测大量背景下 UE 未置位（首拍 12s 内 16 个背景 UE
    // 仅统计到 0~1 个）——只按 attached 计数会让负载观测失效。
    auto BgLoadOf = [](uint16_t cell) -> uint8_t
    {
        if (cell == 0)
        {
            return 0;
        }
        uint32_t n = 0;
        for (const auto& b : g_bg)
        {
            if (b.servingCell == cell)
            {
                ++n;
            }
        }
        return static_cast<uint8_t>(n > 255 ? 255 : n);
    };
    st.load_serving = BgLoadOf(g_servingCell);
    for (int j = 0; j < 3; ++j)
    {
        st.load_nbr[j] = (j < k) ? BgLoadOf(nbrCells[j]) : 0;
    }
    st.in_tunnel = inTunnel ? 1 : 0;

    st.speed_mps = 83.3;
    st.heading_cos = static_cast<float>(cosDir);
    st.dist_to_serving_m = static_cast<float>(dist);
    st.ue_x = ue.x;
    st.throughput_mbps = static_cast<float>(tpMbps);
    // 容量/业务量指标（HSR3）
    // n_ue_cell 的计数器只跟踪背景 UE（ConnEstablished/HoEndOk/RLF 都是背景
    // 分支），控车必然附着于 g_servingCell 故显式 +1 ——「人均速率」的分母
    // 必须是「本小区所有在网用户」，漏掉控车会让分母系统性偏小。
    st.n_ue_cell = static_cast<uint16_t>(g_cellUeCount[g_servingCell] + 1);
    st.n_bg_total = static_cast<uint16_t>(g_bg.size());
    st.tp_agg_mbps = static_cast<float>(g_tpAggMbps);
    st.ctrl_delay_ms = static_cast<float>(g_ctrlDelayMs);
    st.tp_agg_cell_mbps = static_cast<float>(g_tpAggCellMbps);
    st.tp_ue_p05_mbps = static_cast<float>(g_tpUeP05Mbps);
    st.tp_ue_p50_mbps = static_cast<float>(g_tpUeP50Mbps);
    st.jain = static_cast<float>(g_jain);
    st.ho_interrupt_ms = g_hoInterruptMs;
    st.ho_event = g_hoEvent ? 1 : 0;
    st.rlf = g_rlf ? 1 : 0;
    st.done = finalState ? 1 : 0;
    st.ho_request_result = g_requestResult;
    st.ho_in_progress = g_inHo ? 1 : 0;
    st.ho_failed = g_hoFailed ? 1 : 0;
    st.interval_ms = static_cast<float>(dt * 1000.0);
    // g_prevRxPkts was updated above; use interval packet count captured there.
    st.no_rx_interval_ms = static_cast<float>(g_noRxIntervalMs);
    st.no_rx_run_ms = static_cast<float>(g_noRxRunMs);
    if (finalState)
    {
        if (!testMode) { WriteAll(&st, sizeof(st)); }
        return; // no action is owed for final state
    }
    // Clear interval events BEFORE applying the next action.
    g_requestResult = 0;
    g_hoFailed = false;
    g_hoEvent = false;
    g_hoInterruptMs = 0;
    g_rlf = false;

    if (testMode)
    {
        // 自检模式：文本输出 + 内置 A3（偏移 3dB）
        std::printf("%.1fs serv=%u rsrp=%ddBm nbr[%u:%ddBm %u:%ddBm %u:%ddBm] "
                    "dl=%.2fMbps ho=%u rlf=%u tunnel=%d nbg=%u "
                    "load[srv=%u n1=%u n2=%u n3=%u]\n",
                    Simulator::Now().GetSeconds(), st.serving_cell,
                    st.serving_rsrp_dbm,
                    (unsigned)st.nbr_cell[0], (int)st.nbr_rsrp_dbm[0],
                    (unsigned)st.nbr_cell[1], (int)st.nbr_rsrp_dbm[1],
                    (unsigned)st.nbr_cell[2], (int)st.nbr_rsrp_dbm[2],
                    st.throughput_mbps, g_totalHo, (unsigned)st.rlf,
                    inTunnel ? 1 : 0, (unsigned)g_bg.size(),
                    (unsigned)st.load_serving, (unsigned)st.load_nbr[0],
                    (unsigned)st.load_nbr[1], (unsigned)st.load_nbr[2]);
        std::fflush(stdout);
        // 每 5s 打印一次**每小区**附着分布：用于核查 n_ue_cell 这个「人均速率
        // 分母」是否可信。疑点来源：N=32 时 n_ue_cell 均值只有 1.91（含控车），
        // 即服务小区平均不到 1 个背景 UE —— 可能是控车在隧道段附着到洞内微站
        // （背景 UE 极少），也可能是计数器漏计（RLF 递减后未回加）。
        // counted 与 nbg 的差值直接暴露漏计。
        if (g_step % 50 == 0)
        {
            uint32_t tot = 0, macroCnt = 0, microCnt = 0;
            std::string dist;
            for (const auto& kv : g_cellUeCount)
            {
                uint32_t n = kv.second;
                if (n == 0)
                {
                    continue;
                }
                tot += n;
                auto pit = g_enbPos.find(kv.first);
                bool isMicro = (pit != g_enbPos.end() && pit->second.y < 10.0);
                if (isMicro)
                {
                    microCnt += n;
                }
                else
                {
                    macroCnt += n;
                }
                char buf[40];
                std::snprintf(buf, sizeof(buf), " %u:%u%s",
                              (unsigned)kv.first, (unsigned)n, isMicro ? "m" : "");
                dist += buf;
            }
            auto sp = g_enbPos.find(g_servingCell);
            bool srvMicro = (sp != g_enbPos.end() && sp->second.y < 10.0);
            std::printf("[cells] step=%u nbg=%u counted=%u (macro=%u micro=%u) "
                        "srv=%u%s n_ue_cell=%u ue[%s ]\n",
                        g_step, (unsigned)g_bg.size(), tot, macroCnt, microCnt,
                        (unsigned)g_servingCell, srvMicro ? "(micro)" : "(macro)",
                        (unsigned)st.n_ue_cell, dist.c_str());
            std::fflush(stdout);
        }
        if (st.n_nbr > 0 && st.serving_rsrp_dbm > -200
            && st.nbr_rsrp_dbm[0] > st.serving_rsrp_dbm + 3)
        {
            ApplyAction(1, st.serving_cell, nbrCells, st.n_nbr);
        }
    }
    else
    {
        // RL 模式：锁步写状态 → 读动作
        if (!WriteAll(&st, sizeof(st)))
        {
            g_ioFailed = true;
            Simulator::Stop();
            return;
        }
        HsrAction act{};
        if (!ReadAll(&act, sizeof(act)))
        {
            g_ioFailed = true; // Python 结束 episode（EOF）
            Simulator::Stop();
            return;
        }
        ApplyAction(act.action, st.serving_cell, nbrCells, st.n_nbr);
    }

    // 4) advance decision index
    g_step++;
}

// ---------------- main ----------------

int
main(int argc, char* argv[])
{
    uint16_t nEnb = 8;
    double spacing = 1000.0; // eNB 间距 m
    double offset = 50.0;    // eNB 垂直偏移 m
    double speed = 83.3;     // 300 km/h
    double dur = 52.0;       // 仿真时长 s
    double dt = 0.1;         // 决策步长 s
    double cbrRateMb = 100.0;
    uint32_t rngRun = 1;
    uint32_t earfcn = 100;   // 默认 band1 2.1GHz；3.5GHz 需频段表补丁（见 README）
    uint16_t dlRb = 100;     // 100 RB = 20 MHz（spec 要求）
    double txPowerDbm = 43.0;
    double ueNoiseFigure = 7.0;
    bool testMode = false;
    // 背景多用户 / 隧道（高铁真实场景）
    uint32_t nBgUe = 0;
    double bgSpeedMin = 83.0, bgSpeedMax = 138.0;
    double bgSpawnS = 6000.0;
    double voipFrac = 0.2, videoFrac = 0.4;
    double bgClusterFrac = 0.0;
    double coldBoostDb = 0.0;
    bool fastFading = false;
    double fcHz = 2.12e9;
    uint32_t jakesN = 34;
    double tunnelLossDb = 25.0;
    std::string tunnelsStr = ""; // 形如 "2000:2800,5000:5600"
    // v2 洞内微站（漏缆近似）：spacing=0 不部署，退回 v1 全衰减行为
    double tunnelEnbSpacing = 250.0;
    double tunnelEnbTxDbm = 30.0;
    double tunnelMicroLossDb = 3.0;

    CommandLine cmd;
    cmd.AddValue("nEnb", "number of eNBs", nEnb);
    cmd.AddValue("spacing", "eNB spacing (m)", spacing);
    cmd.AddValue("offset", "eNB perpendicular offset (m)", offset);
    cmd.AddValue("speed", "UE speed (m/s)", speed);
    cmd.AddValue("dur", "simulation duration (s)", dur);
    cmd.AddValue("dt", "RL decision step (s)", dt);
    cmd.AddValue("cbrRate", "DL CBR rate (Mb/s)", cbrRateMb);
    cmd.AddValue("rngRun", "RNG run number", rngRun);
    cmd.AddValue("earfcn", "DL/UL EARFCN (default 100=2.1GHz band1; 3.5GHz=43589 needs band42 patch)", earfcn);
    cmd.AddValue("dlRb", "DL bandwidth in RB (100=20MHz)", dlRb);
    cmd.AddValue("txPowerDbm", "eNB TX power dBm", txPowerDbm);
    cmd.AddValue("ueNoiseFigure", "UE noise figure dB", ueNoiseFigure);
    cmd.AddValue("test", "standalone test mode (no IPC, builtin A3)", testMode);
    cmd.AddValue("nBgUe", "number of background high-speed UEs", nBgUe);
    cmd.AddValue("bgSpeedMin", "background UE min speed (m/s)", bgSpeedMin);
    cmd.AddValue("bgSpeedMax", "background UE max speed (m/s)", bgSpeedMax);
    cmd.AddValue("attachStaggerS",
                 "background UE attach stagger window (s); 0 = all attach at t=0",
                 g_attachStaggerS);
    cmd.AddValue("skipLostPkts",
                 "skip FlowMonitor::CheckForLostPackets (perf knob for >=100 UEs)",
                 g_skipLostPkts);
    cmd.AddValue("bgSpawnS", "background UE initial x spread (m)", bgSpawnS);
    cmd.AddValue("voipFrac", "fraction of background UEs with VoIP (QCI1)", voipFrac);
    cmd.AddValue("videoFrac", "fraction with video/eMBB bearer", videoFrac);
    cmd.AddValue("bgClusterFrac",
                 "fraction of bg UEs clustered near even cellId eNBs at low speed "
                 "(load imbalance: even cells congested, odd cells idle)", bgClusterFrac);
    cmd.AddValue("fastFading", "enable Jakes fast fading (Doppler, per-link phase)", fastFading);
    cmd.AddValue("fcHz", "carrier frequency for Doppler computation", fcHz);
    cmd.AddValue("jakesN", "number of Jakes rays", jakesN);
    cmd.AddValue("coldNeighborBoostDb",
                 "Tx power boost (dB) for cold (odd cellId) macro eNBs - makes "
                 "'hand over to the idle neighbour' a coverage-safe choice (S2)",
                 coldBoostDb);
    cmd.AddValue("tunnelLossDb", "extra path loss inside tunnels (dB)", tunnelLossDb);
    cmd.AddValue("tunnels", "tunnel intervals 'x1:x2,x3:x4' (m)", tunnelsStr);
    cmd.AddValue("tunnelEnbSpacing",
                 "in-tunnel micro eNB spacing (m); 0 disables (v1 behavior)",
                 tunnelEnbSpacing);
    cmd.AddValue("tunnelEnbTxDbm", "in-tunnel micro eNB TxPower (dBm)", tunnelEnbTxDbm);
    cmd.AddValue("tunnelMicroLossDb",
                 "extra loss UE<->in-tunnel micro eNB (dB)", tunnelMicroLossDb);
    cmd.Parse(argc, argv);

    g_nBgUe = nBgUe;
    g_bgSpeedMin = bgSpeedMin;
    g_bgSpeedMax = bgSpeedMax;
    g_bgSpawnS = bgSpawnS;
    g_voipFrac = voipFrac;
    g_videoFrac = videoFrac;
    g_bgClusterFrac = bgClusterFrac;
    g_coldBoostDb = coldBoostDb;
    g_fastFading = fastFading;
    g_fcHz = fcHz;
    if (jakesN >= 8)
    {
        g_jakesN = jakesN;
    }
    g_tunnelLossDb = tunnelLossDb;
    g_tunnelEnbSpacing = tunnelEnbSpacing;
    g_tunnelEnbTxDbm = tunnelEnbTxDbm;
    g_tunnelMicroLossDb = tunnelMicroLossDb;
    // 解析隧道区间串 "x1:x2,x3:x4" → g_tunnels
    if (!tunnelsStr.empty())
    {
        std::stringstream ss(tunnelsStr);
        std::string seg;
        while (std::getline(ss, seg, ','))
        {
            std::stringstream ss2(seg);
            double x1 = 0.0, x2 = 0.0;
            char colon = ':';
            if (ss2 >> x1 >> colon >> x2 && x2 > x1)
            {
                g_tunnels.emplace_back(x1, x2);
            }
        }
    }

    Config::SetGlobal("RngRun", UintegerValue(rngRun));
    signal(SIGPIPE, SIG_IGN); // 写已关闭的管道返回错误而非杀进程

    // ---- 射频配置对齐 spec（LENA 默认 25RB/30dBm/9dB 与场景不符）----
    Config::SetDefault("ns3::LteEnbPhy::TxPower", DoubleValue(txPowerDbm));
    Config::SetDefault("ns3::LteUePhy::NoiseFigure", DoubleValue(ueNoiseFigure));
    // 100 RB = 20 MHz；EARFCN 由命令行给（FDD band1: dl=100/ul=18100；
    // 3.5GHz n78 需频段表补丁，dl=ul=43589）
    Config::SetDefault("ns3::LteEnbNetDevice::DlEarfcn", UintegerValue(earfcn));
    Config::SetDefault("ns3::LteEnbNetDevice::UlEarfcn",
                       UintegerValue(earfcn >= 36000 ? earfcn : 18000 + earfcn));
    Config::SetDefault("ns3::LteEnbNetDevice::DlBandwidth", UintegerValue(dlRb));
    Config::SetDefault("ns3::LteEnbNetDevice::UlBandwidth", UintegerValue(dlRb));
    Config::SetDefault("ns3::LteUeNetDevice::DlEarfcn", UintegerValue(earfcn));

    // ---- LTE + EPC ----
    Ptr<LteHelper> lteHelper = CreateObject<LteHelper>();
    Ptr<PointToPointEpcHelper> epcHelper = CreateObject<PointToPointEpcHelper>();
    lteHelper->SetEpcHelper(epcHelper);
    lteHelper->SetSchedulerType("ns3::PfFfMacScheduler");
    // RL 主导：关闭自动切换算法，全部由 HandoverRequest 驱动
    lteHelper->SetHandoverAlgorithmType("ns3::NoOpHandoverAlgorithm");
    // 隧道路损模型：洞外等价 Friis（LENA 默认），洞内额外衰减 tunnelLossDb
    lteHelper->SetPathlossModelType(TunnelLossModel::GetTypeId());
    g_lte = lteHelper;

    Ptr<Node> pgw = epcHelper->GetPgwNode();

    // ---- 节点 ----
    NodeContainer enbNodes, ueNodes;
    enbNodes.Create(nEnb);
    ueNodes.Create(1);

    MobilityHelper enbMob;
    enbMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    enbMob.Install(enbNodes);
    for (uint16_t i = 0; i < nEnb; ++i)
    {
        enbNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(i * spacing, offset, 35.0));
    }

    // ---- v2 洞内微站（漏缆/洞内微小区近似）：每段隧道按 spacing 均布 ----
    // 必须在 InstallEnbDevice / AddX2Interface 之前创建，使微站自动获得设备与 X2。
    uint32_t nEnbTotal = nEnb;
    if (g_tunnelEnbSpacing > 0.0 && !g_tunnels.empty())
    {
        uint32_t nTun = 0;
        for (const auto& t : g_tunnels)
        {
            nTun += std::max(1u,
                             (uint32_t)((t.second - t.first) / g_tunnelEnbSpacing));
        }
        enbNodes.Create(nTun);
        for (uint32_t k = nEnb; k < (uint32_t)nEnb + nTun; ++k)
        {
            enbMob.Install(enbNodes.Get(k)); // 新建节点须单独 Install
        }
        uint32_t k = nEnb;
        for (const auto& t : g_tunnels)
        {
            uint32_t m = std::max(
                1u, (uint32_t)((t.second - t.first) / g_tunnelEnbSpacing));
            for (uint32_t j = 0; j < m; ++j, ++k)
            {
                double x = t.first + (j + 0.5) * (t.second - t.first) / m;
                // y=3（贴近隧道、区别于宏站 offset≥35）、高 5m（洞内低天线）
                enbNodes.Get(k)->GetObject<MobilityModel>()->SetPosition(
                    Vector(x, 3.0, 5.0));
            }
        }
        nEnbTotal = (uint32_t)nEnb + nTun;
        NS_LOG_INFO("tunnel micro eNBs: " << nTun << ", total eNBs "
                                         << nEnbTotal);
    }

    MobilityHelper ueMob;
    ueMob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    ueMob.Install(ueNodes);
    ueNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, 1.5));
    ueNodes.Get(0)->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(
        Vector(speed, 0.0, 0.0));
    g_ueMob = ueNodes.Get(0)->GetObject<MobilityModel>();

    // ---- IP 协议栈（EPC 数据面）----
    InternetStackHelper internet;
    internet.Install(ueNodes);

    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    internet.Install(remoteHostContainer);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetChannelAttribute("Delay", TimeValue(MilliSeconds(10)));
    NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);
    Ipv4AddressHelper ipv4h;
    ipv4h.SetBase("1.0.0.0", "255.0.0.0");
    Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);
    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);

    Ipv4StaticRoutingHelper ipv4RoutingHelper;
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                               Ipv4Mask("255.0.0.0"), 1);

    // ---- LTE 设备 ----
    NetDeviceContainer enbLteDevs = lteHelper->InstallEnbDevice(enbNodes);
    NetDeviceContainer ueLteDevs = lteHelper->InstallUeDevice(ueNodes);
    g_ueDev = ueLteDevs.Get(0);

    // 洞内微站低功率：Config::SetDefault 已把全部 eNB 设为 txPowerDbm(43)，
    // 逐设备覆盖微站（index 从 nEnb 起）。
    for (uint32_t i = nEnb; i < enbLteDevs.GetN(); ++i)
    {
        enbLteDevs.Get(i)
            ->GetObject<LteEnbNetDevice>()
            ->GetPhy()
            ->SetTxPower(g_tunnelEnbTxDbm);
    }

    Ipv4InterfaceContainer ueIpIface =
        epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueLteDevs));
    g_ctrlAddr = ueIpIface.GetAddress(0); // FlowMonitor 只统计控车流
    // 容量测量：登记 UE 索引 0 = 控车
    g_ueIps.clear();
    g_ipToIdx.clear();
    g_ueIps.push_back(g_ctrlAddr);
    g_ipToIdx[g_ctrlAddr.Get()] = 0;
    Ptr<Ipv4StaticRouting> ueStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(ueNodes.Get(0)->GetObject<Ipv4>());
    ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

    lteHelper->AddX2Interface(enbNodes);
    lteHelper->Attach(ueLteDevs.Get(0), enbLteDevs.Get(0));

    // ---- 测量配置（NoOp 算法不自带 report config，需手工添加）----
    // 关键：ns-3.43 LENA 仅支持 EVENT 型上报（PERIODICAL 路径有
    // NS_ASSERT("only triggerType == event is supported")，release 构建下
    // 断言被编译掉，导致 cellsTriggeredList 永远为空 → 邻区 RSRP 不上报）。
    // 用 A3 事件 + a3Offset=-30dB + hysteresis=0 + TTT=0：几乎所有可听邻区
    // 都满足进入条件并常驻触发列表（离开条件需比服务差 30dB），从而每
    // MS120 持续上报最强邻区 RSRP。NoOp 算法不消费这些报告，切换决策仍
    // 100% 由 RL / testMode 内置 A3 显式 HandoverRequest 驱动。
    LteRrcSap::ReportConfigEutra reportConfig;
    reportConfig.triggerType = LteRrcSap::ReportConfigEutra::EVENT;
    reportConfig.eventId = LteRrcSap::ReportConfigEutra::EVENT_A3;
    reportConfig.a3Offset = -30;
    reportConfig.hysteresis = 0;
    reportConfig.timeToTrigger = 0;
    reportConfig.triggerQuantity = LteRrcSap::ReportConfigEutra::RSRP;
    reportConfig.reportInterval = LteRrcSap::ReportConfigEutra::MS120;
    reportConfig.reportAmount = 255;
    // 边界含洞内微站：微站也要 AddUeMeasReportConfig（否则邻区列表无微站
    // cellId，A3/RL 永远切不进去）并进入 g_enbPos/g_enbByCell（背景 UE
    // 最近站附着、HandoverRequest 目标查找都依赖这两张表）。
    for (uint32_t i = 0; i < enbLteDevs.GetN(); ++i)
    {
        Ptr<LteEnbRrc> rrc = enbLteDevs.Get(i)->GetObject<LteEnbNetDevice>()->GetRrc();
        rrc->AddUeMeasReportConfig(reportConfig);
        uint16_t cellId = enbLteDevs.Get(i)->GetObject<LteEnbNetDevice>()->GetCellId();
        g_enbPos[cellId] = enbNodes.Get(i)->GetObject<MobilityModel>()->GetPosition();
        g_enbByCell[cellId] = enbLteDevs.Get(i);
    }

    // ---- 冷热对拓扑（S2）：让「切向空闲小区」成为安全可用的选择 ----
    // 背景 UE 聚集在**偶数号**小区（见下方 clustered 分支）⇒ 偶号热、奇号冷。
    // 但 1000m 站距下「冷邻区」同时就是「弱邻区」，实测负载项只会让 RL 为躲
    // 负载而少切换并掉话（congested 存活 1.00 → 0.33，切换 7.33 → 2.67/集）。
    // 给冷（奇号）宏小区加 Tx 功率，使其既空闲又满足 RSRP 门限 —— 这样负载
    // 分流才是一条**不牺牲覆盖**的逃逸路径。只作用于宏站（cellId ≤ nEnb），
    // 避免改变洞内微站的既有标定。
    if (g_coldBoostDb > 0.0)
    {
        for (const auto& kv : g_enbByCell)
        {
            if (kv.first % 2 == 1 && kv.first <= (uint16_t)enbLteDevs.GetN())
            {
                Ptr<LteEnbNetDevice> enbDev = kv.second->GetObject<LteEnbNetDevice>();
                Ptr<LteEnbPhy> phy = enbDev->GetPhy();
                phy->SetTxPower(phy->GetTxPower() + g_coldBoostDb);
                std::fprintf(stderr, "[hot-cold] cell %u (cold) TxPower -> %.1f dBm\n",
                            (unsigned)kv.first, phy->GetTxPower());
            }
        }
    }

    // ---- 背景多用户：同线高速 UE + 异构 QoS 业务（VoIP/视频/消息）----
    // 控车是第一个 InstallUeDevice（IMSI=1）；背景 UE 全部使用 NoOp 切换算法，
    // 由下方 1.0s 周期事件按"最近 eNB"自行 HandoverRequest，不走 IPC。
    if (g_nBgUe > 0)
    {
        Ptr<UniformRandomVariable> bgRng = CreateObject<UniformRandomVariable>();
        bgRng->SetStream(1000 + rngRun);

        NodeContainer bgNodes;
        bgNodes.Create(g_nBgUe);
        internet.Install(bgNodes); // EPC 数据面协议栈（须在 Assign IP 前）
        MobilityHelper bgMob;
        bgMob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
        bgMob.Install(bgNodes);
        NetDeviceContainer bgDevs = lteHelper->InstallUeDevice(bgNodes);
        Ipv4InterfaceContainer bgIp = epcHelper->AssignUeIpv4Address(NetDeviceContainer(bgDevs));
        // 容量测量：登记背景 UE 索引 1..N（用于逐用户吞吐/公平性/最差用户）
        for (uint32_t i = 0; i < g_nBgUe; ++i)
        {
            Ipv4Address a = bgIp.GetAddress(i);
            g_ueIps.push_back(a);
            g_ipToIdx[a.Get()] = 1 + i;
        }

        for (uint32_t i = 0; i < g_nBgUe; ++i)
        {
            Ptr<Node> node = bgNodes.Get(i);
            Ptr<MobilityModel> mob = node->GetObject<MobilityModel>();
            double x0, y0, v;
            bool clustered = bgRng->GetValue(0.0, 1.0) < g_bgClusterFrac
                             && !g_enbPos.empty();
            if (clustered)
            {
                // 非均匀负载：聚集 UE 初始散布在偶数号小区周围（±300m 纵向、
                // ±50m 横向），速度 0-10 m/s（车站/沿线慢行用户），使偶数站
                // 长时间拥塞、奇数站空闲。RL 因此可学到「牺牲少量 RSRP 切向
                // 空闲次强小区」——纯 RSRP 贪心的 A3 无此决策维度。
                std::vector<uint16_t> evenCells;
                for (const auto& ep : g_enbPos)
                {
                    if (ep.first % 2 == 0)
                    {
                        evenCells.push_back(ep.first);
                    }
                }
                uint16_t c =
                    evenCells[bgRng->GetInteger(0, (int64_t)evenCells.size() - 1)];
                const Vector& cp = g_enbPos[c];
                x0 = cp.x + bgRng->GetValue(-300.0, 300.0);
                y0 = cp.y + bgRng->GetValue(-50.0, 50.0);
                v = bgRng->GetValue(0.0, 10.0);
            }
            else
            {
                x0 = bgRng->GetValue(0.0, g_bgSpawnS);
                y0 = bgRng->GetValue(-3.0, 3.0);
                v = bgRng->GetValue(g_bgSpeedMin, g_bgSpeedMax);
                if (bgRng->GetValue(0.0, 1.0) < 0.2)
                {
                    v = -v; // 20% 对向列车
                }
            }
            mob->SetPosition(Vector(x0, y0, 1.5));
            node->GetObject<ConstantVelocityMobilityModel>()->SetVelocity(
                Vector(v, 0.0, 0.0));

            // 初始接入几何最近 eNB（g_enbByCell 已在上方循环中填好）
            uint16_t bestCell = 0;
            double bestD = std::numeric_limits<double>::max();
            for (const auto& ep : g_enbPos)
            {
                double ddx = ep.second.x - x0;
                double ddy = ep.second.y - y0;
                double d = std::sqrt(ddx * ddx + ddy * ddy);
                if (d < bestD)
                {
                    bestD = d;
                    bestCell = ep.first;
                }
            }
            // 接入错峰（100~1000 用户压测用）：默认 0 = 全部在 t=0 同时发起随机
            // 接入。N 很大时会形成 RRC 建立风暴 —— 既不符合真实列车（用户是陆
            // 续上车的），也可能让接入过程本身成为瓶颈而污染容量结论。
            if (g_attachStaggerS > 0.0 && g_nBgUe > 1)
            {
                double tAt = g_attachStaggerS * (double)i / (double)(g_nBgUe - 1);
                Simulator::Schedule(Seconds(tAt), [=]() {
                    lteHelper->Attach(bgDevs.Get(i), g_enbByCell[bestCell]);
                });
            }
            else
            {
                lteHelper->Attach(bgDevs.Get(i), g_enbByCell[bestCell]);
            }
            Ptr<Ipv4StaticRouting> sr =
                ipv4RoutingHelper.GetStaticRouting(node->GetObject<Ipv4>());
            sr->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);

            BgUe b;
            b.dev = bgDevs.Get(i);
            b.imsi = DynamicCast<LteUeNetDevice>(bgDevs.Get(i))->GetImsi();
            b.servingCell = bestCell; // 目标小区；attached 等 ConnectionEstablished
            b.speed = v;
            b.attached = false;
            b.inHo = false;
            g_bg.push_back(b);

            // ---- 每 UE 一条 QoS 业务流（专用承载 + 端口过滤 TFT）----
            uint16_t bgPort = 10001 + i;
            double r = bgRng->GetValue(0.0, 1.0);
            // ns-3.43 QCI 枚举：QCI9=NGBR_VIDEO_TCP_DEFAULT(best effort)，
            // QCI8=NGBR_VIDEO_TCP_PREMIUM，QCI1=GBR_CONV_VOICE
            EpsBearer bearer(EpsBearer::NGBR_VIDEO_TCP_DEFAULT);
            double rateMb = 0.128;
            uint32_t pktSize = 128;
            bool isVideo = false;
            if (r < g_voipFrac)
            {
                // VoIP：QCI1 GBR 会话语音，64kbps，近恒定码率
                bearer = EpsBearer(EpsBearer::GBR_CONV_VOICE);
                rateMb = 0.064;
                pktSize = 80;
            }
            else if (r < g_voipFrac + g_videoFrac)
            {
                // 视频/eMBB：QCI8 高级 TCP 视频承载，2-5Mbps 突发
                bearer = EpsBearer(EpsBearer::NGBR_VIDEO_TCP_PREMIUM);
                rateMb = bgRng->GetValue(2.0, 5.0);
                pktSize = 1024;
                isVideo = true;
            }
            // 其余：QCI9 best-effort 消息类 128kbps
            Ptr<EpcTft> tft = Create<EpcTft>();
            EpcTft::PacketFilter pf;
            pf.remotePortStart = bgPort;
            pf.remotePortEnd = bgPort;
            tft->Add(pf);
            lteHelper->ActivateDedicatedEpsBearer(bgDevs.Get(i), bearer, tft);

            OnOffHelper bgOn("ns3::UdpSocketFactory",
                             InetSocketAddress(bgIp.GetAddress(i), bgPort));
            bgOn.SetAttribute("DataRate",
                              DataRateValue(DataRate(std::to_string(rateMb) + "Mb/s")));
            bgOn.SetAttribute("PacketSize", UintegerValue(pktSize));
            if (isVideo)
            {
                // 视频流：指数 On/Off 突发
                bgOn.SetAttribute("OnTime",
                                  StringValue("ns3::ExponentialRandomVariable[Mean=2.0]"));
                bgOn.SetAttribute("OffTime",
                                  StringValue("ns3::ExponentialRandomVariable[Mean=1.0]"));
            }
            ApplicationContainer bgApps = bgOn.Install(remoteHost);
            bgApps.Start(Seconds(0.2));
            bgApps.Stop(Seconds(dur - 0.1));

            PacketSinkHelper bgSink("ns3::UdpSocketFactory",
                                    InetSocketAddress(Ipv4Address::GetAny(), bgPort));
            ApplicationContainer bgSinkApps = bgSink.Install(node);
            bgSinkApps.Start(Seconds(0.2));
            bgSinkApps.Stop(Seconds(dur - 0.1));
        }

        // 背景 UE 移动性：每 1.0s 若最近 eNB 变化则 HandoverRequest（不走 IPC；
        // servingCell 只在 ConnectionEstablished / HandoverEndOk 更新，禁止乐观
        // 改写——否则下一拍会以未完成 HO 的目标为源，对 INITIAL_RANDOM_ACCESS
        // 的 UeManager 发 HandoverRequest → NS_FATAL lte-enb-rrc.cc:783）。
        // 首拍 2.0s：给 Attach 留出 RRC 完成窗口。
        for (double t = 2.0; t < dur - 0.1; t += 1.0)
        {
            Simulator::Schedule(Seconds(t), []() {
                for (auto& b : g_bg)
                {
                    if (!b.attached || b.inHo)
                    {
                        continue;
                    }
                    Vector p = b.dev->GetNode()->GetObject<MobilityModel>()->GetPosition();
                    uint16_t bestCell = 0;
                    double bestD = std::numeric_limits<double>::max();
                    for (const auto& ep : g_enbPos)
                    {
                        double ddx = ep.second.x - p.x;
                        double ddy = ep.second.y - p.y;
                        double d = std::sqrt(ddx * ddx + ddy * ddy);
                        if (d < bestD)
                        {
                            bestD = d;
                            bestCell = ep.first;
                        }
                    }
                    if (bestCell != 0 && bestCell != b.servingCell)
                    {
                        auto sit = g_enbByCell.find(b.servingCell);
                        auto eit = g_enbByCell.find(bestCell);
                        if (sit != g_enbByCell.end() && eit != g_enbByCell.end())
                        {
                            // 不在此改 servingCell；等 HoEndOk / ConnEstablished
                            g_lte->HandoverRequest(Seconds(0), b.dev,
                                                   sit->second, bestCell);
                            b.inHo = true;
                        }
                    }
                }
            });
        }
    }

    // ---- trace 连接（ns-3.43 路径已逐条核对）----
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/RecvMeasurementReport",
                    MakeCallback(&RecvMeasurementReport));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/ConnectionEstablished",
                    MakeCallback(&ConnEstablished));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverStart",
                    MakeCallback(&HoStart));
    Config::ConnectWithoutContext("/NodeList/*/DeviceList/*/LteEnbRrc/HandoverEndOk",
                    MakeCallback(&HoEndOk));
    // 控车 + 背景 UE 的 HO 失败 / RLF：背景侧必须清 inHo/attached，否则
    // 失败后仍按乐观 servingCell 发下一拍 HandoverRequest。
    Config::ConnectWithoutContextFailSafe(
        "/NodeList/*/DeviceList/*/LteUeRrc/HandoverEndError",
        MakeCallback(&HoEndError));
    Config::ConnectWithoutContextFailSafe(
        "/NodeList/*/DeviceList/*/LteUeRrc/RadioLinkFailure",
        MakeCallback(&RadioLinkFailure));

    // ---- 下行饱和 UDP（吞吐测量）----
    uint16_t port = 10000;
    OnOffHelper dl("ns3::UdpSocketFactory",
                   InetSocketAddress(ueIpIface.GetAddress(0), port));
    dl.SetAttribute("DataRate", DataRateValue(DataRate(std::to_string(cbrRateMb) + "Mb/s")));
    dl.SetAttribute("PacketSize", UintegerValue(1024));
    ApplicationContainer dlApps = dl.Install(remoteHost);
    dlApps.Start(Seconds(0.1));
    dlApps.Stop(Seconds(dur - 0.1));

    PacketSinkHelper sink("ns3::UdpSocketFactory",
                          InetSocketAddress(Ipv4Address::GetAny(), port));
    ApplicationContainer sinkApps = sink.Install(ueNodes);
    sinkApps.Start(Seconds(0.1));
    sinkApps.Stop(Seconds(dur - 0.1));

    // ---- FlowMonitor ----
    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();
    Ptr<Ipv4FlowClassifier> flowClassifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());

    // ---- IPC 握手 ----
    if (!testMode)
    {
        // HSR3：协议由 72B 扩到 84B（新增 tp_agg_cell_mbps / tp_ue_p05 /
        // tp_ue_p50 / jain，并修正 n_ue_cell 口径）。magic 同步升级，避免旧
        // Python 按 72B 解析新二进制而静默错位。
        if (!WriteAll("HSR4", 4))
        {
            return 1;
        }
    }

    // ---- 锁步调度（首个决策点 0.3s：等接入 + 首次测量上报）----
    for (double t = 0.3; t < dur - dt; t += dt)
    {
        Simulator::Schedule(Seconds(t), &StepCallback, monitor, flowClassifier,
                            dt, testMode, false);
    }

    // Sample full final geometry/metrics before stopping; includes last partial interval.
    Simulator::Schedule(Seconds(dur - 1e-6), &StepCallback, monitor, flowClassifier,
                        dt, testMode, true);
    Simulator::Stop(Seconds(dur));
    Simulator::Run();

    Simulator::Destroy();

    if (testMode)
    {
        std::printf("=== DONE: steps=%u totalHO=%u ===\n", g_step, g_totalHo);
    }
    return 0;
}
