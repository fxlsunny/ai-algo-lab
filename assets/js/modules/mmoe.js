/* 模块：PLE-MMoE */
MCH.register("mmoe", {
  render() {
    const code = `# PLE-style MMoE：共享专家 + 每任务私有专家，缓解多任务梯度冲突
# 来源：src/models/mmoe.py

class Expert(nn.Module):
    """单专家：2 层 MLP 输出 expert_dim"""
    def __init__(self, in_dim, out_dim, hidden=512, dropout=0.1):
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim), nn.GELU(),
        )


class PLEMMoE(nn.Module):
    """Progressive Layered Extraction (Tang et al., RecSys 2020) —— MMoE 的改进版

    普通 MMoE (Ma et al., KDD 2018):  所有任务共享 N 个专家，各任务独立 gate 选择。
    缺陷：强任务（如 category 2000 类）的梯度会"霸占"专家，削弱弱任务（如 trust）。

    PLE: 每个任务额外配 n_task_experts 个私有专家，共享专家 + 私有专家一起参与 gate mixing。
    优势：私有专家确保任务"有自留地"，梯度不互相干扰。
    """
    def __init__(self, in_dim=512, num_shared_experts=4, num_task_experts=1,
                 expert_dim=256, task_names=("risk", "trust", "category")):
        self.shared_experts = nn.ModuleList([
            Expert(in_dim, expert_dim) for _ in range(num_shared_experts)
        ])
        self.task_experts = nn.ModuleDict({
            t: nn.ModuleList([Expert(in_dim, expert_dim) for _ in range(num_task_experts)])
            for t in task_names
        })
        self.gates = nn.ModuleDict({
            t: nn.Sequential(nn.Linear(in_dim, num_shared_experts + num_task_experts),
                             nn.Softmax(dim=-1))
            for t in task_names
        })

    def add_task(self, name):
        """热插拔新任务：共享专家不变，新任务只加私有专家 + gate（零改底座！）"""
        self.task_experts[name] = nn.ModuleList([Expert(...) for _ in range(self.num_task)])
        self.gates[name] = nn.Sequential(...)
        self.task_names.append(name)

    def forward(self, x):
        # 共享专家对所有任务算一次
        shared_out = torch.stack([e(x) for e in self.shared_experts], dim=1)  # (B, S, D')

        outputs = {}
        for t in self.task_names:
            task_out = torch.stack([e(x) for e in self.task_experts[t]], dim=1)  # (B, P, D')
            cat = torch.cat([shared_out, task_out], dim=1)                       # (B, S+P, D')
            gate = self.gates[t](x).unsqueeze(-1)                                # (B, S+P, 1)
            outputs[t] = (cat * gate).sum(dim=1)                                 # (B, D')
        return outputs`;

    return `
      ${MCH.hero({
        icon: "M",
        name: "PLE-MMoE — 共享 + 私有专家的多任务学习",
        en: "Progressive Layered Extraction · Multi-gate MoE",
        tags: ["4 Shared Experts", "1 Private / 任务", "Task Gate Softmax", "热插拔 add_task()"],
        meta: ["◈ in_dim=512", "⚡ expert_dim=256", "◇ 3 + N 个任务"],
      })}

      <div class="section">
        <h2>1. 算法原理</h2>
        <p class="text-sm text-slate-600 leading-relaxed">
          多任务学习的核心矛盾是 <b>正迁移 vs 负迁移</b>。风险 / 可信 / 类目三个任务语义差异较大：
          风险是稀疏二分类问题；类目是 2000 类超长尾；可信介于回归和 5 级排序之间。若共用一套 backbone，<b>梯度冲突</b> 会让弱任务被强任务淹没。
        </p>

        <h3>· MMoE → PLE 演进</h3>
        <div class="mermaid">
flowchart TB
    subgraph MMoE[MMoE Ma et al. 2018]
        x1[unified emb 512-d]
        e1[Expert 1]
        e2[Expert 2]
        e3[Expert 3]
        e4[Expert 4]
        g1[Gate Risk]
        g2[Gate Trust]
        g3[Gate Category]
        x1 --> e1 & e2 & e3 & e4
        e1 & e2 & e3 & e4 --> g1 & g2 & g3
    end
        </div>

        <div class="mermaid">
flowchart TB
    subgraph PLE[PLE Tang et al. 2020 - current方案]
        x[unified emb 512-d]
        S[Shared Experts × 4]
        P1[Risk Private × 1]
        P2[Trust Private × 1]
        P3[Category Private × 1]
        G1[Gate Risk mixes S + P1]
        G2[Gate Trust mixes S + P2]
        G3[Gate Category mixes S + P3]
        x --> S
        x --> P1 & P2 & P3
        S --> G1 & G2 & G3
        P1 --> G1
        P2 --> G2
        P3 --> G3
    end
        </div>

        <div class="formula-block">
          $$ \\text{Output}_t = \\sum_{i=1}^{S+P_t} g_{t,i}(x) \\cdot e_i(x), \\quad g_t(x) = \\text{softmax}\\big(W_t x\\big) $$
          $$ \\text{Loss}_{\\text{total}} = \\sum_{t} \\lambda_t \\cdot L_t, \\quad (\\lambda_t \\text{ 由 GradNorm 动态学习}) $$
        </div>

        ${MCH.info(`
          <b>热插拔 PluginHead 的关键机制</b>：调用 <code>mmoe.add_task("gambling")</code> 时，
          <b>不改底座</b>，只新增 1 个私有专家 + 1 个 gate。底座参数保持稳定，业务方 500 正样本 1 天内即可训练出新子头。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互可视化</h2>
        <div class="grid-2">
          <div>
            <h3>· 任务 Gate Softmax 分布</h3>
            <p class="text-xs text-slate-500">模拟 3 个任务在 4 shared + 1 private 上的 gate 权重分布。</p>
            <div id="chart-gate" style="height:360px;"></div>
          </div>

          <div>
            <h3>· 梯度冲突：PLE vs MMoE vs Shared MLP</h3>
            <div class="ctrl-panel" style="margin-bottom:12px;">
              ${MCH.slider({ id: "ple-epochs", label: "训练 Epoch（模拟）", min: 5, max: 40, step: 1, value: 20 })}
              ${MCH.slider({ id: "ple-noise", label: "任务差异度", min: 0.1, max: 1.0, step: 0.05, value: 0.5 })}
            </div>
            <div id="chart-conflict" style="height:300px;"></div>
          </div>
        </div>

        <h3 style="margin-top:24px;">· 动态调节：专家数 & 任务数</h3>
        <div class="ctrl-panel">
          <div class="grid-3">
            <div>${MCH.slider({ id: "ple-S", label: "共享专家数 S", min: 1, max: 8, step: 1, value: 4 })}</div>
            <div>${MCH.slider({ id: "ple-P", label: "每任务私有数 P", min: 0, max: 4, step: 1, value: 1 })}</div>
            <div>${MCH.slider({ id: "ple-T", label: "任务数 T（含 Plugin）", min: 3, max: 10, step: 1, value: 4 })}</div>
          </div>
        </div>
        <div id="ple-params" class="card mt-4"></div>
      </div>
    `;
  },

  mount() {
    // Gate 分布
    const gateChart = MCH.echart(document.getElementById("chart-gate"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", data: ["Shared 1", "Shared 2", "Shared 3", "Shared 4", "Private"] },
      yAxis: { type: "value", name: "Gate weight", max: 0.6 },
      series: [
        { name: "Risk", type: "bar", data: [0.10, 0.15, 0.22, 0.18, 0.35], color: "#ef4444", barGap: 0.1 },
        { name: "Trust", type: "bar", data: [0.20, 0.22, 0.15, 0.13, 0.30], color: "#f59e0b" },
        { name: "Category", type: "bar", data: [0.18, 0.20, 0.14, 0.20, 0.28], color: "#4f46e5" },
      ],
    });

    // 梯度冲突模拟
    const cChart = MCH.echart(document.getElementById("chart-conflict"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "Epoch" },
      yAxis: { type: "value", name: "平均任务 Loss (标准化)", min: 0 },
      series: [
        { name: "Shared MLP (baseline)", type: "line", smooth: true, data: [], color: "#94a3b8", lineStyle: { width: 2 } },
        { name: "MMoE (shared only)", type: "line", smooth: true, data: [], color: "#f59e0b", lineStyle: { width: 2 } },
        { name: "PLE (shared + private)", type: "line", smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
      ],
    });
    const updateC = () => {
      const epochs = parseInt(document.getElementById("ple-epochs").value);
      const noise = parseFloat(document.getElementById("ple-noise").value);
      const xs = MCH.linspace(1, epochs, epochs);
      const shared = xs.map(e => 1.0 - 0.3 * (1 - Math.exp(-e / 10)) + noise * 0.15 * Math.sin(e * 1.2) + 0.08);
      const mmoe = xs.map(e => 0.9 - 0.4 * (1 - Math.exp(-e / 8)) + noise * 0.09 * Math.sin(e * 0.9) + 0.04);
      const ple = xs.map(e => 0.85 - 0.55 * (1 - Math.exp(-e / 6)) + noise * 0.04 * Math.sin(e * 0.7));
      cChart.setOption({
        series: [
          { data: xs.map((x, i) => [x, shared[i]]) },
          { data: xs.map((x, i) => [x, mmoe[i]]) },
          { data: xs.map((x, i) => [x, ple[i]]) },
        ],
      });
    };
    ["ple-epochs", "ple-noise"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateC();
      });
    });
    updateC();

    // 参数量估算
    const paramsEl = document.getElementById("ple-params");
    const updateP = () => {
      const S = parseInt(document.getElementById("ple-S").value);
      const P = parseInt(document.getElementById("ple-P").value);
      const T = parseInt(document.getElementById("ple-T").value);
      const D_in = 512, D_out = 256, hidden = 512;
      const expertParams = D_in * hidden + hidden * D_out;   // 单个 Expert 参数
      const gateParams = D_in * (S + P);
      const totalExperts = S + T * P;
      const total = totalExperts * expertParams + T * gateParams;
      const speakText = `
        <h4 class="font-semibold text-slate-800 text-sm">· 当前结构：</h4>
        <div class="grid-3 mt-2">
          <div><div class="text-xs text-slate-500">共享专家</div><div class="text-lg font-bold text-indigo-600">${S}</div></div>
          <div><div class="text-xs text-slate-500">每任务私有</div><div class="text-lg font-bold text-indigo-600">${P}</div></div>
          <div><div class="text-xs text-slate-500">任务数</div><div class="text-lg font-bold text-indigo-600">${T}</div></div>
        </div>
        <div class="grid-3 mt-3">
          <div><div class="text-xs text-slate-500">总专家数</div><div class="text-base font-bold text-slate-700">${totalExperts}</div></div>
          <div><div class="text-xs text-slate-500">单专家参数</div><div class="text-base font-bold text-slate-700">${(expertParams / 1e3).toFixed(1)} K</div></div>
          <div><div class="text-xs text-slate-500">MMoE 总参数</div><div class="text-base font-bold text-indigo-700">${(total / 1e6).toFixed(2)} M</div></div>
        </div>
        <div class="text-xs text-slate-500 mt-3">
          <b>备注</b>：PLE 相比普通 MMoE 增加 <code>T·P</code> 个私有专家，代价 <b>${((T * P * expertParams) / 1e6).toFixed(2)} M</b> 换来多任务梯度冲突显著减少。
        </div>
      `;
      paramsEl.innerHTML = speakText;
    };
    ["ple-S", "ple-P", "ple-T"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateP();
      });
    });
    updateP();
  },
});
