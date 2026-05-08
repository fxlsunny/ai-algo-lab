/* 模块：前沿 Mamba / SSM */
MCH.register("frontier_mamba", {
  render() {
    const code = `# Mamba: Selective State Space Model
# Gu & Dao, COLM 2024

# 核心公式：连续 SSM
#   h'(t) = A · h(t) + B · x(t)    # 隐状态演化
#   y(t)  = C · h(t)                # 输出

# 离散化（Zero-Order Hold / Bilinear）：
#   h_k = Ā · h_{k-1} + B̄ · x_k
#   y_k = C · h_k

# 传统 SSM (S4) 问题：A, B, C 与输入无关 → 表达受限
# Mamba 创新：让 A, B, C 依赖输入 x（Selective SSM）
# 代价：失去全局卷积形式，但用 hardware-aware 并行扫描弥补

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        d_inner = expand * d_model
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        # 1D 卷积：短程信息
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv-1, groups=d_inner)
        # Selective SSM 参数（输入依赖）
        self.x_proj = nn.Linear(d_inner, d_state + d_state + 1)   # Δ, B, C
        self.dt_proj = nn.Linear(1, d_inner)
        # A 是固定学习参数（HiPPO 初始化）
        self.A_log = nn.Parameter(-torch.arange(1, d_state+1).float().log().unsqueeze(0).expand(d_inner, -1))
        self.out_proj = nn.Linear(d_inner, d_model)

    def forward(self, x):
        # 1) 分支
        xz = self.in_proj(x).chunk(2, dim=-1)
        x_proj, z = xz
        # 2) 1D conv
        x_proj = F.silu(self.conv1d(x_proj.transpose(1, 2))[:, :, :x.size(1)].transpose(1, 2))
        # 3) 选择性 SSM
        #    Δ, B, C 都依赖 x_proj —— 这是 Mamba 的核心
        proj = self.x_proj(x_proj)
        delta, B, C = proj.split([self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        # 4) Selective Scan (hardware-aware, 需要 CUDA kernel)
        y = selective_scan(x_proj, delta, self.A_log, B, C)
        # 5) Gate
        y = y * F.silu(z)
        return self.out_proj(y)`;

    return `
      ${MCH.hero({
        icon: "🌀",
        name: "Mamba / SSM 状态空间模型",
        en: "Selective State Space Models (2023-24)",
        tags: ["Mamba", "SSM", "Selective Scan", "线性复杂度", "Transformer 替代"],
        meta: ["◈ O(N) vs Transformer O(N²)", "⚡ 长序列友好"],
      })}

      ${MCH.versionSection("frontier_mamba")}

      <div class="section">
        <h2>1. 一句话：挑战 Transformer 的下一代序列架构</h2>
        ${MCH.info(`
          <b>核心洞察</b>：Transformer 的 O(N²) attention 在<b>极长序列（100k+ tokens）</b>下变得不可承受。
          Mamba 用状态空间模型 (SSM) 把复杂度降到 <b>O(N)</b>，同时保留 Transformer 的建模能力。
          <br/><br/>
          <b>历史脉络</b>：
          <ol style="margin-top:6px;padding-left:20px;">
            <li><b>S4 (2022)</b>：首次证明 SSM 可以替代 Transformer，但效果还差</li>
            <li><b>Mamba (2023)</b>：🏆 Selective SSM，让参数依赖输入 → 能力反超</li>
            <li><b>Mamba-2 (2024)</b>：SSD 框架，Transformer 式高效训练</li>
            <li><b>Jamba / Zamba / Samba (2024)</b>：Transformer + Mamba 混合架构成熟</li>
          </ol>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. Mamba vs Transformer vs RNN 核心差异</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>RNN/LSTM</th><th>Transformer</th><th>Mamba (SSM)</th></tr></thead>
          <tbody>
            <tr><td>训练复杂度</td><td>❌ O(N) 串行</td><td>✓ O(N²) 并行</td><td>✓ O(N) 并行</td></tr>
            <tr><td>推理复杂度</td><td>✓ O(1) / token</td><td>❌ O(N) / token</td><td>✓ O(1) / token</td></tr>
            <tr><td>长序列内存</td><td>✓ 常数</td><td>❌ O(N²)</td><td>✓ 常数</td></tr>
            <tr><td>全局依赖</td><td>❌ 梯度消失</td><td>✓✓ 完美</td><td>✓ 状态压缩</td></tr>
            <tr><td>输入依赖参数</td><td>❌</td><td>✓ (Q/K/V 依赖 input)</td><td>✓ (Δ/B/C 依赖 input)</td></tr>
            <tr><td>硬件友好</td><td>串行限制</td><td>GPU 大算力</td><td>🆕 Selective Scan CUDA</td></tr>
            <tr><td>100k+ 序列</td><td>❌</td><td>❌ 需 Flash + 稀疏</td><td>✓ 天然友好</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：不同序列长度下的复杂度对比</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "mamba-N", label: "序列长度 N", min: 1024, max: 131072, step: 1024, value: 8192 })}
              ${MCH.slider({ id: "mamba-d", label: "隐层维度 d", min: 128, max: 4096, step: 128, value: 1024 })}
            </div>
            <div id="mamba-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-mamba" style="height:340px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 现代混合架构</h2>
        <table class="table">
          <thead><tr><th>架构</th><th>组合</th><th>规模</th><th>亮点</th></tr></thead>
          <tbody>
            <tr><td><b>Jamba</b> (AI21)</td><td>Transformer + Mamba + MoE</td><td>52B (12B 激活)</td><td>🏆 256k 上下文</td></tr>
            <tr><td><b>Zamba</b> (Zyphra)</td><td>Mamba 主体 + 共享 Attention</td><td>7B</td><td>开源 + 高性价比</td></tr>
            <tr><td><b>Samba</b> (Microsoft)</td><td>Mamba + SWA (Sliding Attention)</td><td>3.8B</td><td>效率高精度好</td></tr>
            <tr><td><b>Falcon Mamba</b></td><td>纯 Mamba</td><td>7B</td><td>🆕 长上下文开源</td></tr>
            <tr><td><b>🆕 MiniMax-01</b></td><td>Lightning Attn + MoE</td><td>456B</td><td>4M 上下文</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 参考</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://arxiv.org/abs/2312.00752" target="_blank">Mamba 原始论文</a></li>
          <li><a href="https://arxiv.org/abs/2405.21060" target="_blank">Mamba-2 (SSD)</a></li>
          <li><a href="https://github.com/state-spaces/mamba" target="_blank">官方 Mamba GitHub</a></li>
          <li><a href="https://srush.github.io/annotated-mamba/" target="_blank">Annotated Mamba</a> — 逐行解读</li>
          <li><a href="https://huggingface.co/ai21labs/Jamba-v0.1" target="_blank">Jamba on HF</a></li>
          <li><a href="https://huggingface.co/tiiuae/falcon-mamba-7b" target="_blank">Falcon Mamba 7B</a></li>
        </ul>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("frontier_mamba").pros, MCH.getById("frontier_mamba").cons, MCH.getById("frontier_mamba").best_for)}
      </div>
    `;
  },

  mount() {
    const chart = MCH.echart(document.getElementById("chart-mamba"), {
      tooltip: { trigger: "axis", valueFormatter: v => v.toExponential(1) },
      legend: { top: 0 },
      grid: { left: 70, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "log", name: "序列长度 N", logBase: 2, min: 1024, max: 131072 },
      yAxis: { type: "log", name: "FLOPs", logBase: 10 },
      series: [
        { name: "Transformer O(N²·d)", type: "line", smooth: true, showSymbol: false, color: "#ef4444", lineStyle: { width: 3 }, data: [] },
        { name: "🌀 Mamba O(N·d²)", type: "line", smooth: true, showSymbol: false, color: "#d946ef", lineStyle: { width: 3 }, data: [] },
        { name: "当前配置", type: "scatter", symbolSize: 14, data: [] },
      ],
    });
    const update = () => {
      const N = parseInt(document.getElementById("mamba-N").value);
      const d = parseInt(document.getElementById("mamba-d").value);
      const Ns = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
      const tr = Ns.map(n => [n, n * n * d]);
      const mm = Ns.map(n => [n, n * d * d]);
      chart.setOption({
        series: [
          { data: tr },
          { data: mm },
          { data: [{ value: [N, N * N * d], itemStyle: { color: "#ef4444" }, label: { show: true, position: "top", formatter: "Transformer" } }, { value: [N, N * d * d], itemStyle: { color: "#d946ef" }, label: { show: true, position: "bottom", formatter: "Mamba" } }] },
        ],
      });
      const speedup = (N * N * d) / (N * d * d);
      document.getElementById("mamba-info").innerHTML = `
        <b>N=${N.toLocaleString()}</b>, <b>d=${d}</b><br/>
        Transformer FLOPs: <span style="color:#ef4444;">${(N * N * d).toExponential(1)}</span><br/>
        Mamba FLOPs: <span style="color:#d946ef;">${(N * d * d).toExponential(1)}</span><br/>
        <b>Mamba 相对加速</b>: <span style="color:#10b981;font-weight:700;">${speedup.toFixed(1)}×</span><br/>
        <span style="color:#64748b;">N 越大 Mamba 优势越明显；当 N &gt; d 时 Mamba 更优。实际部署还需 Selective Scan CUDA kernel。</span>
      `;
    };
    ["mamba-N", "mamba-d"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
