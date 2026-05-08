/* 模块：GAT */
MCH.register("graph_gat", {
  render() {
    const code = `# GAT — Graph Attention Network (Veličković et al., ICLR 2018)
# 核心：邻居聚合权重由 attention 自适应学习（替代 GCN 的固定 D^(-1/2) 归一化）

class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=8, concat=True):
        self.W = nn.Parameter(torch.randn(n_heads, in_dim, out_dim) * 0.01)
        # attention 参数：对 [Wh_i || Wh_j] 做线性打分
        self.a = nn.Parameter(torch.randn(n_heads, 2 * out_dim) * 0.01)
        self.leaky = nn.LeakyReLU(0.2)
        self.concat = concat

    def forward(self, X, adj):            # X: (N, in_dim)  adj: (N, N) binary
        N = X.size(0)
        outs = []
        for h in range(self.W.size(0)):
            Wh = X @ self.W[h]             # (N, out_dim)
            # 对每条边 (i, j) 计算 attention
            # e_ij = LeakyReLU( a · [Wh_i || Wh_j] )
            e = self.leaky(
                (Wh @ self.a[h][:out_dim, None]).expand(-1, N) +
                (Wh @ self.a[h][out_dim:, None]).T.expand(N, -1)
            )
            # Mask 非边
            e = e.masked_fill(adj == 0, -1e9)
            # softmax 归一化（每行表示 i 对所有邻居的注意力）
            alpha = F.softmax(e, dim=1)
            # 加权聚合
            outs.append(alpha @ Wh)        # (N, out_dim)

        if self.concat:
            return F.elu(torch.cat(outs, dim=-1))       # (N, H·out_dim)
        else:
            return F.elu(torch.stack(outs).mean(0))     # (N, out_dim), 最后一层用


# 2 层 GAT，末层平均多头
class GAT(nn.Module):
    def __init__(self, in_dim, hidden=8, n_classes=7, n_heads=8):
        self.l1 = GATLayer(in_dim, hidden, n_heads=n_heads, concat=True)
        self.l2 = GATLayer(hidden * n_heads, n_classes, n_heads=1, concat=False)
    def forward(self, X, adj):
        h = self.l1(X, adj)
        return self.l2(h, adj)`;

    return `
      ${MCH.hero({ icon: "⬨", name: "GAT — 图注意力网络", en: "Graph Attention Network (Veličković 2018)", tags: ["Attention", "Multi-Head", "自适应权重", "可解释"], meta: ["◈ 每条边学一个权重", "⚡ 可视化 attention"] })}

      ${MCH.versionSection("graph_gat")}

      <div class="section">
        <h2>1. 为什么要 attention？</h2>
        <p class="text-sm text-slate-600">GCN 用 <b>度数归一化</b> 给每个邻居固定权重；但不同邻居对中心节点的"重要性"其实差异很大：</p>
        <ul class="text-sm text-slate-700 list-disc pl-6 mt-2">
          <li>法人关系 ≠ 同地址关系（风险权重不同）</li>
          <li>学术引用同一论文的不同后续论文 ≠ 同样重要</li>
          <li>社交网络中"熟人" vs "弱连接"信号差异大</li>
        </ul>
        <p class="text-sm text-slate-600 mt-2">GAT 让模型<b>自己学</b>每条边的权重：</p>
        <div class="formula-block">
          $$ e_{ij} = \\text{LeakyReLU}\\big(\\vec{a}^\\top [W h_i \\| W h_j]\\big), \\quad \\alpha_{ij} = \\text{softmax}_j (e_{ij}) $$
          $$ h_i' = \\sigma\\Big( \\sum_{j \\in \\mathcal{N}(i)} \\alpha_{ij} W h_j \\Big) $$
        </div>
      </div>

      <div class="section">
        <h2>2. Multi-Head 的价值</h2>
        <p class="text-sm text-slate-600">类似 Transformer：多头让不同头关注不同<b>邻居子集</b>（例如一头关注高度节点、另一头关注同社区）。</p>
        <div id="chart-heads" style="height:300px;"></div>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：GAT 注意力权重可视化</h2>
        <p class="text-sm text-slate-600">中心节点对每个邻居的 <b>attention 权重</b>。调节温度观察 attention 的尖锐/平滑程度。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "gat-temp", label: "Attention 温度 (softmax 除数)", min: 0.1, max: 5, step: 0.1, value: 1 })}
              ${MCH.slider({ id: "gat-focus", label: "某个邻居的重要度 (模拟)", min: 0, max: 5, step: 0.1, value: 2 })}
            </div>
            <div id="gat-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-attn-gat" style="height:360px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. GCN vs GraphSAGE vs GAT 选择</h2>
        <table class="table">
          <thead><tr><th>需求</th><th>推荐</th><th>理由</th></tr></thead>
          <tbody>
            <tr><td>小图、求极致精度</td><td><b>GCN</b></td><td>简单稳定，2 层够用</td></tr>
            <tr><td>超大图、在线推理</td><td><b>GraphSAGE</b></td><td>Inductive、mini-batch</td></tr>
            <tr><td>需要可解释（哪条边重要）</td><td><b>GAT</b></td><td>attention 权重即解释</td></tr>
            <tr><td>异构邻居差异大</td><td><b>GAT</b></td><td>自适应权重</td></tr>
            <tr><td>有节点 + 边属性</td><td>GAT + Edge features</td><td>扩展 attention 支持边特征</td></tr>
            <tr><td>异构关系图</td><td>HGT / R-GCN</td><td>专为异构设计</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("graph_gat").pros, MCH.getById("graph_gat").cons, MCH.getById("graph_gat").best_for)}
      </div>
    `;
  },

  mount() {
    // Multi-head viz: 不同 head 的 attention 偏好
    MCH.echart(document.getElementById("chart-heads"), {
      title: { text: "4 头 Attention 的偏好模式（示意）", left: "center", textStyle: { fontSize: 12 } },
      tooltip: {},
      legend: { bottom: 0 },
      grid: { left: 60, right: 30, top: 50, bottom: 50 },
      xAxis: { type: "category", data: ["法人邻居", "超管邻居", "同主体", "同地址", "聚合码"] },
      yAxis: { type: "value", name: "attention α", max: 1 },
      series: [
        { name: "Head 1 (法人偏好)", type: "bar", data: [0.65, 0.10, 0.08, 0.10, 0.07], color: "#ef4444", barGap: 0.1 },
        { name: "Head 2 (结构相似)", type: "bar", data: [0.10, 0.60, 0.15, 0.10, 0.05], color: "#f59e0b" },
        { name: "Head 3 (地理位置)", type: "bar", data: [0.05, 0.10, 0.10, 0.65, 0.10], color: "#10b981" },
        { name: "Head 4 (均衡)", type: "bar", data: [0.22, 0.18, 0.20, 0.22, 0.18], color: "#4f46e5" },
      ],
    });

    // Attention viz
    const gChart = MCH.echart(document.getElementById("chart-attn-gat"), {
      tooltip: {},
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", data: ["邻居 1", "邻居 2", "邻居 3 ⭐", "邻居 4", "邻居 5", "邻居 6"] },
      yAxis: { type: "value", name: "α (attention)", max: 1 },
      series: [{ type: "bar", barWidth: 30, data: [] }],
    });
    const update = () => {
      const tau = parseFloat(document.getElementById("gat-temp").value);
      const focus = parseFloat(document.getElementById("gat-focus").value);
      const rawLogits = [0.3, 0.5, focus, 0.2, 0.4, 0.1];
      // softmax with temperature
      const z = rawLogits.map(v => v / tau);
      const m = Math.max(...z);
      const e = z.map(v => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      const a = e.map(v => v / s);
      gChart.setOption({
        series: [{
          data: a.map((v, i) => ({ value: +v.toFixed(3), itemStyle: { color: i === 2 ? "#ef4444" : "#4f46e5" } })),
          label: { show: true, position: "top", formatter: "{c}", color: "#64748b" },
        }],
      });
      const entropy = -a.reduce((s, p) => s + p * Math.log2(p + 1e-12), 0);
      document.getElementById("gat-info").innerHTML = `
        <b>Entropy (分散度)</b>：<span style="color:#4f46e5;">${entropy.toFixed(3)} bits</span> (max=${Math.log2(6).toFixed(2)})<br/>
        <b>最大 α</b>：${Math.max(...a).toFixed(3)} · <b>α(邻居 3)</b>：<span style="color:#ef4444;font-weight:700;">${a[2].toFixed(3)}</span><br/>
        <span style="color:#64748b;">温度越小注意力越尖锐（聚焦一个邻居），越大越均匀。</span>
      `;
    };
    ["gat-temp", "gat-focus"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
