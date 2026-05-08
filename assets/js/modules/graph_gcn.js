/* 模块：GCN */
MCH.register("graph_gcn", {
  render() {
    const code = `# GCN — Graph Convolutional Network (Kipf & Welling, ICLR 2017)
# 频谱图理论的工程化：每层用归一化邻接矩阵更新节点特征

def gcn_layer(H, A_hat, W):
    """H: (N, d)   节点特征
       A_hat: (N, N) 归一化邻接 = D^(-1/2) (A + I) D^(-1/2)
       W: (d, d')   可学习权重"""
    return F.relu(A_hat @ H @ W)     # 1 层 = 聚合邻居 + 线性变换 + 激活

# 核心：归一化邻接
# A + I  :  加自环 → 聚合时包含自身特征
# D^(-1/2) · (A+I) · D^(-1/2)  :  对称归一化防高度节点"爆炸"
def normalize_adj(A):
    A_tilde = A + torch.eye(A.size(0))
    D = A_tilde.sum(1)
    D_inv_sqrt = torch.diag(D.pow(-0.5))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt

# 完整 2 层 GCN
class GCN(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        self.W1 = nn.Parameter(torch.randn(in_dim, hidden) * 0.01)
        self.W2 = nn.Parameter(torch.randn(hidden, out_dim) * 0.01)
    def forward(self, X, A_hat):
        H1 = F.relu(A_hat @ X @ self.W1)
        H1 = F.dropout(H1, 0.5)
        return A_hat @ H1 @ self.W2           # softmax for classification

# 训练：半监督节点分类（只有部分节点有标签）
# loss = CE(logits[train_mask], y[train_mask])`;

    return `
      ${MCH.hero({ icon: "⬢", name: "GCN — 图卷积网络", en: "Graph Convolutional Network (Kipf & Welling 2017)", tags: ["频谱图卷积", "Transductive", "半监督", "GNN 奠基"], meta: ["◈ 2 层就很强", "⚡ 代码 50 行"] })}

      ${MCH.versionSection("graph_gcn")}

      <div class="section">
        <h2>1. 核心思想：让每个节点"听邻居说话"</h2>
        <p class="text-sm text-slate-600">每一层 GCN 对每个节点：</p>
        <div class="formula-block">
          $$ H^{(l+1)} = \\sigma\\Big( \\tilde D^{-1/2} \\tilde A \\tilde D^{-1/2} \\, H^{(l)} W^{(l)} \\Big) $$
          其中 $\\tilde A = A + I$（加自环），$\\tilde D$ 是 $\\tilde A$ 的度矩阵。
        </div>
        <p class="text-sm text-slate-600">3 步拆解：</p>
        <ol class="text-sm text-slate-700 list-decimal pl-6">
          <li><b>聚合</b>：加权平均邻居特征（含自己）；</li>
          <li><b>变换</b>：乘可学习的线性层 W；</li>
          <li><b>激活</b>：ReLU / GELU 引入非线性。</li>
        </ol>
        ${MCH.info(`
          <b>对称归一化的意义</b>：朴素平均 $A + I$ 会让高度节点特征值爆炸。
          $\\tilde D^{-1/2}$ 从两侧"除度数"，使每个聚合都是"相对值"，训练更稳定。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：GCN 层数与过平滑问题</h2>
        <p class="text-sm text-slate-600">堆叠过多层会让所有节点 embedding 趋同（<b>Over-smoothing</b>）。通常 2-3 层是最优点。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "gcn-L", label: "层数 L", min: 1, max: 20, step: 1, value: 2 })}
            </div>
          </div>
          <div id="chart-oversmooth" style="height:280px;"></div>
        </div>

        <h3 style="margin-top:18px;">· 2 层 GCN 的感受野（= 2-hop 邻居）</h3>
        <div id="chart-receptive" style="height:340px;"></div>
      </div>

      <div class="section">
        <h2>4. GCN vs 其他 GNN</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>聚合方式</th><th>Inductive</th><th>大图</th><th>相对精度</th></tr></thead>
          <tbody>
            <tr><td><b>GCN</b></td><td>$\\tilde D^{-1/2} \\tilde A \\tilde D^{-1/2}$ 固定归一化</td><td>❌</td><td>❌</td><td>基线</td></tr>
            <tr><td>GraphSAGE</td><td>采样 + mean/max/LSTM</td><td>✓</td><td>✓</td><td>略低</td></tr>
            <tr><td>GAT</td><td>注意力加权</td><td>✓</td><td>中等</td><td>更高</td></tr>
            <tr><td>GIN</td><td>SUM 聚合 + MLP</td><td>✓</td><td>✓</td><td>高（图级）</td></tr>
            <tr><td>HGT</td><td>异构图 attention</td><td>✓</td><td>✓</td><td>最高（异构图）</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("graph_gcn").pros, MCH.getById("graph_gcn").cons, MCH.getById("graph_gcn").best_for)}
      </div>
    `;
  },

  mount() {
    // Over-smoothing 仿真：随层数增加，节点特征相似度增加
    const osChart = MCH.echart(document.getElementById("chart-oversmooth"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "GCN 层数 L" },
      yAxis: { type: "value", name: "指标" },
      series: [
        { name: "节点分类准确率", type: "line", smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
        { name: "节点特征相似度 (cosine avg)", type: "line", smooth: true, data: [], color: "#ef4444", lineStyle: { width: 3 } },
      ],
    });
    const update = () => {
      const L = parseInt(document.getElementById("gcn-L").value);
      const ls = [...Array(20).keys()].map(i => i + 1);
      // acc: peak at 2-3, then drop
      const acc = ls.map(l => 0.55 + 0.3 * Math.exp(-Math.pow(l - 2.5, 2) / 3));
      // similarity: rises to 1
      const sim = ls.map(l => 1 - Math.exp(-l * 0.4));
      osChart.setOption({
        series: [
          { data: ls.map((l, i) => [l, acc[i]]), markPoint: { data: [{ coord: [L, acc[L - 1]], symbolSize: 14, itemStyle: { color: "#f59e0b" } }] } },
          { data: ls.map((l, i) => [l, sim[i]]) },
        ],
      });
    };
    document.getElementById("gcn-L").addEventListener("input", (e) => {
      document.getElementById("gcn-L-val").textContent = e.target.value;
      update();
    });
    update();

    // 感受野可视化
    const N = 24;
    const pos = [];
    for (let i = 0; i < N; i++) {
      const a = (i / N) * 2 * Math.PI;
      pos.push([Math.cos(a) * 130, Math.sin(a) * 130]);
    }
    // 构造一个环状 + 随机跨接
    const edges = [];
    for (let i = 0; i < N; i++) edges.push([i, (i + 1) % N]);
    edges.push([0, 5], [3, 8], [10, 16], [13, 18], [20, 2]);
    const center = 0;
    const hop1 = new Set();
    edges.forEach(([u, v]) => { if (u === center) hop1.add(v); else if (v === center) hop1.add(u); });
    const hop2 = new Set();
    edges.forEach(([u, v]) => {
      if (hop1.has(u) && v !== center && !hop1.has(v)) hop2.add(v);
      if (hop1.has(v) && u !== center && !hop1.has(u)) hop2.add(u);
    });
    const nodes = pos.map((p, i) => ({
      name: `N${i}`, value: p, symbolSize: i === center ? 26 : 18,
      itemStyle: { color: i === center ? "#ef4444" : (hop1.has(i) ? "#4f46e5" : (hop2.has(i) ? "#c7d2fe" : "#e2e8f0")) },
      label: { show: true, formatter: `${i}`, fontSize: 10, fontWeight: 700, color: i === center ? "white" : (hop1.has(i) ? "white" : "#64748b") },
    }));
    MCH.echart(document.getElementById("chart-receptive"), {
      tooltip: {},
      xAxis: { show: false, min: -160, max: 160 },
      yAxis: { show: false, min: -160, max: 160 },
      grid: { left: 20, right: 20, top: 20, bottom: 20 },
      legend: { top: 0, data: [{ name: "中心节点" }, { name: "1-hop (GCN L=1)" }, { name: "2-hop (GCN L=2)" }, { name: "未覆盖" }] },
      series: [
        { name: "中心节点", type: "scatter", data: [[pos[center][0], pos[center][1]]], itemStyle: { color: "#ef4444" }, symbolSize: 24, silent: true },
        { name: "1-hop (GCN L=1)", type: "scatter", data: [...hop1].map(i => [pos[i][0], pos[i][1]]), itemStyle: { color: "#4f46e5" }, symbolSize: 18, silent: true },
        { name: "2-hop (GCN L=2)", type: "scatter", data: [...hop2].map(i => [pos[i][0], pos[i][1]]), itemStyle: { color: "#c7d2fe" }, symbolSize: 18, silent: true },
        { name: "未覆盖", type: "scatter", data: [[1000, 1000]], itemStyle: { color: "#e2e8f0" }, silent: true },
        { type: "graph", coordinateSystem: "cartesian2d", data: nodes, links: edges.map(([u, v]) => ({ source: `N${u}`, target: `N${v}`, lineStyle: { color: "#cbd5e1", width: 1, opacity: 0.5 } })) },
      ],
    });
  },
});
