/* 模块：GraphSAGE */
MCH.register("graph_sage", {
  render() {
    const code = `# GraphSAGE — SAmple and aggreGatE (Hamilton et al., NeurIPS 2017)
# 核心：① 邻居采样（固定数量）  ② 多种聚合器

class GraphSAGELayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggregator="mean"):
        self.agg = aggregator
        if aggregator == "lstm":
            self.lstm = nn.LSTM(in_dim, in_dim, batch_first=True)
        self.fc_self = nn.Linear(in_dim, out_dim)
        self.fc_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, h_self, h_neigh):  # h_neigh: (B, K, in_dim)
        if self.agg == "mean":
            h_n = h_neigh.mean(dim=1)
        elif self.agg == "max":
            h_n = h_neigh.max(dim=1)[0]
        elif self.agg == "lstm":
            # 随机打乱邻居顺序保证置换不变
            idx = torch.randperm(h_neigh.size(1))
            _, (h_n, _) = self.lstm(h_neigh[:, idx])
            h_n = h_n.squeeze(0)

        # 拼接自身 + 邻居聚合
        return F.relu(self.fc_self(h_self) + self.fc_neigh(h_n))


# 训练过程：固定每层采样 K_l 个邻居 → mini-batch 可训练大图
def sample_neighbors(G, node_batch, K_list=[25, 10]):
    """第 1 层采样 25 个 1-hop 邻居，第 2 层对每个 1-hop 再采 10 个 2-hop 邻居"""
    sampled = {0: node_batch}
    for l, K in enumerate(K_list):
        next_layer = []
        for v in sampled[l]:
            nbrs = list(G.neighbors(v))
            if len(nbrs) > K:
                nbrs = random.sample(nbrs, K)           # 关键：只采固定 K 个
            next_layer.extend(nbrs)
        sampled[l + 1] = next_layer
    return sampled  # 反向计算：从最外层聚合回中心

# Inductive：新节点只需计算其 K-hop 邻居，不需要训练全图，真正可扩展`;

    return `
      ${MCH.hero({ icon: "⬣", name: "GraphSAGE — 归纳式图嵌入", en: "Sample And Aggregate", tags: ["Inductive", "邻居采样", "mini-batch", "工业级"], meta: ["◈ 亿级节点可用", "⚡ 新节点无需重训"] })}

      ${MCH.versionSection("graph_sage")}

      <div class="section">
        <h2>1. GCN 的痛点与 GraphSAGE 的解法</h2>
        <table class="table">
          <thead><tr><th>问题</th><th>GCN</th><th>GraphSAGE</th></tr></thead>
          <tbody>
            <tr><td>大图 OOM</td><td>全图邻接矩阵进内存</td><td><b>固定采样 K 个邻居</b></td></tr>
            <tr><td>新节点</td><td>Transductive, 需重训</td><td><b>Inductive</b>, 用现有 embedding 即可</td></tr>
            <tr><td>训练方式</td><td>整图 batch gradient</td><td><b>mini-batch SGD</b></td></tr>
            <tr><td>聚合方式</td><td>固定归一化邻接</td><td>Mean / Max / LSTM / Pool 可选</td></tr>
            <tr><td>适用规模</td><td>&lt; 100w 节点</td><td>&gt; 1亿节点</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. 核心：两跳邻居采样</h2>
        <p class="text-sm text-slate-600">对 batch 中的每个"目标节点"，先采 K₁ 个 1-hop 邻居，再对每个 1-hop 邻居采 K₂ 个 2-hop 邻居。形成一棵"有界深度、有界宽度"的<b>邻居树</b>，这棵树完整可放进 GPU batch。</p>
        <div class="mermaid">
flowchart TB
    C[目标节点 v]
    N1a[邻居 1]
    N1b[邻居 2]
    N1c["... K₁ 个"]
    N2aa[二跳 1-1]
    N2ab[二跳 1-2]
    N2ac["... K₂ 个"]
    C --> N1a & N1b & N1c
    N1a --> N2aa & N2ab & N2ac
        </div>
        <div class="formula-block">
          $$ h_v^{(l+1)} = \\sigma\\Big( W \\cdot [\\,h_v^{(l)} \\;\\|\\; \\text{AGG}_{u \\in \\mathcal{N}_K(v)}(h_u^{(l)})\\,] \\Big) $$
        </div>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：采样策略对计算量的影响</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "sg-K1", label: "K₁ (1-hop 采样数)", min: 5, max: 50, step: 1, value: 25 })}
              ${MCH.slider({ id: "sg-K2", label: "K₂ (2-hop 采样数)", min: 5, max: 50, step: 1, value: 10 })}
              ${MCH.slider({ id: "sg-B", label: "Batch 大小", min: 32, max: 2048, step: 32, value: 512 })}
              ${MCH.slider({ id: "sg-d", label: "hidden dim d", min: 64, max: 1024, step: 32, value: 256 })}
              ${MCH.slider({ id: "sg-N", label: "图节点总数 N", min: 1000, max: 100000000, step: 10000, value: 1000000, format: (v) => parseInt(v).toLocaleString() })}
            </div>
          </div>
          <div>
            <div id="sg-info" class="card"></div>
            <div id="chart-cost" style="height:240px;margin-top:10px;"></div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>5. 聚合器对比</h2>
        <table class="table">
          <thead><tr><th>Aggregator</th><th>公式</th><th>计算开销</th><th>表达力</th><th>置换不变？</th></tr></thead>
          <tbody>
            <tr><td><b>Mean</b></td><td>mean(h_u for u∈N(v))</td><td>低</td><td>中</td><td>✓</td></tr>
            <tr><td>Max Pooling</td><td>max(MLP(h_u))</td><td>中</td><td>较高</td><td>✓</td></tr>
            <tr><td>LSTM</td><td>LSTM(shuffled h_u)</td><td>高</td><td>最高</td><td>⚠ 需打乱</td></tr>
            <tr><td>Attention (类 GAT)</td><td>Σ α_uv · h_u</td><td>高</td><td>最高</td><td>✓</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("graph_sage").pros, MCH.getById("graph_sage").cons, MCH.getById("graph_sage").best_for)}
      </div>
    `;
  },

  mount() {
    const infoEl = document.getElementById("sg-info");
    const costChart = MCH.echart(document.getElementById("chart-cost"), {
      tooltip: { trigger: "axis" },
      grid: { left: 60, right: 30, top: 20, bottom: 30 },
      xAxis: { type: "value", name: "Batch 大小" },
      yAxis: { type: "log", name: "GFLOPs / batch", logBase: 10 },
      series: [{ type: "line", smooth: true, showSymbol: false, color: "#10b981", lineStyle: { width: 3 }, data: [] }],
    });
    const update = () => {
      const K1 = parseInt(document.getElementById("sg-K1").value);
      const K2 = parseInt(document.getElementById("sg-K2").value);
      const B = parseInt(document.getElementById("sg-B").value);
      const d = parseInt(document.getElementById("sg-d").value);
      const N = parseInt(document.getElementById("sg-N").value);
      // 每 batch：涉及节点数 = B * (1 + K1 + K1*K2), 计算 = 节点数 * d²
      const nodesInvolved = B * (1 + K1 + K1 * K2);
      const flopsPerBatch = nodesInvolved * d * d * 2;
      // 对比 GCN 全图：N * d²（每层）
      const gcnFlops = N * d * d * 2 * 2; // 2 层
      const memSage = nodesInvolved * d * 4 / 1024 / 1024;  // MB (FP32)
      const memGcn = N * d * 4 / 1024 / 1024 / 1024;  // GB

      const bs = MCH.linspace(32, 2048, 40);
      costChart.setOption({
        series: [{ data: bs.map(b => [b, b * (1 + K1 + K1 * K2) * d * d * 2 / 1e9]) }],
      });

      infoEl.innerHTML = `
        <h4 class="font-semibold text-sm text-slate-800">GraphSAGE vs GCN</h4>
        <div class="grid-2 mt-3">
          <div>
            <div class="text-xs text-slate-500">SAGE 单 batch 涉及节点</div>
            <div class="text-base font-bold text-emerald-700">${nodesInvolved.toLocaleString()}</div>
            <div class="text-xs text-slate-500 mt-2">SAGE 显存/batch</div>
            <div class="text-base font-bold text-emerald-700">${memSage.toFixed(1)} MB</div>
            <div class="text-xs text-slate-500 mt-2">SAGE GFLOPs/batch</div>
            <div class="text-base font-bold text-emerald-700">${(flopsPerBatch / 1e9).toFixed(2)} G</div>
          </div>
          <div>
            <div class="text-xs text-slate-500">GCN 全图节点</div>
            <div class="text-base font-bold text-red-600">${N.toLocaleString()}</div>
            <div class="text-xs text-slate-500 mt-2">GCN 显存（单层）</div>
            <div class="text-base font-bold text-red-600">${memGcn.toFixed(2)} GB</div>
            <div class="text-xs text-slate-500 mt-2">GCN GFLOPs/步</div>
            <div class="text-base font-bold text-red-600">${(gcnFlops / 1e9).toFixed(1)} G</div>
          </div>
        </div>
        <div class="text-xs text-slate-500 mt-3">
          SAGE 通过固定 K₁·K₂ 采样，把显存/计算与图规模 <b>解耦</b>，再大的图每步成本都一样。
        </div>
      `;
    };
    ["sg-K1", "sg-K2", "sg-B", "sg-d", "sg-N"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        const v = e.target.value;
        document.getElementById(id + "-val").textContent = id === "sg-N" ? parseInt(v).toLocaleString() : v;
        update();
      });
    });
    update();
  },
});
