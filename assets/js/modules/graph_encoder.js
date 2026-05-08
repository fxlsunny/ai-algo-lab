/* 模块：关系图 Encoder */
MCH.register("graph_encoder", {
  render() {
    const code = `# 双模式关系图 Encoder
# 来源：src/models/graph_encoder.py

class GraphEncoder(nn.Module):
    """- precomputed：线上默认。读取 T-1 离线 HGT 跑好的 256d Embedding
       - online_gnn：冷启 / 离线训练时用，简版异构 GraphSAGE"""
    def __init__(self, mode="precomputed", in_dim=256, hidden_dim=512, num_relations=5):
        if mode == "precomputed":
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            )
        elif mode == "online_gnn":
            self.sage = HeteroGraphSAGE(in_dim, hidden_dim, num_relations)
        self.missing_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)


class HeteroGraphSAGE(nn.Module):
    """极简异构 GraphSAGE（冷启兜底，生产建议替换为 DGL HGT）。

    对每种关系 r 用独立的线性层 W_r 对邻居做 transform 后 mean-aggregate，
    再与 self transform 相加，经 ReLU + 输出投影：
        h = MLP( W_self·x + Σ_r mean_r(W_r·N_r(x)) )
    """
    def __init__(self, in_dim, out_dim, num_relations=5):
        hid = max(in_dim, out_dim)
        self.rel_linears = nn.ModuleList([nn.Linear(in_dim, hid) for _ in range(num_relations)])
        self.self_linear = nn.Linear(in_dim, hid)
        self.out = nn.Sequential(nn.Linear(hid, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, batch):
        center = self.self_linear(batch["center_feat"])         # (B, hid)
        agg = 0
        for i, (nf, nm) in enumerate(zip(batch["neighbor_feat_per_rel"],
                                         batch["neighbor_mask_per_rel"])):
            nf = self.rel_linears[i](nf)                        # (B, K_r, hid)
            m = nm.unsqueeze(-1).float()
            agg = agg + (nf * m).sum(1) / m.sum(1).clamp(min=1)
        return self.out(F.relu(center + agg))`;

    return `
      ${MCH.hero({
        icon: "G",
        name: "关系图 Encoder — Precomputed HGT / Online GraphSAGE",
        en: "Heterogeneous Graph Encoder · T-1 precomputed + online fallback",
        tags: ["HGT 离线预计算", "5 种关系 法人/超管/主体/地址/聚合码", "GraphSAGE 冷启兜底", "缺失 missing_token"],
        meta: ["◈ 输入 256-d node emb", "⚡ 在线仅 MLP Refine", "◇ 输出 512-d"],
      })}

      <div class="section">
        <h2>1. 算法原理</h2>
        <p class="text-sm text-slate-600 leading-relaxed">
          商户黑产高度"结团"——一个高风险主体往往牵出 <b>法人</b> / <b>超管 uin</b> / <b>同主体商户</b> / <b>同地址</b> / <b>同聚合码收款</b>
          等多重关系。关系图模块在离线侧用 <b>HGT</b>（Heterogeneous Graph Transformer）对异构图逐夜重算节点 embedding，
          线上侧只做一次 MLP Refine，<b>零图计算时延</b>。
        </p>

        <h3>· 5 种核心关系</h3>
        <table class="table">
          <thead><tr><th>关系</th><th>定义</th><th>风险信号</th></tr></thead>
          <tbody>
            <tr><td><code>legal_person</code></td><td>同法人 uin</td><td>"壳公司群"批量注册</td></tr>
            <tr><td><code>super_admin</code></td><td>同超管 uin</td><td>批量实名/控制</td></tr>
            <tr><td><code>same_subject</code></td><td>同公司主体</td><td>多收款账户同一主体</td></tr>
            <tr><td><code>same_address</code></td><td>同注册地址</td><td>"地址农场"</td></tr>
            <tr><td><code>same_merchant_code</code></td><td>共用聚合收款码</td><td>跑分团伙标志</td></tr>
          </tbody>
        </table>

        <h3>· 为什么 precomputed 模式上线？</h3>
        ${MCH.info(`
          <b>算力 / 时延考量：</b>
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>HGT 全量推理</b>：1 亿节点 × 5 关系 × 平均 20 邻居 = 100 亿条边，在线不可能；</li>
            <li>离线 T-1 用 <b>DGL 分布式 HGT</b>，每日 ~ 24 GPU·h 算完，落 Parquet / HBase；</li>
            <li>在线取 node embedding 只需 1 次 <b>KV 查询</b> + MLP 投影，延迟 &lt; 2 ms；</li>
            <li>冷启（新商户今日注册，T-1 没跑到）→ 走 <code>online_gnn</code> 简版 GraphSAGE 兜底，或用 <code>missing_token</code>。</li>
          </ul>
        `, "tip")}

        <h3>· GraphSAGE 消息传递公式</h3>
        <div class="formula-block">
          $$ h_v = \\sigma\\Big( W_{\\text{self}} x_v \\;+\\; \\sum_{r \\in \\mathcal{R}} \\frac{1}{|\\mathcal{N}_r(v)|} \\sum_{u \\in \\mathcal{N}_r(v)} W_r x_u \\Big) $$
        </div>
      </div>

      <div class="section">
        <h2>2. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互可视化 — 商户关系子图</h2>
        <p class="text-sm text-slate-600">
          下图模拟一个中心商户（节点 0）的 2-hop 邻居关系图。不同颜色代表不同关系类型，节点大小与 GMV 成比例。
          <b>可拖拽节点</b>观察消息传递路径；点击节点查看其聚合权重。
        </p>
        <div id="chart-graph" style="height:520px;"></div>

        <div class="grid-2 mt-4">
          <div class="card">
            <h4 class="font-semibold text-slate-800">▎ 关系计数</h4>
            <div id="rel-stats" class="text-sm text-slate-600 mt-2"></div>
          </div>
          <div class="card">
            <h4 class="font-semibold text-slate-800">▎ 选中节点的风险聚合</h4>
            <div id="node-info" class="text-sm text-slate-600 mt-2">点击任意节点查看其邻居的风险聚合信号 →</div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 业务案例</h2>
        <div class="grid-2">
          <div class="card">
            <div class="tag tag-red">黑产案例</div>
            <h4 class="font-semibold mt-2">跑分矩阵识别</h4>
            <p class="text-xs text-slate-600 mt-2">
              某"咖啡店"商户 GMV 正常，但 2-hop 内出现 <b>12 个</b> 同超管账户，其中 3 个已判定涉赌。
              图模块聚合后风险分 <b>0.12 → 0.87</b>，被风险头升级为"黑"。
            </p>
          </div>
          <div class="card">
            <div class="tag tag-amber">冷启场景</div>
            <h4 class="font-semibold mt-2">新注册主体</h4>
            <p class="text-xs text-slate-600 mt-2">
              今日 T 日新注册的商户 T-1 尚无 HGT embedding，走 online 异构 GraphSAGE 用 T-1
              已有的邻居兜底计算，确保当日可识别。
            </p>
          </div>
        </div>
      </div>
    `;
  },

  mount() {
    const el = document.getElementById("chart-graph");
    if (!el) return;
    // 构造关系图
    const relNames = ["法人", "超管", "主体", "地址", "聚合码"];
    const relColors = ["#ef4444", "#f97316", "#8b5cf6", "#0ea5e9", "#10b981"];
    // 中心节点 + 每种关系若干邻居
    const nodes = [{ id: 0, name: "目标商户 M0", risk: 0.12, gmv: 320, category: 0, symbolSize: 70, itemStyle: { color: "#4f46e5" } }];
    const links = [];
    let idx = 1;
    // 法人：3 邻居，其中 1 已知黑
    [
      { risk: 0.91, gmv: 45, rel: 0, name: "M1" },
      { risk: 0.15, gmv: 120, rel: 0, name: "M2" },
      { risk: 0.08, gmv: 80, rel: 0, name: "M3" },
      { risk: 0.82, gmv: 20, rel: 1, name: "M4" },
      { risk: 0.75, gmv: 30, rel: 1, name: "M5" },
      { risk: 0.31, gmv: 60, rel: 1, name: "M6" },
      { risk: 0.12, gmv: 280, rel: 2, name: "M7" },
      { risk: 0.10, gmv: 190, rel: 2, name: "M8" },
      { risk: 0.05, gmv: 50, rel: 3, name: "M9" },
      { risk: 0.09, gmv: 70, rel: 3, name: "M10" },
      { risk: 0.45, gmv: 85, rel: 4, name: "M11" },
      { risk: 0.58, gmv: 110, rel: 4, name: "M12" },
    ].forEach((n) => {
      nodes.push({
        id: idx, name: n.name, risk: n.risk, gmv: n.gmv, category: n.rel,
        symbolSize: 20 + Math.min(40, n.gmv / 10),
        itemStyle: { color: relColors[n.rel], opacity: 0.4 + n.risk * 0.6 },
      });
      links.push({ source: 0, target: idx, lineStyle: { color: relColors[n.rel], width: 1 + n.risk * 2, opacity: 0.6 } });
      idx++;
    });

    const chart = MCH.echart(el, {
      tooltip: { formatter: (p) => p.dataType === "node" ? `${p.data.name}<br/>风险分=${p.data.risk}<br/>GMV=¥${p.data.gmv}万` : "" },
      legend: [{ data: relNames.map((n) => ({ name: n })), top: 10 }],
      series: [{
        type: "graph", layout: "force", roam: true, draggable: true,
        categories: relNames.map((n, i) => ({ name: n, itemStyle: { color: relColors[i] } })),
        nodes, links,
        force: { repulsion: 600, edgeLength: [80, 160], gravity: 0.05 },
        label: { show: true, position: "right", formatter: "{b}", fontSize: 11 },
        emphasis: { focus: "adjacency" },
      }],
    });

    // Relation stats
    const stats = {};
    links.forEach(l => { const c = nodes[l.target].category; stats[c] = (stats[c] || 0) + 1; });
    const relStats = document.getElementById("rel-stats");
    if (relStats) {
      relStats.innerHTML = relNames.map((n, i) => `
        <div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px dashed #e2e8f0;">
          <span><span style="display:inline-block;width:10px;height:10px;background:${relColors[i]};border-radius:2px;margin-right:6px;"></span>${n}</span>
          <b>${stats[i] || 0} 邻居</b>
        </div>`).join("");
    }

    chart.on("click", (p) => {
      if (p.dataType !== "node") return;
      const n = p.data;
      const info = document.getElementById("node-info");
      // 聚合该节点的邻居风险
      const neighbors = links.filter(l => l.source === n.id || l.target === n.id).map(l => nodes[l.source === n.id ? l.target : l.source]);
      const avgRisk = neighbors.length ? (neighbors.reduce((s, x) => s + x.risk, 0) / neighbors.length) : 0;
      const maxRisk = neighbors.length ? Math.max(...neighbors.map(x => x.risk)) : 0;
      info.innerHTML = `
        <div><b>${n.name}</b> · 自身风险分 <span style="color:#ef4444;font-weight:700;">${n.risk.toFixed(2)}</span></div>
        <div style="margin-top:6px;">关系类型：<span class="tag" style="background:${relColors[n.category]}22;color:${relColors[n.category]};">${relNames[n.category] || "中心"}</span></div>
        <div style="margin-top:6px;">2-hop 邻居数：<b>${neighbors.length}</b></div>
        <div style="margin-top:6px;">邻居平均风险：<b style="color:#f59e0b;">${avgRisk.toFixed(3)}</b> · 最大风险：<b style="color:#ef4444;">${maxRisk.toFixed(3)}</b></div>
        <div style="margin-top:10px;color:#4f46e5;">聚合后输出 = W_self·x + mean(邻居 risk 信号)</div>
      `;
    });
  },
});
