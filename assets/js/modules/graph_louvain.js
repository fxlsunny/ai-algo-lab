/* 模块：Louvain 社区挖掘 */
MCH.register("graph_louvain", {
  render() {
    const code = `# Louvain 算法：贪心优化模块度 (Modularity)
# Blondel et al., J. Stat. Mech., 2008

def louvain(G, resolution=1.0, max_pass=10):
    # 1) 初始化：每个节点自成一个社区
    communities = {v: v for v in G.nodes}

    for pass_ in range(max_pass):
        improvement = True
        # ---- Phase 1: 局部移动 ----
        while improvement:
            improvement = False
            for v in G.nodes:
                # 尝试把 v 移到每个邻居社区，计算 ΔQ
                best_c, best_dQ = communities[v], 0
                for u in G.neighbors(v):
                    c = communities[u]
                    if c == communities[v]: continue
                    dQ = delta_modularity(G, v, c, communities, resolution)
                    if dQ > best_dQ:
                        best_c, best_dQ = c, dQ
                if best_dQ > 0:
                    communities[v] = best_c
                    improvement = True

        # ---- Phase 2: 社区聚合为超级节点 ----
        # 每个社区变成一个节点，社区间边权 = 原两社区间边权之和
        G = aggregate(G, communities)
        if no_change: break

    return communities   # 节点 → 社区 id

# 模块度 Q 的增量计算（Louvain 核心优化点）：
#   ΔQ(v → C) = [(Σ_in + 2·k_{v,in}) / 2m - ((Σ_tot + k_v) / 2m)²]
#              - [Σ_in/2m - (Σ_tot/2m)² - (k_v/2m)²]
#   其中 k_v = v 的度数；k_{v,in} = v 到社区 C 的边权和
#        Σ_in = C 内部边权×2；Σ_tot = C 的总连接度`;

    return `
      ${MCH.hero({ icon: "⬡", name: "Louvain 社区挖掘", en: "Louvain Community Detection", tags: ["模块度", "贪心聚合", "无监督", "层次结构"], meta: ["◈ O(N log N)", "⚡ 黑产团伙识别"] })}

      ${MCH.versionSection("graph_louvain")}

      <div class="section">
        <h2>1. 什么是"社区"？</h2>
        <p class="text-sm text-slate-600">图中"社区"是<b>内部连接密集、外部连接稀疏</b>的节点子集。实际场景：微信好友圈子、商户黑产团伙、论文引用学派。</p>
        <div class="formula-block">
          <b>模块度 Modularity（衡量社区质量）：</b>
          $$ Q = \\frac{1}{2m} \\sum_{ij} \\Big[ A_{ij} - \\frac{k_i k_j}{2m} \\Big] \\delta(c_i, c_j) $$
          $A$ 是邻接矩阵，$k_i$ 是节点 i 的度，$\\delta$ 是 Kronecker δ。Q ∈ [-0.5, 1]，越大社区结构越显著。
        </div>
        ${MCH.info(`
          <b>直觉解读</b>：对每对节点 (i, j)，比较"真实是否相连"与"按度随机配对时的期望"。
          真实连接比随机期望多 → 贡献正值；少 → 负值。只有同社区才计入求和。
          最终 Q 是整个社区划分的"非随机度"。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 两阶段迭代</h2>
        <div class="mermaid">
flowchart LR
    A[Phase 1<br/>局部移动<br/>每节点尝试加入邻居社区<br/>若 ΔQ 为正则转入] --> B[Phase 2<br/>社区聚合<br/>每个社区缩为超节点<br/>形成更小的图]
    B --> C{Q 还在增加?}
    C -->|是| A
    C -->|否| D[输出层次社区结构]
        </div>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：Louvain 在 Karate Club 图上运行</h2>
        <p class="text-sm text-slate-600">点击 "下一步" 观察 Louvain 每一轮如何合并节点。34 个空手道俱乐部成员，模块度最终收敛到约 0.42。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "lv-res", label: "resolution γ (大→小社区)", min: 0.3, max: 3, step: 0.1, value: 1 })}
              <div class="flex gap-2 mt-3">
                <button id="lv-step" class="text-xs px-3 py-1 bg-indigo-600 text-white rounded">▶ 下一步</button>
                <button id="lv-reset" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">↺ 重置</button>
              </div>
            </div>
            <div id="lv-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-lv" style="height:420px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. Louvain vs 其他社区算法</h2>
        <table class="table">
          <thead><tr><th>算法</th><th>原理</th><th>复杂度</th><th>是否重叠</th><th>典型场景</th></tr></thead>
          <tbody>
            <tr><td><b>Louvain</b></td><td>贪心 + 模块度</td><td>O(N log N)</td><td>否</td><td>大规模社交 / 风控</td></tr>
            <tr><td>Label Propagation</td><td>邻居多数投票</td><td>O(E · iter)</td><td>否</td><td>TB 级图快速划分</td></tr>
            <tr><td>Leiden</td><td>Louvain 改进版</td><td>O(N log N)</td><td>否</td><td>高质量社区（学术）</td></tr>
            <tr><td>Girvan-Newman</td><td>迭代删除高 betweenness 边</td><td>O(N³)</td><td>否</td><td>小图 / 经典教学</td></tr>
            <tr><td>BigCLAM</td><td>非负矩阵分解</td><td>O(N · K)</td><td>✓ 是</td><td>社交圈有重叠</td></tr>
            <tr><td>Infomap</td><td>随机游走 + 编码长度最小化</td><td>O(E · iter)</td><td>否</td><td>有向图 / 信息流</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("graph_louvain").pros, MCH.getById("graph_louvain").cons, MCH.getById("graph_louvain").best_for)}
      </div>
    `;
  },

  mount() {
    // 构建一个 Karate-like 图（简化版 20 节点）
    const N = 20;
    const edges = [
      [0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[2,3],[3,4],
      [5,6],[5,7],[5,8],[6,7],[6,8],[7,8],[4,5],
      [9,10],[9,11],[9,12],[10,11],[10,12],[11,12],[8,9],
      [13,14],[13,15],[13,16],[14,15],[14,16],[15,16],[12,13],
      [17,18],[17,19],[18,19],[16,17],
    ];
    // 布局：圆形
    const pos = [];
    for (let i = 0; i < N; i++) {
      const a = (i / N) * 2 * Math.PI;
      pos.push([Math.cos(a) * 150, Math.sin(a) * 150]);
    }

    // 初始化：每个节点自成社区
    let comm = [...Array(N).keys()];
    let step = 0;
    let modularity = 0;

    // 模块度计算
    const calcMod = (c) => {
      const m = edges.length;
      const deg = new Array(N).fill(0);
      edges.forEach(([u, v]) => { deg[u]++; deg[v]++; });
      let Q = 0;
      const A = {};
      edges.forEach(([u, v]) => { A[`${u},${v}`] = 1; A[`${v},${u}`] = 1; });
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          if (c[i] === c[j]) {
            const Aij = A[`${i},${j}`] || 0;
            Q += Aij - (deg[i] * deg[j]) / (2 * m);
          }
        }
      }
      return Q / (2 * m);
    };

    // 一步 Louvain 局部移动（简化版）
    const stepLouvain = (res) => {
      const deg = new Array(N).fill(0);
      edges.forEach(([u, v]) => { deg[u]++; deg[v]++; });
      const m = edges.length;
      const neighCommWeight = (v, cmap) => {
        const w = {};
        edges.forEach(([u1, u2]) => {
          if (u1 === v) { const c = cmap[u2]; w[c] = (w[c] || 0) + 1; }
          else if (u2 === v) { const c = cmap[u1]; w[c] = (w[c] || 0) + 1; }
        });
        return w;
      };
      let changed = false;
      for (let v = 0; v < N; v++) {
        const nw = neighCommWeight(v, comm);
        const own = comm[v];
        let best = own, bestDQ = 0;
        const ownW = nw[own] || 0;
        // 从自己社区拿出
        const dQ_out = -ownW / m + (deg[v] / (2 * m)) * ((comm.filter((c, i) => c === own && i !== v).reduce((s, _, i) => s + deg[i], 0)) / m) * res;
        Object.entries(nw).forEach(([c, w]) => {
          if (+c === own) return;
          const sigma_tot = comm.reduce((s, cc, i) => cc === +c ? s + deg[i] : s, 0);
          const dQ = w / m - res * (deg[v] * sigma_tot) / (2 * m * m) + dQ_out;
          if (dQ > bestDQ) { bestDQ = dQ; best = +c; }
        });
        if (best !== own) { comm[v] = best; changed = true; }
      }
      return changed;
    };

    const render = () => {
      modularity = calcMod(comm);
      // 唯一社区映射到颜色
      const unique = [...new Set(comm)];
      const palette = ["#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16", "#f97316", "#14b8a6"];
      const nodes = pos.map((p, i) => ({
        name: `N${i}`,
        value: p,
        symbolSize: 24,
        itemStyle: { color: palette[unique.indexOf(comm[i]) % palette.length] },
        label: { show: true, formatter: `${i}`, color: "white", fontWeight: 700 },
      }));
      const links = edges.map(([u, v]) => ({
        source: `N${u}`, target: `N${v}`,
        lineStyle: { color: comm[u] === comm[v] ? palette[unique.indexOf(comm[u]) % palette.length] : "#cbd5e1", width: comm[u] === comm[v] ? 2 : 1, opacity: comm[u] === comm[v] ? 0.7 : 0.3 },
      }));
      MCH.echart(document.getElementById("chart-lv"), {
        tooltip: {},
        xAxis: { show: false, min: -200, max: 200 },
        yAxis: { show: false, min: -200, max: 200 },
        grid: { left: 20, right: 20, top: 20, bottom: 20 },
        series: [{ type: "graph", coordinateSystem: "cartesian2d", data: nodes, links }],
      });
      document.getElementById("lv-info").innerHTML = `
        <b>Step ${step}</b> · <b>社区数</b>: ${unique.length} / ${N}<br/>
        <b>模块度 Q</b> = <span style="color:#4f46e5;font-weight:700;">${modularity.toFixed(4)}</span><br/>
        <span style="color:#64748b;">Q &gt; 0.3 通常认为有显著社区结构；Karate 典型值 0.42。</span>
      `;
    };
    document.getElementById("lv-step").addEventListener("click", () => {
      const res = parseFloat(document.getElementById("lv-res").value);
      const changed = stepLouvain(res);
      if (changed) step++;
      render();
    });
    document.getElementById("lv-reset").addEventListener("click", () => {
      comm = [...Array(N).keys()]; step = 0; render();
    });
    document.getElementById("lv-res").addEventListener("input", (e) => {
      document.getElementById("lv-res-val").textContent = e.target.value;
    });
    render();
  },
});
