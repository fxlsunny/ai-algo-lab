/* 模块：PageRank */
MCH.register("graph_pagerank", {
  render() {
    const code = `# PageRank 幂迭代实现（Brin & Page, 1998）
def pagerank(G, damping=0.85, max_iter=100, tol=1e-6):
    """G: 邻接矩阵 (N, N)；A[i,j]=1 表示 i → j
       返回每个节点的 PageRank 值。"""
    N = G.shape[0]
    # 1) 构建转移矩阵 M：每列做 L1 归一化（按出度）
    out_degree = G.sum(axis=0).astype(float)
    out_degree[out_degree == 0] = 1.0          # dangling node 处理
    M = G / out_degree                          # (N, N)

    # 2) Teleport 矩阵（修复不连通 & dangling 问题）
    #    r_{t+1} = d · M · r_t + (1 - d) / N · 1
    r = np.ones(N) / N                          # 均匀初始化
    for it in range(max_iter):
        r_new = damping * (M @ r) + (1 - damping) / N
        if np.abs(r_new - r).sum() < tol:
            return r_new, it + 1
        r = r_new
    return r, max_iter

# 物理含义：在图上做随机游走
#   - 有 d 的概率按出边跳转
#   - 有 (1-d) 的概率"传送"到任意节点（防止陷入死循环）
# 稳态分布 r 就是 PageRank`;

    return `
      ${MCH.hero({ icon: "⊛", name: "PageRank 图重要性算法", en: "PageRank · Random Walk Stationary Distribution", tags: ["幂迭代", "随机游走", "damping=0.85", "Google 发家算法"], meta: ["◈ 无监督", "⚡ O(E·iter)"] })}

      ${MCH.versionSection("graph_pagerank")}

      <div class="section">
        <h2>1. 核心思想</h2>
        <p class="text-sm text-slate-600">PageRank 的直觉：<b>"重要的节点被很多重要的节点指向"</b>。形式化为马尔可夫链稳态分布：</p>
        <div class="formula-block">
          $$ r_i = \\frac{1 - d}{N} + d \\cdot \\sum_{j \\in B(i)} \\frac{r_j}{L(j)} $$
          其中 $B(i)$ 是指向 i 的节点集合，$L(j)$ 是 j 的出度，$d \\approx 0.85$ 是阻尼系数（有 15% 概率"传送"到任意节点）。
        </div>
        ${MCH.info(`
          <b>为什么要阻尼？</b>
          <ul style="margin-top:6px;padding-left:20px;">
            <li>真实网页图有<b>环</b>和<b>死胡同</b>（dangling），纯随机游走不收敛；</li>
            <li>1-d 的"传送"概率相当于给每个节点加一个"最小存在值"；</li>
            <li>数学上让迭代算子变成<b>严格压缩映射</b>，保证收敛唯一。</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 代码解读</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：幂迭代实时演示</h2>
        <p class="text-sm text-slate-600">下面是一个 8 节点有向图，观察 PageRank 在每轮迭代后的演变过程。</p>

        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "pr-d", label: "阻尼系数 d", min: 0, max: 1, step: 0.01, value: 0.85 })}
              ${MCH.slider({ id: "pr-iter", label: "迭代轮数 t", min: 0, max: 30, step: 1, value: 0 })}
              <div class="flex gap-2 mt-3">
                <button id="pr-step" class="text-xs px-3 py-1 bg-indigo-600 text-white rounded">▶ 单步</button>
                <button id="pr-play" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶▶ 自动播放</button>
                <button id="pr-reset" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">↺ 重置</button>
              </div>
            </div>
            <div id="pr-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-graph" style="height:400px;"></div>
        </div>

        <h3 style="margin-top:18px;">· 每个节点 PageRank 随迭代变化</h3>
        <div id="chart-trace" style="height:320px;"></div>
      </div>

      <div class="section">
        <h2>4. 扩展：Personalized PageRank</h2>
        <p class="text-sm text-slate-600">把 teleport 的均匀分布改成"偏向某个目标节点"的分布 <code>p</code>，可得到 <b>Personalized PageRank</b>：</p>
        <div class="formula-block">
          $$ r = (1 - d) \\cdot p + d \\cdot M \\cdot r $$
        </div>
        <p class="text-sm text-slate-600">应用：社交推荐（"与你最相关的朋友的朋友"）、商户关系图的目标节点风险传播。</p>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("graph_pagerank").pros,
          MCH.getById("graph_pagerank").cons,
          MCH.getById("graph_pagerank").best_for,
        )}
      </div>
    `;
  },

  mount() {
    // 构造 8 节点有向图
    const N = 8;
    const edges = [
      [0, 1], [0, 2], [1, 2], [2, 0], [2, 3],
      [3, 4], [4, 3], [4, 5], [5, 6], [6, 5],
      [6, 7], [7, 2], [1, 3], [3, 2], [0, 4],
    ];
    // 布局：圆形
    const pos = [];
    for (let i = 0; i < N; i++) {
      const a = (i / N) * 2 * Math.PI;
      pos.push([Math.cos(a), Math.sin(a)]);
    }
    // 出度
    const outDeg = new Array(N).fill(0);
    edges.forEach(([u, v]) => outDeg[u]++);

    // PR 历史
    let prHist = [];
    const runPR = (d, maxIter) => {
      let r = new Array(N).fill(1 / N);
      prHist = [[...r]];
      for (let t = 0; t < maxIter; t++) {
        const rNew = new Array(N).fill((1 - d) / N);
        edges.forEach(([u, v]) => { rNew[v] += d * r[u] / outDeg[u]; });
        // normalize
        const s = rNew.reduce((a, b) => a + b, 0);
        for (let i = 0; i < N; i++) rNew[i] = rNew[i] / s;
        prHist.push([...rNew]);
        r = rNew;
      }
    };

    const graphEl = document.getElementById("chart-graph");
    const traceEl = document.getElementById("chart-trace");
    const infoEl = document.getElementById("pr-info");

    const renderGraph = (t) => {
      const r = prHist[Math.min(t, prHist.length - 1)];
      const maxR = Math.max(...r);
      const nodes = r.map((val, i) => ({
        name: `N${i}`,
        value: [pos[i][0] * 200, pos[i][1] * 200],
        symbol: "circle",
        symbolSize: 20 + val * 400 / maxR,
        label: { show: true, formatter: `N${i}\n${val.toFixed(3)}`, color: "#fff", fontWeight: 700, fontSize: 11 },
        itemStyle: { color: `hsl(${270 - val / maxR * 60}, 70%, ${65 - val / maxR * 25}%)` },
      }));
      const links = edges.map(([u, v]) => ({ source: `N${u}`, target: `N${v}` }));
      MCH.echart(graphEl, {
        tooltip: { formatter: p => p.dataType === "node" ? `${p.data.name} · PR=${r[parseInt(p.data.name.slice(1))].toFixed(4)}` : "" },
        xAxis: { show: false, min: -300, max: 300 },
        yAxis: { show: false, min: -300, max: 300 },
        grid: { left: 20, right: 20, top: 20, bottom: 20 },
        series: [{
          type: "graph", coordinateSystem: "cartesian2d",
          data: nodes, links,
          lineStyle: { color: "#94a3b8", width: 1 },
          edgeSymbol: ["none", "arrow"],
          edgeSymbolSize: 8,
        }],
      });
    };

    const renderTrace = (t) => {
      const xs = [...Array(prHist.length).keys()];
      const series = [];
      for (let i = 0; i < N; i++) {
        series.push({
          name: `N${i}`, type: "line", showSymbol: false, smooth: true,
          data: prHist.map((r, idx) => [idx, r[i]]),
          lineStyle: { width: 2 },
          color: `hsl(${i * 45}, 70%, 55%)`,
        });
      }
      MCH.echart(traceEl, {
        tooltip: { trigger: "axis" },
        legend: { top: 0 },
        grid: { left: 60, right: 30, top: 40, bottom: 40 },
        xAxis: { type: "value", name: "迭代轮", min: 0, max: Math.max(xs[xs.length - 1], 1) },
        yAxis: { type: "value", name: "PageRank", min: 0 },
        series,
      });
    };

    const updateInfo = (t) => {
      const r = prHist[Math.min(t, prHist.length - 1)];
      const sorted = r.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v);
      const conv = t > 0 ? r.reduce((s, v, i) => s + Math.abs(v - prHist[t - 1][i]), 0) : 1;
      infoEl.innerHTML = `
        <b>迭代 t = ${t}</b> · L1 变化 = <span style="color:${conv < 1e-4 ? "#10b981" : "#f59e0b"};">${conv.toExponential(2)}</span> ${conv < 1e-4 ? "(已收敛)" : ""}<br/>
        <b>排名前 3：</b>
        ${sorted.slice(0, 3).map((x, idx) => `<span class="tag" style="background:hsl(${x.i * 45}, 70%, 90%);color:hsl(${x.i * 45}, 70%, 30%);">#${idx + 1} N${x.i} = ${x.v.toFixed(4)}</span>`).join(" ")}
      `;
    };

    let playTimer = null;
    const doUpdate = () => {
      const d = parseFloat(document.getElementById("pr-d").value);
      const t = parseInt(document.getElementById("pr-iter").value);
      runPR(d, 30);
      renderGraph(t);
      renderTrace(t);
      updateInfo(t);
    };

    document.getElementById("pr-d").addEventListener("input", (e) => {
      document.getElementById("pr-d-val").textContent = e.target.value;
      doUpdate();
    });
    document.getElementById("pr-iter").addEventListener("input", (e) => {
      document.getElementById("pr-iter-val").textContent = e.target.value;
      doUpdate();
    });
    document.getElementById("pr-step").addEventListener("click", () => {
      const el = document.getElementById("pr-iter");
      el.value = Math.min(30, parseInt(el.value) + 1);
      document.getElementById("pr-iter-val").textContent = el.value;
      doUpdate();
    });
    document.getElementById("pr-reset").addEventListener("click", () => {
      document.getElementById("pr-iter").value = 0;
      document.getElementById("pr-iter-val").textContent = "0";
      if (playTimer) { clearInterval(playTimer); playTimer = null; }
      doUpdate();
    });
    document.getElementById("pr-play").addEventListener("click", () => {
      if (playTimer) { clearInterval(playTimer); playTimer = null; return; }
      playTimer = setInterval(() => {
        const el = document.getElementById("pr-iter");
        const t = parseInt(el.value);
        if (t >= 30) { clearInterval(playTimer); playTimer = null; return; }
        el.value = t + 1;
        document.getElementById("pr-iter-val").textContent = el.value;
        doUpdate();
      }, 400);
    });
    doUpdate();
  },
});
