/* 模块：DBSCAN / HDBSCAN */
MCH.register("ml_dbscan", {
  render() {
    const code = `# DBSCAN — 基于密度的聚类
# Ester, Kriegel, Sander, Xu, KDD 1996

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)   # -1 表示噪声

# ============================================================
# 核心概念
# ============================================================
# - 核心点 (Core point)：在 ε-邻域内至少有 min_samples 个点
# - 边界点 (Border point)：非核心点但在某核心点的 ε-邻域内
# - 噪声点 (Noise)：既不是核心点也不是边界点 → label = -1

# 算法流程：
def dbscan(X, eps, min_samples):
    labels = [UNVISITED] * len(X)
    C = 0
    for i in range(len(X)):
        if labels[i] != UNVISITED: continue
        neighbors = region_query(X, i, eps)
        if len(neighbors) < min_samples:
            labels[i] = NOISE
            continue
        C += 1
        # 扩展簇：BFS 所有密度可达点
        expand_cluster(X, labels, i, neighbors, C, eps, min_samples)
    return labels


# ============================================================
# HDBSCAN — 层次化 DBSCAN (Campello et al., 2013)
# ============================================================
# 解决 DBSCAN 对 eps 敏感的问题
# 不要一个固定 eps，而是构建层次结构选最稳定簇
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=5)
labels = clusterer.fit_predict(X)

# HDBSCAN 优势：
# 1) 只需要 min_cluster_size（远比 eps 直观）
# 2) 自动处理不同密度的簇
# 3) 输出每个点的概率（soft clustering）
# 4) 在高维 UMAP 投影后效果极佳`;

    return `
      ${MCH.hero({
        icon: "◎",
        name: "DBSCAN / HDBSCAN 密度聚类",
        en: "Density-Based Spatial Clustering of Applications with Noise",
        tags: ["密度可达", "自动 K", "噪声识别", "任意形状"],
        meta: ["◈ DBSCAN (1996) → HDBSCAN (2013)", "⚡ UMAP+HDBSCAN 现代组合拳"],
      })}

      ${MCH.versionSection("ml_dbscan")}

      <div class="section">
        <h2>1. 核心概念：密度可达</h2>
        <div class="grid-3">
          <div class="card"><h3 class="text-sm font-bold text-indigo-700">🔵 核心点 (Core)</h3><p class="text-xs text-slate-600 mt-2">ε-邻域内至少有 <code>min_samples</code> 个点。</p></div>
          <div class="card"><h3 class="text-sm font-bold text-amber-700">🟡 边界点 (Border)</h3><p class="text-xs text-slate-600 mt-2">不是核心但在某核心的 ε-邻域内。</p></div>
          <div class="card"><h3 class="text-sm font-bold text-red-700">⚫ 噪声点 (Noise)</h3><p class="text-xs text-slate-600 mt-2">label = -1，不属于任何簇。</p></div>
        </div>

        <div class="formula-block">
          <b>密度直达</b>：q 在 p 的 ε-邻域内，且 p 是核心点 → q 从 p 直接可达<br/>
          <b>密度可达</b>：存在 p=p₁, p₂, ..., pₙ=q，pᵢ 从 p_{i-1} 直达<br/>
          <b>密度相连</b>：存在 o，p 和 q 都从 o 密度可达 → 属于同一簇
        </div>
      </div>

      <div class="section">
        <h2>2. DBSCAN vs HDBSCAN：eps 难题的解决</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>DBSCAN</th><th>HDBSCAN</th></tr></thead>
          <tbody>
            <tr><td>关键超参</td><td>eps + min_samples</td><td><b>min_cluster_size</b> (更直观)</td></tr>
            <tr><td>簇密度适应性</td><td>❌ 要求所有簇同密度</td><td>✓ 支持不同密度</td></tr>
            <tr><td>软聚类</td><td>❌</td><td>✓ 输出每点归属概率</td></tr>
            <tr><td>层次结构</td><td>❌</td><td>✓ 可用于多粒度分析</td></tr>
            <tr><td>计算复杂度</td><td>O(n log n) 空间索引</td><td>O(n log n) 近似</td></tr>
            <tr><td>对维度</td><td>&gt; 20 维效果差</td><td>稍好，但仍推荐先降维</td></tr>
            <tr><td>推荐场景</td><td>2D/3D 空间聚类</td><td>通用首选</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：DBSCAN 超参调节</h2>
        <p class="text-sm text-slate-600">下面数据是两个月牙形 + 一个球形簇（非凸形状，K-Means 会失败）。调节 eps 和 min_samples 观察 DBSCAN 效果。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "db-eps", label: "eps (邻域半径)", min: 0.05, max: 1.5, step: 0.05, value: 0.3 })}
              ${MCH.slider({ id: "db-min", label: "min_samples", min: 2, max: 20, step: 1, value: 5 })}
            </div>
            <div id="db-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-db" style="height:400px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. K-Means vs DBSCAN — 何时用哪个？</h2>
        <div id="chart-compare" style="height:320px;"></div>
        <table class="table mt-3">
          <thead><tr><th>问题</th><th>K-Means</th><th>DBSCAN</th></tr></thead>
          <tbody>
            <tr><td>簇数未知</td><td>❌ 需指定 K</td><td>✓ 自动</td></tr>
            <tr><td>非球形簇（月牙 / 环）</td><td>❌ 失败</td><td>✓ 完美</td></tr>
            <tr><td>噪声识别</td><td>❌ 所有点强制归簇</td><td>✓ label=-1</td></tr>
            <tr><td>簇大小差距大</td><td>⚠ 大簇挤压小簇</td><td>✓ 只看密度</td></tr>
            <tr><td>超大数据（&gt; 100 万）</td><td>✓ MiniBatch</td><td>⚠ 内存压力</td></tr>
            <tr><td>高维（&gt; 50）</td><td>✓ 但效果下降</td><td>❌ 距离失效</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ml_dbscan").pros, MCH.getById("ml_dbscan").cons, MCH.getById("ml_dbscan").best_for)}
      </div>
    `;
  },

  mount() {
    // 生成两个月牙形 + 中央球形
    const data = [];
    for (let i = 0; i < 60; i++) {
      const a = Math.PI * i / 60;
      const r = MCH.randn(2, i);
      data.push([Math.cos(a) * 1.5 + r[0] * 0.1, Math.sin(a) * 1.5 + r[1] * 0.1]);
      data.push([Math.cos(a + Math.PI) * 1.5 + r[0] * 0.1 + 1, -Math.sin(a + Math.PI) * 1.5 + r[1] * 0.1 + 0.5]);
    }
    // 中央噪声点
    for (let i = 0; i < 15; i++) {
      const r = MCH.randn(2, i + 200);
      data.push([r[0] * 0.3, r[1] * 0.3]);
    }

    // 简易 DBSCAN
    const runDBSCAN = (eps, minPts) => {
      const n = data.length;
      const labels = new Array(n).fill(-2);  // -2: unvisited, -1: noise, 0+: cluster
      const regionQuery = (i) => {
        const neighbors = [];
        for (let j = 0; j < n; j++) {
          if (Math.hypot(data[i][0] - data[j][0], data[i][1] - data[j][1]) <= eps) neighbors.push(j);
        }
        return neighbors;
      };
      let C = 0;
      for (let i = 0; i < n; i++) {
        if (labels[i] !== -2) continue;
        const nb = regionQuery(i);
        if (nb.length < minPts) { labels[i] = -1; continue; }
        labels[i] = C;
        const queue = [...nb];
        while (queue.length) {
          const q = queue.shift();
          if (labels[q] === -1) labels[q] = C;
          if (labels[q] !== -2) continue;
          labels[q] = C;
          const nbQ = regionQuery(q);
          if (nbQ.length >= minPts) queue.push(...nbQ);
        }
        C++;
      }
      return { labels, nClusters: C };
    };

    const colors = ["#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4"];
    const update = () => {
      const eps = parseFloat(document.getElementById("db-eps").value);
      const minPts = parseInt(document.getElementById("db-min").value);
      const { labels, nClusters } = runDBSCAN(eps, minPts);
      const noiseCnt = labels.filter(l => l === -1).length;
      const series = [];
      for (let c = 0; c < nClusters; c++) {
        series.push({
          name: `C${c + 1}`, type: "scatter",
          data: data.filter((_, i) => labels[i] === c),
          itemStyle: { color: colors[c % colors.length] }, symbolSize: 7,
        });
      }
      series.push({
        name: "噪声", type: "scatter",
        data: data.filter((_, i) => labels[i] === -1),
        itemStyle: { color: "#94a3b8", opacity: 0.5 }, symbol: "cross", symbolSize: 8,
      });
      MCH.echart(document.getElementById("chart-db"), {
        tooltip: {},
        legend: { top: 0, show: false },
        grid: { left: 50, right: 30, top: 40, bottom: 40 },
        xAxis: { type: "value", min: -2, max: 3 },
        yAxis: { type: "value", min: -2, max: 2.5 },
        series,
      });
      document.getElementById("db-info").innerHTML = `
        <b>eps</b>=${eps}, <b>min_samples</b>=${minPts}<br/>
        <b>发现簇数</b>: <span style="color:#4f46e5;font-weight:700;">${nClusters}</span><br/>
        <b>噪声点</b>: <span style="color:#94a3b8;">${noiseCnt}</span> / ${data.length}<br/>
        <span style="color:#64748b;">理想：2 个月牙簇 + 中央 ~15 个噪声点。</span>
        ${eps < 0.15 ? '<br/><span style="color:#ef4444;">⚠ eps 过小 → 碎片化</span>' : ''}
        ${eps > 1.0 ? '<br/><span style="color:#f59e0b;">⚠ eps 过大 → 所有合并为一簇</span>' : ''}
      `;
    };
    ["db-eps", "db-min"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();

    // K-Means vs DBSCAN 对比示意图
    MCH.echart(document.getElementById("chart-compare"), {
      title: { text: "在月牙数据上：K-Means (失败) vs DBSCAN (成功)", left: "center", textStyle: { fontSize: 12 } },
      tooltip: {},
      legend: { bottom: 0 },
      grid: [
        { left: "6%", right: "55%", top: 40, bottom: 40 },
        { left: "55%", right: "6%", top: 40, bottom: 40 },
      ],
      xAxis: [
        { type: "value", gridIndex: 0, min: -2, max: 3, axisLabel: { show: false } },
        { type: "value", gridIndex: 1, min: -2, max: 3, axisLabel: { show: false } },
      ],
      yAxis: [
        { type: "value", gridIndex: 0, min: -2, max: 2.5, axisLabel: { show: false } },
        { type: "value", gridIndex: 1, min: -2, max: 2.5, axisLabel: { show: false } },
      ],
      series: [
        { name: "K-Means 簇 A", type: "scatter", xAxisIndex: 0, yAxisIndex: 0, data: data.filter(p => p[0] < 0.5), itemStyle: { color: "#ef4444" }, symbolSize: 6 },
        { name: "K-Means 簇 B", type: "scatter", xAxisIndex: 0, yAxisIndex: 0, data: data.filter(p => p[0] >= 0.5), itemStyle: { color: "#f59e0b" }, symbolSize: 6 },
        { name: "DBSCAN 月牙 1", type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: data.slice(0, 120).filter((_, i) => i % 2 === 0), itemStyle: { color: "#4f46e5" }, symbolSize: 6 },
        { name: "DBSCAN 月牙 2", type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: data.slice(0, 120).filter((_, i) => i % 2 === 1), itemStyle: { color: "#10b981" }, symbolSize: 6 },
        { name: "DBSCAN 噪声", type: "scatter", xAxisIndex: 1, yAxisIndex: 1, data: data.slice(120), itemStyle: { color: "#94a3b8" }, symbol: "cross", symbolSize: 7 },
      ],
    });
  },
});
