/* 模块：K-Means 聚类 */
MCH.register("ml_kmeans", {
  render() {
    const code = `# K-Means 算法（Lloyd's Algorithm, 1957）
# 目标：最小化簇内平方和 WCSS (Within-Cluster Sum of Squares)

def kmeans(X, K, max_iter=100, init='kmeans++'):
    # ---- 1) 初始化 K 个中心点 ----
    if init == 'random':
        centers = X[np.random.choice(len(X), K, replace=False)]
    elif init == 'kmeans++':
        centers = kmeans_plus_plus(X, K)       # 概率加权初始化
    
    for _ in range(max_iter):
        # ---- 2) Assignment step：每个样本分到最近的中心 ----
        dists = euclidean(X, centers)            # (n, K)
        labels = dists.argmin(axis=1)

        # ---- 3) Update step：重新计算每簇的均值作为新中心 ----
        new_centers = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        # ---- 4) 收敛检查 ----
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers
    
    return labels, centers


# ============================================================
# K-Means++ 初始化（Arthur & Vassilvitskii, 2007）
# ============================================================
# 解决 K-Means 对初始化敏感的问题
def kmeans_plus_plus(X, K):
    centers = [X[np.random.randint(len(X))]]      # 1) 随机选第一个中心
    for _ in range(K - 1):
        # 2) 计算每个点到已有中心的最近距离 D(x)
        D2 = np.min([np.sum((X - c)**2, axis=1) for c in centers], axis=0)
        # 3) 概率正比于 D²(x) 选下一个中心 → 倾向于远处的点
        probs = D2 / D2.sum()
        next_center = X[np.random.choice(len(X), p=probs)]
        centers.append(next_center)
    return np.array(centers)


# 大数据替代：MiniBatch KMeans（每次只用一小批更新）
from sklearn.cluster import MiniBatchKMeans
mb = MiniBatchKMeans(n_clusters=20, batch_size=1024, max_iter=100)
mb.fit(X)    # 百万样本 2 分钟搞定`;

    return `
      ${MCH.hero({
        icon: "◉",
        name: "K-Means 聚类",
        en: "K-Means (Lloyd 1957) + K-Means++ (2007)",
        tags: ["无监督", "中心点", "EM 思想", "K-Means++"],
        meta: ["◈ 最常用聚类算法", "⚡ MiniBatch 版可扩展到亿级"],
      })}

      ${MCH.versionSection("ml_kmeans")}

      <div class="section">
        <h2>1. 核心思想：交替最小化 WCSS</h2>
        <div class="formula-block">
          <b>目标函数（簇内平方和）</b>：
          $$ \\min_{\\{C_k, \\mu_k\\}} \\sum_{k=1}^{K} \\sum_{x \\in C_k} \\|x - \\mu_k\\|^2 $$
          <b>EM 式两步迭代</b>：
          <ol style="padding-left:20px;margin-top:6px;">
            <li><b>E-step</b>：每个点分配到最近中心 $c(x) = \\arg\\min_k \\|x - \\mu_k\\|^2$</li>
            <li><b>M-step</b>：每簇中心更新为其成员均值 $\\mu_k = \\text{mean}(C_k)$</li>
          </ol>
          保证单调下降 → 局部最优收敛。
        </div>
      </div>

      <div class="section">
        <h2>2. K-Means++ 初始化的意义</h2>
        <p class="text-sm text-slate-600">随机初始化 K 个中心在数据紧密 cluster 上很糟糕 — 常陷入差的局部最优。<b>K-Means++</b> 用"远者优先"的概率采样：</p>
        <div class="mermaid">
flowchart LR
    S[随机选一个点作第一中心] --> L[对每个点 x 计算到最近中心的距离 D x]
    L --> P[按 D² 正比的概率采样下一中心]
    P --> Check{选够 K 个?}
    Check -->|否| L
    Check -->|是| Done[开始标准 K-Means 迭代]
        </div>
        <p class="text-sm text-slate-600 mt-3">理论上 K-Means++ 保证<b>期望 8×ln(K) 逼近</b>最优 WCSS，而随机初始化没有保证。</p>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：K-Means 实时迭代</h2>
        <p class="text-sm text-slate-600">点击"下一步"观察 K-Means 每轮的分配 + 更新过程。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "km-K", label: "K (簇数)", min: 2, max: 8, step: 1, value: 4 })}
              <label class="text-xs mt-2 block">初始化：
                <select id="km-init" class="text-xs border rounded p-1 ml-2">
                  <option value="random">随机</option>
                  <option value="kmeans++" selected>K-Means++</option>
                  <option value="bad">刻意差（集中一角）</option>
                </select>
              </label>
              <div class="flex gap-2 mt-3">
                <button id="km-step" class="text-xs px-3 py-1 bg-indigo-600 text-white rounded">▶ 下一步</button>
                <button id="km-play" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶▶ 播放</button>
                <button id="km-reset" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">↺ 重置</button>
              </div>
            </div>
            <div id="km-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-km" style="height:400px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 如何选择 K？— Elbow 法 vs Silhouette</h2>
        <div class="grid-2">
          <div>
            <h3>· Elbow Method（肘部法）</h3>
            <p class="text-xs text-slate-600">画 K vs WCSS 曲线，拐点（Elbow）处 K 是一个好选择。简单直观但主观。</p>
          </div>
          <div>
            <h3>· Silhouette Score</h3>
            <p class="text-xs text-slate-600">每个点的 Silhouette s(i) = (b-a)/max(a,b)，a=簇内距离，b=最近外簇距离。取均值最大的 K。</p>
          </div>
        </div>
        <div id="chart-k-select" style="height:300px;margin-top:12px;"></div>
      </div>

      <div class="section">
        <h2>6. 何时 K-Means 会失败？</h2>
        ${MCH.info(`
          K-Means 默认假设簇是<b>球形、大小相近、密度均匀</b>。不符时会失败：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>非球形簇</b>（月牙/环形）→ 用 <b>DBSCAN</b>；</li>
            <li><b>簇大小差距大</b> → 用 GMM（软聚类）或 <b>HDBSCAN</b>；</li>
            <li><b>密度差异大</b> → 用 DBSCAN / Spectral Clustering；</li>
            <li><b>未知 K</b> → 用 DBSCAN / HDBSCAN / Affinity Propagation。</li>
          </ul>
        `, "warn")}
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ml_kmeans").pros, MCH.getById("ml_kmeans").cons, MCH.getById("ml_kmeans").best_for)}
      </div>
    `;
  },

  mount() {
    // 生成 4 cluster 数据
    const trueCenters = [[2, 2], [-2, 2], [-2, -2], [2, -2], [0, 0], [3, 0], [-3, 0], [0, 3]];
    const data = [];
    for (let c = 0; c < 8; c++) {
      for (let i = 0; i < 20; i++) {
        const r = MCH.randn(2, c * 100 + i);
        data.push([trueCenters[c][0] + r[0] * 0.5, trueCenters[c][1] + r[1] * 0.5]);
      }
    }
    let centers = [];
    let labels = new Array(data.length).fill(0);
    let iter = 0;

    const initCenters = (K, mode) => {
      if (mode === "random") {
        const idx = [];
        while (idx.length < K) {
          const j = Math.floor(Math.random() * data.length);
          if (!idx.includes(j)) idx.push(j);
        }
        return idx.map(j => [...data[j]]);
      } else if (mode === "kmeans++") {
        const cs = [[...data[Math.floor(Math.random() * data.length)]]];
        while (cs.length < K) {
          const D2 = data.map(x => Math.min(...cs.map(c => (x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2)));
          const total = D2.reduce((a, b) => a + b, 0);
          let r = Math.random() * total;
          let picked = 0;
          for (let i = 0; i < D2.length; i++) { r -= D2[i]; if (r <= 0) { picked = i; break; } }
          cs.push([...data[picked]]);
        }
        return cs;
      } else {
        // bad：集中一角
        return [...Array(K).keys()].map(i => [3.5 + i * 0.1, 3.5 + i * 0.05]);
      }
    };

    const stepKMeans = () => {
      // Assignment
      labels = data.map(x => {
        let best = 0, bd = Infinity;
        centers.forEach((c, i) => { const d = (x[0] - c[0]) ** 2 + (x[1] - c[1]) ** 2; if (d < bd) { bd = d; best = i; } });
        return best;
      });
      // Update
      const newCenters = centers.map((_, k) => {
        const members = data.filter((_, i) => labels[i] === k);
        if (members.length === 0) return centers[k];
        const sum = members.reduce((a, p) => [a[0] + p[0], a[1] + p[1]], [0, 0]);
        return [sum[0] / members.length, sum[1] / members.length];
      });
      const changed = centers.some((c, i) => Math.hypot(c[0] - newCenters[i][0], c[1] - newCenters[i][1]) > 0.001);
      centers = newCenters;
      iter++;
      return changed;
    };

    const colors = ["#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"];

    const render = () => {
      const K = centers.length;
      const series = [];
      for (let k = 0; k < K; k++) {
        series.push({
          name: `C${k + 1}`, type: "scatter",
          data: data.filter((_, i) => labels[i] === k),
          itemStyle: { color: colors[k] }, symbolSize: 8,
        });
      }
      series.push({
        name: "簇中心", type: "scatter",
        data: centers,
        itemStyle: { color: "#000", borderColor: "#fff", borderWidth: 2 },
        symbol: "diamond", symbolSize: 18,
      });
      MCH.echart(document.getElementById("chart-km"), {
        tooltip: {},
        legend: { top: 0, show: false },
        grid: { left: 50, right: 30, top: 40, bottom: 40 },
        xAxis: { type: "value", min: -5, max: 5 },
        yAxis: { type: "value", min: -5, max: 5 },
        series,
      });
      const wcss = data.reduce((s, x, i) => s + ((x[0] - centers[labels[i]][0]) ** 2 + (x[1] - centers[labels[i]][1]) ** 2), 0);
      document.getElementById("km-info").innerHTML = `
        <b>迭代</b>: ${iter} · <b>K</b>: ${K}<br/>
        <b>WCSS（簇内平方和）</b>: <span style="color:#4f46e5;font-weight:700;">${wcss.toFixed(2)}</span><br/>
        <span style="color:#64748b;">WCSS 每轮递减；收敛后停止。</span>
      `;
    };

    const reset = () => {
      const K = parseInt(document.getElementById("km-K").value);
      const mode = document.getElementById("km-init").value;
      centers = initCenters(K, mode);
      labels = new Array(data.length).fill(0);
      iter = 0;
      render();
    };

    let timer = null;
    document.getElementById("km-step").addEventListener("click", () => { stepKMeans(); render(); });
    document.getElementById("km-play").addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; return; }
      timer = setInterval(() => {
        const changed = stepKMeans();
        render();
        if (!changed || iter > 30) { clearInterval(timer); timer = null; }
      }, 500);
    });
    document.getElementById("km-reset").addEventListener("click", reset);
    document.getElementById("km-K").addEventListener("input", (e) => {
      document.getElementById("km-K-val").textContent = e.target.value;
      reset();
    });
    document.getElementById("km-init").addEventListener("change", reset);
    reset();

    // Elbow + Silhouette
    const ks = [2, 3, 4, 5, 6, 7, 8, 9, 10];
    const wcss = ks.map(k => {
      // 模拟：Zipf 式衰减，拐点在真实 K=8 附近
      return 800 * Math.exp(-k / 3) + 30;
    });
    const silhouette = ks.map(k => {
      return 0.3 + 0.3 * Math.exp(-Math.pow((k - 4) / 2, 2));
    });
    MCH.echart(document.getElementById("chart-k-select"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 50, top: 40, bottom: 40 },
      xAxis: { type: "category", name: "K", data: ks },
      yAxis: [
        { type: "value", name: "WCSS", position: "left" },
        { type: "value", name: "Silhouette", position: "right", max: 1 },
      ],
      series: [
        { name: "WCSS (Elbow)", type: "line", smooth: true, yAxisIndex: 0, color: "#4f46e5", lineStyle: { width: 3 }, data: wcss, markPoint: { data: [{ name: "Elbow", coord: [2, wcss[2]], symbolSize: 14, itemStyle: { color: "#ef4444" }, label: { formatter: "Elbow K=4" } }] } },
        { name: "Silhouette Score", type: "line", smooth: true, yAxisIndex: 1, color: "#10b981", lineStyle: { width: 3 }, data: silhouette.map(v => +v.toFixed(3)) },
      ],
    });
  },
});
