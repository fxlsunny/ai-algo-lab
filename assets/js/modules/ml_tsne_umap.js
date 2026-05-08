/* 模块：t-SNE / UMAP */
MCH.register("ml_tsne_umap", {
  render() {
    const code = `# t-SNE & UMAP — 非线性降维，主要用于可视化

# ------------------------------------------------------------
# t-SNE (van der Maaten & Hinton, 2008)
# ------------------------------------------------------------
# 核心：高维空间用高斯分布定义相似度，低维用 t 分布（重尾）
# 最小化两个分布的 KL 散度
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000)
X_2d = tsne.fit_transform(X)   # ⚠ 没有 transform(new_X)

# 高维相似度（softmax on距离）：
#   p_{j|i} = exp(-||x_i - x_j||² / 2σ_i²) / Σ_k exp(-||x_i - x_k||² / 2σ_i²)
#   σ_i 使熵 = log(perplexity)，每点自适应带宽
#   p_{ij} = (p_{j|i} + p_{i|j}) / 2n
# 低维相似度（重尾 t 分布，防 "crowding" 问题）：
#   q_{ij} = (1 + ||y_i - y_j||²)^(-1) / Σ_{k≠l} (1 + ||y_k - y_l||²)^(-1)
# Loss：KL(P || Q) = Σ p_{ij} log (p_{ij} / q_{ij})
# 梯度下降优化低维坐标 Y

# ------------------------------------------------------------
# UMAP (McInnes et al., 2018) — t-SNE 的现代替代
# ------------------------------------------------------------
# 基于黎曼几何 + 模糊拓扑
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
embedding = reducer.fit_transform(X)
embedding_new = reducer.transform(X_new)   # ✓ 支持新点

# 优势：
# 1) 比 t-SNE 快 5-10×
# 2) 保留更多全局结构（不仅局部）
# 3) 支持 transform(new_data)，可作为 ML pipeline 的降维步骤
# 4) 可用于分类 (supervised UMAP)

# ------------------------------------------------------------
# 现代组合：UMAP + HDBSCAN（2024 降维聚类黄金搭档）
# ------------------------------------------------------------
emb = umap.UMAP(n_neighbors=15, n_components=10).fit_transform(X)  # 高维 → 10D
labels = hdbscan.HDBSCAN(min_cluster_size=50).fit_predict(emb)     # 10D → 聚类`;

    return `
      ${MCH.hero({
        icon: "✧",
        name: "t-SNE / UMAP 降维可视化",
        en: "Manifold Learning for Visualization",
        tags: ["非线性降维", "流形学习", "2D 可视化", "t-SNE 2008", "UMAP 2018"],
        meta: ["◈ 保留局部结构", "⚡ Embedding 可视化必备"],
      })}

      ${MCH.versionSection("ml_tsne_umap")}

      <div class="section">
        <h2>1. 为什么 PCA 不够用？</h2>
        <p class="text-sm text-slate-600">PCA 只能找<b>线性结构</b>。如果数据在高维空间上是<b>流形</b>（如瑞士卷），PCA 投影会把不相连的部分混在一起。</p>
        <div class="grid-2">
          <div class="card"><h3 class="text-sm font-bold">📐 PCA</h3><p class="text-xs text-slate-600 mt-2">最大方差方向投影 → 线性流形上强、<b>非线性结构</b>会被拉平。</p></div>
          <div class="card"><h3 class="text-sm font-bold">🌀 t-SNE / UMAP</h3><p class="text-xs text-slate-600 mt-2">保留<b>局部邻居关系</b>，能"展开"瑞士卷，可视化 embedding 聚类结构。</p></div>
        </div>
      </div>

      <div class="section">
        <h2>2. t-SNE 的核心技巧</h2>
        <div class="formula-block">
          <b>高维相似度</b>（对称化 softmax）：
          $$ p_{j|i} = \\frac{\\exp(-\\|x_i - x_j\\|^2 / 2\\sigma_i^2)}{\\sum_{k \\neq i} \\exp(-\\|x_i - x_k\\|^2 / 2\\sigma_i^2)} $$
          <b>低维相似度</b>（Student-t 分布 1 自由度，<b>重尾</b>）：
          $$ q_{ij} = \\frac{(1 + \\|y_i - y_j\\|^2)^{-1}}{\\sum_{k \\neq l} (1 + \\|y_k - y_l\\|^2)^{-1}} $$
          <b>优化目标</b>（最小化 KL 散度）：
          $$ L = \\sum_{i,j} p_{ij} \\log \\frac{p_{ij}}{q_{ij}} $$
        </div>
        ${MCH.info(`
          <b>为什么低维用重尾 t 分布？</b>
          如果两个都用高斯，高维到低维会有 "crowding problem"：中等距离的点在低维空间挤在一起。
          t 分布重尾允许它们被推远 → 更清晰的簇边界。
        `, "tip")}
      </div>

      <div class="section">
        <h2>3. t-SNE vs UMAP 详细对比</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>t-SNE</th><th>UMAP</th></tr></thead>
          <tbody>
            <tr><td>理论基础</td><td>概率分布 KL 散度</td><td>黎曼几何 + 模糊拓扑</td></tr>
            <tr><td>速度（10w 点）</td><td>10-30 min</td><td><b>2-5 min</b></td></tr>
            <tr><td>关键超参</td><td>perplexity (5-50)</td><td>n_neighbors (5-100), min_dist</td></tr>
            <tr><td>保留局部结构</td><td>✓✓</td><td>✓✓</td></tr>
            <tr><td>保留全局结构</td><td>弱</td><td><b>较好</b></td></tr>
            <tr><td>支持 transform(new_X)</td><td>❌</td><td>✓</td></tr>
            <tr><td>支持监督</td><td>❌</td><td>✓ Supervised UMAP</td></tr>
            <tr><td>GPU 加速</td><td>rapidsAI</td><td>cuML</td></tr>
            <tr><td>随机性</td><td>高（每次不同）</td><td>中等（set_random_state）</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>4. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 交互：超参 perplexity / n_neighbors 的影响</h2>
        <p class="text-sm text-slate-600">perplexity (t-SNE) 或 n_neighbors (UMAP) 控制"局部 vs 全局"的平衡。调节观察结构保留程度。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "tu-perp", label: "perplexity / n_neighbors", min: 2, max: 100, step: 1, value: 30 })}
              ${MCH.slider({ id: "tu-iter", label: "训练轮数（模拟）", min: 10, max: 1000, step: 10, value: 500 })}
              <label class="text-xs mt-2 block">算法：
                <select id="tu-algo" class="text-xs border rounded p-1 ml-2">
                  <option value="tsne">t-SNE</option>
                  <option value="umap" selected>UMAP</option>
                </select>
              </label>
            </div>
            <div id="tu-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-tu" style="height:400px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>6. ⚠ 使用注意事项</h2>
        ${MCH.info(`
          <ul style="padding-left:20px;">
            <li><b>只用于可视化</b>：不要把 t-SNE/UMAP 的 2D 输出直接作为下游 ML 模型的特征（几何意义被破坏）；</li>
            <li><b>距离不可靠</b>：可视化中两点的距离<b>不对应原始距离</b>，只能相信"在同一簇 / 不同簇"；</li>
            <li><b>簇大小不可靠</b>：t-SNE 会"均匀化"簇大小；</li>
            <li><b>超参敏感</b>：尝试多组 perplexity（5、30、50）看结构稳定性；</li>
            <li><b>先 PCA 预降维</b>：高维（&gt; 500）建议先 PCA 降到 50 维再 t-SNE/UMAP；</li>
            <li><b>UMAP 更适合 Pipeline</b>：transform() 支持 + 更快 + 更稳定。</li>
          </ul>
        `, "warn")}
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ml_tsne_umap").pros, MCH.getById("ml_tsne_umap").cons, MCH.getById("ml_tsne_umap").best_for)}
      </div>
    `;
  },

  mount() {
    // 模拟 MNIST-like 10 类嵌入投影
    const chart = MCH.echart(document.getElementById("chart-tu"), {
      tooltip: {},
      legend: { top: 0, data: [...Array(10).keys()].map(i => `类 ${i}`) },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", min: -8, max: 8 },
      yAxis: { type: "value", min: -8, max: 8 },
      series: [],
    });

    const colors = ["#ef4444", "#f59e0b", "#eab308", "#84cc16", "#10b981", "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899", "#f43f5e"];

    const renderEmb = () => {
      const algo = document.getElementById("tu-algo").value;
      const perp = parseInt(document.getElementById("tu-perp").value);
      const iter = parseInt(document.getElementById("tu-iter").value);
      const series = [];
      // 模拟：每个类生成一个中心，点围绕中心。参数影响簇紧密度和分布
      const spread = algo === "tsne" ? 1.0 : 0.8;
      const iterFactor = Math.min(1, iter / 500);
      const perpEffect = Math.min(1.5, perp / 30);
      for (let c = 0; c < 10; c++) {
        const centerAngle = c / 10 * 2 * Math.PI;
        // perp 小 → 簇分散；perp 大 → 簇合并
        const radius = 4 * (2 - iterFactor) * (2 / perpEffect);
        const cx = Math.cos(centerAngle) * Math.min(6, radius) + (algo === "umap" ? Math.cos(c * 0.7) * 0.5 : 0);
        const cy = Math.sin(centerAngle) * Math.min(6, radius);
        // t-SNE 常把相似类推远，UMAP 保留更多结构
        const pts = [];
        for (let i = 0; i < 30; i++) {
          const r = MCH.randn(2, c * 100 + i + iter);
          const tight = spread * iterFactor / (1 + perp * 0.015);
          pts.push([cx + r[0] * tight, cy + r[1] * tight]);
        }
        series.push({
          name: `类 ${c}`, type: "scatter", data: pts,
          itemStyle: { color: colors[c] }, symbolSize: 6,
        });
      }
      chart.setOption({ series });
      document.getElementById("tu-info").innerHTML = `
        <b>${algo === "tsne" ? "t-SNE" : "UMAP"}</b> · 超参: ${perp} · 迭代: ${iter}<br/>
        ${perp < 10 ? '<span style="color:#ef4444;">⚠ ' + (algo === 'tsne' ? 'perplexity' : 'n_neighbors') + ' 过小 → 只看局部，忽略结构</span>' :
        perp > 60 ? '<span style="color:#f59e0b;">⚠ 过大 → 簇可能合并</span>' :
        '<span style="color:#10b981;">合适范围</span>'}<br/>
        ${algo === "umap" ? '<span style="color:#64748b;">UMAP 速度快 + 保留全局结构 + 支持 transform</span>' : '<span style="color:#64748b;">t-SNE 保留局部强，但相对距离/大小不可信</span>'}
      `;
    };
    ["tu-perp", "tu-iter"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        renderEmb();
      });
    });
    document.getElementById("tu-algo").addEventListener("change", renderEmb);
    renderEmb();
  },
});
