/* 模块：PCA 主成分分析 */
MCH.register("ml_pca", {
  render() {
    const code = `# PCA 两种求解方式
# ① 协方差矩阵特征分解
# ② SVD 分解（更数值稳定）

import numpy as np

def pca_eigen(X, n_components):
    # 1) 中心化
    X_centered = X - X.mean(axis=0)
    # 2) 协方差矩阵 (d, d)
    cov = (X_centered.T @ X_centered) / (len(X) - 1)
    # 3) 特征分解
    eigvals, eigvecs = np.linalg.eigh(cov)
    # 4) 按特征值降序取前 k
    idx = np.argsort(eigvals)[::-1]
    components = eigvecs[:, idx[:n_components]]
    # 5) 投影
    X_pca = X_centered @ components
    return X_pca, components, eigvals[idx]

def pca_svd(X, n_components):
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # S² / (n-1) = 特征值；Vt 的前 k 行 = 主成分
    components = Vt[:n_components]
    X_pca = X_centered @ components.T
    return X_pca, components, S[:n_components]**2 / (len(X) - 1)


# scikit-learn：自动选择最优求解器
from sklearn.decomposition import PCA, IncrementalPCA, KernelPCA

# 小数据 + 精确解
pca = PCA(n_components=0.95)   # 保留 95% 方差的最少主成分
X_pca = pca.fit_transform(X)
print(pca.explained_variance_ratio_)  # 每个主成分解释的方差比

# 流式 / 大数据：Incremental PCA
ipca = IncrementalPCA(n_components=50, batch_size=1024)
for batch in data_loader:
    ipca.partial_fit(batch)

# 非线性：Kernel PCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)`;

    return `
      ${MCH.hero({
        icon: "⊿",
        name: "PCA 主成分分析",
        en: "Principal Component Analysis (Pearson 1901)",
        tags: ["SVD", "方差最大化", "线性降维", "无监督"],
        meta: ["◈ 闭式解", "⚡ 最经典降维", "◇ 可逆变换"],
      })}

      ${MCH.versionSection("ml_pca")}

      <div class="section">
        <h2>1. 核心思想：沿"方差最大方向"投影</h2>
        <p class="text-sm text-slate-600">PCA 寻找一组正交方向（主成分），使得数据投影后方差依次最大化。可证明这些方向就是<b>协方差矩阵的特征向量</b>，特征值就是对应的方差。</p>
        <div class="formula-block">
          <b>协方差矩阵</b>（中心化后）：
          $$ C = \\frac{1}{n-1} X_c^\\top X_c, \\quad X_c = X - \\bar X $$
          <b>特征分解</b>：$C v_i = \\lambda_i v_i$, 其中 $\\lambda_1 \\geq \\lambda_2 \\geq \\dots$<br/>
          <b>主成分</b>：$v_1, v_2, \\dots, v_k$（前 k 个最大特征值对应的特征向量）<br/>
          <b>投影</b>：$X_{\\text{pca}} = X_c V_k \\in \\mathbb{R}^{n \\times k}$
        </div>
        ${MCH.info(`
          <b>为什么等价于 SVD？</b>
          对 $X_c = U \\Sigma V^\\top$ 做 SVD，则 $C = \\frac{1}{n-1} V \\Sigma^2 V^\\top$。
          V 的列就是 PCA 的主成分，Σ²/(n-1) 就是特征值。
          SVD 方式避免显式构造 d×d 协方差矩阵，<b>数值更稳定，对 d &gt; n 的高维小样本友好</b>。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：2D 数据的 PCA 动态演示</h2>
        <p class="text-sm text-slate-600">调节数据相关性，观察两个主成分方向（沿最大方差）。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "pca-corr", label: "数据相关性 ρ", min: -0.95, max: 0.95, step: 0.05, value: 0.8 })}
              ${MCH.slider({ id: "pca-scale-y", label: "Y 方向伸缩", min: 0.3, max: 3, step: 0.1, value: 1 })}
            </div>
            <div id="pca-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-pca" style="height:360px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. Scree Plot —— 解释方差比</h2>
        <p class="text-sm text-slate-600">模拟一个 20 维数据集，观察 PCA 各主成分解释的方差比。<b>累积方差曲线</b>帮助选择 k（通常取累积 ≥ 95%）。</p>
        <div id="chart-scree" style="height:320px;"></div>
      </div>

      <div class="section">
        <h2>5. PCA 实现变体对比</h2>
        <table class="table">
          <thead><tr><th>变体</th><th>适用场景</th><th>时间复杂度</th><th>特点</th></tr></thead>
          <tbody>
            <tr><td>PCA (full SVD)</td><td>中小数据，需要精确解</td><td>O(n·d²) + O(d³)</td><td>sklearn 默认（小数据）</td></tr>
            <tr><td><b>Randomized PCA</b></td><td>大矩阵 + 只取前 k</td><td>O(n·d·k) 近似</td><td>Halko 2011 算法，快 10×</td></tr>
            <tr><td><b>Incremental PCA</b></td><td>流式 / 大数据</td><td>mini-batch</td><td>支持 partial_fit，内存可控</td></tr>
            <tr><td>Sparse PCA</td><td>稀疏载荷、可解释性强</td><td>慢</td><td>诱导 L1 正则</td></tr>
            <tr><td><b>Kernel PCA</b></td><td>非线性降维</td><td>O(n³)</td><td>核函数版，但不可扩展</td></tr>
            <tr><td>ICA</td><td>源分离 / 盲信号</td><td>同 PCA</td><td>找统计独立的方向（非正交）</td></tr>
            <tr><td>NMF</td><td>非负数据</td><td>迭代</td><td>主题建模、图像分解</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. PCA 的应用场景</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6 space-y-1">
          <li><b>可视化</b>：把高维数据投到 2D/3D 散点图；</li>
          <li><b>去噪</b>：丢弃低方差主成分可以滤掉噪声；</li>
          <li><b>特征去相关</b>：下游模型（尤其 LR）假设特征独立时；</li>
          <li><b>图像压缩</b>：Eigenfaces 人脸识别经典方法；</li>
          <li><b>加速</b>：把 d=1000 降到 d=100，下游训练大幅提速；</li>
          <li><b>Embedding 初步分析</b>：Word2Vec / BERT 前 2/3 主成分可视化语义结构。</li>
        </ul>
        ${MCH.info(`
          <b>⚠ 重要</b>：PCA 前必须<b>标准化</b>（StandardScaler），否则不同量纲的特征会让结果被"大数值特征"主导。
        `, "warn")}
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ml_pca").pros, MCH.getById("ml_pca").cons, MCH.getById("ml_pca").best_for)}
      </div>
    `;
  },

  mount() {
    // 2D PCA viz
    const chart = MCH.echart(document.getElementById("chart-pca"), {
      tooltip: {},
      grid: { left: 50, right: 20, top: 30, bottom: 30 },
      xAxis: { type: "value", min: -4, max: 4 },
      yAxis: { type: "value", min: -4, max: 4 },
      series: [],
    });
    const update = () => {
      const rho = parseFloat(document.getElementById("pca-corr").value);
      const sy = parseFloat(document.getElementById("pca-scale-y").value);
      // 生成相关数据
      const n = 200;
      const pts = [];
      for (let i = 0; i < n; i++) {
        const r = MCH.randn(2, i * 3 + 1);
        const x = r[0];
        const y = rho * r[0] + Math.sqrt(1 - rho * rho) * r[1];
        pts.push([x * 1.5, y * sy * 1.5]);
      }
      // 计算 PCA
      const mx = pts.reduce((s, p) => s + p[0], 0) / n;
      const my = pts.reduce((s, p) => s + p[1], 0) / n;
      const cxx = pts.reduce((s, p) => s + (p[0] - mx) ** 2, 0) / (n - 1);
      const cyy = pts.reduce((s, p) => s + (p[1] - my) ** 2, 0) / (n - 1);
      const cxy = pts.reduce((s, p) => s + (p[0] - mx) * (p[1] - my), 0) / (n - 1);
      // 2x2 协方差矩阵特征值/向量
      const tr = cxx + cyy, det = cxx * cyy - cxy * cxy;
      const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
      const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
      const v1 = [cxy, l1 - cxx];
      const v1n = Math.hypot(...v1);
      const v1u = [v1[0] / v1n, v1[1] / v1n];
      const v2u = [-v1u[1], v1u[0]];
      const r1 = Math.sqrt(l1) * 2.5, r2 = Math.sqrt(l2) * 2.5;
      chart.setOption({
        series: [
          { type: "scatter", data: pts, itemStyle: { color: "#94a3b8", opacity: 0.6 }, symbolSize: 5 },
          { type: "lines", data: [{ coords: [[mx, my], [mx + v1u[0] * r1, my + v1u[1] * r1]] }], lineStyle: { color: "#4f46e5", width: 4 }, effect: { show: true, symbol: "arrow", color: "#4f46e5" }, label: { show: true, position: "end", formatter: "PC1", fontSize: 12 } },
          { type: "lines", data: [{ coords: [[mx, my], [mx + v2u[0] * r2, my + v2u[1] * r2]] }], lineStyle: { color: "#f59e0b", width: 3 }, effect: { show: true, symbol: "arrow", color: "#f59e0b" }, label: { show: true, position: "end", formatter: "PC2", fontSize: 12 } },
        ],
      });
      const var1_ratio = l1 / (l1 + l2);
      document.getElementById("pca-info").innerHTML = `
        <b>PC1 特征值</b>: ${l1.toFixed(3)} · <b>PC2 特征值</b>: ${l2.toFixed(3)}<br/>
        <b>PC1 解释方差比</b>: <span style="color:#4f46e5;font-weight:700;">${(var1_ratio * 100).toFixed(1)}%</span><br/>
        <b>PC2 解释方差比</b>: <span style="color:#f59e0b;">${((1 - var1_ratio) * 100).toFixed(1)}%</span><br/>
        ${var1_ratio > 0.9 ? '<span style="color:#10b981;">强相关，PC1 几乎包含全部信息，可以安全降到 1D</span>' : ''}
        ${var1_ratio < 0.6 ? '<span style="color:#f59e0b;">相关性弱，两个 PC 都重要，降维要谨慎</span>' : ''}
      `;
    };
    ["pca-corr", "pca-scale-y"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();

    // Scree plot
    const d = 20;
    const ev = [...Array(d).keys()].map(i => 15 * Math.exp(-i * 0.3) + 0.3 * Math.random());
    const total = ev.reduce((a, b) => a + b, 0);
    const ratios = ev.map(v => v / total);
    const cumulative = ratios.reduce((acc, v, i) => { acc.push((acc[i - 1] || 0) + v); return acc; }, []);
    MCH.echart(document.getElementById("chart-scree"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 60, top: 40, bottom: 40 },
      xAxis: { type: "category", name: "主成分索引", data: [...Array(d).keys()].map(i => `PC${i + 1}`) },
      yAxis: [
        { type: "value", name: "单个方差比", max: 1, position: "left" },
        { type: "value", name: "累积方差比", max: 1, position: "right" },
      ],
      series: [
        { name: "解释方差比", type: "bar", yAxisIndex: 0, barWidth: 16, data: ratios.map(v => +v.toFixed(3)), color: "#4f46e5" },
        { name: "累积", type: "line", smooth: true, yAxisIndex: 1, color: "#10b981", lineStyle: { width: 3 }, data: cumulative.map(v => +v.toFixed(3)), markLine: { symbol: "none", silent: true, data: [{ yAxis: 0.95, label: { formatter: "95% 阈值" }, lineStyle: { color: "#ef4444", type: "dashed" } }] } },
      ],
    });
  },
});
