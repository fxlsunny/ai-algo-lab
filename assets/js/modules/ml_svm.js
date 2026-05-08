/* 模块：SVM 支持向量机 */
MCH.register("ml_svm", {
  render() {
    const code = `# SVM：寻找最大间隔超平面
# Cortes & Vapnik, Machine Learning, 1995

from sklearn.svm import SVC, LinearSVC, SVR

# 线性 SVM：找一个超平面 w·x + b = 0 使两类样本间隔最大
# min  (1/2) ||w||² + C · Σ ξ_i
# s.t. y_i (w·x_i + b) >= 1 - ξ_i,  ξ_i >= 0
#        ξ 是松弛变量，允许少量误分（软间隔）
#        C 越大容忍越少 → 过拟合风险；C 越小间隔越大 → 欠拟合

clf = SVC(C=1.0, kernel='rbf', gamma='scale')
clf.fit(X, y)

# ============================================================
# 核技巧 Kernel Trick —— SVM 的灵魂
# ============================================================
# 用 K(x_i, x_j) = φ(x_i)·φ(x_j) 替代高维空间的点积
# 不需要显式计算 φ(x)，可以处理无穷维特征空间

# 常见核函数：
# - Linear:   K(x, y) = x·y                              —— 文本分类首选
# - Polynomial: K(x, y) = (γ x·y + r)^d                  —— 处理特征交互
# - RBF (Gaussian): K(x, y) = exp(-γ ||x - y||²)         —— 默认，处理非线性
# - Sigmoid:  K(x, y) = tanh(γ x·y + r)                  —— 近似神经网络

# 支持向量：只有位于间隔边界上/内部的样本才影响决策边界
# 这些样本的 α_i > 0，其余 α_i = 0 → 稀疏性

# 大数据替代：
# - LinearSVC: 用 liblinear，比 SVC(kernel='linear') 快 10-100×
# - SGDClassifier(loss='hinge'): 等价于 SVM，支持百万级样本
# - ThunderSVM: GPU 版 SVM，A100 上可训百万数据`;

    return `
      ${MCH.hero({
        icon: "⊕",
        name: "SVM 支持向量机",
        en: "Support Vector Machine (Cortes & Vapnik 1995)",
        tags: ["最大间隔", "核技巧", "凸优化", "SVC/SVR"],
        meta: ["◈ 小/中数据 SOTA（前深度学习时代）", "⚡ 理论坚实"],
      })}

      ${MCH.versionSection("ml_svm")}

      <div class="section">
        <h2>1. 核心思想：最大间隔</h2>
        <p class="text-sm text-slate-600">SVM 的核心直觉：<b>不仅要把两类分开，还要让分界超平面离最近的样本尽可能远</b>（大间隔 = 好泛化）。</p>
        <div class="formula-block">
          <b>硬间隔（线性可分）</b>：
          $$ \\min_{w, b} \\frac{1}{2} \\|w\\|^2 \\quad \\text{s.t.} \\quad y_i (w \\cdot x_i + b) \\geq 1 $$
          <b>软间隔（含噪声）</b>：引入松弛变量 ξ 和惩罚系数 C：
          $$ \\min_{w, b, \\xi} \\frac{1}{2} \\|w\\|^2 + C \\sum_i \\xi_i $$
        </div>
        ${MCH.info(`
          <b>为什么叫"支持向量"机？</b>
          只有位于间隔边界上或间隔内的样本（α_i &gt; 0）才影响决策边界 —— 这些样本称为<b>支持向量</b>。
          移除其他样本（远离边界的"安全"样本）对模型无影响，这就是 SVM 的<b>稀疏性</b>。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 核函数（Kernel Trick）— SVM 的灵魂</h2>
        <p class="text-sm text-slate-600">对非线性问题，SVM 用 <b>核函数</b> 隐式地把数据映射到更高维空间，在那里变为线性可分：</p>
        <div class="formula-block">
          $$ K(x_i, x_j) = \\phi(x_i) \\cdot \\phi(x_j) \\;\\;\\; \\text{(无需显式计算 } \\phi\\text{)} $$
        </div>

        <div class="grid-4 mt-3">
          <div class="card"><h4 class="font-semibold text-sm">Linear</h4><p class="text-xs text-slate-600 mt-1">K = x·y<br/>文本 TF-IDF 首选</p></div>
          <div class="card"><h4 class="font-semibold text-sm">RBF / 高斯</h4><p class="text-xs text-slate-600 mt-1">K = exp(-γ‖x-y‖²)<br/>默认，处理非线性</p></div>
          <div class="card"><h4 class="font-semibold text-sm">Polynomial</h4><p class="text-xs text-slate-600 mt-1">K = (γx·y+r)^d<br/>特征交互建模</p></div>
          <div class="card"><h4 class="font-semibold text-sm">Sigmoid</h4><p class="text-xs text-slate-600 mt-1">K = tanh(γx·y+r)<br/>近似神经网络</p></div>
        </div>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：C 和 γ 对决策边界的影响</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "svm-C", label: "C (惩罚强度)", min: 0.01, max: 100, step: 0.5, value: 1, format: (v) => parseFloat(v).toFixed(2) })}
              ${MCH.slider({ id: "svm-gamma", label: "γ (RBF 核宽度)", min: 0.01, max: 10, step: 0.1, value: 1, format: (v) => parseFloat(v).toFixed(2) })}
              <label class="text-xs text-slate-600 mr-3 mt-3 block">Kernel:</label>
              <select id="svm-kernel" class="text-xs border rounded p-1">
                <option value="linear">Linear</option>
                <option value="rbf" selected>RBF (默认)</option>
                <option value="poly">Polynomial (d=3)</option>
              </select>
            </div>
            <div id="svm-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-svm" style="height:400px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. SVM 适用性判断</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>推荐</th><th>原因</th></tr></thead>
          <tbody>
            <tr><td>文本分类（TF-IDF 维度 10w+）</td><td><b>LinearSVC</b></td><td>文本高维稀疏，线性核已经够强；LinearSVC 比 SVC 快 10-100×</td></tr>
            <tr><td>样本 &lt; 10w，特征 &lt; 100</td><td><b>SVC (RBF)</b></td><td>小数据 + 非线性，SVM 的舒适区</td></tr>
            <tr><td>样本 &gt; 10w</td><td>XGBoost / LightGBM</td><td>SVM 训练 O(n²) 慢到不可用</td></tr>
            <tr><td>需要概率输出</td><td>LogisticRegression 或 SVC(probability=True)</td><td>SVM 原本不输出概率，Platt 标定有开销</td></tr>
            <tr><td>多类问题（类别 &gt; 2）</td><td>OvR / OvO</td><td>SVC 默认 OvO（精度高但 K(K-1)/2 个模型）</td></tr>
            <tr><td>GPU 加速</td><td><code>ThunderSVM</code></td><td>支持 A100 训练</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ml_svm").pros, MCH.getById("ml_svm").cons, MCH.getById("ml_svm").best_for)}
      </div>
    `;
  },

  mount() {
    // 决策边界可视化 (2D, 模拟)
    const chartEl = document.getElementById("chart-svm");
    const chart = MCH.echart(chartEl, {
      tooltip: {},
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", min: -4, max: 4 },
      yAxis: { type: "value", min: -4, max: 4 },
      legend: { top: 0, data: ["类别 +1", "类别 -1", "决策边界区域"] },
      series: [],
    });

    // 生成数据：2 个 cluster
    const rng = MCH.seedRng(42);
    const class1 = [], class2 = [];
    for (let i = 0; i < 30; i++) {
      const r1 = MCH.randn(2, i * 3 + 1);
      const r2 = MCH.randn(2, i * 7 + 2);
      class1.push([1.2 + r1[0] * 0.7, 1.0 + r1[1] * 0.7]);
      class2.push([-1.2 + r2[0] * 0.7, -1.0 + r2[1] * 0.7]);
    }

    const update = () => {
      const C = parseFloat(document.getElementById("svm-C").value);
      const gamma = parseFloat(document.getElementById("svm-gamma").value);
      const kernel = document.getElementById("svm-kernel").value;

      // 模拟决策函数（非真实 SVM 训练，仅直观示意）
      const G = 40;
      const heatmapData = [];
      for (let i = 0; i < G; i++) {
        for (let j = 0; j < G; j++) {
          const x = -4 + (i / G) * 8;
          const y = -4 + (j / G) * 8;
          // 计算与两类 cluster 中心的相对距离（模拟决策值）
          let score = 0;
          if (kernel === "linear") {
            score = x + y;
          } else if (kernel === "rbf") {
            // 用每个支持向量的加权
            const pos = Math.exp(-gamma * ((x - 1.2) ** 2 + (y - 1) ** 2));
            const neg = Math.exp(-gamma * ((x + 1.2) ** 2 + (y + 1) ** 2));
            score = (pos - neg) * (1 + Math.log(C + 1) * 0.2);
          } else if (kernel === "poly") {
            const cx = x * 1.2 + y * 1.0, cn = x * (-1.2) + y * (-1.0);
            score = Math.pow(cx + 1, 3) - Math.pow(cn + 1, 3);
            score = score / (1 + Math.abs(score));
          }
          heatmapData.push([x, y, Math.tanh(score * 2)]);
        }
      }
      chart.setOption({
        visualMap: { show: false, min: -1, max: 1, inRange: { color: ["#fecaca", "#fef3c7", "#bbf7d0"] } },
        series: [
          { name: "决策边界区域", type: "heatmap", data: heatmapData, progressive: 0, silent: true, coordinateSystem: "cartesian2d" },
          { name: "类别 +1", type: "scatter", data: class1, symbolSize: 11, itemStyle: { color: "#10b981" } },
          { name: "类别 -1", type: "scatter", data: class2, symbolSize: 11, itemStyle: { color: "#ef4444" } },
        ],
      });

      // 估计支持向量数（示意）
      const nSV = kernel === "linear" ? 4 : kernel === "rbf" ? Math.round(10 + gamma * 3) : 12;
      document.getElementById("svm-info").innerHTML = `
        <b>当前配置</b>: kernel=${kernel}, C=${C.toFixed(2)}, γ=${gamma.toFixed(2)}<br/>
        <b>估计支持向量数</b>: ~${nSV} / 60<br/>
        <b>解读</b>：
        ${C > 10 ? '<span style="color:#ef4444;">C 很大 → 近似硬间隔，易过拟合</span>' : C < 0.1 ? '<span style="color:#f59e0b;">C 很小 → 间隔很大但允许很多误分</span>' : '<span style="color:#10b981;">C 合适</span>'}
        ${kernel === 'rbf' && gamma > 3 ? '<br/><span style="color:#ef4444;">γ 过大 → 每个点只影响自身附近 → 过拟合</span>' : ''}
        ${kernel === 'rbf' && gamma < 0.1 ? '<br/><span style="color:#f59e0b;">γ 过小 → 边界过于平滑 → 欠拟合</span>' : ''}
      `;
    };
    ["svm-C", "svm-gamma"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = parseFloat(e.target.value).toFixed(2);
        update();
      });
    });
    document.getElementById("svm-kernel").addEventListener("change", update);
    update();
  },
});
