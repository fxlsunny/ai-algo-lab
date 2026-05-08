/* 模块：逻辑回归 */
MCH.register("ml_logistic", {
  render() {
    const code = `# 逻辑回归：线性模型 + sigmoid + 交叉熵
# 用 SGD / FTRL 训练，支持在线增量学习

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic(X, y, lr=0.01, n_iter=100, lam_l2=0.01):
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for epoch in range(n_iter):
        z = X @ w + b
        p = sigmoid(z)                      # 预测概率
        # 梯度：(p - y) 是 log-loss 对 z 的导数
        grad_w = X.T @ (p - y) / n + 2 * lam_l2 * w   # L2 正则 = 权重收缩
        grad_b = (p - y).mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b

# FTRL-Proximal (Follow The Regularized Leader)：Google 2013，CTR 在线学习神器
#   每来一个样本立即更新，L1 正则天然稀疏化，适合十亿级稀疏特征`;

    return `
      ${MCH.hero({ icon: "σ", name: "逻辑回归 Logistic Regression", en: "Linear Model + Sigmoid", tags: ["线性", "概率输出", "L1/L2 正则", "在线学习"], meta: ["◈ 风控评分卡核心", "⚡ 训练/推理极快"] })}

      ${MCH.versionSection("ml_logistic")}

      <div class="section">
        <h2>1. 核心模型</h2>
        <div class="formula-block">
          <b>模型：</b>
          $$ P(y=1 \\mid x) = \\sigma(w^\\top x + b), \\quad \\sigma(z) = \\frac{1}{1 + e^{-z}} $$
          <b>损失（负对数似然，即 Binary Cross-Entropy）：</b>
          $$ \\mathcal{L} = -\\frac{1}{n} \\sum_i \\big[y_i \\log p_i + (1 - y_i) \\log(1 - p_i)\\big] + \\lambda_1 \\|w\\|_1 + \\tfrac{\\lambda_2}{2} \\|w\\|_2^2 $$
          <b>梯度（漂亮的闭式）：</b>
          $$ \\nabla_w \\mathcal{L} = \\frac{1}{n} X^\\top (p - y) + 2 \\lambda_2 w $$
        </div>
        ${MCH.info(`
          <b>为什么"Logistic"叫回归不叫分类？</b>
          虽然用于二分类，但数学上它建模的是"对数几率"（log-odds）：
          $\\log \\frac{P(y=1)}{P(y=0)} = w^\\top x + b$，是对连续量的线性回归。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 代码解读</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：Sigmoid 曲线 / 决策边界</h2>
        <div class="grid-2">
          <div>
            <h3>· Sigmoid 函数形状</h3>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "lr-slope", label: "w（斜率）", min: 0.1, max: 5, step: 0.1, value: 1 })}
              ${MCH.slider({ id: "lr-bias", label: "b（偏置）", min: -5, max: 5, step: 0.1, value: 0 })}
            </div>
            <div id="chart-sigmoid" style="height:280px;"></div>
          </div>
          <div>
            <h3>· L1 vs L2 正则的稀疏化</h3>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "lr-reg", label: "正则强度", min: 0, max: 5, step: 0.1, value: 1 })}
            </div>
            <div id="chart-reg" style="height:280px;"></div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 应用场景：评分卡建模</h2>
        <p class="text-sm text-slate-600">金融风控把逻辑回归系数转换为<b>评分</b>：</p>
        <div class="formula-block">
          $$ \\text{Score} = A - B \\cdot \\log \\frac{P(\\text{bad})}{P(\\text{good})} = A - B \\cdot (w^\\top x + b) $$
        </div>
        <p class="text-sm text-slate-600">每个特征的<b>分箱 WOE 编码</b>乘以对应系数 w_i，即该特征的评分贡献，完全可解释，审批友好。</p>
        <table class="table mt-3">
          <thead><tr><th>特征</th><th>分箱</th><th>WOE</th><th>LR 系数</th><th>评分贡献</th></tr></thead>
          <tbody>
            <tr><td>近 30d GMV</td><td>&lt; 1k</td><td>-0.82</td><td>0.76</td><td class="text-red-600"><b>-34</b></td></tr>
            <tr><td>近 30d GMV</td><td>1k ~ 10k</td><td>0.21</td><td>0.76</td><td>+13</td></tr>
            <tr><td>近 30d GMV</td><td>&gt; 10k</td><td>0.89</td><td>0.76</td><td class="text-emerald-600"><b>+55</b></td></tr>
            <tr><td>对手方唯一度</td><td>&lt; 5</td><td>-1.12</td><td>0.43</td><td class="text-red-600"><b>-27</b></td></tr>
            <tr><td>对手方唯一度</td><td>&gt;= 20</td><td>0.67</td><td>0.43</td><td>+16</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("ml_logistic").pros,
          MCH.getById("ml_logistic").cons,
          MCH.getById("ml_logistic").best_for,
        )}
      </div>

      <div class="section">
        <h2>6. 关键超参 & 工程</h2>
        ${MCH.hyperTable([
          ["C (正则强度倒数)", "0.01 ~ 10", "C 越小正则越强", "GridSearch 对数间隔"],
          ["penalty", "l1 / l2 / elasticnet", "L1 稀疏、L2 稳定", "特征超多用 L1 筛选"],
          ["solver", "liblinear / saga / lbfgs", "不同求解器", "大数据用 saga（支持 L1+L2+online）"],
          ["class_weight", "'balanced' / 自定义", "类不均衡加权", "长尾必调"],
          ["max_iter", "100 ~ 1000", "最大迭代轮数", "不收敛时增大"],
          ["fit_intercept", "True", "是否学偏置", "通常 True"],
        ])}
      </div>
    `;
  },

  mount() {
    // Sigmoid
    const sigChart = MCH.echart(document.getElementById("chart-sigmoid"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "x", min: -10, max: 10 },
      yAxis: { type: "value", name: "P(y=1)", min: 0, max: 1 },
      series: [{ type: "line", smooth: true, showSymbol: false, color: "#4f46e5", lineStyle: { width: 3 }, data: [] }],
    });
    const updateSig = () => {
      const w = parseFloat(document.getElementById("lr-slope").value);
      const b = parseFloat(document.getElementById("lr-bias").value);
      const xs = MCH.linspace(-10, 10, 200);
      sigChart.setOption({
        series: [{ data: xs.map(x => [x, 1 / (1 + Math.exp(-(w * x + b)))]) }],
      });
    };
    ["lr-slope", "lr-bias"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateSig();
      });
    });
    updateSig();

    // Reg comparison
    const regChart = MCH.echart(document.getElementById("chart-reg"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", name: "特征维度 d", data: [...Array(20).keys()].map(i => `w${i + 1}`) },
      yAxis: { type: "value", name: "权重" },
      series: [
        { name: "无正则", type: "bar", data: [], color: "#94a3b8", barGap: 0.1 },
        { name: "L2 (岭回归)", type: "bar", data: [], color: "#4f46e5" },
        { name: "L1 (Lasso, 稀疏)", type: "bar", data: [], color: "#f59e0b" },
      ],
    });
    const updateReg = () => {
      const lam = parseFloat(document.getElementById("lr-reg").value);
      // 真实系数
      const wTrue = [2.5, 1.8, -1.5, 2.1, 0.9, -0.6, 1.2, -0.4, 0.2, -0.1, 0.05, -0.03, 0.01, 0.002, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
      // L2：按 1/(1+λ) 收缩
      const wL2 = wTrue.map(w => w / (1 + lam * 0.3));
      // L1：soft thresholding
      const wL1 = wTrue.map(w => Math.sign(w) * Math.max(0, Math.abs(w) - lam * 0.2));
      regChart.setOption({
        series: [{ data: wTrue }, { data: wL2.map(v => +v.toFixed(3)) }, { data: wL1.map(v => +v.toFixed(3)) }],
      });
    };
    document.getElementById("lr-reg").addEventListener("input", (e) => {
      document.getElementById("lr-reg-val").textContent = e.target.value;
      updateReg();
    });
    updateReg();
  },
});
