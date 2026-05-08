/* 模块：随机森林 */
MCH.register("ml_random_forest", {
  render() {
    const code = `# 随机森林（Bagging + 特征采样）
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features="sqrt"):
        self.trees = []
        self.n = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features

    def fit(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.n):
            # 1) Bootstrap：有放回采样 n 个样本
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[idx], y[idx]

            # 2) 特征随机：每棵树只用部分特征（sqrt(d) 或 log2(d)）
            tree = DecisionTree(
                max_depth=self.max_depth,
                max_features=self.max_features,   # 每次分裂也随机选
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        # 分类：多数投票；回归：平均
        all_preds = np.stack([t.predict(X) for t in self.trees])  # (T, n)
        return np.apply_along_axis(lambda p: np.bincount(p).argmax(), 0, all_preds)

    @property
    def feature_importances_(self):
        """特征重要性 = 所有树中该特征带来的不纯度下降加权平均"""
        fi = np.zeros(n_features)
        for tree in self.trees:
            fi += tree.feature_importances_
        return fi / fi.sum()`;

    return `
      ${MCH.hero({ icon: "🌲", name: "随机森林 Random Forest", en: "Bootstrap Aggregating of Decision Trees", tags: ["Bagging", "特征采样", "OOB", "特征重要性"], meta: ["◈ T 棵独立树并行", "⚡ 投票 / 平均聚合"] })}

      ${MCH.versionSection("ml_random_forest")}

      <div class="section">
        <h2>1. 核心思想：双重随机 + 聚合</h2>
        <p class="text-sm text-slate-600">单棵决策树方差高、易过拟合。随机森林通过 <b>两层随机</b> 降方差：</p>
        <div class="grid-2">
          <div class="card"><h3 class="text-emerald-700 text-sm font-bold">① 样本 Bootstrap</h3><p class="text-xs text-slate-600 mt-1">每棵树从 N 个样本有放回采样 N 次（期望覆盖 63.2% 独立样本），未被采样的 <b>OOB (Out-of-Bag)</b> 可作天然验证集。</p></div>
          <div class="card"><h3 class="text-emerald-700 text-sm font-bold">② 特征随机</h3><p class="text-xs text-slate-600 mt-1">每次分裂只从 √d (分类) 或 d/3 (回归) 个随机特征中选最优，去除强特征对树结构的主导。</p></div>
        </div>
        <div class="formula-block">
          $$ \\hat y(x) = \\text{mode}_{t=1..T}\\{\\hat y_t(x)\\} \\;(\\text{分类}), \\quad \\hat y(x) = \\frac{1}{T} \\sum_t \\hat y_t(x) \\;(\\text{回归}) $$
          <b>方差缩减：</b> 单树方差 $\\sigma^2$，T 棵<b>近独立</b>树平均方差 → $\\sigma^2 / T$
        </div>
      </div>

      <div class="section">
        <h2>2. 代码解读</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互可视化</h2>
        <div class="grid-2">
          <div>
            <h3>· 树数量 vs 精度收敛</h3>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "rf-T", label: "n_estimators T", min: 1, max: 500, step: 1, value: 100 })}
              ${MCH.slider({ id: "rf-corr", label: "树间相关度 ρ", min: 0, max: 1, step: 0.05, value: 0.3 })}
            </div>
            <div id="chart-conv" style="height:280px;"></div>
          </div>
          <div>
            <h3>· Bootstrap 覆盖率（63.2% 法则）</h3>
            <div id="chart-oob" style="height:280px;"></div>
            <p class="text-xs text-slate-500 mt-2">$P(\\text{被选中}) = 1 - (1 - 1/N)^N \\to 1 - e^{-1} \\approx 63.2\\%$ 当 N 大时。</p>
          </div>
        </div>

        <h3 style="margin-top:18px;">· 特征重要性（示例）</h3>
        <div id="chart-fi" style="height:280px;"></div>
      </div>

      <div class="section">
        <h2>4. 与单棵决策树 / XGBoost 的关系</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>单棵决策树</th><th>随机森林 (Bagging)</th><th>XGBoost (Boosting)</th></tr></thead>
          <tbody>
            <tr><td>集成方式</td><td>单模型</td><td>并行投票</td><td>串行加法</td></tr>
            <tr><td>降方差 or 降偏差</td><td>-</td><td>降<b>方差</b></td><td>降<b>偏差</b></td></tr>
            <tr><td>树间关系</td><td>-</td><td>独立</td><td>依赖（逐棵纠错）</td></tr>
            <tr><td>并行性</td><td>-</td><td>天然并行</td><td>树内并行、树间串行</td></tr>
            <tr><td>过拟合</td><td>严重</td><td>弱</td><td>需正则</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("ml_random_forest").pros,
          MCH.getById("ml_random_forest").cons,
          MCH.getById("ml_random_forest").best_for,
        )}
      </div>

      <div class="section">
        <h2>6. 关键超参</h2>
        ${MCH.hyperTable([
          ["n_estimators", "100~500", "树数量，越多方差越低但收益递减", "通常 200~300 够，看 OOB 曲线"],
          ["max_features", "sqrt / log2 / 0.3", "每次分裂候选特征数", "sqrt(d) 经验最佳"],
          ["max_depth", "None 或 10~30", "单树深度", "None 让树充分生长，靠 bagging 降方差"],
          ["min_samples_leaf", "1~10", "叶子最少样本", "增大可进一步防过拟合"],
          ["bootstrap", "True", "有放回采样", "False 会损失 OOB 优势"],
        ])}
      </div>
    `;
  },

  mount() {
    // Convergence
    const convChart = MCH.echart(document.getElementById("chart-conv"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "T (树数)", min: 1 },
      yAxis: { type: "value", name: "测试误差" },
      series: [
        { name: "独立树平均（低相关度）", type: "line", smooth: true, showSymbol: false, color: "#10b981", lineStyle: { width: 3 }, data: [] },
        { name: "当前相关度 ρ", type: "line", smooth: true, showSymbol: false, color: "#4f46e5", lineStyle: { width: 3 }, data: [] },
      ],
    });
    const updateConv = () => {
      const T = parseInt(document.getElementById("rf-T").value);
      const rho = parseFloat(document.getElementById("rf-corr").value);
      const ts = MCH.linspace(1, 500, 80);
      const low = ts.map(t => 0.15 + 0.02 * Math.sqrt(1 / t));
      const cur = ts.map(t => 0.15 * (rho + (1 - rho) / t) + 0.02);
      convChart.setOption({
        series: [
          { data: ts.map((t, i) => [t, low[i]]) },
          { data: ts.map((t, i) => [t, cur[i]]), markPoint: { data: [{ coord: [T, 0.15 * (rho + (1 - rho) / T) + 0.02], symbol: "circle", symbolSize: 10, itemStyle: { color: "#f59e0b" } }] } },
        ],
      });
    };
    ["rf-T", "rf-corr"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateConv();
      });
    });
    updateConv();

    // Bootstrap coverage
    MCH.echart(document.getElementById("chart-oob"), {
      tooltip: { trigger: "axis" },
      grid: { left: 60, right: 30, top: 30, bottom: 40 },
      xAxis: { type: "log", name: "N (样本量)", logBase: 10 },
      yAxis: { type: "value", name: "P(被选中)", min: 0, max: 1 },
      series: [
        { type: "line", smooth: true, showSymbol: false, color: "#4f46e5", lineStyle: { width: 3 }, data: MCH.linspace(1, 6, 60).map(v => [Math.pow(10, v), 1 - Math.pow(1 - 1 / Math.pow(10, v), Math.pow(10, v))]) },
        { type: "line", smooth: true, showSymbol: false, color: "#94a3b8", lineStyle: { type: "dashed" }, data: MCH.linspace(1, 6, 60).map(v => [Math.pow(10, v), 1 - 1 / Math.E]), markLine: { symbol: "none", data: [{ yAxis: 1 - 1 / Math.E, label: { formatter: "1 - 1/e ≈ 63.2%" } }] } },
      ],
    });

    // Feature importance
    const fi = [0.32, 0.18, 0.14, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.01];
    const fnames = ["GMV_30d", "日均笔数", "场景熵", "对手方唯一度", "夜间占比", "资金入出比", "首笔天数", "退款率", "跨境占比", "同IP商户"];
    MCH.echart(document.getElementById("chart-fi"), {
      tooltip: {},
      grid: { left: 120, right: 40, top: 20, bottom: 30 },
      xAxis: { type: "value", name: "重要性" },
      yAxis: { type: "category", data: fnames.reverse() },
      series: [{ type: "bar", data: fi.reverse(), barWidth: 18, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [{ offset: 0, color: "#a7f3d0" }, { offset: 1, color: "#10b981" }]) }, label: { show: true, position: "right", formatter: v => v.value.toFixed(2) } }],
    });
  },
});
