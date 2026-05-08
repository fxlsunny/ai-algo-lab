/* 模块：XGBoost */
MCH.register("ml_xgboost", {
  render() {
    const code = `# XGBoost 的核心：二阶泰勒 + 正则化
# Chen & Guestrin, KDD 2016

# Step 0：初始预测 y_hat_0 = 常数（类别频率 log-odds 或均值）

for t in range(num_boost_round):          # 串行增加第 t 棵树
    # Step 1：在当前预测上计算每个样本的一阶 / 二阶导数
    g = [ dL/d y_hat  for each sample ]   # 一阶梯度
    h = [ d²L/d y_hat²  for each sample ] # 二阶 Hessian

    # Step 2：用 g, h 作为"软标签"训练一棵 CART 树
    #         分裂准则：增益 Gain 不再用 Gini，而是由 g,h 推导的最大化下式
    #         Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) - (G_L+G_R)²/(H_L+H_R+λ) ] - γ
    tree_t = build_tree_using(g, h)

    # Step 3：更新预测（带学习率收缩 η 防止单棵树过度主导）
    y_hat += eta * tree_t.predict(X)

# 正则化目标函数：
#   Obj = Σ L(y, y_hat) + Σ Ω(f_t),
#   Ω(f) = γ·T + ½·λ·||w||²
#   —— γ 控制叶子数、λ 控制叶子权重 L2，双重防过拟合`;

    return `
      ${MCH.hero({ icon: "🚀", name: "XGBoost", en: "eXtreme Gradient Boosting", tags: ["二阶泰勒", "正则化", "Kaggle 冠军", "缺失值方向"], meta: ["◈ Chen & Guestrin KDD'16", "⚡ GPU 加速"] })}

      ${MCH.versionSection("ml_xgboost")}

      <div class="section">
        <h2>1. 从 GBDT 到 XGBoost 的三大升级</h2>
        <div class="grid-3">
          <div class="card"><h3 class="text-indigo-700 font-bold text-sm">① 二阶泰勒</h3><p class="text-xs text-slate-600 mt-2">GBDT 只用一阶梯度拟合，XGBoost 使用 <b>(g, h)</b> 两阶信息，类比牛顿法，收敛更快。</p></div>
          <div class="card"><h3 class="text-indigo-700 font-bold text-sm">② 正则化</h3><p class="text-xs text-slate-600 mt-2">目标函数显式加入 $\\gamma T + \\tfrac{1}{2}\\lambda \\|w\\|^2$，从数学上防过拟合，而非只靠早停。</p></div>
          <div class="card"><h3 class="text-indigo-700 font-bold text-sm">③ 系统优化</h3><p class="text-xs text-slate-600 mt-2">列块并行、缓存感知、稀疏感知（自动学习缺失值方向）、近似直方图。</p></div>
        </div>
        <div class="formula-block">
          <b>目标函数：</b>
          $$ \\mathcal{L}^{(t)} = \\sum_i l\\Big(y_i, \\hat y_i^{(t-1)} + f_t(x_i)\\Big) + \\Omega(f_t) $$
          <b>二阶泰勒展开（t 轮）：</b>
          $$ \\mathcal{L}^{(t)} \\approx \\sum_i [\\,g_i f_t(x_i) + \\tfrac{1}{2} h_i f_t^2(x_i)\\,] + \\gamma T + \\tfrac{1}{2}\\lambda \\sum_j w_j^2 $$
          <b>叶子最优权重与 Gain：</b>
          $$ w_j^* = -\\frac{\\sum_{i \\in I_j} g_i}{\\sum_{i \\in I_j} h_i + \\lambda}, \\quad \\text{Gain} = \\frac{1}{2}\\Big[ \\frac{G_L^2}{H_L + \\lambda} + \\frac{G_R^2}{H_R + \\lambda} - \\frac{(G_L+G_R)^2}{H_L + H_R + \\lambda} \\Big] - \\gamma $$
        </div>
      </div>

      <div class="section">
        <h2>2. 核心代码逻辑</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：Boosting 迭代收敛</h2>
        <p class="text-sm text-slate-600">模拟 1D 回归问题：每棵新树拟合前一轮的残差（经 g/h 加权）。观察 learning_rate 与 num_boost_round 的权衡。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "xgb-T", label: "num_boost_round T", min: 1, max: 100, step: 1, value: 20 })}
              ${MCH.slider({ id: "xgb-eta", label: "learning_rate η", min: 0.01, max: 1.0, step: 0.01, value: 0.3 })}
              ${MCH.slider({ id: "xgb-depth", label: "max_depth", min: 1, max: 10, step: 1, value: 4 })}
            </div>
            <div id="xgb-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-boost" style="height:360px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. 缺失值方向自动学习</h2>
        <p class="text-sm text-slate-600">XGBoost 在每次分裂时，会 <b>分别尝试"缺失值全去左"和"缺失值全去右"</b>，选择收益最大的方向。这让模型对稀疏特征/缺失值天然鲁棒。</p>
        <div class="mermaid">
flowchart LR
    N[分裂节点 feat=gmv]
    N -->|非缺失 gmv 不大于 1000| L[左子]
    N -->|非缺失 gmv 大于 1000| R[右子]
    N -->|缺失值 default| D{"default_direction<br/>= arg max (Gain_left, Gain_right)"}
    D -.-> L
    D -.-> R
        </div>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("ml_xgboost").pros,
          MCH.getById("ml_xgboost").cons,
          MCH.getById("ml_xgboost").best_for,
        )}
      </div>

      <div class="section">
        <h2>6. 关键超参</h2>
        ${MCH.hyperTable([
          ["learning_rate (η)", "0.01 ~ 0.3", "每棵树贡献收缩", "小 η 需配大 n_estimators，通常 0.05 起步"],
          ["max_depth", "4 ~ 10", "单树深度，复杂度核心控制", "结构化数据 6 最常用"],
          ["n_estimators", "100 ~ 3000", "树数量", "配合 early_stopping_rounds"],
          ["subsample", "0.7 ~ 1.0", "行采样比例", "增加多样性防过拟合"],
          ["colsample_bytree", "0.6 ~ 1.0", "列采样比例", "类似 RF 特征随机"],
          ["min_child_weight", "1 ~ 20", "叶子最小样本 Hessian 和", "大数据可增大"],
          ["reg_alpha (α)", "0 ~ 1", "L1 正则（权重稀疏）", "类别特征多时增大"],
          ["reg_lambda (λ)", "0 ~ 10", "L2 正则（权重收缩）", "默认 1，通常足够"],
          ["gamma (γ)", "0 ~ 5", "最小分裂收益阈值", "大数据增大防噪声分裂"],
          ["tree_method", "hist / gpu_hist", "树构建方式", "大数据必用直方图"],
        ])}
      </div>
    `;
  },

  mount() {
    const boostChart = MCH.echart(document.getElementById("chart-boost"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "x" },
      yAxis: { type: "value" },
      series: [
        { name: "真实函数 f*", type: "line", smooth: true, showSymbol: false, color: "#94a3b8", lineStyle: { width: 2, type: "dashed" }, data: [] },
        { name: "Boosting 集成预测", type: "line", smooth: true, showSymbol: false, color: "#4f46e5", lineStyle: { width: 3 }, data: [] },
        { name: "当前残差", type: "line", smooth: true, showSymbol: false, color: "#ef4444", lineStyle: { width: 1.5, type: "dotted" }, data: [] },
      ],
    });
    const truFn = x => Math.sin(x * 2) + 0.5 * Math.cos(x * 5) * Math.exp(-x / 3);
    const xs = MCH.linspace(0, 5, 200);
    const update = () => {
      const T = parseInt(document.getElementById("xgb-T").value);
      const eta = parseFloat(document.getElementById("xgb-eta").value);
      const depth = parseInt(document.getElementById("xgb-depth").value);
      const tru = xs.map(truFn);
      const pred = xs.map(() => 0);
      // 模拟 Boosting：每轮拟合残差的 piecewise constant
      const splitN = Math.pow(2, depth);
      for (let t = 0; t < T; t++) {
        // 当前残差
        const resid = xs.map((x, i) => tru[i] - pred[i]);
        // 按区间均值拟合
        const buckets = new Array(splitN).fill(0);
        const counts = new Array(splitN).fill(0);
        xs.forEach((x, i) => {
          const b = Math.min(splitN - 1, Math.floor(x / 5 * splitN));
          buckets[b] += resid[i]; counts[b]++;
        });
        const means = buckets.map((s, i) => s / Math.max(1, counts[i]));
        xs.forEach((x, i) => {
          const b = Math.min(splitN - 1, Math.floor(x / 5 * splitN));
          pred[i] += eta * means[b];
        });
      }
      const finalResid = xs.map((x, i) => tru[i] - pred[i]);
      const rmse = Math.sqrt(finalResid.reduce((s, r) => s + r * r, 0) / xs.length);
      boostChart.setOption({
        series: [
          { data: xs.map((x, i) => [x, tru[i]]) },
          { data: xs.map((x, i) => [x, pred[i]]) },
          { data: xs.map((x, i) => [x, finalResid[i]]) },
        ],
      });
      document.getElementById("xgb-info").innerHTML = `
        <b>当前 RMSE</b> = <span style="color:#4f46e5;">${rmse.toFixed(4)}</span><br/>
        η=${eta}, T=${T}, depth=${depth}，有效模型容量 ≈ ${splitN} × ${T} = ${splitN * T} 段<br/>
        <span class="text-slate-500">减小 η + 增加 T 通常泛化更好，但训练慢；增加 depth 则树更复杂。</span>
      `;
    };
    ["xgb-T", "xgb-eta", "xgb-depth"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
