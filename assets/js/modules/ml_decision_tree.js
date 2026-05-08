/* 模块：决策树 */
MCH.register("ml_decision_tree", {
  render() {
    const code = `# 决策树（CART 版本）核心逻辑 —— sklearn 风格伪代码
def build_tree(X, y, depth=0, max_depth=10, min_samples_split=2):
    # 终止条件：纯节点 / 样本过少 / 达到最大深度
    if len(y) < min_samples_split or len(set(y)) == 1 or depth >= max_depth:
        return Leaf(prediction=majority_class(y))

    # 遍历每个特征和候选分裂点，找最小不纯度分裂
    best_gain, best_feat, best_thr = 0, None, None
    for feat in range(X.shape[1]):
        thresholds = unique_sorted(X[:, feat])
        for thr in thresholds:
            left_mask = X[:, feat] <= thr
            gain = impurity(y) - weighted_impurity(y[left_mask], y[~left_mask])
            if gain > best_gain:
                best_gain, best_feat, best_thr = gain, feat, thr

    if best_gain <= 0:
        return Leaf(prediction=majority_class(y))

    # 递归分裂
    left_mask = X[:, best_feat] <= best_thr
    return Node(
        feature=best_feat, threshold=best_thr,
        left =build_tree(X[left_mask],  y[left_mask],  depth+1, ...),
        right=build_tree(X[~left_mask], y[~left_mask], depth+1, ...),
    )

# 不纯度指标
def gini(y):      # CART 默认
    probs = np.bincount(y) / len(y)
    return 1.0 - np.sum(probs ** 2)

def entropy(y):   # ID3 / C4.5 用
    probs = np.bincount(y) / len(y)
    return -np.sum(probs * np.log2(probs + 1e-12))`;

    return `
      ${MCH.hero({ icon: "🌳", name: "决策树 Decision Tree", en: "CART / ID3 / C4.5", tags: ["贪心", "不纯度最小", "可解释", "非参数"], meta: ["◈ 训练 O(n·d·log n)", "⚡ 推理 O(tree depth)"] })}

      ${MCH.versionSection("ml_decision_tree")}

      <div class="section">
        <h2>1. 核心思想</h2>
        <p class="text-sm text-slate-600">决策树递归地选择"最能分开两边样本"的特征和阈值，通过贪心搜索把特征空间切成一系列矩形区域，每个区域用多数类或均值作为预测。</p>
        <div class="formula-block">
          <b>Gini 不纯度（CART）：</b>
          $$ \\text{Gini}(S) = 1 - \\sum_{k=1}^{K} p_k^2, \\quad p_k = \\frac{|\\{y=k\\}|}{|S|} $$
          <b>信息增益（ID3）：</b>
          $$ \\text{Gain}(S, A) = H(S) - \\sum_{v} \\frac{|S_v|}{|S|} H(S_v), \\quad H(S) = -\\sum_k p_k \\log_2 p_k $$
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
            <h3>· 不纯度函数对比</h3>
            <div class="ctrl-panel">
              <p class="text-xs text-slate-500">二分类 p=p(y=1)，对比 Gini / Entropy / Misclassification。</p>
            </div>
            <div id="chart-impurity" style="height:300px;"></div>
          </div>
          <div>
            <h3>· 深度 vs 过拟合</h3>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "dt-depth", label: "max_depth", min: 1, max: 20, step: 1, value: 5 })}
              ${MCH.slider({ id: "dt-n", label: "样本量 log10(N)", min: 2, max: 6, step: 0.1, value: 3 })}
            </div>
            <div id="chart-depth" style="height:260px;"></div>
          </div>
        </div>

        <h3 style="margin-top:18px;">· 2D 决策边界（模拟）</h3>
        <p class="text-xs text-slate-500">改变 max_depth 观察决策边界如何从简单矩形逐步细化为复杂阶梯。</p>
        <div id="chart-boundary" style="height:380px;"></div>
      </div>

      <div class="section">
        <h2>4. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("ml_decision_tree").pros,
          MCH.getById("ml_decision_tree").cons,
          MCH.getById("ml_decision_tree").best_for,
        )}
      </div>

      <div class="section">
        <h2>5. 关键超参</h2>
        ${MCH.hyperTable([
          ["max_depth", "3~10", "限制树深，控制复杂度", "从小往大试，配合交叉验证"],
          ["min_samples_split", "2~50", "节点分裂最小样本数", "大数据增大此值防过拟合"],
          ["criterion", "gini / entropy", "不纯度度量", "通常 gini 更快、entropy 更准确但差异小"],
          ["max_features", "None / sqrt / log2", "每次分裂的特征候选数", "随机森林中常用 sqrt"],
          ["min_impurity_decrease", "0.0~0.01", "停止分裂的最小收益", "防止微小增益导致的过拟合"],
        ])}
      </div>
    `;
  },

  mount() {
    // 不纯度对比
    MCH.echart(document.getElementById("chart-impurity"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "p(y=1)", min: 0, max: 1 },
      yAxis: { type: "value", name: "impurity" },
      series: [
        { name: "Gini = 2p(1-p)", type: "line", showSymbol: false, smooth: true, color: "#4f46e5", lineStyle: { width: 3 }, data: MCH.linspace(0.001, 0.999, 100).map(p => [p, 2 * p * (1 - p)]) },
        { name: "Entropy = -p·log2 p - (1-p)·log2(1-p)", type: "line", showSymbol: false, smooth: true, color: "#f59e0b", lineStyle: { width: 3 }, data: MCH.linspace(0.001, 0.999, 100).map(p => [p, -(p * Math.log2(p) + (1 - p) * Math.log2(1 - p))]) },
        { name: "Misclass = 1 - max(p, 1-p)", type: "line", showSymbol: false, smooth: true, color: "#10b981", lineStyle: { width: 3 }, data: MCH.linspace(0.001, 0.999, 100).map(p => [p, 1 - Math.max(p, 1 - p)]) },
      ],
    });

    // Depth vs Overfit
    const depthChart = MCH.echart(document.getElementById("chart-depth"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "max_depth", min: 1, max: 20 },
      yAxis: { type: "value", name: "准确率", min: 0.5, max: 1 },
      series: [
        { name: "训练准确率", type: "line", smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
        { name: "测试准确率", type: "line", smooth: true, data: [], color: "#ef4444", lineStyle: { width: 3 } },
      ],
    });
    const updateDepth = () => {
      const N = Math.pow(10, parseFloat(document.getElementById("dt-n").value));
      const depthMark = parseInt(document.getElementById("dt-depth").value);
      const ds = MCH.linspace(1, 20, 20);
      const train = ds.map(d => 1 - Math.exp(-d * 0.6));
      // test accuracy curve depends on N: small N peaks earlier
      const peak = 3 + Math.log10(N) * 2;
      const test = ds.map(d => 1 - 0.4 * Math.exp(-d * 0.5) - 0.04 * Math.max(0, d - peak));
      depthChart.setOption({
        series: [
          { data: ds.map((d, i) => [d, train[i]]), markLine: { symbol: "none", data: [{ xAxis: depthMark, lineStyle: { color: "#94a3b8", type: "dashed" }, label: { formatter: "当前" } }] } },
          { data: ds.map((d, i) => [d, test[i]]) },
        ],
      });
    };
    ["dt-depth", "dt-n"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateDepth();
      });
    });
    updateDepth();

    // 2D Boundary: 用方块模拟 depth 细化的决策边界
    const boundEl = document.getElementById("chart-boundary");
    const renderBoundary = () => {
      const depth = parseInt(document.getElementById("dt-depth").value);
      const G = 40;
      const data = [];
      const splitAt = Math.max(1, Math.pow(2, Math.min(depth, 5)));
      for (let i = 0; i < G; i++) {
        for (let j = 0; j < G; j++) {
          const x = i / G, y = j / G;
          // 模拟多层嵌套分裂：不同 depth 对应不同块状边界
          let label;
          if (depth <= 1) label = (x > 0.5) ? 1 : 0;
          else if (depth <= 3) label = ((x > 0.5) !== (y > 0.5)) ? 1 : 0;
          else {
            const bx = Math.floor(x * splitAt) / splitAt;
            const by = Math.floor(y * splitAt) / splitAt;
            const target = Math.pow((bx + by - 1), 2) + Math.pow(bx - by, 2);
            label = target < 0.3 ? 1 : 0;
            if (Math.random() < 0.03) label = 1 - label;
          }
          data.push([i, j, label]);
        }
      }
      MCH.echart(boundEl, {
        tooltip: { formatter: p => `区域 (${p.data[0]}, ${p.data[1]})<br/>预测: ${p.data[2]}` },
        grid: { left: 30, right: 30, top: 20, bottom: 30 },
        xAxis: { type: "category", data: [...Array(G).keys()], axisLabel: { show: false } },
        yAxis: { type: "category", data: [...Array(G).keys()], axisLabel: { show: false } },
        visualMap: { min: 0, max: 1, calculable: false, orient: "horizontal", left: "center", bottom: 0, inRange: { color: ["#e0e7ff", "#4f46e5"] } },
        series: [{ type: "heatmap", data, progressive: 0 }],
      });
    };
    document.getElementById("dt-depth").addEventListener("input", renderBoundary);
    renderBoundary();
  },
});
