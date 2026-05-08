/* 模块：Isolation Forest */
MCH.register("ml_isolation_forest", {
  render() {
    const code = `# Isolation Forest — 完全不同的异常检测思路
# Liu, Ting, Zhou, ICDM 2008

# 核心洞察：异常点 vs 正常点
#   - 正常点：被正常点包围，"隔离"它需要很多次随机切分
#   - 异常点：稀疏、远离主体，"隔离"它只需要几次切分
# 所以：异常分数 ∝ 1 / 路径长度

from sklearn.ensemble import IsolationForest
iforest = IsolationForest(n_estimators=100, contamination=0.01, max_samples=256)
iforest.fit(X)
anomaly_score = iforest.decision_function(X)   # 越小越异常
predictions = iforest.predict(X)                # -1=异常, +1=正常

# ============================================================
# 算法细节
# ============================================================
def isolation_tree(X, max_depth):
    """构建一棵 Isolation Tree（完全随机划分）"""
    if len(X) <= 1 or current_depth >= max_depth:
        return Leaf(size=len(X))
    # 随机选一个特征 + 随机选一个分裂值（特征范围内均匀）
    feature = random.randint(0, X.shape[1])
    v_min, v_max = X[:, feature].min(), X[:, feature].max()
    threshold = random.uniform(v_min, v_max)
    left = X[X[:, feature] < threshold]
    right = X[X[:, feature] >= threshold]
    return Node(feature, threshold, isolation_tree(left), isolation_tree(right))


def path_length(x, tree, current_depth=0):
    """一个点穿过树到达叶子的路径长度"""
    if tree.is_leaf:
        return current_depth + c(tree.size)     # c(n) 是 BST 平均路径长度修正
    if x[tree.feature] < tree.threshold:
        return path_length(x, tree.left, current_depth + 1)
    return path_length(x, tree.right, current_depth + 1)


def anomaly_score(x, forest, n_samples=256):
    """异常分数 s(x, n) = 2^(-E[h(x)] / c(n))"""
    avg_path = mean(path_length(x, tree) for tree in forest)
    return 2 ** (-avg_path / c(n_samples))
# s → 1  表示异常（路径短，容易隔离）
# s → 0.5 表示正常`;

    return `
      ${MCH.hero({
        icon: "⚠",
        name: "Isolation Forest 异常检测",
        en: "Isolation Forest (Liu et al., ICDM 2008)",
        tags: ["无监督异常检测", "随机切分", "路径长度", "O(n log n)"],
        meta: ["◈ 工业级异常检测首选", "⚡ 比距离/密度类算法快"],
      })}

      ${MCH.versionSection("ml_isolation_forest")}

      <div class="section">
        <h2>1. 颠覆性思路：不建模"正常"，而是"直接找孤立的"</h2>
        <p class="text-sm text-slate-600">传统异常检测（LOF、One-Class SVM 等）先建模"正常"分布，再看偏差。但异常本身很稀少，建模数据集本身困难。</p>
        ${MCH.info(`
          <b>Isolation Forest 的反向思维</b>：异常点天然"容易隔离"——
          它们稀疏、远离主体，随机切分几次就能把它与其他点分开；
          正常点则被密集邻居包围，需要很多次才能切开。
          <br/><br/>
          所以<b>隔离所需的次数（树路径长度）本身就是异常信号</b> — 这是算法核心洞察。
        `, "tip")}
        <div class="formula-block">
          <b>异常分数</b>：$s(x, n) = 2^{-\\frac{E[h(x)]}{c(n)}}$<br/>
          $h(x)$ 是 x 在多棵树上的平均路径长度；$c(n) = 2 H(n-1) - 2(n-1)/n$ 是 BST 平均路径<br/>
          → $s \\to 1$ : 路径很短 → <b>异常</b> / $s \\to 0.5$ : 正常
        </div>
      </div>

      <div class="section">
        <h2>2. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：Isolation Forest 异常打分</h2>
        <p class="text-sm text-slate-600">下面数据是 2 个高斯 cluster + 若干离群点。调节 contamination 观察哪些点被判为异常。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "if-cont", label: "contamination", min: 0.01, max: 0.3, step: 0.01, value: 0.1 })}
              ${MCH.slider({ id: "if-trees", label: "n_estimators T", min: 10, max: 500, step: 10, value: 100 })}
              ${MCH.slider({ id: "if-max", label: "max_samples ψ", min: 32, max: 512, step: 32, value: 256 })}
            </div>
            <div id="if-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-if" style="height:400px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. 与其他异常检测算法对比</h2>
        <table class="table">
          <thead><tr><th>算法</th><th>原理</th><th>复杂度</th><th>适合</th></tr></thead>
          <tbody>
            <tr><td><b>Isolation Forest</b></td><td>随机切分路径长度</td><td>O(n log n)</td><td>大数据、高维、工业通用</td></tr>
            <tr><td>LOF (Local Outlier Factor)</td><td>局部密度偏差</td><td>O(n²) 精确 / O(n log n) 近似</td><td>局部异常（密度变化）</td></tr>
            <tr><td>One-Class SVM</td><td>找一个超平面隔离主体</td><td>O(n²)~O(n³)</td><td>小数据高维</td></tr>
            <tr><td>Elliptic Envelope</td><td>椭球鲁棒协方差估计</td><td>O(n·d²)</td><td>高斯近似数据</td></tr>
            <tr><td>AutoEncoder</td><td>重构误差</td><td>DL 训练</td><td>图像 / 大数据</td></tr>
            <tr><td>PyOD 框架</td><td>统一接口，集成 30+</td><td>-</td><td>算法选型 / baseline 对比</td></tr>
            <tr><td>Extended IF</td><td>随机超平面代替坐标轴切分</td><td>同 IF</td><td>IF 2.0 版本</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 参数调优经验</h2>
        <table class="table">
          <thead><tr><th>参数</th><th>典型范围</th><th>调优建议</th></tr></thead>
          <tbody>
            <tr><td><code>n_estimators</code></td><td>100-500</td><td>更多树更稳定，通常 100 够</td></tr>
            <tr><td><code>max_samples</code></td><td>"auto" (=256)</td><td>论文推荐固定 256，大数据不用增加</td></tr>
            <tr><td><code>contamination</code></td><td>0.01-0.1</td><td>估计的异常比例，若未知用 "auto"</td></tr>
            <tr><td><code>max_features</code></td><td>1.0 默认</td><td>高维可调小，增加树间多样性</td></tr>
            <tr><td><code>random_state</code></td><td>固定值</td><td>保证结果可复现</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 业务应用场景</h2>
        <div class="grid-2">
          <div class="card"><div class="tag tag-red">🛡 风控</div><h3 class="font-bold mt-2">信用卡异常交易</h3><p class="text-xs text-slate-600 mt-1">特征：金额 / 时间 / 商户类型 / 地理位置。IF 无需历史异常标签，适合冷启动。</p></div>
          <div class="card"><div class="tag tag-amber">🌐 网络</div><h3 class="font-bold mt-2">入侵检测 (IDS)</h3><p class="text-xs text-slate-600 mt-1">流量五元组特征，IF 识别异常连接模式。</p></div>
          <div class="card"><div class="tag">🏭 IoT</div><h3 class="font-bold mt-2">传感器异常</h3><p class="text-xs text-slate-600 mt-1">工业设备故障预测，IF 对未知异常模式友好。</p></div>
          <div class="card"><div class="tag tag-green">💰 商户</div><h3 class="font-bold mt-2">支付商户行为异常</h3><p class="text-xs text-slate-600 mt-1">交易序列 + 统计特征，识别刷单 / 套现等。</p></div>
        </div>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ml_isolation_forest").pros, MCH.getById("ml_isolation_forest").cons, MCH.getById("ml_isolation_forest").best_for)}
      </div>
    `;
  },

  mount() {
    // 生成数据：2 cluster + outliers
    const data = [];
    for (let i = 0; i < 60; i++) {
      const r = MCH.randn(2, i + 1);
      data.push({ pt: [2 + r[0] * 0.6, 2 + r[1] * 0.6], truth: "normal" });
    }
    for (let i = 0; i < 60; i++) {
      const r = MCH.randn(2, i + 100);
      data.push({ pt: [-2 + r[0] * 0.6, -1 + r[1] * 0.6], truth: "normal" });
    }
    // outliers
    const outliers = [[4, 4], [4, -3], [-4, 3], [0, 4], [-3.5, -3.5], [5, 0], [-4.5, 0], [0, -4], [4.5, -4], [2, -4]];
    outliers.forEach(p => data.push({ pt: p, truth: "outlier" }));

    const computeScore = (pt) => {
      // 模拟 IF score：离 cluster 中心越远分数越高
      const d1 = Math.hypot(pt[0] - 2, pt[1] - 2);
      const d2 = Math.hypot(pt[0] + 2, pt[1] + 1);
      const minD = Math.min(d1, d2);
      // 归一化到 [0, 1]，异常 → 1
      return Math.min(1, minD / 5);
    };

    const update = () => {
      const cont = parseFloat(document.getElementById("if-cont").value);
      const trees = parseInt(document.getElementById("if-trees").value);
      const psi = parseInt(document.getElementById("if-max").value);
      // 计算分数 + 取 top-k 作为异常
      const scored = data.map(d => ({ ...d, score: computeScore(d.pt) + (Math.random() - 0.5) * 0.05 * (500 / trees) }));
      const sortedByScore = [...scored].sort((a, b) => b.score - a.score);
      const threshold = Math.floor(data.length * cont);
      const anomalySet = new Set(sortedByScore.slice(0, threshold).map(d => d.pt.join(",")));
      const normal = scored.filter(d => !anomalySet.has(d.pt.join(",")));
      const anomalies = scored.filter(d => anomalySet.has(d.pt.join(",")));

      MCH.echart(document.getElementById("chart-if"), {
        tooltip: { formatter: p => p.data.truth ? `(${p.data.pt[0].toFixed(2)}, ${p.data.pt[1].toFixed(2)})<br/>分数=${p.data.score.toFixed(3)}<br/>真实=${p.data.truth}` : "" },
        legend: { top: 0, data: ["判为正常", "判为异常"] },
        grid: { left: 50, right: 30, top: 40, bottom: 40 },
        xAxis: { type: "value", min: -6, max: 6 },
        yAxis: { type: "value", min: -5, max: 5 },
        series: [
          { name: "判为正常", type: "scatter", data: normal.map(d => ({ value: d.pt, pt: d.pt, score: d.score, truth: d.truth })), itemStyle: { color: "#94a3b8" }, symbolSize: 6 },
          { name: "判为异常", type: "scatter", data: anomalies.map(d => ({ value: d.pt, pt: d.pt, score: d.score, truth: d.truth })), itemStyle: { color: "#ef4444" }, symbol: "pin", symbolSize: 18 },
        ],
      });

      // 统计 TP / FP
      const tp = anomalies.filter(d => d.truth === "outlier").length;
      const fp = anomalies.filter(d => d.truth === "normal").length;
      const trueOutliers = data.filter(d => d.truth === "outlier").length;
      const recall = trueOutliers ? (tp / trueOutliers * 100).toFixed(0) : 0;
      const precision = (tp + fp) ? (tp / (tp + fp) * 100).toFixed(0) : 0;
      document.getElementById("if-info").innerHTML = `
        <b>contamination</b>=${cont} · <b>T</b>=${trees} · <b>ψ</b>=${psi}<br/>
        <b>检测出异常</b>: ${anomalies.length} 个 (True Pos=${tp}, False Pos=${fp})<br/>
        <b>Recall (召回)</b>: <span style="color:#10b981;">${recall}%</span> ·
        <b>Precision (精度)</b>: <span style="color:#4f46e5;">${precision}%</span><br/>
        <span style="color:#64748b;">contamination 设为真实异常比例时，recall/precision 最平衡。</span>
      `;
    };
    ["if-cont", "if-trees", "if-max"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
