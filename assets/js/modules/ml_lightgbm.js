/* 模块：LightGBM */
MCH.register("ml_lightgbm", {
  render() {
    const code = `# LightGBM 的四大创新
# Ke et al., NeurIPS 2017

# ① Histogram-based 直方图决策：
#    把连续特征离散化到 k 个桶（默认 255），分裂时只扫 k 个候选
#    而不是 n 个样本，时间复杂度 O(n·d) → O(d·bins)
def histogram_split(X_col, g, h, bins=255):
    hist_g = np.zeros(bins); hist_h = np.zeros(bins); hist_n = np.zeros(bins)
    for i in range(len(X_col)):
        b = int(X_col[i])       # 预先分箱
        hist_g[b] += g[i]; hist_h[b] += h[i]; hist_n[b] += 1
    # 用前缀和求最优分裂点
    best_gain, best_bin = -np.inf, None
    G_sum, H_sum = hist_g.sum(), hist_h.sum()
    G_L = H_L = 0
    for b in range(bins - 1):
        G_L += hist_g[b]; H_L += hist_h[b]
        G_R = G_sum - G_L; H_R = H_sum - H_L
        gain = 0.5 * (G_L**2/(H_L+lam) + G_R**2/(H_R+lam) - G_sum**2/(H_sum+lam)) - gamma
        if gain > best_gain: best_gain, best_bin = gain, b
    return best_gain, best_bin

# ② Leaf-wise（而非 level-wise）：每次从所有叶子里选增益最大的一个分裂
#    → 同参数下深度可能更深、收敛更快，但小数据易过拟合（需 max_depth 约束）

# ③ GOSS (Gradient-based One-Side Sampling)：
#    梯度大的样本 100% 保留（关键样本）
#    梯度小的样本随机采 b% 并乘以 (1-a)/b 权重修正
def goss_sample(g, top_rate=0.2, other_rate=0.1):
    abs_g = np.abs(g)
    top_n = int(len(g) * top_rate)
    top_idx = np.argsort(abs_g)[-top_n:]                 # Top-a 梯度
    other_idx = np.random.choice(len(g) - top_n, int(len(g) * other_rate))
    weights = np.ones(len(g))
    weights[other_idx] = (1 - top_rate) / other_rate     # 权重修正
    return np.concatenate([top_idx, other_idx]), weights

# ④ EFB (Exclusive Feature Bundling)：
#    把"几乎不同时非零"的稀疏特征捆绑到一个特征，降低有效特征维度`;

    return `
      ${MCH.hero({ icon: "⚡", name: "LightGBM", en: "Light Gradient Boosting Machine", tags: ["直方图", "Leaf-wise", "GOSS", "EFB"], meta: ["◈ 比 XGB 快 3-10×", "⚡ 工业首选"] })}

      ${MCH.versionSection("ml_lightgbm")}

      <div class="section">
        <h2>1. LightGBM vs XGBoost 核心差异</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>XGBoost (hist)</th><th>LightGBM</th><th>收益</th></tr></thead>
          <tbody>
            <tr><td>树生长</td><td>Level-wise（层序）</td><td><b>Leaf-wise</b>（最大增益优先）</td><td>同参数下拟合更充分</td></tr>
            <tr><td>特征采样</td><td>-</td><td><b>GOSS</b> 保留大梯度 + 采小梯度</td><td>大数据训练 2× 快</td></tr>
            <tr><td>稀疏特征</td><td>缺失方向学习</td><td><b>EFB</b> 互斥特征捆绑</td><td>有效特征维度下降</td></tr>
            <tr><td>类别特征</td><td>需手动 one-hot</td><td><b>原生支持</b>（many-vs-many 分裂）</td><td>类别基数高时大幅提速</td></tr>
            <tr><td>分布式</td><td>允许</td><td>支持 <b>Voting Parallel</b> 通信量小</td><td>TB 级数据可跑</td></tr>
            <tr><td>GPU</td><td>gpu_hist</td><td>cuda_exp 实验</td><td>不相上下</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. 代码解读（四大创新）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：Leaf-wise vs Level-wise 对比</h2>
        <p class="text-sm text-slate-600">同样分裂 N 次：Level-wise 按层展开，得到平衡树；Leaf-wise 按最大增益展开，可能得到不对称但更深的树。</p>
        <div class="ctrl-panel">
          ${MCH.slider({ id: "lgb-splits", label: "总分裂次数 N", min: 0, max: 20, step: 1, value: 8 })}
        </div>
        <div class="grid-2 mt-3">
          <div><h3>Level-wise（XGBoost hist / 传统）</h3><div id="chart-level" style="height:320px;"></div></div>
          <div><h3>Leaf-wise（LightGBM）</h3><div id="chart-leaf" style="height:320px;"></div></div>
        </div>
      </div>

      <div class="section">
        <h2>4. 交互：直方图分桶提速</h2>
        <p class="text-sm text-slate-600">同样在 N 个样本上找最优分裂点：Exact 方法扫 N-1 个候选；Histogram 方法只扫 bins 个候选。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "lgb-N", label: "样本量 log10(N)", min: 3, max: 8, step: 0.1, value: 6 })}
              ${MCH.slider({ id: "lgb-bins", label: "bins", min: 32, max: 512, step: 32, value: 255 })}
            </div>
            <div id="lgb-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-hist-speed" style="height:300px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("ml_lightgbm").pros,
          MCH.getById("ml_lightgbm").cons,
          MCH.getById("ml_lightgbm").best_for,
        )}
      </div>

      <div class="section">
        <h2>6. 关键超参（LightGBM 专用）</h2>
        ${MCH.hyperTable([
          ["num_leaves", "31 ~ 255", "叶子数（取代 max_depth）", "num_leaves ≤ 2^max_depth，默认 31；大数据可调 127"],
          ["min_data_in_leaf", "20 ~ 200", "叶子最小样本", "大数据增大防过拟合，关键超参"],
          ["learning_rate", "0.01 ~ 0.1", "同 XGBoost", "大数据用 0.05"],
          ["feature_fraction", "0.7 ~ 0.9", "列采样", "同 XGB colsample_bytree"],
          ["bagging_fraction", "0.7 ~ 0.9", "行采样", "+ bagging_freq=5 启用"],
          ["lambda_l1 / lambda_l2", "0 ~ 10", "L1/L2 正则", "类别特征多时 L1 偏大"],
          ["categorical_feature", "'auto' / 列名", "原生类别特征", "不要提前 label-encode 再标为 categorical"],
          ["boost_from_average", "True", "回归任务从均值起步", "一般保持 True"],
          ["early_stopping_round", "50 ~ 200", "验证集停止轮数", "必用，防止过拟合"],
        ])}
      </div>
    `;
  },

  mount() {
    // Tree structure viz
    const renderTree = (el, mode, N) => {
      // 生成节点/连线
      const nodes = [{ id: 0, x: 0.5, y: 0, label: "root", gain: 1.0 }];
      const links = [];
      const leafs = [0];
      for (let s = 0; s < N; s++) {
        let pickIdx;
        if (mode === "level") {
          // 最浅的叶子先分裂
          pickIdx = 0;
          let minDepth = Infinity;
          leafs.forEach((lid, idx) => {
            const d = nodes[lid].y;
            if (d < minDepth) { minDepth = d; pickIdx = idx; }
          });
        } else {
          // 最大 gain 的叶子
          pickIdx = 0;
          let maxG = -Infinity;
          leafs.forEach((lid, idx) => { if (nodes[lid].gain > maxG) { maxG = nodes[lid].gain; pickIdx = idx; } });
        }
        const parentId = leafs[pickIdx];
        const p = nodes[parentId];
        const spread = 0.5 / Math.pow(2, p.y + 1);
        const L = { id: nodes.length, x: p.x - spread, y: p.y + 1, gain: p.gain * (0.4 + Math.random() * 0.6) };
        const R = { id: nodes.length + 1, x: p.x + spread, y: p.y + 1, gain: p.gain * (0.4 + Math.random() * 0.6) };
        nodes.push(L, R);
        links.push({ source: parentId, target: L.id }, { source: parentId, target: R.id });
        leafs.splice(pickIdx, 1);
        leafs.push(L.id, R.id);
      }
      const maxDepth = Math.max(...nodes.map(n => n.y));
      MCH.echart(el, {
        tooltip: { formatter: p => p.dataType === "node" ? `depth=${p.data.y}<br/>gain=${p.data.gain?.toFixed(2) || "-"}` : "" },
        xAxis: { show: false, min: 0, max: 1 },
        yAxis: { show: false, min: -0.5, max: maxDepth + 0.5, inverse: true },
        grid: { left: 10, right: 10, top: 10, bottom: 10 },
        series: [{
          type: "graph", coordinateSystem: "cartesian2d",
          data: nodes.map(n => ({
            name: "n" + n.id,
            value: [n.x, n.y],
            symbol: "circle",
            symbolSize: leafs.includes(n.id) ? 12 : 18,
            itemStyle: {
              color: n.id === 0 ? "#4f46e5" : (leafs.includes(n.id) ? "#10b981" : "#c7d2fe"),
            },
          })),
          links: links.map(l => ({ source: "n" + l.source, target: "n" + l.target })),
          lineStyle: { color: "#94a3b8", width: 1.5 },
        }],
      });
    };
    const updateTree = () => {
      const N = parseInt(document.getElementById("lgb-splits").value);
      renderTree(document.getElementById("chart-level"), "level", N);
      renderTree(document.getElementById("chart-leaf"), "leaf", N);
    };
    document.getElementById("lgb-splits").addEventListener("input", (e) => {
      document.getElementById("lgb-splits-val").textContent = e.target.value;
      updateTree();
    });
    updateTree();

    // Histogram speed
    const speedChart = MCH.echart(document.getElementById("chart-hist-speed"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "log", name: "样本量 N", logBase: 10 },
      yAxis: { type: "log", name: "分裂候选扫描次数", logBase: 10 },
      series: [
        { name: "Exact (XGB pre-sorted)", type: "line", smooth: true, showSymbol: false, color: "#ef4444", lineStyle: { width: 3 }, data: [] },
        { name: "Histogram (LightGBM)", type: "line", smooth: true, showSymbol: false, color: "#4f46e5", lineStyle: { width: 3 }, data: [] },
      ],
    });
    const updateSpeed = () => {
      const N = Math.pow(10, parseFloat(document.getElementById("lgb-N").value));
      const bins = parseInt(document.getElementById("lgb-bins").value);
      const ns = MCH.linspace(3, 8, 40).map(v => Math.pow(10, v));
      speedChart.setOption({
        series: [
          { data: ns.map(n => [n, n]) },
          { data: ns.map(n => [n, bins]) },
        ],
      });
      const speedup = N / bins;
      document.getElementById("lgb-info").innerHTML = `
        N = <b>${N.toExponential(1)}</b> 样本 · bins = <b>${bins}</b><br/>
        Exact 方法扫 <b>${(N - 1).toExponential(1)}</b> 个分裂候选<br/>
        Histogram 方法只扫 <b>${bins}</b> 个<br/>
        <b style="color:#10b981;">理论加速 ≈ ${speedup.toFixed(0)}×</b>（实测因预分箱开销约为 3-10×）
      `;
    };
    ["lgb-N", "lgb-bins"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateSpeed();
      });
    });
    updateSpeed();
  },
});
