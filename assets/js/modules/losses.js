/* 模块：损失函数族 */
MCH.register("losses", {
  render() {
    return `
      ${MCH.hero({
        icon: "L",
        name: "损失函数族 — 长尾 × 多任务协同",
        en: "Imbalance-aware Losses · Focal / LDAM / Seesaw / SupCon / GMV / GradNorm",
        tags: ["Focal", "LDAM + DRW", "Seesaw", "OrdinalCE", "SupCon", "GMV-Weighted", "GradNorm", "Uncertainty"],
        meta: ["◈ 7 种损失 + 2 种多任务 balancer", "◇ 来源：src/losses/losses.py"],
      })}

      ${MCH.versionSection("losses")}

      <div class="section">
        <h2>1. 损失设计理念</h2>
        ${MCH.info(`
          商户识别的核心难题是 <b>三重不均衡</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>类别不均衡</b>：黑样本 : 白样本 ≈ 1 : 10,000；</li>
            <li><b>业务量纲不均衡</b>：Top 1% 商户贡献 80% 交易额，误判成本差 100×；</li>
            <li><b>任务梯度不均衡</b>：风险/可信/类目任务 loss 量级差 5~10×。</li>
          </ul>
          本模块提供 <b>7 种基础损失 + 2 种多任务 balancer</b>，组合使用。
        `, "biz")}
      </div>

      <!-- Focal -->
      <div class="section">
        <h2>2. Focal Loss — 风险 3 分类</h2>
        <p class="text-sm text-slate-600">Lin et al., ICCV 2017。用 <code>(1-p_t)^γ</code> 降权易样本、聚焦难样本。</p>
        <div class="formula-block">
          $$ L_{\\text{focal}} = -(1 - p_t)^\\gamma \\log p_t, \\quad p_t = \\text{softmax}(z)_{y} $$
        </div>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "focal-gamma", label: "γ (focusing)", min: 0, max: 5, step: 0.1, value: 2 })}
            </div>
            ${MCH.code(`# Focal Loss 核心 (简化)
p = F.softmax(logits, dim=-1)
p_t = p.gather(1, y.unsqueeze(1)).squeeze(1)
loss = -((1 - p_t) ** gamma) * torch.log(p_t)`, "python")}
          </div>
          <div id="chart-focal" style="height:320px;"></div>
        </div>
      </div>

      <!-- LDAM -->
      <div class="section">
        <h2>3. LDAM Loss + DRW — 长尾 Margin</h2>
        <p class="text-sm text-slate-600">Cao et al., NeurIPS 2019。按类频率给少样本类 <b>更大的 margin</b>：</p>
        <div class="formula-block">
          $$ m_c = \\frac{C}{n_c^{1/4}}, \\quad z'_y = z_y - s \\cdot m_y, \\quad L = \\text{CE}(z', y) $$
        </div>
        <p class="text-xs text-slate-500">DRW (Deferred Re-Weighting)：前 11 epoch 不加权（学表征），第 12 起切 class-balanced weight（调分类器）。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "ldam-maxm", label: "max_m（最大 margin）", min: 0.1, max: 1.0, step: 0.05, value: 0.5 })}
              ${MCH.slider({ id: "ldam-s", label: "scale s", min: 10, max: 60, step: 1, value: 30 })}
              ${MCH.slider({ id: "ldam-drw", label: "当前 Epoch", min: 0, max: 30, step: 1, value: 10 })}
            </div>
            <div id="ldam-info" class="text-xs text-slate-600 mt-2"></div>
          </div>
          <div id="chart-ldam" style="height:320px;"></div>
        </div>
      </div>

      <!-- Seesaw -->
      <div class="section">
        <h2>4. Seesaw Loss — 2000 类叶子类目专用</h2>
        <p class="text-sm text-slate-600">Wang et al., CVPR 2021。对"稀少类 → 高频类"和"高概率错类"双向调节：</p>
        <div class="formula-block">
          $$ M_{ij} = \\underbrace{\\min\\left(1, \\left(\\tfrac{N_j}{N_i}\\right)^{p}\\right)}_{\\text{Mitigation: 稀少→高频 降权}} \\cdot \\underbrace{\\max\\left(1, \\left(\\tfrac{p_j}{p_i}\\right)^{q}\\right)}_{\\text{Compensation: 错判高概率类 加重}} $$
          $$ L = \\text{CE}\\big(z + \\log M,\\; y\\big) $$
        </div>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "ss-p", label: "Mitigation 强度 p", min: 0, max: 2, step: 0.05, value: 0.8 })}
              ${MCH.slider({ id: "ss-q", label: "Compensation 强度 q", min: 0, max: 4, step: 0.1, value: 2 })}
            </div>
            <p class="text-xs text-slate-500 mt-2">固定真实类 = 一个中频类（#=100），观察对其他类的 <code>log M</code> 调节量。</p>
          </div>
          <div id="chart-seesaw" style="height:320px;"></div>
        </div>
      </div>

      <!-- SupCon -->
      <div class="section">
        <h2>5. SupCon — 监督对比学习</h2>
        <p class="text-sm text-slate-600">Khosla et al., NeurIPS 2020。<b>同类拉近、异类推远</b>，显著提升类目表征：</p>
        <div class="formula-block">
          $$ L_{\\text{supcon}} = -\\sum_{i} \\frac{1}{|P(i)|} \\sum_{p \\in P(i)} \\log \\frac{\\exp(z_i \\cdot z_p / \\tau)}{\\sum_{a \\neq i} \\exp(z_i \\cdot z_a / \\tau)} $$
        </div>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "sc-T", label: "温度 τ", min: 0.05, max: 1.0, step: 0.02, value: 0.1 })}
              ${MCH.slider({ id: "sc-epoch", label: "训练 Epoch", min: 0, max: 20, step: 1, value: 0 })}
            </div>
            <button id="sc-btn" class="mt-3 px-3 py-1.5 bg-indigo-600 text-white text-xs rounded-md">▶ 播放收敛动画</button>
          </div>
          <div id="chart-supcon" style="height:360px;"></div>
        </div>
      </div>

      <!-- GMV -->
      <div class="section">
        <h2>6. GMV-Weighted Wrapper — 业务量纲加权</h2>
        <p class="text-sm text-slate-600">对任意 base loss，按 <code>w = 1 + λ·log(1 + gmv)</code> 逐样本加权，使大商户误判代价更高。</p>
        <div class="formula-block">
          $$ L = \\text{mean}\\big(\\, w_i \\cdot L_i \\,\\big), \\quad w_i = \\frac{1 + \\lambda \\log(1 + \\text{GMV}_i)}{\\mathbb{E}\\big[1 + \\lambda \\log(1+\\text{GMV})\\big]} $$
        </div>
        <div class="grid-2">
          <div class="ctrl-panel">
            ${MCH.slider({ id: "gmv-lambda", label: "λ（加权强度）", min: 0, max: 3, step: 0.1, value: 1 })}
          </div>
          <div id="chart-gmv" style="height:280px;"></div>
        </div>
      </div>

      <!-- GradNorm & Uncertainty -->
      <div class="section">
        <h2>7. 多任务 Balancer：GradNorm / Uncertainty Weighting</h2>
        <div class="grid-2">
          <div class="card">
            <h4 class="font-semibold">GradNorm (Chen et al., ICML 2018)</h4>
            <p class="text-xs text-slate-600 mt-2">按各任务梯度范数与训练进度动态调整权重：</p>
            <div class="formula-block text-xs">
              $$ L_{\\text{grad}} = \\sum_t \\big| G_t - \\bar G \\cdot r_t^{\\alpha} \\big|,\\;\\;r_t = \\tilde L_t / \\overline{\\tilde L} $$
            </div>
            <p class="text-xs text-slate-500">代码：<code>losses.py::GradNormBalancer</code></p>
          </div>
          <div class="card">
            <h4 class="font-semibold">Uncertainty Weighting (Kendall et al., CVPR 2018)</h4>
            <p class="text-xs text-slate-600 mt-2">可学习 log σ²，等价于对每任务做精度加权：</p>
            <div class="formula-block text-xs">
              $$ L_{\\text{total}} = \\sum_t \\frac{1}{2\\sigma_t^2} L_t + \\log \\sigma_t $$
            </div>
            <p class="text-xs text-slate-500">代码：<code>losses.py::UncertaintyWeighting</code></p>
          </div>
        </div>
        <h3 style="margin-top:18px;">· 任务权重演化（模拟）</h3>
        <div id="chart-gradnorm" style="height:320px;"></div>
      </div>

      <!-- 组合策略 -->
      <div class="section">
        <h2>8. 损失组合策略（实战配方）</h2>
        <table class="table">
          <thead><tr><th>任务</th><th>Loss 组合</th><th>业务动机</th></tr></thead>
          <tbody>
            <tr><td>风险 3 分类</td><td><b>FocalLoss(γ=2)</b> + GMVWeighted(λ=1)</td><td>长尾 + 大商户代价高</td></tr>
            <tr><td>风险 5 级 Ordinal</td><td><b>OrdinalCE</b> (系数 0.5)</td><td>惩罚跨级错判</td></tr>
            <tr><td>可信 5 级</td><td><b>LDAM</b> + DRW(从 E12 起加权)</td><td>长尾边界 + 稳定表征</td></tr>
            <tr><td>类目 L1 (20)</td><td>标准 CE · 权重 0.3</td><td>大类辅助分层</td></tr>
            <tr><td>类目 L2 (2000)</td><td><b>Seesaw(p=0.8, q=2)</b> · 权重 0.7</td><td>2000 类超长尾</td></tr>
            <tr><td>类目表征</td><td><b>SupCon(τ=0.1)</b> · 权重 0.3</td><td>同类聚类、异类分散</td></tr>
            <tr><td>多任务聚合</td><td><b>GradNorm(α=1.5)</b> 或 Uncertainty</td><td>自动平衡</td></tr>
          </tbody>
        </table>
      </div>
    `;
  },

  mount() {
    // ---------- Focal ----------
    const focalChart = MCH.echart(document.getElementById("chart-focal"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "p_t (真实类概率)", min: 0, max: 1 },
      yAxis: { type: "value", name: "loss" },
      series: [
        { name: "CE (γ=0)", type: "line", showSymbol: false, smooth: true, data: [], color: "#94a3b8" },
        { name: "Focal", type: "line", showSymbol: false, smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
      ],
    });
    const updateFocal = () => {
      const gamma = parseFloat(document.getElementById("focal-gamma").value);
      const ps = MCH.linspace(0.01, 0.99, 100);
      focalChart.setOption({
        series: [
          { data: ps.map(p => [p, -Math.log(p)]) },
          { data: ps.map(p => [p, -Math.pow(1 - p, gamma) * Math.log(p)]) },
        ],
      });
    };
    document.getElementById("focal-gamma").addEventListener("input", (e) => {
      document.getElementById("focal-gamma-val").textContent = e.target.value;
      updateFocal();
    });
    updateFocal();

    // ---------- LDAM ----------
    const ldamChart = MCH.echart(document.getElementById("chart-ldam"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", name: "类别 (按频率升序)", data: [] },
      yAxis: [
        { type: "value", name: "样本数", position: "left" },
        { type: "value", name: "margin", position: "right" },
      ],
      series: [
        { name: "样本数 n_c", type: "bar", yAxisIndex: 0, data: [], color: "#94a3b8", barWidth: 10 },
        { name: "margin m_c", type: "line", yAxisIndex: 1, data: [], color: "#4f46e5", lineStyle: { width: 3 }, smooth: true },
        { name: "DRW weight", type: "line", yAxisIndex: 1, data: [], color: "#f59e0b", lineStyle: { width: 2, type: "dashed" }, smooth: true },
      ],
    });
    const updateLdam = () => {
      const maxM = parseFloat(document.getElementById("ldam-maxm").value);
      const s = parseFloat(document.getElementById("ldam-s").value);
      const epoch = parseInt(document.getElementById("ldam-drw").value);
      const counts = [5, 15, 30, 60, 120, 240, 500, 1000, 2000, 5000];
      const mRaw = counts.map(n => 1 / Math.pow(n, 0.25));
      const mMax = Math.max(...mRaw);
      const m = mRaw.map(v => (v * maxM / mMax));
      const beta = 0.9999;
      const drwActive = epoch >= 12;
      let drw = counts.map(n => (1 - beta) / (1 - Math.pow(beta, n)));
      const norm = drw.reduce((a, b) => a + b, 0) / drw.length;
      drw = drw.map(w => w / norm);
      ldamChart.setOption({
        xAxis: { data: counts.map((_, i) => `C${i}`) },
        series: [
          { data: counts },
          { data: m.map(v => v.toFixed(3)) },
          { data: drwActive ? drw.map(v => v.toFixed(2)) : drw.map(() => 1) },
        ],
      });
      document.getElementById("ldam-info").innerHTML = `
        当前 epoch = ${epoch}，DRW ${drwActive ? '<span style="color:#f59e0b;font-weight:700;">已激活</span>' : '<span style="color:#94a3b8;">未激活（用 uniform=1）</span>'}<br/>
        min margin = ${Math.min(...m).toFixed(3)} · max margin = ${Math.max(...m).toFixed(3)}（类越少，margin 越大）
      `;
    };
    ["ldam-maxm", "ldam-s", "ldam-drw"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateLdam();
      });
    });
    updateLdam();

    // ---------- Seesaw ----------
    const ssChart = MCH.echart(document.getElementById("chart-seesaw"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", name: "类别（按频率）", data: [] },
      yAxis: { type: "value", name: "log M (logit 调节)" },
      series: [
        { name: "Mitigation ∝ (N_j/N_i)^p", type: "line", smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
        { name: "Compensation ∝ (p_j/p_i)^q", type: "line", smooth: true, data: [], color: "#f59e0b", lineStyle: { width: 3 } },
        { name: "合计 log M", type: "line", smooth: true, data: [], color: "#10b981", lineStyle: { width: 3 } },
      ],
    });
    const updateSs = () => {
      const p = parseFloat(document.getElementById("ss-p").value);
      const q = parseFloat(document.getElementById("ss-q").value);
      const counts = [5, 15, 30, 60, 100, 200, 500, 1000, 3000, 8000];
      const Ni = 100; // 真实类设为中频
      const pi = 0.15;
      // 模拟其他类的预测概率：高频类被错预测概率更高
      const probs = counts.map((n, i) => Math.max(0.02, 0.05 + (i > 4 ? (i - 4) * 0.06 : 0)));
      const mit = counts.map(n => Math.log(Math.min(1, Math.pow(n / Ni, p))));
      const comp = probs.map(pj => Math.log(Math.max(1, Math.pow(pj / pi, q))));
      const total = mit.map((v, i) => v + comp[i]);
      ssChart.setOption({
        xAxis: { data: counts.map((c, i) => `n=${c}`) },
        series: [
          { data: mit.map(v => v.toFixed(3)) },
          { data: comp.map(v => v.toFixed(3)) },
          { data: total.map(v => v.toFixed(3)) },
        ],
      });
    };
    ["ss-p", "ss-q"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateSs();
      });
    });
    updateSs();

    // ---------- SupCon ----------
    const scChart = MCH.echart(document.getElementById("chart-supcon"), {
      grid: { left: 40, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", min: -2, max: 2, name: "x" },
      yAxis: { type: "value", min: -2, max: 2, name: "y" },
      legend: { top: 0 },
      series: [],
    });
    let scClasses = [
      { c: [1, 0.8], n: 10 },
      { c: [-1, 0.3], n: 10 },
      { c: [0.2, -1.2], n: 10 },
    ];
    const regen = () => {
      const epoch = parseInt(document.getElementById("sc-epoch").value);
      const tau = parseFloat(document.getElementById("sc-T").value);
      const t = epoch / 20;
      const compactness = 0.7 - 0.55 * t * Math.max(0.2, (0.15 / tau));
      const scatter = scClasses.map((cl, i) => {
        const pts = [];
        for (let j = 0; j < cl.n; j++) {
          const g = MCH.randn(2, j * 7 + i * 11 + epoch);
          pts.push([cl.c[0] + g[0] * compactness, cl.c[1] + g[1] * compactness]);
        }
        return {
          name: `类 ${i + 1}`, type: "scatter",
          data: pts, symbolSize: 10,
          color: ["#4f46e5", "#f59e0b", "#10b981"][i],
        };
      });
      scChart.setOption({ series: scatter });
    };
    ["sc-T", "sc-epoch"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        regen();
      });
    });
    document.getElementById("sc-btn").addEventListener("click", () => {
      let e = 0;
      const timer = setInterval(() => {
        e++;
        document.getElementById("sc-epoch").value = e;
        document.getElementById("sc-epoch-val").textContent = e;
        regen();
        if (e >= 20) clearInterval(timer);
      }, 250);
    });
    regen();

    // ---------- GMV-weighted ----------
    const gmvChart = MCH.echart(document.getElementById("chart-gmv"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "log", name: "GMV (元)", logBase: 10 },
      yAxis: { type: "value", name: "权重 w" },
      series: [{ name: "1 + λ·log(1+GMV)", type: "line", smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } }],
    });
    const updateGmv = () => {
      const lam = parseFloat(document.getElementById("gmv-lambda").value);
      const gmvs = MCH.linspace(0, 8, 100).map(v => Math.pow(10, v));
      gmvChart.setOption({
        series: [{ data: gmvs.map(g => [g, 1 + lam * Math.log1p(g)]) }],
      });
    };
    document.getElementById("gmv-lambda").addEventListener("input", (e) => {
      document.getElementById("gmv-lambda-val").textContent = e.target.value;
      updateGmv();
    });
    updateGmv();

    // ---------- GradNorm Sim ----------
    const gnChart = MCH.echart(document.getElementById("chart-gradnorm"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "Step" },
      yAxis: { type: "value", name: "任务权重", min: 0 },
      series: [],
    });
    const steps = MCH.linspace(0, 200, 100);
    const risk = steps.map(s => 1.3 - 0.3 * (1 - Math.exp(-s / 60)) + 0.05 * Math.sin(s / 20));
    const trust = steps.map(s => 0.7 + 0.4 * (1 - Math.exp(-s / 50)) + 0.05 * Math.cos(s / 15));
    const category = steps.map(s => 1.0 + 0.2 * Math.sin(s / 25));
    gnChart.setOption({
      legend: { data: ["risk", "trust", "category"], top: 0 },
      series: [
        { name: "risk", type: "line", smooth: true, showSymbol: false, color: "#ef4444", lineStyle: { width: 2 }, data: steps.map((s, i) => [s, risk[i]]) },
        { name: "trust", type: "line", smooth: true, showSymbol: false, color: "#f59e0b", lineStyle: { width: 2 }, data: steps.map((s, i) => [s, trust[i]]) },
        { name: "category", type: "line", smooth: true, showSymbol: false, color: "#4f46e5", lineStyle: { width: 2 }, data: steps.map((s, i) => [s, category[i]]) },
      ],
    });
  },
});
