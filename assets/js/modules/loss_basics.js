/* 模块：基础损失函数 */
MCH.register("loss_basics", {
  render() {
    return `
      ${MCH.hero({ icon: "L", name: "基础损失函数 — 从 MSE 到 Triplet", en: "Basic Loss Functions · MSE / CE / BCE / KL / Huber / Triplet", tags: ["回归", "分类", "对比学习", "分布拟合"], meta: ["◈ 数学基础", "⚡ 所有学习的起点"] })}

      ${MCH.versionSection("loss_basics")}

      <div class="section">
        <h2>概览：按任务类型组织</h2>
        <table class="table">
          <thead><tr><th>任务类型</th><th>损失</th><th>建模假设</th><th>典型应用</th></tr></thead>
          <tbody>
            <tr><td rowspan="3"><b>回归</b></td><td>MSE / L2</td><td>误差服从高斯分布</td><td>房价预测、CTR 连续值</td></tr>
            <tr><td>MAE / L1</td><td>误差服从拉普拉斯分布</td><td>对离群值鲁棒场景</td></tr>
            <tr><td>Huber</td><td>MSE + MAE 融合</td><td>工业界回归首选</td></tr>
            <tr><td rowspan="3"><b>分类</b></td><td>Cross Entropy</td><td>最大似然 (KL to one-hot)</td><td>多分类标准损失</td></tr>
            <tr><td>Binary CE (BCE)</td><td>二分类 log-loss</td><td>二分类 / 多标签</td></tr>
            <tr><td>Hinge (SVM)</td><td>最大间隔</td><td>SVM、排序</td></tr>
            <tr><td rowspan="2"><b>分布匹配</b></td><td>KL Divergence</td><td>两个概率分布的距离</td><td>蒸馏、VAE</td></tr>
            <tr><td>JS Divergence</td><td>对称化 KL</td><td>GAN 原始版</td></tr>
            <tr><td rowspan="2"><b>度量 / 对比</b></td><td>Triplet Loss</td><td>anchor/pos 近、anchor/neg 远</td><td>人脸 / 商品检索</td></tr>
            <tr><td>Contrastive (InfoNCE)</td><td>softmax 形式对比</td><td>SimCLR / CLIP</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>1. 回归损失：MSE / MAE / Huber</h2>
        <div class="formula-block">
          $$ L_{\\text{MSE}}(y, \\hat y) = (y - \\hat y)^2, \\quad L_{\\text{MAE}} = |y - \\hat y| $$
          $$ L_{\\text{Huber}} = \\begin{cases} \\tfrac{1}{2}(y - \\hat y)^2 & |y - \\hat y| \\leq \\delta \\\\ \\delta (|y - \\hat y| - \\tfrac{1}{2}\\delta) & \\text{otherwise} \\end{cases} $$
        </div>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "reg-delta", label: "Huber δ", min: 0.1, max: 5, step: 0.1, value: 1 })}
            </div>
          </div>
          <div id="chart-reg-loss" style="height:280px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>2. 分类损失：CE / BCE / Hinge</h2>
        <div class="formula-block">
          $$ L_{\\text{CE}} = -\\sum_k y_k \\log p_k, \\quad L_{\\text{BCE}} = -[y \\log p + (1-y) \\log(1-p)] $$
          $$ L_{\\text{Hinge}} = \\max(0, 1 - y \\cdot \\hat y), \\; y \\in \\{-1, +1\\} $$
        </div>
        <div class="grid-2">
          <div>
            <p class="text-xs text-slate-500">二分类场景：真实类 y=1，观察预测概率 p 与损失的关系。</p>
          </div>
          <div id="chart-cls-loss" style="height:280px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>3. KL 散度 — 分布距离的"非对称尺子"</h2>
        <div class="formula-block">
          $$ D_{\\text{KL}}(P \\| Q) = \\sum_x P(x) \\log \\frac{P(x)}{Q(x)}, \\quad \\geq 0,\\; =0 \\iff P=Q $$
        </div>
        ${MCH.info(`
          <b>"非对称"是什么意思？</b>
          $D_{KL}(P \\| Q) \\neq D_{KL}(Q \\| P)$。用 <b>forward KL</b>（P → Q）会让 Q 覆盖 P 的全部支撑集（mode-covering）；
          用 <b>reverse KL</b>（Q → P）会让 Q 挑选 P 的一个 mode（mode-seeking，VAE/强化学习常用）。
        `, "tip")}
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "kl-mu2", label: "Q 均值 μ_Q", min: -3, max: 3, step: 0.1, value: 1 })}
              ${MCH.slider({ id: "kl-sig2", label: "Q 标准差 σ_Q", min: 0.3, max: 3, step: 0.1, value: 1 })}
            </div>
            <div id="kl-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-kl" style="height:300px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. Triplet & Contrastive — 度量学习损失</h2>
        <p class="text-sm text-slate-600">不直接学类别，而是学一个 embedding 空间，使"同类近、异类远"：</p>
        <div class="formula-block">
          <b>Triplet Loss (FaceNet 2015)：</b>
          $$ L = \\max\\big(0,\\; d(a, p) - d(a, n) + \\alpha\\big) $$
          <b>InfoNCE (SimCLR / CLIP)：</b>
          $$ L = -\\log \\frac{\\exp(z_a \\cdot z_p / \\tau)}{\\sum_{k} \\exp(z_a \\cdot z_k / \\tau)} $$
        </div>
        <div id="chart-triplet" style="height:340px;"></div>
      </div>

      <div class="section">
        <h2>5. 损失函数选型速查</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>首选</th><th>备选</th><th>注意</th></tr></thead>
          <tbody>
            <tr><td>标准回归</td><td><b>Huber(δ=1)</b></td><td>MSE</td><td>有离群值时 Huber &gt; MSE</td></tr>
            <tr><td>多分类</td><td><b>CE</b></td><td>Label Smoothing CE</td><td>LS 常配 0.1 提升泛化</td></tr>
            <tr><td>长尾分类</td><td>Focal / LDAM</td><td>CE + class_weight</td><td>见<a href="#/losses" class="text-indigo-600">长尾损失</a>模块</td></tr>
            <tr><td>多标签</td><td>BCE per-label</td><td>Asymmetric Loss</td><td>不要用 softmax CE</td></tr>
            <tr><td>蒸馏</td><td>KL(T=4) + CE</td><td>MSE on features</td><td>Hinton 2015 经典组合</td></tr>
            <tr><td>度量学习</td><td>InfoNCE / SupCon</td><td>Triplet</td><td>SimCLR 批大小要够大</td></tr>
            <tr><td>图像分割</td><td>Dice + BCE</td><td>Focal + Dice</td><td>小目标用 Dice 防样本不均</td></tr>
          </tbody>
        </table>
      </div>
    `;
  },

  mount() {
    // Regression losses
    const regChart = MCH.echart(document.getElementById("chart-reg-loss"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "误差 y - ŷ" },
      yAxis: { type: "value", name: "loss" },
      series: [
        { name: "MSE", type: "line", showSymbol: false, smooth: true, color: "#4f46e5", lineStyle: { width: 3 }, data: [] },
        { name: "MAE", type: "line", showSymbol: false, smooth: true, color: "#10b981", lineStyle: { width: 3 }, data: [] },
        { name: "Huber (当前)", type: "line", showSymbol: false, smooth: true, color: "#f59e0b", lineStyle: { width: 3 }, data: [] },
      ],
    });
    const updateReg = () => {
      const delta = parseFloat(document.getElementById("reg-delta").value);
      const xs = MCH.linspace(-3, 3, 200);
      const huber = (x) => Math.abs(x) <= delta ? 0.5 * x * x : delta * (Math.abs(x) - 0.5 * delta);
      regChart.setOption({
        series: [
          { data: xs.map(x => [x, x * x]) },
          { data: xs.map(x => [x, Math.abs(x)]) },
          { data: xs.map(x => [x, huber(x)]) },
        ],
      });
    };
    document.getElementById("reg-delta").addEventListener("input", (e) => {
      document.getElementById("reg-delta-val").textContent = e.target.value;
      updateReg();
    });
    updateReg();

    // Classification losses
    MCH.echart(document.getElementById("chart-cls-loss"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "p (预测概率, y=1)", min: 0.001, max: 0.999 },
      yAxis: { type: "value", name: "loss" },
      series: [
        { name: "BCE = -log(p)", type: "line", showSymbol: false, smooth: true, color: "#4f46e5", lineStyle: { width: 3 }, data: MCH.linspace(0.001, 0.999, 100).map(p => [p, -Math.log(p)]) },
        { name: "0-1 Loss", type: "line", step: "middle", showSymbol: false, color: "#94a3b8", lineStyle: { width: 2 }, data: MCH.linspace(0.001, 0.999, 100).map(p => [p, p < 0.5 ? 1 : 0]) },
        { name: "Hinge (y=1, s=2p-1)", type: "line", showSymbol: false, smooth: true, color: "#f59e0b", lineStyle: { width: 3 }, data: MCH.linspace(0.001, 0.999, 100).map(p => [p, Math.max(0, 1 - (2 * p - 1))]) },
      ],
    });

    // KL divergence
    const klChart = MCH.echart(document.getElementById("chart-kl"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "x" },
      yAxis: { type: "value", name: "density" },
      series: [
        { name: "P ~ N(0, 1)", type: "line", showSymbol: false, smooth: true, color: "#4f46e5", areaStyle: { opacity: 0.2 }, data: [] },
        { name: "Q ~ N(μ, σ)", type: "line", showSymbol: false, smooth: true, color: "#f59e0b", areaStyle: { opacity: 0.2 }, data: [] },
      ],
    });
    const updateKL = () => {
      const mu = parseFloat(document.getElementById("kl-mu2").value);
      const sig = parseFloat(document.getElementById("kl-sig2").value);
      const gauss = (x, m, s) => Math.exp(-0.5 * Math.pow((x - m) / s, 2)) / (s * Math.sqrt(2 * Math.PI));
      const xs = MCH.linspace(-6, 6, 200);
      klChart.setOption({
        series: [
          { data: xs.map(x => [x, gauss(x, 0, 1)]) },
          { data: xs.map(x => [x, gauss(x, mu, sig)]) },
        ],
      });
      // KL(P||Q) for two gaussians: log(σ_Q/σ_P) + (σ_P² + (μ_P - μ_Q)²)/(2σ_Q²) - 1/2
      const klPQ = Math.log(sig / 1) + (1 + Math.pow(0 - mu, 2)) / (2 * sig * sig) - 0.5;
      const klQP = Math.log(1 / sig) + (sig * sig + Math.pow(mu - 0, 2)) / 2 - 0.5;
      document.getElementById("kl-info").innerHTML = `
        <b>KL(P || Q)</b> = <span style="color:#4f46e5;font-weight:700;">${klPQ.toFixed(3)}</span> · <b>KL(Q || P)</b> = <span style="color:#f59e0b;font-weight:700;">${klQP.toFixed(3)}</span><br/>
        <span style="color:#64748b;">两者不相等，KL 非对称。</span>
      `;
    };
    ["kl-mu2", "kl-sig2"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateKL();
      });
    });
    updateKL();

    // Triplet
    const anchor = [0, 0];
    const positives = [[0.3, 0.2], [-0.2, 0.4], [0.4, -0.1], [0.1, 0.3]];
    const negatives = [[1.3, 1.2], [1.5, -0.8], [-1.3, 1.1], [-1.0, -1.4], [1.7, 0.3]];
    MCH.echart(document.getElementById("chart-triplet"), {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", min: -2, max: 2 },
      yAxis: { type: "value", min: -2, max: 2 },
      series: [
        { name: "Anchor", type: "scatter", data: [anchor], color: "#ef4444", symbolSize: 20 },
        { name: "Positives (同类)", type: "scatter", data: positives, color: "#10b981", symbolSize: 14 },
        { name: "Negatives (异类)", type: "scatter", data: negatives, color: "#64748b", symbolSize: 14 },
        { name: "margin α", type: "scatter", data: [], markArea: { silent: true, itemStyle: { color: "rgba(251,191,36,0.08)" }, data: [[{ coord: [-0.5, -0.5] }, { coord: [0.5, 0.5] }]] } },
      ],
    });
  },
});
