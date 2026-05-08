/* 模块：采样器 */
MCH.register("samplers", {
  render() {
    const code = `# Class-Balanced + GMV-Weighted 采样器
# 来源：src/data/samplers.py

class ClassBalancedSampler(Sampler):
    """Cui et al., CVPR'19 — Effective Number of Samples.

    核心思想：不是简单按 1/n_c 逆频采样，因为当样本数大时"边际信息"会饱和。
    定义 effective_num(c) = (1 - β^n_c) / (1 - β)，取 β ∈ [0.99, 0.9999]。
    β → 1 时退化为 1/n_c；β → 0 时所有类权重相等。
    """
    def __init__(self, labels, beta=0.9999):
        counts = np.bincount(labels).astype(np.float64)
        effective = 1.0 - np.power(beta, counts)
        per_class_w = (1.0 - beta) / effective         # 稀少类权重更高
        self.weights = per_class_w[labels] / per_class_w[labels].sum()

    def __iter__(self):
        g = np.random.default_rng(self.seed)
        return iter(g.choice(len(self.labels), size=self.num_samples,
                             replace=True, p=self.weights).tolist())


class GmvWeightedSampler(Sampler):
    """在 Class-Balanced 基础上叠加 1 + λ·log(1+gmv) 权重，大商户更频繁被采样。"""
    def __init__(self, labels, gmv, beta=0.9999, gmv_lambda=1.0):
        counts = np.bincount(labels)
        cb_w = ((1 - beta) / (1 - np.power(beta, counts)))[labels]
        gmv_w = 1.0 + gmv_lambda * np.log1p(np.maximum(gmv, 0))
        w = cb_w * gmv_w
        self.weights = w / w.sum()`;

    return `
      ${MCH.hero({
        icon: "R",
        name: "采样器 — Class-Balanced × GMV-Weighted",
        en: "Weighted Samplers · Effective Number + GMV",
        tags: ["CB β=0.9999", "Effective Number", "GMV log 加权", "Decoupled Stage 2"],
        meta: ["◈ 替代 WeightedRandomSampler", "◇ 来源 src/data/samplers.py"],
      })}

      <div class="section">
        <h2>1. 算法原理</h2>
        <p class="text-sm text-slate-600">
          采样层面干预是对抗长尾最朴素有效的手段。但"简单逆频采样"在极端长尾下会导致<b>过度重复稀少类</b>，
          反而让模型记住噪声。CVPR'19 提出 <b>Effective Number of Samples</b>：
        </p>
        <div class="formula-block">
          $$ E_n = \\frac{1 - \\beta^n}{1 - \\beta}, \\quad w_c \\propto \\frac{1}{E_{n_c}} = \\frac{1 - \\beta}{1 - \\beta^{n_c}} $$
        </div>
        <p class="text-sm text-slate-600">
          物理含义：样本带来的边际信息随 n 递减（几何衰减），β 控制衰减速度。β=0.9999 是经验最佳。
        </p>

        ${MCH.info(`
          <b>GMV 加权的业务动机</b>：Top 1% 商户贡献 80% 交易额。若 CB 采样把所有类目等权对待，
          会导致"小众商家过度曝光、大商家训练不足"。叠加 <code>1 + λ·log(1+GMV)</code> 权重可让
          训练分布更贴近<b>业务损失分布</b>。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：Effective Number 曲线</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "eff-beta", label: "β", min: 0.9, max: 0.99999, step: 0.001, value: 0.9999, format: (v) => parseFloat(v).toFixed(4) })}
            </div>
            <p class="text-xs text-slate-500 mt-2">对比 β 不同时 w_c 的曲线形状。β=0 → uniform；β=1 → 1/n。</p>
          </div>
          <div id="chart-eff" style="height:300px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. 交互：采样前/采样后分布（2000 叶子类目）</h2>
        <p class="text-sm text-slate-600">
          模拟 2000 类的真实 Zipf 分布（α=1.1），观察 uniform / CB / CB+GMV 三种采样策略后的类别频率变化。
        </p>
        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="grid-3">
            <div>${MCH.slider({ id: "samp-beta", label: "CB β", min: 0.9, max: 0.99999, step: 0.001, value: 0.9999, format: (v) => parseFloat(v).toFixed(4) })}</div>
            <div>${MCH.slider({ id: "samp-lam", label: "GMV λ", min: 0, max: 3, step: 0.1, value: 1 })}</div>
            <div>${MCH.slider({ id: "samp-N", label: "采样批量 N", min: 10000, max: 500000, step: 10000, value: 100000, format: (v) => parseInt(v).toLocaleString() })}</div>
          </div>
        </div>
        <div id="chart-samp" style="height:360px;"></div>
        <div id="samp-stats" class="grid-4 mt-4"></div>
      </div>

      <div class="section">
        <h2>5. 使用时机（Decoupled cRT 的 Stage 策略）</h2>
        <table class="table">
          <thead><tr><th>Stage</th><th>Sampler</th><th>学什么</th><th>动机</th></tr></thead>
          <tbody>
            <tr><td>Stage 1 · 表征学习</td><td>Uniform / 随机</td><td>backbone + fusion</td><td>原始分布 → 最真实的 feature</td></tr>
            <tr><td>Stage 2 · 分类器重训</td><td><b>Class-Balanced</b></td><td>heads + mmoe</td><td>消除分类器对头部的偏袒</td></tr>
            <tr><td>Stage 3 · 联合微调</td><td>GMV-Weighted</td><td>全网小 lr</td><td>贴近业务损失分布</td></tr>
          </tbody>
        </table>
      </div>
    `;
  },

  mount() {
    // Effective Number 曲线
    const effChart = MCH.echart(document.getElementById("chart-eff"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "log", name: "n_c (类样本数)", logBase: 10 },
      yAxis: { type: "log", name: "权重 w_c（相对）", logBase: 10 },
      series: [
        { name: "Uniform w=1", type: "line", smooth: true, data: [], color: "#94a3b8", lineStyle: { type: "dashed" } },
        { name: "1/n (inverse freq)", type: "line", smooth: true, data: [], color: "#ef4444" },
        { name: "CB β=0.9", type: "line", smooth: true, data: [], color: "#10b981" },
        { name: "CB β=0.99", type: "line", smooth: true, data: [], color: "#f59e0b" },
        { name: "CB β 当前", type: "line", smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
      ],
    });
    const ns = MCH.linspace(0, 5, 40).map(v => Math.pow(10, v));
    const cbW = (b) => ns.map(n => [n, (1 - b) / (1 - Math.pow(b, n))]);
    const updateEff = () => {
      const beta = parseFloat(document.getElementById("eff-beta").value);
      effChart.setOption({
        series: [
          { data: ns.map(n => [n, 1]) },
          { data: ns.map(n => [n, 1 / n]) },
          { data: cbW(0.9) },
          { data: cbW(0.99) },
          { data: cbW(beta) },
        ],
      });
    };
    document.getElementById("eff-beta").addEventListener("input", (e) => {
      document.getElementById("eff-beta-val").textContent = parseFloat(e.target.value).toFixed(4);
      updateEff();
    });
    updateEff();

    // 分布采样
    const sampChart = MCH.echart(document.getElementById("chart-samp"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 70, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "log", name: "类别 ID（按频率降序）", logBase: 10 },
      yAxis: { type: "log", name: "出现次数", logBase: 10 },
      series: [
        { name: "原始（Zipf）", type: "line", smooth: true, showSymbol: false, data: [], color: "#94a3b8", lineStyle: { width: 2 } },
        { name: "Class-Balanced", type: "line", smooth: true, showSymbol: false, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
        { name: "CB + GMV", type: "line", smooth: true, showSymbol: false, data: [], color: "#10b981", lineStyle: { width: 3 } },
      ],
    });

    // 生成类分布
    const numClasses = 2000;
    const zipf_alpha = 1.1;
    const rawCounts = new Array(numClasses);
    for (let i = 0; i < numClasses; i++) rawCounts[i] = 10000 / Math.pow(i + 1, zipf_alpha);
    const totalRaw = rawCounts.reduce((a, b) => a + b, 0);
    const rawProb = rawCounts.map(c => c / totalRaw);
    // 模拟 GMV 每类平均值：头部类 gmv 也更高
    const gmvPerClass = new Array(numClasses);
    for (let i = 0; i < numClasses; i++) gmvPerClass[i] = 100000 / Math.pow(i + 1, 0.5);

    const binCount = 50;
    const bins = MCH.linspace(0, Math.log10(numClasses), binCount + 1).map(v => Math.pow(10, v));
    const binify = (counts) => {
      const out = new Array(binCount).fill(0);
      for (let i = 0; i < numClasses; i++) {
        const bin = Math.min(binCount - 1, Math.floor(Math.log10(i + 1) / Math.log10(numClasses) * binCount));
        out[bin] += counts[i];
      }
      return out.map((v, i) => [bins[i], v + 1e-3]);
    };

    const updateSamp = () => {
      const beta = parseFloat(document.getElementById("samp-beta").value);
      const lam = parseFloat(document.getElementById("samp-lam").value);
      const N = parseInt(document.getElementById("samp-N").value);

      const cbW = rawCounts.map(n => (1 - beta) / (1 - Math.pow(beta, n)));
      const cbWNorm = cbW.map((w, i) => w * rawProb[i]);
      const sCb = cbWNorm.reduce((a, b) => a + b, 0);
      const cbProb = cbWNorm.map(w => w / sCb);

      const gmvMul = gmvPerClass.map(g => 1 + lam * Math.log1p(g));
      const cbGmv = cbProb.map((p, i) => p * gmvMul[i]);
      const sCg = cbGmv.reduce((a, b) => a + b, 0);
      const cbGmvProb = cbGmv.map(p => p / sCg);

      const rawSampled = rawProb.map(p => p * N);
      const cbSampled = cbProb.map(p => p * N);
      const cbGmvSampled = cbGmvProb.map(p => p * N);

      sampChart.setOption({
        series: [
          { data: binify(rawSampled) },
          { data: binify(cbSampled) },
          { data: binify(cbGmvSampled) },
        ],
      });

      // 显示 head / tail 统计
      const topPct = 0.01;
      const topK = Math.ceil(numClasses * topPct);
      const sum = arr => arr.reduce((a, b) => a + b, 0);
      const rawTop = sum(rawSampled.slice(0, topK)) / N * 100;
      const cbTop = sum(cbSampled.slice(0, topK)) / N * 100;
      const cgTop = sum(cbGmvSampled.slice(0, topK)) / N * 100;
      const rawTail = sum(rawSampled.slice(-Math.floor(numClasses * 0.5))) / N * 100;
      const cbTail = sum(cbSampled.slice(-Math.floor(numClasses * 0.5))) / N * 100;
      const cgTail = sum(cbGmvSampled.slice(-Math.floor(numClasses * 0.5))) / N * 100;

      document.getElementById("samp-stats").innerHTML = `
        <div class="card"><div class="text-xs text-slate-500">Top 1% 类占比</div>
          <div class="text-sm mt-2"><span class="tag tag-slate">Raw</span> <b>${rawTop.toFixed(1)}%</b></div>
          <div class="text-sm"><span class="tag">CB</span> <b>${cbTop.toFixed(1)}%</b></div>
          <div class="text-sm"><span class="tag tag-green">CB+GMV</span> <b>${cgTop.toFixed(1)}%</b></div>
        </div>
        <div class="card"><div class="text-xs text-slate-500">Tail 50% 类占比</div>
          <div class="text-sm mt-2"><span class="tag tag-slate">Raw</span> <b>${rawTail.toFixed(2)}%</b></div>
          <div class="text-sm"><span class="tag">CB</span> <b>${cbTail.toFixed(2)}%</b></div>
          <div class="text-sm tag-green"><span class="tag tag-green">CB+GMV</span> <b>${cgTail.toFixed(2)}%</b></div>
        </div>
        <div class="card"><div class="text-xs text-slate-500">类覆盖率（每批至少 1 次）</div>
          <div class="text-sm mt-2">理论覆盖 ${Math.min(100, (N / numClasses) * 100).toFixed(0)}% 类别</div>
          <div class="text-xs text-slate-500 mt-1">CB 下尾部类出现概率 ↑</div>
        </div>
        <div class="card"><div class="text-xs text-slate-500">KS 距离 (CB vs Raw)</div>
          <div class="text-sm mt-2"><b>${Math.abs(cbTop - rawTop).toFixed(1)}%</b> 头部变化</div>
          <div class="text-xs text-slate-500 mt-1">数值越大 → 采样对分布改造越强</div>
        </div>
      `;
    };
    ["samp-beta", "samp-lam", "samp-N"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        const v = e.target.value;
        const fmt = id === "samp-beta" ? parseFloat(v).toFixed(4) : (id === "samp-N" ? parseInt(v).toLocaleString() : v);
        document.getElementById(id + "-val").textContent = fmt;
        updateSamp();
      });
    });
    updateSamp();
  },
});
