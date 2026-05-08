/* 模块：深度异常检测 */
MCH.register("ad_deep", {
  render() {
    const code = `# 深度异常检测三大范式

# ① AutoEncoder — 基于重构误差
class AnomalyAE(nn.Module):
    def __init__(self, in_dim, hidden_dim=32):
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 128), nn.ReLU(),
            nn.Linear(128, in_dim),
        )
    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

# 训练：仅用正常样本最小化 MSE
# 推理：异常分数 = ||x - AE(x)||²  (重构差大 = 异常)


# ② VAE — 变分自编码器，给概率异常分
class VAE(nn.Module):
    def __init__(self, in_dim, z_dim=10):
        self.encoder = nn.Linear(in_dim, 64)
        self.fc_mu = nn.Linear(64, z_dim)
        self.fc_logvar = nn.Linear(64, z_dim)
        self.decoder = nn.Linear(z_dim, in_dim)
    def forward(self, x):
        h = F.relu(self.encoder(x))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # 重参数化
        return self.decoder(z), mu, logvar

# 损失 = 重构损失 + KL 散度（ELBO）
# 异常分数 = -log p(x) ≈ 重构误差 + KL 项


# ③ DeepSVDD — One-Class 超球学习
class DeepSVDD(nn.Module):
    def __init__(self, c):
        self.c = c               # 超球心（预训练 + 固定）
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 32),
        )
    def forward(self, x):
        return torch.sum((self.net(x) - self.c) ** 2, dim=1)
# 训练：最小化 ||net(x) - c||²（所有样本压到超球中心附近）
# 推理：dist 越大 = 越异常`;

    return `
      ${MCH.hero({
        icon: "🔬",
        name: "深度异常检测 — AutoEncoder / VAE / DeepSVDD",
        en: "Deep Learning for Anomaly Detection",
        tags: ["AutoEncoder", "VAE", "DeepSVDD", "GANomaly", "重构误差"],
        meta: ["◈ 高维数据友好", "⚡ 图像/时序/日志通用"],
      })}

      ${MCH.versionSection("ad_deep")}

      <div class="section">
        <h2>1. 深度异常检测的核心思路</h2>
        <p class="text-sm text-slate-600">本质：<b>"学习正常，识别偏离"</b>。只用正常样本训练，让模型对<b>未见过的异常模式</b>表现不佳：</p>
        <div class="grid-3">
          <div class="card" style="border-top:3px solid #8b5cf6;">
            <h3 class="font-bold text-violet-700">🔄 重构派</h3>
            <p class="text-xs text-slate-600 mt-2">AE / VAE：学一个低维 bottleneck，正常样本能重构好，异常重构差。</p>
          </div>
          <div class="card" style="border-top:3px solid #06b6d4;">
            <h3 class="font-bold text-cyan-700">🎯 单类派</h3>
            <p class="text-xs text-slate-600 mt-2">DeepSVDD：把所有正常样本压进超球，越远越异常。</p>
          </div>
          <div class="card" style="border-top:3px solid #ec4899;">
            <h3 class="font-bold text-pink-700">🎭 对抗派</h3>
            <p class="text-xs text-slate-600 mt-2">GANomaly：生成器+判别器联合，生成器重构异常样本困难。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 三大方法对比</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>原理</th><th>输出</th><th>适合场景</th></tr></thead>
          <tbody>
            <tr><td><b>AutoEncoder</b></td><td>最小化重构误差</td><td>重构 MSE 作为分数</td><td>结构化/时序/图像通用</td></tr>
            <tr><td><b>VAE</b></td><td>ELBO = 重构 + KL</td><td>概率分数 −log p(x)</td><td>需要概率输出场景</td></tr>
            <tr><td><b>DAGMM</b> (2018)</td><td>AE + GMM 联合</td><td>GMM 似然</td><td>高维稀疏</td></tr>
            <tr><td><b>DeepSVDD</b></td><td>压缩到超球内</td><td>距离超球心</td><td>一分类明确</td></tr>
            <tr><td><b>GANomaly</b></td><td>生成 + 判别器</td><td>重构 + 判别分数</td><td>医疗图像</td></tr>
            <tr><td><b>PatchCore</b> (CVPR'22)</td><td>预训练 CNN 特征匹配</td><td>最近邻距离</td><td>🏆 工业缺陷 MVTec SOTA</td></tr>
            <tr><td><b>Anomaly Transformer</b></td><td>Series/Prior Attention</td><td>关联差异</td><td>时序异常</td></tr>
            <tr><td><b>🆕 EfficientAD (2024)</b></td><td>学生-教师蒸馏</td><td>特征差</td><td>工业视觉 实时 SOTA</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：AutoEncoder 异常检测可视化</h2>
        <p class="text-sm text-slate-600">2D 数据的 AE 重构：调节瓶颈维度看正常区域的"重构地图"，异常点分数高。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "ad-bottleneck", label: "Bottleneck z_dim", min: 1, max: 20, step: 1, value: 3 })}
              ${MCH.slider({ id: "ad-threshold", label: "异常阈值 percentile", min: 80, max: 99.5, step: 0.5, value: 95 })}
            </div>
            <div id="ad-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-ad" style="height:380px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 深度 vs 经典异常检测选型</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>首选</th><th>备选</th></tr></thead>
          <tbody>
            <tr><td>表格数据 / 数千样本</td><td><b>Isolation Forest</b></td><td>LOF / One-Class SVM</td></tr>
            <tr><td>高维图像 (工业缺陷)</td><td><b>PatchCore / EfficientAD</b></td><td>DRAEM</td></tr>
            <tr><td>医疗图像 (罕见病)</td><td><b>GANomaly / f-AnoGAN</b></td><td>VAE</td></tr>
            <tr><td>时序异常 (IoT/监控)</td><td><b>Anomaly Transformer / AE</b></td><td>Chronos + 重构</td></tr>
            <tr><td>日志异常</td><td><b>LLM-based (LogGPT)</b></td><td>AE + sequence model</td></tr>
            <tr><td>需要解释</td><td>Isolation Forest (SHAP)</td><td>SHAP + AE</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 开源工具</h2>
        <table class="table">
          <thead><tr><th>工具</th><th>亮点</th></tr></thead>
          <tbody>
            <tr><td><a href="https://github.com/yzhao062/pyod" target="_blank"><b>PyOD</b></a></td><td>🏆 30+ 异常检测统一接口（经典 + 深度）</td></tr>
            <tr><td><a href="https://github.com/openvinotoolkit/anomalib" target="_blank"><b>Anomalib</b></a> (Intel)</td><td>图像异常检测框架，集成 PatchCore/EfficientAD</td></tr>
            <tr><td><a href="https://github.com/salesforce/DeepOD" target="_blank"><b>DeepOD</b></a></td><td>Salesforce 深度异常检测库</td></tr>
            <tr><td><a href="https://github.com/thuml/Anomaly-Transformer" target="_blank"><b>Anomaly Transformer</b></a></td><td>清华，时序异常 ICLR'22</td></tr>
            <tr><td><a href="https://github.com/logpai/loghub" target="_blank"><b>LogHub</b></a></td><td>日志异常检测数据集 + 基线</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ad_deep").pros, MCH.getById("ad_deep").cons, MCH.getById("ad_deep").best_for)}
      </div>
    `;
  },

  mount() {
    const chart = MCH.echart(document.getElementById("chart-ad"), {
      tooltip: { formatter: p => p.data.truth ? `(${p.data.pt[0].toFixed(2)}, ${p.data.pt[1].toFixed(2)})<br/>重构误差=${p.data.score?.toFixed(3) || '-'}<br/>真实=${p.data.truth}` : "" },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", min: -4, max: 4 },
      yAxis: { type: "value", min: -4, max: 4 },
      legend: { top: 0 },
      series: [],
    });
    // 生成两个 cluster 的正常数据 + 异常点
    const data = [];
    for (let i = 0; i < 80; i++) {
      const r = MCH.randn(2, i + 10);
      data.push({ pt: [1 + r[0] * 0.5, 1 + r[1] * 0.5], truth: "normal" });
    }
    for (let i = 0; i < 80; i++) {
      const r = MCH.randn(2, i + 200);
      data.push({ pt: [-1 + r[0] * 0.5, -1 + r[1] * 0.5], truth: "normal" });
    }
    const outliers = [[3, 3], [-3, 2.5], [3, -3], [-3, -3], [2, -2], [-2, 3], [0, 3.5]];
    outliers.forEach(p => data.push({ pt: p, truth: "outlier" }));

    const update = () => {
      const z = parseInt(document.getElementById("ad-bottleneck").value);
      const pct = parseFloat(document.getElementById("ad-threshold").value);
      // 模拟 AE 重构误差（距离 cluster 中心）
      data.forEach(d => {
        const d1 = Math.hypot(d.pt[0] - 1, d.pt[1] - 1);
        const d2 = Math.hypot(d.pt[0] + 1, d.pt[1] + 1);
        const minD = Math.min(d1, d2);
        // bottleneck 小 → 重构能力强 → 正常样本误差更小，但异常样本的误差仍大
        const capacity = 1 + 0.3 * (20 - z) / 19;
        d.score = minD * capacity + (Math.random() - 0.5) * 0.1;
      });
      const scores = data.map(d => d.score).sort((a, b) => a - b);
      const threshold = scores[Math.floor(scores.length * pct / 100)];
      const normal = data.filter(d => d.score < threshold);
      const detected = data.filter(d => d.score >= threshold);
      chart.setOption({
        legend: { data: ["正常", "检测为异常"], top: 0 },
        series: [
          { name: "正常", type: "scatter", data: normal.map(d => ({ value: d.pt, pt: d.pt, score: d.score, truth: d.truth })), itemStyle: { color: "#94a3b8" }, symbolSize: 6 },
          { name: "检测为异常", type: "scatter", data: detected.map(d => ({ value: d.pt, pt: d.pt, score: d.score, truth: d.truth })), itemStyle: { color: "#ef4444" }, symbol: "pin", symbolSize: 16 },
        ],
      });
      const tp = detected.filter(d => d.truth === "outlier").length;
      const fp = detected.filter(d => d.truth === "normal").length;
      const real = data.filter(d => d.truth === "outlier").length;
      document.getElementById("ad-info").innerHTML = `
        <b>Bottleneck z_dim</b>: ${z} / 20<br/>
        <b>阈值 P${pct}</b> : score ≥ ${threshold.toFixed(3)}<br/>
        <b>检测出异常</b>: ${detected.length} 个 (TP=${tp}, FP=${fp})<br/>
        <b>Recall</b>: <span style="color:#10b981;">${(tp / real * 100).toFixed(0)}%</span> · 
        <b>Precision</b>: <span style="color:#4f46e5;">${detected.length ? (tp / detected.length * 100).toFixed(0) : 0}%</span><br/>
        <span style="color:#64748b;">瓶颈越小约束越强，异常重构误差放大；但太小会伤害正常样本。</span>
      `;
    };
    ["ad-bottleneck", "ad-threshold"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
