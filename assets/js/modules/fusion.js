/* 模块：Perceiver-IO 融合 */
MCH.register("fusion", {
  render() {
    const code = `# Perceiver-IO 跨模态融合 + Modality Dropout
# 来源：src/models/fusion.py

class PerceiverIOFusion(nn.Module):
    """5 模态 Token → K 个 Latent 查询 → 4 层 Cross-Attn + Self-Attn → 512-d 统一表征

    核心思想（Jaegle et al., 2021）:
    1) 用 K 个可学习 Latent Token 作为 Query；
    2) Cross-Attention(K, ΣN_i) 把所有模态信息"吸收"到 Latent 里；
    3) Latent 之间再做 Self-Attention 让模态互相交互；
    4) 复杂度 O(K·ΣN) ≪ O((ΣN)²)  — 当 ΣN 大、K 小时收益显著
    """
    def __init__(self, dim=512, num_modalities=5, num_latents=32,
                 num_layers=4, num_heads=8, modality_dropout=0.15,
                 keep_text_always=True):
        # 可学习 Latent Query（Perceiver 的核心）
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim) * 0.02)
        # 模态类型 embedding（告诉 attention "这是 text / seq / graph / image / aux"）
        self.modality_type_emb = nn.Parameter(torch.randn(num_modalities, dim) * 0.02)
        # 整模态缺失时的占位 token（可学习兜底，避免 NaN）
        self.missing_tokens = nn.Parameter(torch.randn(num_modalities, dim) * 0.02)

        self.cross_layers = nn.ModuleList([CrossAttnBlock(dim, num_heads)
                                           for _ in range(num_layers)])
        self.self_layers  = nn.ModuleList([SelfAttnBlock(dim, num_heads)
                                           for _ in range(num_layers)])
        # Gated pooling over K latents
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def _apply_modality_dropout(self, tokens, masks):
        """训练期随机丢 1~2 个模态，提升生产鲁棒性（keep_text_always=True 保护文本）。"""
        if not self.training or self.modality_dropout <= 0:
            return masks
        new_masks = []
        for i, m in enumerate(masks):
            if self.keep_text_always and i == 0:       # 文本永远保留
                new_masks.append(m); continue
            keep = (torch.rand(B) > self.modality_dropout).unsqueeze(-1)
            new_masks.append(m & keep.expand_as(m))
        return new_masks

    def forward(self, modality_tokens, modality_masks):
        # 1) Modality Dropout（仅训练期）
        modality_masks = self._apply_modality_dropout(modality_tokens, modality_masks)

        # 2) 整模态缺失 → 用 missing_token 占位 + 加上 modality type embedding
        projected = []
        for i, (t, m) in enumerate(zip(modality_tokens, modality_masks)):
            any_valid = m.any(1, keepdim=True).float().unsqueeze(-1)
            t = t * any_valid + self.missing_tokens[i] * (1 - any_valid)
            t = t + self.modality_type_emb[i]
            projected.append(t)

        inputs = torch.cat(projected, dim=1)                  # (B, ΣN, D)
        mask   = torch.cat(modality_masks, dim=1)

        # 3) K 个 Latent 逐层 Cross-Attn + Self-Attn
        lat = self.latents.expand(B, -1, -1)                  # (B, K, D)
        for cross, selfa in zip(self.cross_layers, self.self_layers):
            lat = cross(lat, inputs, mask)                    # Q=lat, K=V=inputs
            lat = selfa(lat)                                  # lat 内部交互

        # 4) Gated pooling — softmax 权重聚合 K 个 latent
        g = self.gate(lat).squeeze(-1)                        # (B, K)
        alpha = torch.softmax(g, -1).unsqueeze(-1)
        return (lat * alpha).sum(1)                           # (B, D)`;

    return `
      ${MCH.hero({
        icon: "F",
        name: "跨模态融合 — Perceiver-IO + Modality Dropout",
        en: "Multi-Modal Fusion · K Latents × Cross-Attention × 4 Layers",
        tags: ["K=32 Latent", "4 × Cross+Self Attn", "Modality Dropout p=0.15", "缺失 missing_token", "O(K·ΣN)"],
        meta: ["◈ 5 模态输入", "⚡ 显存 ÷ 10 vs full attn", "◇ 输出 512-d 统一表征"],
      })}

      <div class="section">
        <h2>1. 算法原理</h2>
        <p class="text-sm text-slate-600 leading-relaxed">
          5 路模态 Token 长度差异巨大（文本 256 · 交易 512 · 图/图像/辅助 1），直接拼起来走 self-attention：
        </p>
        <div class="formula-block">
          $$ \\text{Complexity}_{\\text{full}} = O\\Big( \\big(\\textstyle\\sum N_i\\big)^2 \\cdot D \\Big) $$
        </div>
        <p class="text-sm text-slate-600">
          当 ΣN ≈ 770 时 self-attn ≈ <b>60 万</b> 对点积，训练/推理都吃不消。Perceiver-IO 用 K 个 Latent Token 作为"信息瓶颈"查询：
        </p>
        <div class="formula-block">
          $$ \\text{Complexity}_{\\text{perceiver}} = O\\Big( K \\cdot \\textstyle\\sum N_i \\cdot D \\cdot L \\Big) \\quad (K = 32,\\; L = 4) $$
        </div>

        <h3>· 核心架构</h3>
        <div class="mermaid">
flowchart LR
    subgraph Inputs[5 路模态 Token]
        T1[Text 256×D]
        T2[Seq 512×D]
        T3[Graph 1×D]
        T4[Image 1×D]
        T5[Aux 1×D]
    end
    L[Latent Queries<br/>K=32 × D]
    CA["CrossAttn<br/>Q=Latent, K=V=Inputs"]
    SA[Self-Attn over Latents]
    G[Gated Pool]
    Out[统一表征 D=512]
    T1 & T2 & T3 & T4 & T5 --> CA
    L --> CA --> SA --> CA
    SA --> G --> Out
        </div>

        ${MCH.info(`
          <b>Modality Dropout 的秘密</b>：训练期以 p=0.15 随机丢非文本模态，模拟生产环境"图/关系缺失"；
          整模态缺失时用可学习的 <code>missing_tokens[i]</code> 占位而非 0 向量 —— 这让模型<b>对缺失敏感</b>但不崩溃。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互可视化</h2>
        <div class="grid-2">
          <div>
            <h3>· 复杂度对比：Full-Attn vs Perceiver-IO</h3>
            <div class="ctrl-panel" style="margin-bottom:12px;">
              ${MCH.slider({ id: "fus-K", label: "Latent 数 K", min: 4, max: 256, step: 4, value: 32 })}
              ${MCH.slider({ id: "fus-L", label: "层数 L", min: 1, max: 12, step: 1, value: 4 })}
              ${MCH.slider({ id: "fus-D", label: "隐层维度 D", min: 64, max: 1024, step: 64, value: 512 })}
            </div>
            <div id="chart-complexity" style="height:320px;"></div>
          </div>

          <div>
            <h3>· Modality Dropout 效果</h3>
            <div class="ctrl-panel" style="margin-bottom:12px;">
              ${MCH.slider({ id: "md-p", label: "Modality Dropout p", min: 0, max: 0.5, step: 0.01, value: 0.15, format: (v) => (parseFloat(v) * 100).toFixed(0) + "%" })}
              <label class="text-xs text-slate-600 flex items-center gap-2 mt-3">
                <input type="checkbox" id="md-keep-text" checked /> keep_text_always=True（保护文本模态）
              </label>
            </div>
            <div class="card">
              <h4 class="font-semibold text-slate-800 text-sm">· 训练期每 step 期望保留模态数（5 模态）</h4>
              <div id="md-expected" class="text-xs text-slate-600 mt-2"></div>
            </div>
            <div id="chart-md" style="height:260px;margin-top:10px;"></div>
          </div>
        </div>

        <h3 style="margin-top:24px;">· Latent × Modality 的 Cross-Attention 热图（模拟）</h3>
        <div id="chart-attn-fusion" style="height:380px;"></div>
      </div>

      <div class="section">
        <h2>4. 消融实验（预期）</h2>
        <table class="table">
          <thead><tr><th>配置</th><th>风险 AUC</th><th>类目 Acc</th><th>显存占用</th><th>说明</th></tr></thead>
          <tbody>
            <tr><td>无 Fusion (concat + MLP)</td><td>0.881</td><td>0.724</td><td>低</td><td>baseline</td></tr>
            <tr><td>Full Self-Attention</td><td>0.893</td><td>0.746</td><td>24 GB × B=32</td><td>精度好但不可扩展</td></tr>
            <tr><td><b>Perceiver-IO K=32</b></td><td><b>0.895</b></td><td><b>0.752</b></td><td>8 GB × B=32</td><td>当前方案</td></tr>
            <tr><td>+ Modality Dropout</td><td>0.901</td><td>0.758</td><td>同上</td><td>显著提升生产环境鲁棒性</td></tr>
            <tr><td>+ Missing Token</td><td><b>0.905</b></td><td><b>0.761</b></td><td>同上</td><td>对图/关系缺失用户友好</td></tr>
          </tbody>
        </table>
      </div>
    `;
  },

  mount() {
    // 复杂度对比
    const cChart = MCH.echart(document.getElementById("chart-complexity"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 80, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "ΣN (Token 总数)", min: 0, max: 2000 },
      yAxis: { type: "log", name: "算力 (点积次数)", logBase: 10 },
      series: [
        { name: "Full Self-Attn: O(ΣN² · D · L)", type: "line", showSymbol: false, smooth: true, data: [], color: "#ef4444", lineStyle: { width: 3 } },
        { name: "Perceiver-IO: O(K · ΣN · D · L)", type: "line", showSymbol: false, smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
      ],
    });
    const updateC = () => {
      const K = parseInt(document.getElementById("fus-K").value);
      const L = parseInt(document.getElementById("fus-L").value);
      const D = parseInt(document.getElementById("fus-D").value);
      const Ns = MCH.linspace(10, 2000, 50);
      cChart.setOption({
        series: [
          { data: Ns.map(N => [N, N * N * D * L]) },
          { data: Ns.map(N => [N, K * N * D * L]) },
        ],
      });
    };
    ["fus-K", "fus-L", "fus-D"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateC();
      });
    });
    updateC();

    // Modality Dropout 分布
    const mdChart = MCH.echart(document.getElementById("chart-md"), {
      tooltip: { trigger: "axis" },
      grid: { left: 50, right: 20, top: 30, bottom: 40 },
      xAxis: { type: "category", name: "保留模态数", data: [1, 2, 3, 4, 5] },
      yAxis: { type: "value", name: "概率", max: 1 },
      series: [{ type: "bar", barWidth: 40, data: [], label: { show: true, position: "top", formatter: (p) => (p.value * 100).toFixed(1) + "%" } }],
    });
    const updateMd = () => {
      const p = parseFloat(document.getElementById("md-p").value);
      const keepText = document.getElementById("md-keep-text").checked;
      const N = 5;
      const droppable = keepText ? 4 : 5;
      const fixedKeep = keepText ? 1 : 0;
      // P(保留 m 个在 droppable 里) = C(droppable, m-fixedKeep) * p^(droppable - m + fixedKeep) * (1-p)^(m-fixedKeep)
      const C = (n, k) => { let r = 1; for (let i = 0; i < k; i++) r = r * (n - i) / (i + 1); return r; };
      const probs = [];
      let expected = 0;
      for (let m = 1; m <= 5; m++) {
        const k = m - fixedKeep;
        if (k < 0 || k > droppable) { probs.push(0); continue; }
        const prob = C(droppable, k) * Math.pow(1 - p, k) * Math.pow(p, droppable - k);
        probs.push(prob);
        expected += m * prob;
      }
      mdChart.setOption({
        series: [{ data: probs.map((v, i) => ({ value: v, itemStyle: { color: (i + 1 >= 4) ? "#10b981" : (i + 1 >= 2 ? "#f59e0b" : "#ef4444") } })) }],
      });
      document.getElementById("md-expected").innerHTML = `期望保留模态数 E[m] = <b style="color:#4f46e5;">${expected.toFixed(2)}</b> / 5`;
    };
    document.getElementById("md-p").addEventListener("input", (e) => {
      document.getElementById("md-p-val").textContent = (parseFloat(e.target.value) * 100).toFixed(0) + "%";
      updateMd();
    });
    document.getElementById("md-keep-text").addEventListener("change", updateMd);
    updateMd();

    // Cross-Attn 热图
    const modalityNames = ["Text[0-7]", "Seq[0-23]", "Graph", "Image", "Aux"];
    const tokenRanges = [8, 24, 1, 1, 1];  // 模拟缩减版
    const totalN = tokenRanges.reduce((a, b) => a + b, 0);
    const K = 16;  // 缩减的 latent
    const data = [];
    for (let k = 0; k < K; k++) {
      let idx = 0;
      for (let m = 0; m < tokenRanges.length; m++) {
        for (let t = 0; t < tokenRanges[m]; t++) {
          // 模拟：每个 latent 偏好特定模态
          const affinity = [0.7, 0.5, 0.3, 0.4, 0.2][(k + m) % 5];
          const v = Math.max(0, affinity * (0.5 + 0.5 * Math.sin(k * 0.7 + t * 0.3)) + Math.random() * 0.1);
          data.push([idx, k, v]);
          idx++;
        }
      }
    }
    const boundary = [];
    let acc = 0;
    tokenRanges.forEach((r, i) => { acc += r; if (i < tokenRanges.length - 1) boundary.push(acc - 0.5); });
    MCH.echart(document.getElementById("chart-attn-fusion"), {
      tooltip: { formatter: (p) => `Latent ${p.data[1]} ← Token ${p.data[0]}<br/>attn = ${p.data[2].toFixed(3)}` },
      grid: { left: 80, right: 60, top: 40, bottom: 60 },
      xAxis: { type: "category", name: "Input Token (模态拼接)", data: [...Array(totalN).keys()] },
      yAxis: { type: "category", data: [...Array(K).keys()].map(i => `L${i}`), name: "Latent", inverse: true },
      visualMap: { min: 0, max: 1, orient: "horizontal", left: "center", bottom: 10, calculable: true, inRange: { color: ["#f8fafc", "#c7d2fe", "#6366f1", "#3730a3"] } },
      series: [{
        type: "heatmap", data,
        markLine: {
          symbol: "none", silent: true,
          lineStyle: { color: "#ef4444", width: 2, type: "dashed" },
          data: boundary.map(x => ({ xAxis: x })),
        },
      }],
    });
  },
});
