/* 模块：Attention / Transformer */
MCH.register("nn_attention", {
  render() {
    const code = `# Scaled Dot-Product Attention (Vaswani et al., 2017)
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q: (B, H, N_q, d_k)  K/V: (B, H, N_k, d_k)
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)   # (B, H, N_q, N_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)                    # 每个 query 在 keys 上的分布
    return attn @ V                                      # (B, H, N_q, d_v)

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

    def forward(self, x, mask=None):
        B, N, _ = x.shape
        # 拆分多头：(B, N, d) → (B, H, N, d/H)
        q = self.Wq(x).view(B, N, H, d_k).transpose(1, 2)
        k = self.Wk(x).view(B, N, H, d_k).transpose(1, 2)
        v = self.Wv(x).view(B, N, H, d_k).transpose(1, 2)
        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, N, d_model)
        return self.Wo(out)


# Transformer Block = MHA + MLP + 两个残差 + 两个 LN
class TransformerBlock(nn.Module):
    def forward(self, x, mask=None):
        x = x + self.mha(self.ln1(x), mask)      # Pre-Norm（LLaMA/GPT 主流）
        x = x + self.mlp(self.ln2(x))
        return x`;

    return `
      ${MCH.hero({ icon: "✶", name: "Attention / Transformer 基础", en: "Self-Attention · Q/K/V · Multi-Head", tags: ["Q/K/V", "Multi-Head", "Scaled Dot-Product", "LayerNorm", "残差"], meta: ["◈ Vaswani 2017", "⚡ LLM / CV / Graph 统一架构"] })}

      ${MCH.versionSection("nn_attention")}

      <div class="section">
        <h2>1. 核心公式</h2>
        <div class="formula-block">
          $$ \\text{Attention}(Q, K, V) = \\text{softmax}\\Big(\\frac{QK^\\top}{\\sqrt{d_k}}\\Big) V $$
          <b>Multi-Head</b>：把 d_model 拆成 H 个头，每头独立 attention，再拼回投影：
          $$ \\text{MHA}(X) = W_O \\cdot \\text{concat}\\big[\\text{Attn}_h(Q_h, K_h, V_h)\\big]_{h=1}^{H} $$
        </div>
        ${MCH.info(`
          <b>"Scaled" 中的 √d_k 为什么重要？</b>
          当 d_k 大时，内积 Q·K 的方差会随 d_k 线性增长，softmax 前的 logits 太大会让分布变成"硬 one-hot"，
          梯度接近 0。除以 √d_k 后方差归一化到 1，训练更稳定。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 代码解读</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：自注意力可视化</h2>
        <p class="text-sm text-slate-600">下面是一个长度 N=12 的序列的自注意力热图。点击 <b>Query 位置</b>观察它对哪些 Keys 给予注意力。</p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <label class="text-xs text-slate-600 mr-3">模式：</label>
          <select id="attn-mode" class="text-xs border rounded p-1">
            <option value="local">局部模式 (window attention)</option>
            <option value="diag">对角偏好 (近邻依赖)</option>
            <option value="causal">因果掩码 (GPT 自回归)</option>
            <option value="global">全局关注 (BERT 双向)</option>
            <option value="sparse">稀疏模式 (Longformer)</option>
          </select>
          ${MCH.slider({ id: "attn-Q", label: "选中 Query 位置", min: 0, max: 11, step: 1, value: 6 })}
          ${MCH.slider({ id: "attn-temp", label: "温度（softmax 平滑度）", min: 0.1, max: 5, step: 0.1, value: 1 })}
        </div>

        <div class="grid-2">
          <div>
            <h3>· 自注意力矩阵（Q × K）</h3>
            <div id="chart-attn-mat" style="height:400px;"></div>
          </div>
          <div>
            <h3>· 当前 Query 的 attention 权重</h3>
            <div id="chart-attn-row" style="height:400px;"></div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 多头的意义</h2>
        <p class="text-sm text-slate-600">多头不是简单"加宽"，而是让不同头关注不同<b>子空间</b>的关系：语法 / 语义 / 位置 / 实体等。实际研究发现一些头专门做"指代消解"、"依赖关系"等。</p>
        <div id="chart-multihead" style="height:300px;"></div>
      </div>

      <div class="section">
        <h2>5. Transformer vs CNN vs RNN</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>RNN (LSTM/GRU)</th><th>CNN</th><th>Transformer</th></tr></thead>
          <tbody>
            <tr><td>长程依赖</td><td>⚠️ 梯度消失</td><td>⚠️ 需要深层</td><td>✓ 直接 O(1) 跳</td></tr>
            <tr><td>并行度</td><td>❌ 时序串行</td><td>✓ 高</td><td>✓ 最高</td></tr>
            <tr><td>复杂度</td><td>O(N·d²)</td><td>O(N·k·d²)</td><td>O(N²·d)</td></tr>
            <tr><td>位置信息</td><td>天然</td><td>隐式（感受野）</td><td>需位置编码</td></tr>
            <tr><td>数据需求</td><td>小</td><td>中</td><td>大（≥10w）</td></tr>
            <tr><td>现代首选</td><td>❌ 基本被取代</td><td>✓ 实时 CV</td><td>✓ NLP/LLM/多模态</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("nn_attention").pros,
          MCH.getById("nn_attention").cons,
          MCH.getById("nn_attention").best_for,
        )}
      </div>
    `;
  },

  mount() {
    const N = 12;
    const makeScores = (mode) => {
      const scores = [];
      for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < N; j++) {
          let s = 0;
          switch (mode) {
            case "local": s = Math.exp(-Math.abs(i - j) * 0.8); break;
            case "diag": s = Math.exp(-Math.abs(i - j) * 0.3) * (1 + 0.3 * Math.sin(i + j)); break;
            case "causal": s = j <= i ? Math.exp(-Math.abs(i - j) * 0.2) : 0; break;
            case "global": s = 0.3 + 0.7 * Math.exp(-Math.abs(i - j) * 0.4) + 0.2 * Math.cos(i * 0.5) * Math.cos(j * 0.5); break;
            case "sparse": s = (j === 0 || j === N - 1 || j === i || Math.abs(i - j) <= 1) ? 1 : 0.01; break;
          }
          row.push(s);
        }
        scores.push(row);
      }
      return scores;
    };

    const matEl = document.getElementById("chart-attn-mat");
    const rowEl = document.getElementById("chart-attn-row");
    const matChart = MCH.echart(matEl, {
      tooltip: { formatter: p => `Query ${p.data[1]} · Key ${p.data[0]}<br/>attn=${p.data[2].toFixed(3)}` },
      grid: { left: 50, right: 40, top: 30, bottom: 60 },
      xAxis: { type: "category", name: "Key", data: [...Array(N).keys()] },
      yAxis: { type: "category", name: "Query", data: [...Array(N).keys()], inverse: true },
      visualMap: { min: 0, max: 1, calculable: true, orient: "horizontal", left: "center", bottom: 10, inRange: { color: ["#f8fafc", "#c7d2fe", "#6366f1", "#3730a3"] } },
      series: [{ type: "heatmap", data: [] }],
    });
    const rowChart = MCH.echart(rowEl, {
      tooltip: {},
      grid: { left: 50, right: 30, top: 30, bottom: 40 },
      xAxis: { type: "category", data: [...Array(N).keys()], name: "Key 位置" },
      yAxis: { type: "value", name: "attention 权重", min: 0, max: 1 },
      series: [{ type: "bar", barWidth: 20, data: [] }],
    });

    const update = () => {
      const mode = document.getElementById("attn-mode").value;
      const q = parseInt(document.getElementById("attn-Q").value);
      const tau = parseFloat(document.getElementById("attn-temp").value);
      const raw = makeScores(mode);
      // softmax row-wise with temperature
      const softmax = raw.map(r => {
        const m = Math.max(...r);
        const e = r.map(v => Math.exp((v - m) / tau));
        const s = e.reduce((a, b) => a + b, 0) || 1;
        return e.map(v => v / s);
      });
      const matData = [];
      for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) matData.push([j, i, softmax[i][j]]);
      matChart.setOption({
        series: [{ data: matData, markArea: { itemStyle: { borderColor: "#ef4444", borderWidth: 2, color: "transparent" }, data: [[{ coord: [0, q] }, { coord: [N - 1, q] }]] } }],
      });
      rowChart.setOption({
        series: [{ data: softmax[q].map((v, i) => ({ value: v.toFixed(3), itemStyle: { color: i === q ? "#ef4444" : "#4f46e5" } })), label: { show: true, position: "top", formatter: p => (p.value * 100).toFixed(0) + "%", color: "#64748b" } }],
      });
    };
    ["attn-mode", "attn-Q", "attn-temp"].forEach(id => {
      const el = document.getElementById(id);
      el.addEventListener(id === "attn-mode" ? "change" : "input", (e) => {
        if (id !== "attn-mode") document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();

    // Multi-head concept
    MCH.echart(document.getElementById("chart-multihead"), {
      title: { text: "多头 Attention 的不同关注模式（示意）", left: "center", textStyle: { fontSize: 12 } },
      tooltip: {},
      legend: { bottom: 0 },
      grid: { left: 50, right: 30, top: 70, bottom: 60 },
      xAxis: { type: "category", data: ["语法结构", "语义相关", "实体关系", "位置信息", "共指消解", "话题连贯"] },
      yAxis: { type: "value", name: "某头注意强度", max: 1 },
      series: [
        { name: "Head 1 (语法)", type: "bar", data: [0.85, 0.2, 0.15, 0.6, 0.1, 0.1], color: "#4f46e5", barGap: 0.1 },
        { name: "Head 2 (语义)", type: "bar", data: [0.15, 0.9, 0.3, 0.1, 0.2, 0.4], color: "#7c3aed" },
        { name: "Head 3 (位置)", type: "bar", data: [0.2, 0.1, 0.1, 0.95, 0.1, 0.1], color: "#10b981" },
        { name: "Head 4 (共指)", type: "bar", data: [0.1, 0.3, 0.5, 0.1, 0.88, 0.2], color: "#f59e0b" },
      ],
    });
  },
});
