/* 模块：深度时序模型 */
MCH.register("ts_deep", {
  render() {
    const code = `# 现代深度时序：PatchTST (ICLR 2023) 核心思想
# - 把时序切 patch（类似 ViT 把图像切块）
# - Channel Independence：每个 channel 独立建模
# - 普通 Transformer Encoder → 预测

class PatchTSTModel(nn.Module):
    def __init__(self, context_len=336, patch_len=16, stride=8,
                 pred_len=96, num_vars=7, d_model=128, n_heads=16, n_layers=3):
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (context_len - patch_len) // stride + 1
        # 每个 patch 作为 token
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True), n_layers)
        self.head = nn.Linear(self.num_patches * d_model, pred_len)

    def forward(self, x):          # x: (B, num_vars, context_len)
        B, V, L = x.shape
        # Channel Independence：把 variable 当 batch
        x = x.reshape(B * V, L)
        # Patching
        patches = x.unfold(-1, self.patch_len, self.stride)   # (B*V, num_patches, patch_len)
        tokens = self.patch_embed(patches) + self.pos_embed
        encoded = self.encoder(tokens)                         # (B*V, num_patches, d_model)
        flat = encoded.flatten(1)
        out = self.head(flat).reshape(B, V, -1)                # (B, V, pred_len)
        return out


# iTransformer (ICLR 2024) — 更简单更有效的设计
# 关键洞察：反转 Transformer，把 variable 当 token！
class iTransformer(nn.Module):
    def __init__(self, context_len, num_vars, d_model, n_heads, n_layers, pred_len):
        # 每个 variable 的整条历史作为一个 token
        self.embed = nn.Linear(context_len, d_model)
        self.encoder = nn.TransformerEncoder(...)
        self.project = nn.Linear(d_model, pred_len)

    def forward(self, x):         # x: (B, context_len, num_vars)
        x = x.permute(0, 2, 1)    # (B, num_vars, context_len)
        tokens = self.embed(x)    # 🏆 这里反转：variable-as-token
        encoded = self.encoder(tokens)
        return self.project(encoded).permute(0, 2, 1)  # (B, pred_len, num_vars)`;

    return `
      ${MCH.hero({
        icon: "⏱",
        name: "深度时序模型 — DeepAR / TFT / PatchTST / iTransformer",
        en: "Deep Learning for Time Series Forecasting",
        tags: ["DeepAR", "N-BEATS", "TFT", "PatchTST", "iTransformer", "长序列"],
        meta: ["◈ 多序列联合训练", "⚡ 工业 SOTA"],
      })}

      ${MCH.versionSection("ts_deep")}

      <div class="section">
        <h2>1. 深度时序模型演进</h2>
        <div class="mermaid">
flowchart LR
    A[RNN 时代<br/>LSTM/GRU 2016-18] --> B[DeepAR 2020<br/>Amazon<br/>概率预测] --> C[Transformer 时代<br/>Informer 2021<br/>Autoformer 2021]
    C --> D[N-BEATS 2020<br/>纯 MLP<br/>M4 冠军] --> E[TFT 2021<br/>Google<br/>多变量 + 可解释]
    E --> F[🏆 PatchTST 2023<br/>Patching + 独立 channel] --> G[🆕 iTransformer 2024<br/>反转 token 轴]
    G --> H[🆕 TimeMixer 2024<br/>纯 MLP 多尺度]
        </div>
      </div>

      <div class="section">
        <h2>2. 四大架构范式对比</h2>
        <table class="table">
          <thead><tr><th>架构</th><th>代表模型</th><th>亮点</th><th>局限</th></tr></thead>
          <tbody>
            <tr><td>📻 RNN 系</td><td>DeepAR, DeepState</td><td>天然处理变长，概率预测</td><td>长序列慢；已被 Transformer 超越</td></tr>
            <tr><td>🪜 MLP 系</td><td>N-BEATS, N-HiTS, TimeMixer</td><td>🏆 简单极致，训练快</td><td>长序列不如 Transformer</td></tr>
            <tr><td>🔄 Transformer 系</td><td>Informer, Autoformer, TFT</td><td>建模远程依赖</td><td>O(N²) 复杂度</td></tr>
            <tr><td>🧩 Patch 系（2023+）</td><td>🏆 PatchTST, iTransformer</td><td>🆕 简单超越 SOTA</td><td>仍是 Transformer 基础</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 关键突破：从 Vanilla Transformer 到 PatchTST</h2>
        ${MCH.info(`
          <b>早期 Transformer 时序的教训</b>：2021-22 年很多论文（Informer/Autoformer/FEDformer）在为"时序专用复杂 attention"努力。
          <br/><br/>
          <b>2023 年翻转</b>：Nie 等人的 PatchTST 证明 <b>"朴素 Transformer + patching 就能超越那些花哨架构"</b>。
          关键创新仅两点：
          <ol style="margin-top:6px;padding-left:20px;">
            <li><b>Patching</b>：把 336 步序列切成 42 个 16 步 patch，token 长度 ÷8，attention 复杂度 ÷64</li>
            <li><b>Channel Independence</b>：每个变量独立建模，而不是把所有变量拼接（反直觉但有效）</li>
          </ol>
        `, "tip")}
      </div>

      <div class="section">
        <h2>4. 核心代码（PatchTST + iTransformer）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 交互：不同方法预测对比（模拟）</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              <label class="text-xs block mb-2">选择方法：</label>
              <select id="ts-method" class="text-xs border rounded p-1 w-full">
                <option value="naive">Naive (上一期 baseline)</option>
                <option value="arima">ARIMA</option>
                <option value="lstm">DeepAR / LSTM</option>
                <option value="tft">TFT</option>
                <option value="patchtst" selected>🏆 PatchTST</option>
                <option value="itrans">🆕 iTransformer</option>
              </select>
              ${MCH.slider({ id: "ts-noise", label: "数据噪声强度", min: 0, max: 1, step: 0.05, value: 0.3 })}
            </div>
            <div id="ts-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-ts" style="height:340px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>6. 开源工具</h2>
        <table class="table">
          <thead><tr><th>框架</th><th>亮点</th><th>链接</th></tr></thead>
          <tbody>
            <tr><td><b>Darts</b></td><td>unit8，sklearn 风格，50+ 模型</td><td><a href="https://github.com/unit8co/darts" target="_blank">GitHub</a></td></tr>
            <tr><td><b>NeuralForecast</b></td><td>🏆 Nixtla，PyTorch Lightning 底层，30+ 深度模型</td><td><a href="https://github.com/nixtla/neuralforecast" target="_blank">GitHub</a></td></tr>
            <tr><td><b>PyTorch Forecasting</b></td><td>TFT 原作者维护</td><td><a href="https://pytorch-forecasting.readthedocs.io/" target="_blank">文档</a></td></tr>
            <tr><td><b>GluonTS</b></td><td>Amazon，概率预测强</td><td><a href="https://ts.gluon.ai/" target="_blank">官网</a></td></tr>
            <tr><td><b>TimeSeriesLibrary</b></td><td>清华，最新论文官方实现集合</td><td><a href="https://github.com/thuml/Time-Series-Library" target="_blank">GitHub</a></td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ts_deep").pros, MCH.getById("ts_deep").cons, MCH.getById("ts_deep").best_for)}
      </div>
    `;
  },

  mount() {
    const chart = MCH.echart(document.getElementById("chart-ts"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "时间步" },
      yAxis: { type: "value" },
      series: [
        { name: "历史观测", type: "line", smooth: true, showSymbol: false, color: "#94a3b8", data: [] },
        { name: "真实未来", type: "line", smooth: true, showSymbol: false, color: "#64748b", lineStyle: { type: "dashed" }, data: [] },
        { name: "模型预测", type: "line", smooth: true, showSymbol: false, color: "#06b6d4", lineStyle: { width: 3 }, data: [] },
        { name: "预测区间", type: "line", smooth: true, showSymbol: false, color: "#cffafe", areaStyle: { opacity: 0.5 }, data: [] },
      ],
    });
    const methodQuality = {
      naive: { bias: 2, variance: 4 },
      arima: { bias: 1.5, variance: 2.5 },
      lstm: { bias: 1.2, variance: 2 },
      tft: { bias: 0.7, variance: 1.2 },
      patchtst: { bias: 0.4, variance: 0.8 },
      itrans: { bias: 0.3, variance: 0.6 },
    };
    const update = () => {
      const method = document.getElementById("ts-method").value;
      const noise = parseFloat(document.getElementById("ts-noise").value);
      const N = 120, split = 90;
      const xs = [...Array(N).keys()];
      // 生成真实序列
      const truth = xs.map(t => {
        return 50 + 0.3 * t + 10 * Math.sin(2 * Math.PI * t / 12) + 5 * Math.sin(2 * Math.PI * t / 30) + MCH.randn(1, t + 100)[0] * noise * 3;
      });
      const q = methodQuality[method];
      const pred = xs.slice(split).map((t, i) => {
        const shift = q.bias * (noise + 0.3);
        const err = MCH.randn(1, t + 200)[0] * q.variance * (noise + 0.3);
        return truth[t] + shift * Math.sin(t * 0.5) + err;
      });
      chart.setOption({
        series: [
          { data: xs.slice(0, split).map(t => [t, truth[t]]) },
          { data: xs.slice(split).map(t => [t, truth[t]]) },
          { data: xs.slice(split).map((t, i) => [t, pred[i]]) },
          { data: xs.slice(split).map((t, i) => [t, pred[i] + q.variance * 2]) },
        ],
      });
      const mse = pred.reduce((s, v, i) => s + Math.pow(v - truth[split + i], 2), 0) / pred.length;
      document.getElementById("ts-info").innerHTML = `
        <b>方法</b>：${method.toUpperCase()}<br/>
        <b>MSE</b>：<span style="color:#06b6d4;font-weight:700;">${mse.toFixed(2)}</span><br/>
        <b>偏差量级</b>：${q.bias} · <b>方差量级</b>：${q.variance}<br/>
        <span style="color:#64748b;">${method === 'patchtst' ? '🏆 PatchTST 在多变量长程上普遍 SOTA' : method === 'itrans' ? '🆕 iTransformer 简单超越复杂架构' : method === 'tft' ? '可解释 + 多变量友好' : method === 'naive' ? 'baseline：如果 Naive 都能打败你的模型，就有问题了' : '相对基础方法'}</span>
      `;
    };
    document.getElementById("ts-method").addEventListener("change", update);
    document.getElementById("ts-noise").addEventListener("input", (e) => {
      document.getElementById("ts-noise-val").textContent = e.target.value;
      update();
    });
    update();
  },
});
