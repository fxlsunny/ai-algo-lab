/* 模块：交易序列 Encoder — BST + Time2Vec */
MCH.register("seq_encoder", {
  render() {
    const code = `# 核心结构：BST (Behavior-Sequence-Transformer) + Time2Vec 连续时间
# 来源：src/models/seq_encoder.py

class Time2Vec(nn.Module):
    """Kazemi et al., 2019:  v[0] = w0·t + b0  (线性分量)
                              v[i] = sin(wi·t + bi)  for i≥1  (周期分量)"""
    def __init__(self, in_dim=1, out_dim=16):
        self.w = nn.Parameter(torch.randn(in_dim, out_dim) * 0.02)
        self.b = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        v = x @ self.w + self.b                              # (B, T, out_dim)
        return torch.cat([v[..., :1], torch.sin(v[..., 1:])], -1)


class TxnTokenEmbedding(nn.Module):
    """每笔交易的 7 个离散字段独立 Embedding，sum 成一个 Token。"""
    def __init__(self, cfg, token_dim=128):
        self.amount_emb = nn.Embedding(cfg.amount_buckets, token_dim)   # log2 分桶
        self.hour_emb   = nn.Embedding(24, token_dim)
        self.dow_emb    = nn.Embedding(7,  token_dim)
        self.scene_emb  = nn.Embedding(200, token_dim)
        self.counter_emb= nn.Embedding(1<<20, token_dim)  # 对手方 hash 2^20
        self.gap_emb    = nn.Embedding(16, token_dim)      # Δt log2 分桶
        self.is_night_emb = nn.Embedding(2, token_dim)
        self.t2v = Time2Vec(1, 16)
        self.t2v_proj = nn.Linear(16, token_dim)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, seq):
        x = (self.amount_emb(seq["amount_bucket"])
           + self.hour_emb(seq["hour"]) + self.dow_emb(seq["dow"])
           + self.scene_emb(seq["scene_id"])
           + self.counter_emb(seq["counter_hash"])
           + self.gap_emb(seq["gap_bucket"])
           + self.is_night_emb(seq["is_night"]))
        if "delta_t" in seq:
            t2v = self.t2v(seq["delta_t"].float().unsqueeze(-1))
            x = x + self.t2v_proj(t2v)                       # 加入连续 Δt 编码
        return self.norm(x)


class TransactionSeqEncoder(nn.Module):
    """BST：4 层 Transformer Encoder + [CLS] 池化"""
    def __init__(self, cfg, token_dim=128, hidden_dim=512, num_layers=4, num_heads=8):
        self.token = TxnTokenEmbedding(cfg, token_dim)
        self.cls   = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)
        self.pos   = nn.Embedding(max_len + 1, token_dim)
        self.empty_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)  # 冷启兜底
        layer = nn.TransformerEncoderLayer(token_dim, num_heads, batch_first=True,
                                            norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.proj = nn.Sequential(nn.Linear(token_dim, hidden_dim),
                                  nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, seq, pad_mask):
        tok = self.token(seq)
        tok = torch.cat([self.cls.expand(B, -1, -1), tok], 1)
        tok = tok + self.pos(torch.arange(T+1))
        h = self.encoder(tok, src_key_padding_mask=~pad_mask)
        vec = self.proj(h[:, 0])                              # [CLS] 池化
        # 全部无交易的冷启商户 → empty_token 兜底
        any_valid = pad_mask.any(1, keepdim=True).float()
        return vec * any_valid + self.empty_token * (1 - any_valid)`;

    return `
      ${MCH.hero({
        icon: "S",
        name: "交易序列 Encoder — BST + Time2Vec",
        en: "Behavior Sequence Transformer · with continuous time encoding",
        tags: ["BST 4 层 Transformer", "Time2Vec 线性+正弦", "7 字段离散桶", "T ≤ 512", "冷启 empty_token"],
        meta: ["◈ token_dim=128", "⚡ 4 heads × 4 layers", "◇ 输出 512-d"],
      })}

      <div class="section">
        <h2>1. 算法原理</h2>
        <p class="text-sm text-slate-600 leading-relaxed">
          交易行为是商户风险最强的"动态信号"。本模块继承阿里 <b>BST</b>（Behavior-Sequence-Transformer）思路，
          将每笔交易抽象为一个 token，叠加 <b>Time2Vec</b> 对 Δt 做连续编码后送入 4 层 Transformer 提取序列表征。
        </p>

        <h3>· 每笔交易的 7 个离散字段</h3>
        <table class="table">
          <thead><tr><th>字段</th><th>桶数</th><th>分桶方式</th><th>业务语义</th></tr></thead>
          <tbody>
            <tr><td>amount_bucket</td><td>32</td><td>$\\lfloor \\log_2(\\text{amt}+1) \\rfloor$</td><td>对数金额，覆盖小额→大额</td></tr>
            <tr><td>hour</td><td>24</td><td>小时</td><td>日周期</td></tr>
            <tr><td>dow</td><td>7</td><td>星期</td><td>周周期</td></tr>
            <tr><td>scene_id</td><td>200</td><td>MD5 hash % 200</td><td>收款场景（微信/支付宝/聚合码/...）</td></tr>
            <tr><td>counter_hash</td><td>2²⁰</td><td>MD5 hash % 2²⁰</td><td>对手方 uin，高基数哈希</td></tr>
            <tr><td>gap_bucket</td><td>16</td><td>$\\lfloor \\log_2(\\Delta t + 1) \\rfloor$</td><td>与前笔间隔</td></tr>
            <tr><td>is_night</td><td>2</td><td>0 / 1</td><td>夜间 [0:00, 6:00)</td></tr>
          </tbody>
        </table>

        <h3>· Time2Vec：连续时间编码</h3>
        <div class="formula-block">
          $$ \\text{Time2Vec}(t) = \\big[\\,w_0 t + b_0\\,,\\; \\sin(w_1 t + b_1),\\; \\sin(w_2 t + b_2),\\; \\dots\\big] $$
        </div>
        ${MCH.info(`
          <b>为什么用 Time2Vec 而非绝对位置 Embedding？</b>
          <ul style="margin-top:6px;padding-left:20px;">
            <li>位置 Embedding 要求离散索引，对<b>非均匀采样</b>的交易流不自然；</li>
            <li>Time2Vec 的 <b>线性分量</b> 可学习一次性趋势；<b>正弦分量</b> 可学习多尺度周期（日/周/月）；</li>
            <li>上线前的 Δt 直接来自交易时间戳差值，<b>无须对齐时间片</b>。</li>
          </ul>
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
            <h3>· Time2Vec 波形（可调频率 / 相位）</h3>
            <div class="ctrl-panel" style="margin-bottom:12px;">
              ${MCH.slider({ id: "t2v-w0", label: "线性斜率 w0", min: -0.5, max: 0.5, step: 0.01, value: 0.05 })}
              ${MCH.slider({ id: "t2v-w1", label: "正弦频率 w1（日周期）", min: 0.1, max: 5, step: 0.05, value: 0.8 })}
              ${MCH.slider({ id: "t2v-w2", label: "正弦频率 w2（周周期）", min: 0.01, max: 2, step: 0.01, value: 0.12 })}
              ${MCH.slider({ id: "t2v-w3", label: "正弦频率 w3（细粒度）", min: 1, max: 10, step: 0.1, value: 3 })}
            </div>
            <div id="chart-t2v" style="height:320px;"></div>
          </div>

          <div>
            <h3>· 金额分桶（log2 bucket）</h3>
            <div class="ctrl-panel" style="margin-bottom:12px;">
              ${MCH.slider({ id: "amt-sample-N", label: "采样金额笔数 N", min: 200, max: 5000, step: 100, value: 1000 })}
            </div>
            <div id="chart-amount" style="height:320px;"></div>
          </div>
        </div>

        <h3 style="margin-top:18px;">· 模拟交易序列 — Transformer 自注意力热图</h3>
        <p class="text-xs text-slate-500">模拟一段 24 笔交易的自注意力热图（对角强 + 场景聚类 + 异常点远程依赖）</p>
        <div id="chart-attn" style="height:400px;"></div>
      </div>

      <div class="section">
        <h2>4. 业务洞察</h2>
        <div class="grid-3">
          <div class="card">
            <div class="tag tag-amber">异常识别</div>
            <h4 class="font-semibold mt-2 text-slate-800">夜间金额集中</h4>
            <p class="text-xs text-slate-600 mt-1"><code>is_night=1 &amp; amount_bucket≥20</code> + 对手方分散 → 疑似<b>赌博/跑分</b>代收。</p>
          </div>
          <div class="card">
            <div class="tag tag-red">高危模式</div>
            <h4 class="font-semibold mt-2 text-slate-800">Δt 极密集</h4>
            <p class="text-xs text-slate-600 mt-1"><code>gap_bucket≤2 (Δt&lt;8s)</code> 持续数十笔 → 疑似<b>机器刷单/养商户</b>。</p>
          </div>
          <div class="card">
            <div class="tag tag-green">正常模式</div>
            <h4 class="font-semibold mt-2 text-slate-800">咖啡店日峰</h4>
            <p class="text-xs text-slate-600 mt-1"><code>hour∈{8,12,15,19}</code> + amount_bucket=5~8 + scene=wxpay_qrcode → 典型餐饮节律。</p>
          </div>
        </div>
      </div>
    `;
  },

  mount() {
    // Time2Vec 可视化
    const el = document.getElementById("chart-t2v");
    if (el) {
      const chart = MCH.echart(el, {
        tooltip: { trigger: "axis" },
        legend: { top: 0, data: ["linear w0·t", "sin(w1·t)", "sin(w2·t)", "sin(w3·t)"] },
        grid: { left: 50, right: 30, top: 40, bottom: 40 },
        xAxis: { type: "value", name: "t" },
        yAxis: { type: "value" },
        series: [],
      });
      const update = () => {
        const ts = MCH.linspace(0, 24, 200);
        const w0 = parseFloat(document.getElementById("t2v-w0").value);
        const w1 = parseFloat(document.getElementById("t2v-w1").value);
        const w2 = parseFloat(document.getElementById("t2v-w2").value);
        const w3 = parseFloat(document.getElementById("t2v-w3").value);
        chart.setOption({
          series: [
            { name: "linear w0·t", type: "line", showSymbol: false, data: ts.map(t => [t, w0 * t]), smooth: true, color: "#4f46e5" },
            { name: "sin(w1·t)", type: "line", showSymbol: false, data: ts.map(t => [t, Math.sin(w1 * t)]), smooth: true, color: "#10b981" },
            { name: "sin(w2·t)", type: "line", showSymbol: false, data: ts.map(t => [t, Math.sin(w2 * t)]), smooth: true, color: "#f59e0b" },
            { name: "sin(w3·t)", type: "line", showSymbol: false, data: ts.map(t => [t, Math.sin(w3 * t)]), smooth: true, color: "#ef4444" },
          ],
        });
      };
      ["t2v-w0", "t2v-w1", "t2v-w2", "t2v-w3"].forEach(id => {
        document.getElementById(id).addEventListener("input", (e) => {
          document.getElementById(id + "-val").textContent = e.target.value;
          update();
        });
      });
      update();
    }

    // 金额分桶
    const el2 = document.getElementById("chart-amount");
    if (el2) {
      const chart2 = MCH.echart(el2, {
        tooltip: { trigger: "axis" },
        legend: { top: 0 },
        grid: { left: 50, right: 30, top: 40, bottom: 40 },
        xAxis: { type: "category", name: "amount_bucket", data: [...Array(32).keys()] },
        yAxis: { type: "value", name: "count" },
        series: [{ name: "原始金额分布 (log bucket)", type: "bar", barWidth: 14, data: [] }],
      });
      const update2 = () => {
        const N = parseInt(document.getElementById("amt-sample-N").value);
        const buckets = new Array(32).fill(0);
        // 模拟重尾分布：lognormal
        for (let i = 0; i < N; i++) {
          const g = MCH.randn(1, i + 1)[0];
          const amt = Math.exp(3 + g * 1.5);
          const b = Math.min(31, Math.floor(Math.log2(amt + 1)));
          buckets[b]++;
        }
        chart2.setOption({
          series: [{
            data: buckets.map((v, i) => ({ value: v, itemStyle: { color: i >= 20 ? "#ef4444" : (i >= 12 ? "#f59e0b" : "#4f46e5") } })),
          }],
        });
      };
      document.getElementById("amt-sample-N").addEventListener("input", (e) => {
        document.getElementById("amt-sample-N-val").textContent = e.target.value;
        update2();
      });
      update2();
    }

    // Attention 热图
    const el3 = document.getElementById("chart-attn");
    if (el3) {
      const N = 24;
      const data = [];
      // 构造一个结构化的注意力矩阵
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < N; j++) {
          let v = 0;
          // 对角
          v += Math.exp(-Math.abs(i - j) * 0.3) * 0.8;
          // 同场景聚类（每 6 笔一个 scene）
          if (Math.floor(i / 6) === Math.floor(j / 6)) v += 0.3;
          // 异常 long-range
          if ((i === 3 && j === 19) || (j === 3 && i === 19)) v += 0.7;
          v += Math.random() * 0.12;
          data.push([j, i, Math.min(1, v)]);
        }
      }
      MCH.echart(el3, {
        tooltip: { formatter: (p) => `Q token ${p.data[1]} · K token ${p.data[0]}<br/>attn = ${p.data[2].toFixed(3)}` },
        grid: { left: 60, right: 60, top: 30, bottom: 50 },
        xAxis: { type: "category", data: [...Array(N).keys()], name: "Key Token" },
        yAxis: { type: "category", data: [...Array(N).keys()], name: "Query Token", inverse: true },
        visualMap: { min: 0, max: 1, calculable: true, orient: "horizontal", left: "center", bottom: 10, inRange: { color: ["#f8fafc", "#c7d2fe", "#6366f1", "#3730a3"] } },
        series: [{ type: "heatmap", data, progressive: 0 }],
      });
    }
  },
});
