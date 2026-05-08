/* 模块：GPT / Decoder-only Transformer */
MCH.register("nn_gpt", {
  render() {
    const code = `# GPT 架构：Decoder-only Transformer
# 核心：Causal Mask + Next-Token Prediction + 位置编码

class GPTBlock(nn.Module):
    """一个 Transformer Block：Pre-LN + MHA(causal) + MLP + 两次残差"""
    def __init__(self, d_model=768, n_heads=12, ffn_mult=4, dropout=0.1):
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),                              # 或 SwiGLU (LLaMA)
            nn.Linear(ffn_mult * d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    """关键：causal mask 让 position i 只能看 0..i，不能看未来"""
    def __init__(self, d_model, n_heads, dropout):
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
    def forward(self, x):                     # x: (B, T, D)
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # each (B, H, T, d)
        # 用 PyTorch 2.0+ 的 F.scaled_dot_product_attention（自动用 Flash Attention）
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class GPT(nn.Module):
    """完整 GPT：Embedding + PosEmb + N×Block + LMHead"""
    def __init__(self, vocab_size=50257, d_model=768, n_layers=12, n_heads=12,
                 max_seq_len=2048, dropout=0.1):
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)        # 绝对位置（GPT-2）
        # 现代：RoPE (Rotary Position Embedding, LLaMA) / ALiBi (BLOOM)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # 权重共享：输入 embedding 与输出投影矩阵共享
        self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)                # (B, T, D)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                  # (B, T, V)
        return logits

    def generate(self, idx, max_new=100, temp=0.8, top_p=0.9):
        """自回归解码：一次生成一个 token（配 KV Cache 大幅加速）"""
        for _ in range(max_new):
            logits = self(idx[:, -self.max_seq_len:])[:, -1] / temp
            # top-p 采样
            probs = F.softmax(logits, -1)
            # ... (top-p filter)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx`;

    return `
      ${MCH.hero({ icon: "⫶", name: "GPT / Decoder-only Transformer", en: "Generative Pre-trained Transformer (Radford 2018)", tags: ["Causal Mask", "Next-Token", "自回归", "LLM 主流架构"], meta: ["◈ GPT-2/3/4, LLaMA, Mistral", "⚡ 预训练 + 微调范式", "◇ 涌现能力"] })}

      ${MCH.versionSection("nn_gpt")}

      <div class="section">
        <h2>1. GPT 与 BERT 的核心差异</h2>
        <p class="text-sm text-slate-600">都是 Transformer，但方向不同决定了用途：</p>
        <table class="table">
          <thead><tr><th>维度</th><th>GPT (Decoder-only)</th><th>BERT (Encoder-only)</th></tr></thead>
          <tbody>
            <tr><td>Attention Mask</td><td><b>Causal (下三角)</b></td><td>双向 (全矩阵)</td></tr>
            <tr><td>预训练目标</td><td>Next-Token Prediction<br/>$P(x_t \\mid x_{&lt;t})$</td><td>Masked LM<br/>$P(x_{\\text{mask}} \\mid x_{\\setminus \\text{mask}})$</td></tr>
            <tr><td>能生成文本</td><td>✓ 天然</td><td>❌ (需改 Encoder-Decoder)</td></tr>
            <tr><td>理解能力</td><td>中等</td><td>强</td></tr>
            <tr><td>可规模化</td><td>✓ 已到万亿参数</td><td>⚠ 通常 &lt; 1B</td></tr>
            <tr><td>代表模型</td><td>GPT-4, LLaMA, Mistral, Qwen</td><td>BERT, RoBERTa, DeBERTa</td></tr>
            <tr><td>零样本能力</td><td>✓ 很强</td><td>弱</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. 核心：Causal Self-Attention</h2>
        <div class="formula-block">
          $$ \\text{CausalAttn}(Q, K, V) = \\text{softmax}\\Big(\\frac{QK^\\top}{\\sqrt{d_k}} + M\\Big) V $$
          其中 M 是下三角 mask：$M_{ij} = 0$ 若 $i \\geq j$，否则 $-\\infty$。让 position i 只能看到 $\\leq i$ 的 tokens。
        </div>
        <div id="chart-causal" style="height:320px;"></div>
      </div>

      <div class="section">
        <h2>3. 核心代码（简化版 GPT）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：模型参数量 Scaling Law</h2>
        <p class="text-sm text-slate-600">Kaplan 2020 / Hoffmann 2022 (Chinchilla) 给出了 LLM 的<b>规模化规律</b>：loss 随参数量/数据量/计算量呈幂律下降。</p>

        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "gpt-d", label: "d_model", min: 128, max: 12288, step: 128, value: 768 })}
              ${MCH.slider({ id: "gpt-L", label: "num_layers", min: 4, max: 128, step: 2, value: 12 })}
              ${MCH.slider({ id: "gpt-H", label: "num_heads", min: 4, max: 128, step: 4, value: 12 })}
              ${MCH.slider({ id: "gpt-V", label: "vocab_size", min: 10000, max: 200000, step: 1000, value: 50257 })}
              ${MCH.slider({ id: "gpt-S", label: "max_seq_len", min: 128, max: 131072, step: 128, value: 2048 })}
            </div>
            <div id="gpt-info" class="card mt-3"></div>
          </div>
          <div id="chart-scale" style="height:360px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 现代 GPT 的关键改进</h2>
        <table class="table">
          <thead><tr><th>组件</th><th>GPT-2 2019</th><th>LLaMA-2 2023</th><th>LLaMA-3 / Mistral 2024</th></tr></thead>
          <tbody>
            <tr><td>位置编码</td><td>绝对位置 Embedding</td><td>RoPE (旋转位置编码)</td><td>RoPE + YaRN 长度外推</td></tr>
            <tr><td>Normalization</td><td>Post-LN</td><td>Pre-RMSNorm</td><td>Pre-RMSNorm</td></tr>
            <tr><td>激活函数</td><td>GELU</td><td>SwiGLU</td><td>SwiGLU</td></tr>
            <tr><td>MHA</td><td>标准</td><td>GQA (分组 Q-Attention)</td><td>GQA + Flash Attention v2</td></tr>
            <tr><td>Tokenizer</td><td>BPE (50k)</td><td>SentencePiece (32k)</td><td>Tiktoken (100-200k)</td></tr>
            <tr><td>上下文长度</td><td>1024</td><td>4096</td><td>8k - 128k+</td></tr>
            <tr><td>训练目标</td><td>CE</td><td>CE + SFT + RLHF</td><td>CE + SFT + DPO / RLAIF</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. In-Context Learning 与涌现能力</h2>
        ${MCH.info(`
          <b>In-Context Learning (ICL)</b>：GPT 可以通过 prompt 中的示例完成新任务，<b>无需更新参数</b>。
          这是模型规模跨过阈值（通常 &gt; 10B）后的涌现能力。
          <br/><br/>
          <b>Chain-of-Thought (CoT)</b>：让模型"一步一步想"显著提升数学/逻辑题正确率。
          <br/>
          <b>工具调用 / Agent</b>：更大的模型能学会解析工具描述并选择调用。
        `, "biz")}
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("nn_gpt").pros, MCH.getById("nn_gpt").cons, MCH.getById("nn_gpt").best_for)}
      </div>
    `;
  },

  mount() {
    // Causal mask viz
    const N = 8;
    const data = [];
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      data.push([j, i, j <= i ? (0.3 + 0.7 * Math.exp(-Math.abs(i - j) * 0.4)) : 0]);
    }
    MCH.echart(document.getElementById("chart-causal"), {
      title: { text: "Causal Mask 自注意力 — 下三角", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { formatter: p => `Q ${p.data[1]} · K ${p.data[0]}<br/>attn = ${p.data[2].toFixed(3)}${p.data[2] === 0 ? " (masked: 看不到未来)" : ""}` },
      grid: { left: 50, right: 60, top: 40, bottom: 50 },
      xAxis: { type: "category", name: "Key (被看的 token)", data: [...Array(N).keys()].map(i => `t${i}`) },
      yAxis: { type: "category", name: "Query (当前 token)", data: [...Array(N).keys()].map(i => `t${i}`), inverse: true },
      visualMap: { min: 0, max: 1, orient: "horizontal", left: "center", bottom: 0, inRange: { color: ["#f1f5f9", "#c7d2fe", "#6366f1", "#312e81"] } },
      series: [{ type: "heatmap", data }],
    });

    // Scaling
    const paramChart = MCH.echart(document.getElementById("chart-scale"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 70, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "log", name: "参数量 (B)", logBase: 10 },
      yAxis: { type: "value", name: "Test Loss", inverse: false },
      series: [
        { name: "Kaplan Scaling (chinchilla optimal)", type: "line", smooth: true, showSymbol: false, color: "#ec4899", lineStyle: { width: 3 }, data: [] },
        { name: "当前配置", type: "scatter", data: [], symbolSize: 14, itemStyle: { color: "#7c3aed" } },
      ],
    });
    const famous = [
      { name: "GPT-2", params: 1.5, loss: 2.8 },
      { name: "GPT-3", params: 175, loss: 1.9 },
      { name: "LLaMA-2 70B", params: 70, loss: 2.0 },
      { name: "GPT-4 (est)", params: 1800, loss: 1.5 },
    ];
    const update = () => {
      const d = parseInt(document.getElementById("gpt-d").value);
      const L = parseInt(document.getElementById("gpt-L").value);
      const H = parseInt(document.getElementById("gpt-H").value);
      const V = parseInt(document.getElementById("gpt-V").value);
      const S = parseInt(document.getElementById("gpt-S").value);

      // 粗略参数量：emb (V·d) + pos (S·d) + per-layer [4d² (attn qkv+proj) + 8d² (ffn 4×MLP)] × L
      const embParams = V * d + S * d;
      const blockParams = 12 * d * d + 4 * d;  // attn + mlp + LN
      const total = embParams + L * blockParams;
      // Scaling law approx: L(N) ≈ (8.8e13 / N)^0.076 + 1.7
      const lossEst = Math.pow(8.8e13 / total, 0.076) + 1.7;

      const pts = MCH.linspace(7, 12.5, 60).map(v => Math.pow(10, v));
      paramChart.setOption({
        series: [
          { data: pts.map(n => [n / 1e9, Math.pow(8.8e13 / n, 0.076) + 1.7]) },
          { data: [[total / 1e9, lossEst], ...famous.map(f => ({ value: [f.params, f.loss], label: { show: true, position: "top", formatter: f.name } }))] },
        ],
      });

      // KV cache per batch × seq: 2 * L * 2(bytes fp16) * S * d
      const kvCacheBytes = 2 * L * 2 * S * d;
      const flops = 6 * total * S;  // 6N per token

      document.getElementById("gpt-info").innerHTML = `
        <div class="grid-3">
          <div><div class="text-xs text-slate-500">总参数量</div><div class="text-lg font-bold text-pink-700">${(total / 1e9).toFixed(2)} B</div></div>
          <div><div class="text-xs text-slate-500">单 token FLOPs</div><div class="text-lg font-bold text-pink-700">${(flops / 1e9).toFixed(2)} G</div></div>
          <div><div class="text-xs text-slate-500">KV Cache/batch</div><div class="text-lg font-bold text-pink-700">${(kvCacheBytes / 1024 / 1024).toFixed(1)} MB</div></div>
        </div>
        <div class="text-xs text-slate-500 mt-2">
          Scaling Law 估计 test loss ≈ <b>${lossEst.toFixed(3)}</b>（越低越好）<br/>
          FP16 权重 ≈ <b>${(total * 2 / 1024 / 1024 / 1024).toFixed(1)} GB</b>（仅模型，训练需 4×）
        </div>
      `;
    };
    ["gpt-d", "gpt-L", "gpt-H", "gpt-V", "gpt-S"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
