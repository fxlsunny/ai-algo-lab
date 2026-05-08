/* 模块：Transformer 完整架构 */
MCH.register("nn_transformer", {
  render() {
    const code = `# 完整 Transformer（Encoder-Decoder，现代化版本）
# Vaswani 2017 → LLaMA-3 现代组件

class TransformerBlock(nn.Module):
    """Pre-LN + RMSNorm + Multi-Head Attention + SwiGLU FFN"""
    def __init__(self, d_model=768, n_heads=12, ffn_mult=4):
        self.norm1 = RMSNorm(d_model)          # 现代 normalization（比 LayerNorm 快）
        self.attn = MultiHeadAttention(d_model, n_heads, rope=True)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_mult * d_model)

    def forward(self, x, attn_mask=None):
        # Pre-LN：norm 在残差之前 → 训练更稳定（大模型必用）
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x

class RMSNorm(nn.Module):
    """Root Mean Square Norm (LLaMA 采用)：比 LayerNorm 少算 mean，快 10-30%"""
    def __init__(self, dim, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x * rms           # 无 bias，无减均值

class SwiGLU(nn.Module):
    """FFN: W_2 · (Swish(W_1 x) * W_3 x)
    相比标准 FFN(W_2 · GELU(W_1 x))，有 3 个矩阵但效果显著更好"""
    def __init__(self, d_in, d_hidden):
        self.w1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w2 = nn.Linear(d_hidden, d_in, bias=False)
        self.w3 = nn.Linear(d_in, d_hidden, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

def apply_rope(q, k, theta=10000):
    """RoPE (Rotary Position Embedding, Su 2021)：
    把位置信息旋转编码到 Q/K 中，不需要独立的位置 embedding
    性质：q_m · k_n 的点积只依赖相对位置 (m - n)"""
    # 实现省略，核心是对 Q/K 做按位旋转
    return rotated_q, rotated_k


# Encoder-Decoder 完整结构（T5 / 机器翻译）
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_layers=6, n_heads=8):
        self.src_embed = nn.Embedding(vocab_size, d_model)
        self.tgt_embed = nn.Embedding(vocab_size, d_model)
        # Encoder：双向自注意力
        self.encoder = nn.ModuleList([EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        # Decoder：causal 自注意力 + encoder-decoder cross attention
        self.decoder = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt):
        enc_out = self.src_embed(src)
        for layer in self.encoder:
            enc_out = layer(enc_out)               # (B, T_src, D)
        dec_out = self.tgt_embed(tgt)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)      # cross-attend 到 enc_out
        return self.out(dec_out)`;

    return `
      ${MCH.hero({
        icon: "⟁",
        name: "Transformer 完整架构",
        en: "Full Transformer: Encoder-Decoder + Modern Components",
        tags: ["Encoder-Decoder", "RoPE", "SwiGLU", "RMSNorm", "Pre-LN", "GQA"],
        meta: ["◈ Vaswani 2017 → LLaMA 2024", "⚡ NLP / CV / 多模态统一架构", "◇ 与 GPT/BERT 的关系"],
      })}

      ${MCH.versionSection("nn_transformer")}

      <div class="section">
        <h2>1. Transformer 三大分支</h2>
        <p class="text-sm text-slate-600">原始 Vaswani 2017 的 Transformer 是 <b>Encoder-Decoder</b>，后续衍生出三大分支：</p>
        <div class="grid-3">
          <div class="card" style="border-top:3px solid #4f46e5;">
            <h3 class="font-bold text-indigo-700 text-sm">🔷 Encoder-only</h3>
            <div class="text-xs text-slate-600 mt-2">
              <b>代表</b>：BERT, RoBERTa, DeBERTa, ViT<br/>
              <b>预训练</b>：MLM (Masked Language Model)<br/>
              <b>强项</b>：理解、分类、检索、回归<br/>
              <b>弱项</b>：生成能力差<br/>
              <b>Attention</b>：双向无掩码
            </div>
          </div>
          <div class="card" style="border-top:3px solid #f59e0b;">
            <h3 class="font-bold text-amber-700 text-sm">🟡 Encoder-Decoder</h3>
            <div class="text-xs text-slate-600 mt-2">
              <b>代表</b>：原始 Transformer, T5, BART<br/>
              <b>预训练</b>：Seq2Seq / Span Corruption<br/>
              <b>强项</b>：翻译、摘要、QA（输入→输出）<br/>
              <b>弱项</b>：参数双倍，推理稍复杂<br/>
              <b>Attention</b>：Enc 双向 + Dec causal + cross
            </div>
          </div>
          <div class="card" style="border-top:3px solid #ec4899;">
            <h3 class="font-bold text-pink-700 text-sm">🔴 Decoder-only</h3>
            <div class="text-xs text-slate-600 mt-2">
              <b>代表</b>：GPT 系列, LLaMA, Mistral, Qwen<br/>
              <b>预训练</b>：Next-Token Prediction<br/>
              <b>强项</b>：生成、指令跟随、ICL<br/>
              <b>弱项</b>：理解任务略逊<br/>
              <b>Attention</b>：单向 causal
            </div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 架构总览</h2>
        <div class="mermaid">
flowchart LR
    subgraph ENC[Encoder Stack × N]
        E1[Input Embed + PE/RoPE] --> E2[Self-Attention 双向]
        E2 --> E3[Add &amp; Norm]
        E3 --> E4[FFN SwiGLU]
        E4 --> E5[Add &amp; Norm]
    end
    subgraph DEC[Decoder Stack × N]
        D1[Output Embed + PE] --> D2[Masked Self-Attn 单向]
        D2 --> D3[Add &amp; Norm]
        D3 --> D4[Cross-Attn Q=dec K V=enc]
        D4 --> D5[Add &amp; Norm]
        D5 --> D6[FFN SwiGLU]
        D6 --> D7[Add &amp; Norm]
    end
    E5 -.K V.-> D4
    D7 --> OUT[Linear → Softmax → token]
        </div>
      </div>

      <div class="section">
        <h2>3. 现代化组件（2017 → 2024）</h2>
        <table class="table">
          <thead><tr><th>组件</th><th>原始 (2017)</th><th>现代 (LLaMA-3 / Mistral 2024)</th><th>收益</th></tr></thead>
          <tbody>
            <tr><td>位置编码</td><td>绝对 Sinusoidal PE</td><td><b>RoPE</b> / ALiBi</td><td>长度外推、相对位置</td></tr>
            <tr><td>Normalization</td><td>LayerNorm</td><td><b>RMSNorm</b></td><td>快 10-30%，大模型同等效果</td></tr>
            <tr><td>LN 位置</td><td>Post-LN（残差后）</td><td><b>Pre-LN</b>（残差前）</td><td>深层训练稳定</td></tr>
            <tr><td>激活函数</td><td>ReLU / GELU</td><td><b>SwiGLU</b></td><td>+1-2% 基准精度</td></tr>
            <tr><td>FFN</td><td>W₂·GELU(W₁ x)</td><td>W₂·(SiLU(W₁x) ⊙ W₃x)</td><td>门控机制</td></tr>
            <tr><td>Attention</td><td>Standard MHA</td><td><b>GQA</b> / MQA</td><td>KV Cache ÷ 4-8</td></tr>
            <tr><td>计算优化</td><td>-</td><td><b>Flash Attn v2/v3</b></td><td>2-4× 吞吐</td></tr>
            <tr><td>Bias</td><td>有偏置</td><td>无偏置</td><td>减参数 + 轻微加速</td></tr>
            <tr><td>Dropout</td><td>0.1</td><td>接近 0（大模型）</td><td>大数据不需要 dropout</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>4. 核心代码（现代化版本）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 交互：位置编码对比</h2>
        <p class="text-sm text-slate-600">不同位置编码方式的行为差异。选择一种观察其模式：</p>
        <div class="ctrl-panel">
          <label class="text-xs text-slate-600 mr-3">编码方式：</label>
          <select id="pe-type" class="text-xs border rounded p-1">
            <option value="sinusoidal">Sinusoidal (原始)</option>
            <option value="learned">Learned Absolute (GPT-2)</option>
            <option value="rope" selected>RoPE (LLaMA/Qwen)</option>
            <option value="alibi">ALiBi (BLOOM)</option>
          </select>
          ${MCH.slider({ id: "pe-len", label: "序列长度 N", min: 32, max: 512, step: 8, value: 128 })}
          ${MCH.slider({ id: "pe-d", label: "维度 d", min: 32, max: 256, step: 8, value: 64 })}
        </div>
        <div class="grid-2 mt-3">
          <div>
            <h3>· 位置向量可视化（d=32 维前 4 维）</h3>
            <div id="chart-pe" style="height:260px;"></div>
          </div>
          <div>
            <h3>· 相对位置 m-n 下的 Q·K 衰减</h3>
            <div id="chart-pe-decay" style="height:260px;"></div>
          </div>
        </div>
        <div id="pe-info" class="card mt-3 text-xs"></div>
      </div>

      <div class="section">
        <h2>6. 交互：参数量估算 + Attention vs FFN 参数占比</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "tr-d", label: "d_model", min: 128, max: 8192, step: 128, value: 2048 })}
              ${MCH.slider({ id: "tr-L", label: "num_layers", min: 2, max: 96, step: 2, value: 24 })}
              ${MCH.slider({ id: "tr-H", label: "num_heads", min: 4, max: 96, step: 4, value: 16 })}
              ${MCH.slider({ id: "tr-V", label: "vocab_size (×1000)", min: 8, max: 200, step: 4, value: 32 })}
              ${MCH.slider({ id: "tr-ffn", label: "FFN 扩展倍数", min: 2, max: 8, step: 0.5, value: 4 })}
              <label class="text-xs mt-3 flex items-center gap-2">
                <input type="checkbox" id="tr-swiglu" checked /> 使用 SwiGLU (×1.5 FFN 参数)
              </label>
            </div>
            <div id="tr-info" class="card mt-3"></div>
          </div>
          <div id="chart-params-breakdown" style="height:420px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>7. 训练技巧（现代大模型必备）</h2>
        <table class="table">
          <thead><tr><th>技巧</th><th>目的</th><th>推荐设置</th></tr></thead>
          <tbody>
            <tr><td>Learning Rate Warmup</td><td>防止早期梯度不稳</td><td>前 2000-10000 steps 线性 warmup</td></tr>
            <tr><td>Cosine LR Schedule</td><td>平滑收敛</td><td>到最大 lr 后余弦衰减到 10% 最大</td></tr>
            <tr><td>Weight Decay</td><td>正则化</td><td>AdamW, wd=0.1（LLM 常用）</td></tr>
            <tr><td>β₂ 调低</td><td>稳定训练</td><td>0.95 而非默认 0.999（LLaMA/GPT-4）</td></tr>
            <tr><td>Gradient Clipping</td><td>防爆炸</td><td>max_norm=1.0</td></tr>
            <tr><td>Mixed Precision</td><td>训练提速</td><td>bf16 (A100/H100), fp16 + loss scale</td></tr>
            <tr><td>ZeRO / FSDP</td><td>显存分片</td><td>ZeRO-2/3, FSDP full_shard</td></tr>
            <tr><td>Gradient Checkpointing</td><td>换显存</td><td>重计算激活，内存 ÷ 3-5（训练慢 30%）</td></tr>
            <tr><td>Activation Offload</td><td>进一步省显存</td><td>激活卸载到 CPU</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>8. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("nn_transformer").pros, MCH.getById("nn_transformer").cons, MCH.getById("nn_transformer").best_for)}
      </div>
    `;
  },

  mount() {
    // PE 可视化
    const peChart = MCH.echart(document.getElementById("chart-pe"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "位置 pos" },
      yAxis: { type: "value" },
      series: [],
    });
    const decayChart = MCH.echart(document.getElementById("chart-pe-decay"), {
      tooltip: { trigger: "axis" },
      grid: { left: 60, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "相对位置 |m-n|" },
      yAxis: { type: "value", name: "Q·K 相似度" },
      series: [{ type: "line", smooth: true, showSymbol: false, data: [], color: "#4f46e5", lineStyle: { width: 3 } }],
    });
    const update = () => {
      const type = document.getElementById("pe-type").value;
      const N = parseInt(document.getElementById("pe-len").value);
      const d = parseInt(document.getElementById("pe-d").value);
      const ps = [...Array(N).keys()];
      const dims = Math.min(4, d);
      const series = [];
      const colors = ["#4f46e5", "#f59e0b", "#10b981", "#ef4444"];
      let decay;
      let info = "";
      if (type === "sinusoidal") {
        for (let i = 0; i < dims; i++) {
          const data = ps.map(p => {
            const val = (i % 2 === 0)
              ? Math.sin(p / Math.pow(10000, i / d))
              : Math.cos(p / Math.pow(10000, (i - 1) / d));
            return [p, val];
          });
          series.push({ name: `dim ${i}`, type: "line", showSymbol: false, smooth: true, data, color: colors[i] });
        }
        decay = ps.map(r => [r, Math.max(0.05, Math.cos(r / 10) * Math.exp(-r * 0.005))]);
        info = "<b>Sinusoidal</b>：不同维度对应不同频率。优点：无参数、理论上可外推；缺点：实践外推差。";
      } else if (type === "learned") {
        for (let i = 0; i < dims; i++) {
          const seed = (i + 1) * 13;
          const base = MCH.randn(N, seed);
          series.push({ name: `dim ${i}`, type: "line", showSymbol: false, smooth: true, data: ps.map((p, j) => [p, base[j] * 0.8]), color: colors[i] });
        }
        decay = ps.map(r => [r, Math.max(0.02, 1 - r / N)]);
        info = "<b>Learned Absolute</b>：每个位置一个可学习 d 维向量。GPT-2 用。缺点：训练长度外不可用。";
      } else if (type === "rope") {
        for (let i = 0; i < dims; i += 2) {
          const theta = Math.pow(10000, -i / d);
          const d1 = ps.map(p => [p, Math.cos(p * theta)]);
          const d2 = ps.map(p => [p, Math.sin(p * theta)]);
          series.push({ name: `dim ${i} (cos)`, type: "line", showSymbol: false, smooth: true, data: d1, color: colors[i % 4] });
          series.push({ name: `dim ${i + 1} (sin)`, type: "line", showSymbol: false, smooth: true, data: d2, color: colors[(i + 1) % 4], lineStyle: { type: "dashed" } });
        }
        decay = ps.map(r => [r, Math.max(0.1, Math.cos(r * 0.05))]);
        info = "<b>RoPE (旋转位置编码)</b>：通过在 Q/K 上做 2D 旋转引入相对位置信息。LLaMA/Qwen 用。长度外推能力强（YaRN 可扩到 128k）。";
      } else if (type === "alibi") {
        // ALiBi: linearly biased attention scores
        for (let i = 0; i < dims; i++) {
          const m = Math.pow(2, -(i + 1));
          series.push({ name: `head ${i} slope m=${m.toFixed(3)}`, type: "line", showSymbol: false, smooth: true, data: ps.map(p => [p, -m * p]), color: colors[i] });
        }
        decay = ps.map(r => [r, Math.max(0, 1 - r * 0.008)]);
        info = "<b>ALiBi</b>：不添加位置向量，直接在 attention scores 上加线性偏置 -m·|m-n|。BLOOM/MPT 用。外推极好。";
      }
      peChart.setOption({ series });
      decayChart.setOption({ series: [{ data: decay }] });
      document.getElementById("pe-info").innerHTML = info;
    };
    ["pe-type"].forEach(id => document.getElementById(id).addEventListener("change", update));
    ["pe-len", "pe-d"].forEach(id => document.getElementById(id).addEventListener("input", (e) => {
      document.getElementById(id + "-val").textContent = e.target.value;
      update();
    }));
    update();

    // 参数量拆解
    const pChart = MCH.echart(document.getElementById("chart-params-breakdown"), {
      tooltip: { formatter: p => `${p.name}<br/>参数 = ${(p.value / 1e6).toFixed(2)} M (${p.percent}%)` },
      legend: { top: 0 },
      series: [{
        type: "pie", radius: ["35%", "70%"], center: ["50%", "55%"],
        data: [], label: { formatter: "{b}\n{d}%" },
        itemStyle: { borderColor: "#fff", borderWidth: 2 },
      }],
    });
    const updateP = () => {
      const d = parseInt(document.getElementById("tr-d").value);
      const L = parseInt(document.getElementById("tr-L").value);
      const V = parseInt(document.getElementById("tr-V").value) * 1000;
      const ffn = parseFloat(document.getElementById("tr-ffn").value);
      const swiglu = document.getElementById("tr-swiglu").checked;

      const embed = V * d;
      const attnPerLayer = 4 * d * d;                                                 // Q/K/V/O
      const ffnPerLayer = swiglu ? 3 * d * (ffn * d) : 2 * d * (ffn * d);           // 2 or 3 matrices
      const lnPerLayer = 4 * d;                                                        // RMSNorm pre/post
      const perLayer = attnPerLayer + ffnPerLayer + lnPerLayer;
      const total = embed + L * perLayer;

      pChart.setOption({
        series: [{
          data: [
            { name: "Token Embedding (V × d)", value: embed, itemStyle: { color: "#4f46e5" } },
            { name: `Attention × ${L} 层 (Q/K/V/O)`, value: L * attnPerLayer, itemStyle: { color: "#f59e0b" } },
            { name: `FFN × ${L} 层 ${swiglu ? "(SwiGLU 3 矩阵)" : "(2 矩阵)"}`, value: L * ffnPerLayer, itemStyle: { color: "#10b981" } },
            { name: `LayerNorm 参数`, value: L * lnPerLayer, itemStyle: { color: "#94a3b8" } },
          ],
        }],
      });

      const flops_per_token = 6 * total;  // Kaplan 公式
      document.getElementById("tr-info").innerHTML = `
        <div class="grid-3 mt-2">
          <div><div class="text-xs text-slate-500">总参数</div><div class="text-base font-bold text-indigo-700">${(total / 1e9).toFixed(2)} B</div></div>
          <div><div class="text-xs text-slate-500">Per-token FLOPs</div><div class="text-base font-bold text-indigo-700">${(flops_per_token / 1e9).toFixed(2)} G</div></div>
          <div><div class="text-xs text-slate-500">FP16 权重</div><div class="text-base font-bold text-indigo-700">${(total * 2 / 1024 / 1024 / 1024).toFixed(1)} GB</div></div>
        </div>
        <div class="text-xs text-slate-500 mt-2">
          常见配比：<br/>
          · GPT-3 175B: d=12288, L=96, H=96<br/>
          · LLaMA-2 7B: d=4096, L=32, H=32<br/>
          · LLaMA-3 70B: d=8192, L=80, H=64 (GQA: 8 KV 头)
        </div>
      `;
    };
    ["tr-d", "tr-L", "tr-H", "tr-V", "tr-ffn"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateP();
      });
    });
    document.getElementById("tr-swiglu").addEventListener("change", updateP);
    updateP();
  },
});
