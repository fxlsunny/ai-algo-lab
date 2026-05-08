/* 模块：LLM 量化 & KV Cache */
MCH.register("llm_quantization", {
  render() {
    const code = `# ---------- INT8 动态量化（推理最快落地方案）----------
from torch.ao.quantization import quantize_dynamic
q_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# ---------- GPTQ (Frantar 2023) ----------
# 基于 Hessian 的 4-bit 量化，精度最好
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("TheBloke/Llama-2-7B-GPTQ",
                                            use_safetensors=True,
                                            device="cuda:0",
                                            quantize_config=None)

# ---------- AWQ (Lin 2023) ----------
# Activation-aware 量化：保护"重要权重通道"
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized("TheBloke/Llama-2-7B-AWQ")


# ---------- KV Cache 实现（加速自回归推理）----------
# 朴素推理：每生成 1 个 token 都重算整个序列 → O(N²) 重复计算
# KV Cache：缓存 每层的 K, V → 新 token 只需 append → O(N) 增量

class CachedAttention(nn.Module):
    def forward(self, x, past_kv=None):
        q = self.Wq(x)
        k = self.Wk(x); v = self.Wv(x)
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=-2)  # 拼接历史 K
            v = torch.cat([past_kv[1], v], dim=-2)  # 拼接历史 V
        # attn 计算：Q (new 1 个) vs K/V (所有历史 + 新)
        attn = F.softmax(q @ k.transpose(-2, -1) / sqrt(d_k), -1)
        return attn @ v, (k, v)  # 返回更新后的 KV Cache


# ---------- Flash Attention (Dao 2022) ----------
# 融合 attention kernel，避免 O(N²) 中间结果写回 HBM
# 训练/推理都能加速 2-4×，显存大幅下降
import torch.nn.functional as F
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)  # PyTorch 2.0+ 自带`;

    return `
      ${MCH.hero({ icon: "📦", name: "LLM 量化 & KV Cache & Flash Attention", en: "LLM Inference Optimization", tags: ["INT8", "INT4", "GPTQ", "AWQ", "KV Cache", "Flash Attn"], meta: ["◈ 显存 ÷ 4", "⚡ 延迟 2-5× 下降"] })}

      ${MCH.versionSection("llm_quantization")}

      <div class="section">
        <h2>1. 量化方法速览</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>精度</th><th>PPL 损失</th><th>推理速度</th><th>显存</th><th>适用场景</th></tr></thead>
          <tbody>
            <tr><td>FP16 (baseline)</td><td>16-bit</td><td>-</td><td>1×</td><td>2×</td><td>训练/高精度推理</td></tr>
            <tr><td>BF16</td><td>16-bit</td><td>&lt; 0.01</td><td>1×</td><td>2×</td><td>A100/H100 训练首选</td></tr>
            <tr><td>INT8 (dynamic)</td><td>8-bit</td><td>0.05 - 0.2</td><td>1.5-2×</td><td>1×</td><td>CPU / 常规推理</td></tr>
            <tr><td><b>GPTQ 4-bit</b></td><td>4-bit</td><td>0.3 - 0.8</td><td>2-3×</td><td>0.5×</td><td>端侧 / 消费 GPU</td></tr>
            <tr><td><b>AWQ 4-bit</b></td><td>4-bit</td><td>0.2 - 0.5</td><td>2-3×</td><td>0.5×</td><td>端侧 / 对齐 GPU 友好</td></tr>
            <tr><td>GGUF Q5_K_M</td><td>5-bit</td><td>0.1 - 0.3</td><td>2×</td><td>0.62×</td><td>llama.cpp CPU</td></tr>
            <tr><td>INT4 + GPTQ Sym</td><td>4-bit</td><td>1.0+</td><td>3×</td><td>0.5×</td><td>极致场景（有损失）</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：量化误差与分布可视化</h2>
        <p class="text-sm text-slate-600">模拟权重分布（近似正态），观察不同量化位数的误差分布。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "q-bits", label: "量化位数 (bits)", min: 2, max: 16, step: 1, value: 4 })}
              ${MCH.slider({ id: "q-range", label: "权重范围（% 分位）", min: 90, max: 100, step: 0.1, value: 99.9, format: (v) => parseFloat(v).toFixed(1) })}
              <label class="text-xs mt-2 flex items-center gap-2">
                <input type="checkbox" id="q-symmetric" checked /> 对称量化
              </label>
            </div>
            <div id="q-stats" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-qerror" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. KV Cache — 自回归推理加速的关键</h2>
        <p class="text-sm text-slate-600">没有 KV Cache 时，生成第 N 个 token 需要重算前 N-1 个 token 的 K/V，<b>O(N²)</b> 冗余计算。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "kv-N", label: "生成 token 数 N", min: 10, max: 4096, step: 10, value: 512 })}
              ${MCH.slider({ id: "kv-d", label: "hidden_dim d", min: 512, max: 8192, step: 256, value: 4096 })}
              ${MCH.slider({ id: "kv-L", label: "层数 L", min: 6, max: 80, step: 2, value: 32 })}
              ${MCH.slider({ id: "kv-H", label: "num_heads H", min: 8, max: 128, step: 8, value: 32 })}
            </div>
            <div id="kv-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-kv" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. Flash Attention — 算法级 IO 优化</h2>
        <p class="text-sm text-slate-600">朴素 Attention 的瓶颈在于 <b>GPU 显存带宽</b>：QK^T 的中间矩阵 N×N 要反复读写 HBM。Flash Attention 用<b>分块 (tiling) + 重计算</b>技巧，把 softmax 融合到 SRAM 中：</p>
        <div class="grid-2">
          <div class="card">
            <h3 class="font-bold text-sm">朴素 Attention</h3>
            <ol class="text-xs text-slate-600 mt-2" style="padding-left:20px;list-style:decimal;">
              <li>计算 S = QK^T，写回 HBM (N×N)</li>
              <li>计算 softmax(S) = P，写回 HBM</li>
              <li>计算 PV，写回 HBM</li>
            </ol>
            <p class="text-xs text-red-600 mt-2">⚠️ HBM IO: O(N²) 读写</p>
          </div>
          <div class="card">
            <h3 class="font-bold text-sm">Flash Attention</h3>
            <ol class="text-xs text-slate-600 mt-2" style="padding-left:20px;list-style:decimal;">
              <li>把 Q、K、V 分块载入 SRAM</li>
              <li>在 SRAM 中融合计算 softmax 和 P·V</li>
              <li>只把最终 O 写回 HBM</li>
            </ol>
            <p class="text-xs text-emerald-600 mt-2">✓ HBM IO: O(N·d) 线性</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>6. 推理加速组合拳</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>推荐组合</th><th>预期收益</th></tr></thead>
          <tbody>
            <tr><td>云端高并发 Serving</td><td>vLLM + FP16 + Flash Attn + PagedAttn + Continuous Batching</td><td>吞吐 ×10-20</td></tr>
            <tr><td>单机单卡大模型</td><td>AWQ 4-bit + Flash Attn</td><td>显存 ÷ 4 / 速度 ×2</td></tr>
            <tr><td>CPU / 端侧</td><td>GGUF Q5_K_M + llama.cpp</td><td>7B 可跑 4GB 内存机器</td></tr>
            <tr><td>超长上下文</td><td>FP16 + Flash Attn + KV Cache 压缩 (H2O)</td><td>100k+ 序列可用</td></tr>
            <tr><td>离线批量推理</td><td>INT8 + torch.compile</td><td>速度 ×2-3</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("llm_quantization").pros,
          MCH.getById("llm_quantization").cons,
          MCH.getById("llm_quantization").best_for,
        )}
      </div>
    `;
  },

  mount() {
    // Quantization error dist
    const qChart = MCH.echart(document.getElementById("chart-qerror"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "w 原始值" },
      yAxis: { type: "value", name: "量化后 w_q" },
      series: [
        { name: "y = x (理想)", type: "line", showSymbol: false, color: "#94a3b8", lineStyle: { type: "dashed" }, data: [] },
        { name: "量化还原", type: "line", step: "middle", showSymbol: false, color: "#ec4899", lineStyle: { width: 2 }, data: [] },
      ],
    });
    const updateQ = () => {
      const bits = parseInt(document.getElementById("q-bits").value);
      const pct = parseFloat(document.getElementById("q-range").value);
      const sym = document.getElementById("q-symmetric").checked;
      // 假设权重范围基于 percentile 的最大绝对值
      const maxAbs = 0.5 + (pct - 90) * 0.1;  // 0.5 - 1.5 粗估
      const levels = Math.pow(2, bits);
      const step = (sym ? 2 * maxAbs : maxAbs) / (levels - 1);
      const xs = MCH.linspace(-maxAbs, maxAbs, 500);
      const quant = xs.map(x => {
        const q = Math.round(x / step) * step;
        return Math.max(-maxAbs, Math.min(maxAbs, q));
      });
      qChart.setOption({
        series: [
          { data: xs.map(x => [x, x]) },
          { data: xs.map((x, i) => [x, quant[i]]) },
        ],
      });
      const errs = xs.map((x, i) => Math.abs(quant[i] - x));
      const meanErr = errs.reduce((a, b) => a + b, 0) / errs.length;
      const maxErr = Math.max(...errs);
      document.getElementById("q-stats").innerHTML = `
        bits = <b>${bits}</b> → <b>${levels}</b> 档量化<br/>
        range = ±${maxAbs.toFixed(2)} · step = <b>${step.toExponential(2)}</b><br/>
        <b>平均量化误差</b>: <span style="color:#ec4899;">${meanErr.toExponential(2)}</span><br/>
        <b>最大量化误差</b>: <span style="color:#ef4444;">${maxErr.toExponential(2)}</span><br/>
        <span style="color:#64748b;">每降 1 bit，误差约 ×2；AWQ/GPTQ 等通过保护重要通道可压低 70%+ 损失。</span>
      `;
    };
    ["q-bits", "q-range"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        const v = e.target.value;
        document.getElementById(id + "-val").textContent = id === "q-range" ? parseFloat(v).toFixed(1) : v;
        updateQ();
      });
    });
    document.getElementById("q-symmetric").addEventListener("change", updateQ);
    updateQ();

    // KV Cache
    const kvChart = MCH.echart(document.getElementById("chart-kv"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "已生成 token 数" },
      yAxis: { type: "value", name: "累积计算量" },
      series: [
        { name: "无 KV Cache O(N²)", type: "line", smooth: true, showSymbol: false, color: "#ef4444", lineStyle: { width: 3 }, data: [] },
        { name: "有 KV Cache O(N)", type: "line", smooth: true, showSymbol: false, color: "#10b981", lineStyle: { width: 3 }, data: [] },
      ],
    });
    const updateKV = () => {
      const N = parseInt(document.getElementById("kv-N").value);
      const d = parseInt(document.getElementById("kv-d").value);
      const L = parseInt(document.getElementById("kv-L").value);
      const H = parseInt(document.getElementById("kv-H").value);
      const ns = MCH.linspace(1, N, 60);
      kvChart.setOption({
        series: [
          { data: ns.map(n => [n, n * n * d / 1e6]) },
          { data: ns.map(n => [n, n * d / 1e6]) },
        ],
      });
      // KV cache memory = 2 (K+V) * N * L * H * (d/H) * 2 bytes (FP16)
      const kvMemBytes = 2 * N * L * d * 2;
      const kvMemGB = kvMemBytes / 1024 / 1024 / 1024;
      const totalFLOPs_noCache = N * (N + 1) / 2 * d;
      const totalFLOPs_cache = N * d;
      document.getElementById("kv-info").innerHTML = `
        <b>KV Cache 显存占用</b>: <span style="color:#10b981;font-weight:700;">${kvMemGB.toFixed(2)} GB</span><br/>
        （= 2 × N × L × d × 2B (FP16)）<br/>
        <b>累积 FLOPs 对比</b>:<br/>
        &nbsp;&nbsp;无 Cache: <span style="color:#ef4444;">${(totalFLOPs_noCache / 1e9).toFixed(2)} G</span><br/>
        &nbsp;&nbsp;有 Cache: <span style="color:#10b981;">${(totalFLOPs_cache / 1e9).toFixed(3)} G</span><br/>
        <b>加速比</b>: <b style="color:#4f46e5;">${(totalFLOPs_noCache / totalFLOPs_cache).toFixed(1)}×</b><br/>
        <span style="color:#64748b;">注：长序列 KV Cache 占用大，需 PagedAttention (vLLM) / KV 量化进一步压缩。</span>
      `;
    };
    ["kv-N", "kv-d", "kv-L", "kv-H"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateKV();
      });
    });
    updateKV();
  },
});
