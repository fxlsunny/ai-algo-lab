/* 模块：LLM 高效微调 */
MCH.register("llm_finetune", {
  render() {
    const code = `# LoRA：Low-Rank Adaptation (Hu et al., 2021)
#   把 ΔW 分解为两个低秩矩阵：W_new = W_0 + (α/r) · B·A
#   其中 B ∈ R^(d × r), A ∈ R^(r × d), r << d
#   训练时只更新 A 和 B，原始 W_0 冻结 → 显存 ÷ 10-100

class LoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16, dropout=0.1):
        self.base = base_linear       # 冻结
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base_linear.in_features, base_linear.out_features
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 零初始化 → 训练开始时 ΔW = 0，等价于只用 base
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.base(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

# QLoRA (Dettmers 2023)：在 LoRA 基础上把 base model 量化到 4-bit
#   显存进一步 ÷ 4，13B 可单卡 24GB GPU 训！
#   技巧：
#     - NF4 (NormalFloat 4-bit) 量化格式
#     - 双量化 (double quant)：量化量化常数本身
#     - Paged Optimizer：优化器状态溢出到 CPU

# 使用 peft 库（HuggingFace 官方）：
from peft import LoraConfig, get_peft_model, TaskType
config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 只给 attention 加
    lora_dropout=0.05,
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(base_model, config)
model.print_trainable_parameters()
# "trainable params: 8M | total: 7B | trainable%: 0.11%"`;

    return `
      ${MCH.hero({ icon: "🎯", name: "LLM 高效微调 — LoRA / QLoRA / Adapter", en: "Parameter-Efficient Fine-Tuning (PEFT)", tags: ["LoRA", "QLoRA", "Adapter", "Prefix-Tuning", "P-Tuning v2"], meta: ["◈ 可训参数 < 1%", "⚡ 显存 ÷ 10-100"] })}

      ${MCH.versionSection("llm_finetune")}

      <div class="section">
        <h2>1. 为什么要 PEFT？</h2>
        <p class="text-sm text-slate-600">7B-70B 大模型全参微调的问题：</p>
        <div class="grid-3">
          <div class="card"><h3 class="text-pink-700 font-bold text-sm">💾 显存爆炸</h3><p class="text-xs text-slate-600 mt-1">7B 模型 FP16 训练需要 <b>112 GB</b>（权重+梯度+Adam 状态），A100 80G 也吃不下。</p></div>
          <div class="card"><h3 class="text-pink-700 font-bold text-sm">⏳ 训练慢</h3><p class="text-xs text-slate-600 mt-1">全量 backward 对每个参数都要存梯度 + 优化器状态。</p></div>
          <div class="card"><h3 class="text-pink-700 font-bold text-sm">🗂️ 存储成本</h3><p class="text-xs text-slate-600 mt-1">每个下游任务存 7B × 2字节 = 14GB 一份 checkpoint，50 个任务就要 700GB。</p></div>
        </div>
      </div>

      <div class="section">
        <h2>2. 主流 PEFT 方法对比</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>可训参数位置</th><th>典型占比</th><th>推理开销</th><th>备注</th></tr></thead>
          <tbody>
            <tr><td><b>LoRA</b></td><td>为 Q/K/V/O 加低秩旁路</td><td>0.1 - 1%</td><td>可 merge 回权重，零开销</td><td>最主流</td></tr>
            <tr><td><b>QLoRA</b></td><td>LoRA + 4-bit 量化基座</td><td>0.1 - 1%</td><td>需反量化</td><td>13B 单卡可训</td></tr>
            <tr><td>Adapter</td><td>每层插入小 MLP</td><td>2 - 5%</td><td>增加 1-2 层计算</td><td>Houlsby 2019 原始版</td></tr>
            <tr><td>Prefix-Tuning</td><td>在 KV 前拼接可训 prefix</td><td>0.1 - 0.5%</td><td>序列长度 +20 token</td><td>Li & Liang 2021</td></tr>
            <tr><td>P-Tuning v2</td><td>每层加可训 prefix</td><td>0.1 - 0.3%</td><td>同上</td><td>适合分类</td></tr>
            <tr><td>IA³</td><td>可训向量 rescale activations</td><td>0.01%</td><td>点乘，极小</td><td>Liu 2022</td></tr>
            <tr><td>DoRA</td><td>LoRA 分解幅度 / 方向</td><td>0.1 - 1%</td><td>同 LoRA</td><td>2024 SOTA PEFT</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 代码解读：LoRA 核心实现</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：LoRA 参数量 vs 性能权衡</h2>

        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "ft-model", label: "基座大小 (B 参数)", min: 1, max: 70, step: 1, value: 7 })}
              ${MCH.slider({ id: "ft-r", label: "LoRA rank r", min: 2, max: 128, step: 2, value: 16 })}
              ${MCH.slider({ id: "ft-modules", label: "目标模块数 L", min: 4, max: 64, step: 4, value: 32 })}
              ${MCH.slider({ id: "ft-dim", label: "隐层维度 d", min: 512, max: 8192, step: 256, value: 4096 })}
              <label class="text-xs mt-2 flex items-center gap-2">
                <input type="checkbox" id="ft-qlora" /> 使用 QLoRA (4-bit 量化)
              </label>
            </div>
          </div>
          <div>
            <div id="ft-info" class="card"></div>
          </div>
        </div>

        <h3 style="margin-top:18px;">· 显存占用对比 (FP16)</h3>
        <div id="chart-memory" style="height:280px;"></div>
      </div>

      <div class="section">
        <h2>5. 训练配方（开箱即用）</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>r</th><th>alpha</th><th>target_modules</th><th>lr</th><th>epochs</th></tr></thead>
          <tbody>
            <tr><td>通用指令微调 Alpaca-style</td><td>16</td><td>32</td><td>q/k/v/o</td><td>3e-4</td><td>3</td></tr>
            <tr><td>风格 / 人格调整</td><td>8</td><td>16</td><td>q/v</td><td>2e-4</td><td>5-10</td></tr>
            <tr><td>垂直领域（医疗/金融/法律）</td><td>32</td><td>64</td><td>q/k/v/o + mlp.up_proj</td><td>1e-4</td><td>3-5</td></tr>
            <tr><td>小数据 (&lt; 1k 样本)</td><td>4-8</td><td>16</td><td>q/v</td><td>5e-5 (小)</td><td>10-20 early stop</td></tr>
            <tr><td>对话 SFT 大数据</td><td>64</td><td>128</td><td>全 linear</td><td>2e-4</td><td>1-2</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("llm_finetune").pros,
          MCH.getById("llm_finetune").cons,
          MCH.getById("llm_finetune").best_for,
        )}
      </div>
    `;
  },

  mount() {
    const infoEl = document.getElementById("ft-info");
    const memChart = MCH.echart(document.getElementById("chart-memory"), {
      tooltip: { trigger: "axis", valueFormatter: v => v.toFixed(1) + " GB" },
      legend: { top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", data: ["全参 FP16", "LoRA FP16", "QLoRA 4-bit + LoRA"] },
      yAxis: { type: "value", name: "显存 (GB)", type: "log", logBase: 10 },
      series: [
        { name: "模型权重", type: "bar", stack: "m", data: [], color: "#7c3aed" },
        { name: "梯度", type: "bar", stack: "m", data: [], color: "#ec4899" },
        { name: "Adam 状态 (×2)", type: "bar", stack: "m", data: [], color: "#f59e0b" },
        { name: "激活 (est)", type: "bar", stack: "m", data: [], color: "#10b981" },
      ],
    });
    const update = () => {
      const modelB = parseFloat(document.getElementById("ft-model").value);
      const r = parseInt(document.getElementById("ft-r").value);
      const L = parseInt(document.getElementById("ft-modules").value);
      const d = parseInt(document.getElementById("ft-dim").value);
      const isQLora = document.getElementById("ft-qlora").checked;

      // LoRA 新增参数 = 2 · r · d · L (B: d×r + A: r×d, 每个目标模块)
      const loraParams = 2 * r * d * L;
      const baseParams = modelB * 1e9;
      const pct = loraParams / baseParams * 100;

      // 显存（GB） - FP16 weights (2B per param)
      const fullWeights = baseParams * 2 / 1e9;
      const fullGrads = fullWeights;  // same size
      const fullAdam = fullWeights * 2;  // Adam has m, v (each same as param)
      const activ = 8;  // rough for 7B

      const loraWeights = fullWeights;  // base still FP16
      const loraGrads = loraParams * 2 / 1e9;
      const loraAdam = loraGrads * 2;

      const qloraWeights = baseParams * 0.5 / 1e9;  // 4-bit ≈ 0.5 byte/param
      const qloraGrads = loraGrads;
      const qloraAdam = loraAdam;

      memChart.setOption({
        series: [
          { data: [fullWeights.toFixed(1), loraWeights.toFixed(1), qloraWeights.toFixed(1)] },
          { data: [fullGrads.toFixed(1), loraGrads.toFixed(3), qloraGrads.toFixed(3)] },
          { data: [fullAdam.toFixed(1), loraAdam.toFixed(3), qloraAdam.toFixed(3)] },
          { data: [activ, activ, activ] },
        ],
      });

      const displayParams = isQLora ? loraParams : loraParams;
      const displayTotal = isQLora ? (qloraWeights + loraGrads + loraAdam + activ) : (loraWeights + loraGrads + loraAdam + activ);
      const fullTotal = fullWeights + fullGrads + fullAdam + activ;

      infoEl.innerHTML = `
        <h4 class="font-semibold text-slate-800 text-sm">📊 当前配置</h4>
        <div class="grid-3 mt-3">
          <div><div class="text-xs text-slate-500">基座参数</div><div class="text-xl font-bold text-pink-700">${modelB} B</div></div>
          <div><div class="text-xs text-slate-500">LoRA 可训参数</div><div class="text-xl font-bold text-pink-700">${(loraParams / 1e6).toFixed(2)} M</div></div>
          <div><div class="text-xs text-slate-500">占比</div><div class="text-xl font-bold text-pink-700">${pct.toFixed(3)}%</div></div>
        </div>
        <div class="text-xs text-slate-500 mt-3">
          <b>${isQLora ? "QLoRA" : "LoRA"}</b> 总显存 ≈ <b style="color:#ec4899;">${displayTotal.toFixed(1)} GB</b><br/>
          相比全参微调（<b>${fullTotal.toFixed(1)} GB</b>）节省 <b style="color:#10b981;">${((1 - displayTotal / fullTotal) * 100).toFixed(1)}%</b><br/>
          ${isQLora ? "QLoRA 4-bit 量化后的 base model 反量化有 ~30% 速度开销" : "LoRA 训练后可 merge 回 base weight，推理零开销"}
        </div>
      `;
    };
    ["ft-model", "ft-r", "ft-modules", "ft-dim"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    document.getElementById("ft-qlora").addEventListener("change", update);
    update();
  },
});
