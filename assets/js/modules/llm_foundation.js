/* 模块：LLM 基座模型原理 */
MCH.register("llm_foundation", {
  render() {
    const code = `# 预训练目标三大流派
# ========================================================================

# ① Autoregressive LM (GPT 系) —— 当前主流
#    L = -Σ_t log P(x_t | x_{<t}; θ)
#    优势：直接对应生成任务；数据 = 任意文本
def ar_loss(logits, labels):
    # logits: (B, T, V)   labels: (B, T)
    return F.cross_entropy(logits[:, :-1].flatten(0, 1), labels[:, 1:].flatten())

# ② Masked LM (BERT 系)
#    L = -Σ log P(x_mask | x_observed; θ)
#    随机 mask 15% tokens
def mlm_loss(logits, labels, mask):
    return F.cross_entropy(logits[mask], labels[mask])

# ③ Span Corruption (T5)
#    mask 连续 span → Seq2Seq 预测被 mask 的部分
#    输入：The <X> brown <Y> jumps   →   输出：<X> quick <Y> fox


# ========================================================================
# Dense vs Mixture of Experts (MoE)
# ========================================================================

class DenseFFN(nn.Module):
    """普通 Transformer：每个 token 走所有参数"""
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoEFFN(nn.Module):
    """MoE：每个 token 只激活 top-k 个专家（Switch / Mixtral / DeepSeek-MoE）
       总参数量大，但实际计算量 = top_k/N × Dense"""
    def __init__(self, d_model, d_hidden, num_experts=8, top_k=2):
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_hidden) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        logits = self.router(x)                        # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_idx = probs.topk(self.top_k, dim=-1)
        # 对每个 token 只调用 top-k 个专家（稀疏激活）
        out = torch.zeros_like(x)
        for e in range(len(self.experts)):
            mask = (top_k_idx == e).any(-1)
            if mask.any():
                expert_out = self.experts[e](x[mask])
                # 根据 router 权重加权
                weights = top_k_probs[mask] * (top_k_idx[mask] == e).float()
                out[mask] += (expert_out * weights.sum(-1, keepdim=True))
        return out


# ========================================================================
# Chinchilla Scaling Law (Hoffmann 2022)
# ========================================================================
# 给定计算预算 C，最优模型大小 N 和训练 token 数 D 应该：
#    N_optimal = G · C^a,  D_optimal = G' · C^b,  a ≈ b ≈ 0.5
#    实际经验：每 1B 参数 用 ≈ 20B tokens 训练
# 启示：与其训一个 175B 只喂 300B tokens 的模型，不如训一个 70B 喂 1.4T tokens
# 这就是 LLaMA-2/3 选择参数量较小但 token 数大的原因。`;

    return `
      ${MCH.hero({
        icon: "🏛",
        name: "基座模型原理 — Foundation Models",
        en: "Pre-training Objectives · Scaling Laws · Modern LLM Family",
        tags: ["Autoregressive LM", "MoE", "Chinchilla Law", "Tokenizer", "RoPE/GQA/SwiGLU"],
        meta: ["◈ GPT-4 / LLaMA-3 / Qwen / DeepSeek / Mistral", "⚡ 涌现能力的根源"],
      })}

      ${MCH.versionSection("llm_foundation")}

      <div class="section">
        <h2>1. 三种预训练目标</h2>
        <table class="table">
          <thead><tr><th>目标</th><th>代表</th><th>数据消耗</th><th>优势</th><th>劣势</th></tr></thead>
          <tbody>
            <tr><td><b>Autoregressive LM</b></td><td>GPT 系, LLaMA, Qwen</td><td>1×（每 token 全量计算）</td><td>天然生成；数据 = 任何文本</td><td>单向 attention</td></tr>
            <tr><td>Masked LM</td><td>BERT, RoBERTa</td><td>6.7× (只预测 15%)</td><td>双向理解强</td><td>生成能力弱</td></tr>
            <tr><td>Span Corruption</td><td>T5, BART, UL2</td><td>中等</td><td>统一理解+生成</td><td>实现复杂</td></tr>
            <tr><td>Denoising AE</td><td>BART</td><td>-</td><td>文本恢复 / 摘要</td><td>-</td></tr>
            <tr><td>Prefix LM</td><td>UL2, T5</td><td>-</td><td>前缀双向 + 后缀单向</td><td>-</td></tr>
          </tbody>
        </table>
        ${MCH.info(`
          <b>当前共识（2024）</b>：Decoder-only + Autoregressive LM 是最易规模化的目标，能用最少的工程复杂度换取最强的能力。
          GPT-4 / Gemini / Claude / LLaMA-3 / Qwen / DeepSeek 无一例外都走这条路。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. Chinchilla Scaling Law — 参数量 vs 数据量的权衡</h2>
        <p class="text-sm text-slate-600">DeepMind 2022 提出的关键发现：<b>同样计算预算下，参数与数据应约等量级增长</b>。之前的模型（如 GPT-3 175B 只用 300B tokens）实际上"参数过剩、数据不足"。</p>
        <div class="formula-block">
          $$ \\mathcal{L}(N, D) = E + \\frac{A}{N^\\alpha} + \\frac{B}{D^\\beta}, \\quad \\alpha \\approx \\beta \\approx 0.34 $$
          <b>最优配比</b>：$D^* \\approx 20 \\cdot N$（每 1B 参数约 20B tokens）
        </div>

        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "cl-budget", label: "计算预算 log10(FLOPs)", min: 18, max: 26, step: 0.2, value: 22, format: (v) => "10^" + parseFloat(v).toFixed(1) })}
            </div>
            <div id="cl-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-chinchilla" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>3. Dense vs MoE — 参数效率革命</h2>
        <p class="text-sm text-slate-600"><b>MoE (Mixture of Experts)</b>：每个 FFN 层拆成 N 个专家，每 token 只激活 top-k 个。总参数量大但单 token 计算量小 → <b>"稀疏激活"</b>。</p>
        <div class="grid-2">
          <div class="card" style="border-top:3px solid #4f46e5;">
            <h3 class="font-bold text-sm">Dense (传统)</h3>
            <p class="text-xs text-slate-600 mt-2">每 token 走所有参数。计算 = 参数。</p>
            <div class="text-xs text-slate-500 mt-2">
              · LLaMA-2 70B：每 token 激活 70B<br/>
              · GPT-3 175B：每 token 激活 175B
            </div>
          </div>
          <div class="card" style="border-top:3px solid #ec4899;">
            <h3 class="font-bold text-sm">MoE (稀疏激活)</h3>
            <p class="text-xs text-slate-600 mt-2">每 token 只走 top-k 个专家。总参数 &gt;&gt; 激活参数。</p>
            <div class="text-xs text-slate-500 mt-2">
              · Mixtral 8×7B：总 47B，激活 13B<br/>
              · DeepSeek-V3 671B：总 671B，激活 37B<br/>
              · Grok-1 314B：总 314B，激活 86B
            </div>
          </div>
        </div>
        <h3 style="margin-top:18px;">· MoE 路由可视化</h3>
        <div class="ctrl-panel">
          ${MCH.slider({ id: "moe-experts", label: "专家数 N", min: 4, max: 64, step: 2, value: 8 })}
          ${MCH.slider({ id: "moe-topk", label: "top-k 激活", min: 1, max: 8, step: 1, value: 2 })}
        </div>
        <div id="chart-moe" style="height:280px;margin-top:10px;"></div>
      </div>

      <div class="section">
        <h2>4. 核心代码：预训练 + MoE</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 现代 LLM 谱系（2024）</h2>
        <table class="table">
          <thead><tr><th>模型</th><th>发布方</th><th>架构</th><th>参数</th><th>训练 tokens</th><th>上下文</th><th>亮点</th></tr></thead>
          <tbody>
            <tr><td><b>GPT-4</b></td><td>OpenAI</td><td>Dense (推测)</td><td>~1.8T</td><td>13T</td><td>128k</td><td>综合能力最强</td></tr>
            <tr><td><b>Claude 3.5</b></td><td>Anthropic</td><td>Dense</td><td>-</td><td>-</td><td>200k</td><td>长文 / 代码</td></tr>
            <tr><td><b>Gemini 1.5 Pro</b></td><td>Google</td><td>MoE</td><td>-</td><td>-</td><td>1M-2M</td><td>超长上下文</td></tr>
            <tr><td><b>LLaMA-3.1 70B</b></td><td>Meta</td><td>Dense, GQA</td><td>70B</td><td>15T</td><td>128k</td><td>开源 SOTA</td></tr>
            <tr><td><b>Qwen-2.5 72B</b></td><td>阿里</td><td>Dense</td><td>72B</td><td>18T</td><td>128k</td><td>中文开源 SOTA</td></tr>
            <tr><td><b>DeepSeek-V3</b></td><td>深度求索</td><td>MoE (256 专家)</td><td>671B / 37B 激活</td><td>14.8T</td><td>64k-128k</td><td>训练成本极低</td></tr>
            <tr><td><b>Mistral Large 2</b></td><td>Mistral</td><td>Dense</td><td>123B</td><td>-</td><td>128k</td><td>欧洲旗舰</td></tr>
            <tr><td><b>Mixtral 8×22B</b></td><td>Mistral</td><td>MoE</td><td>141B / 39B 激活</td><td>-</td><td>64k</td><td>MoE 开源标杆</td></tr>
            <tr><td><b>Yi-1.5 34B</b></td><td>零一万物</td><td>Dense</td><td>34B</td><td>3.6T</td><td>16-200k</td><td>训练高效</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. Tokenizer 演进</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>原理</th><th>词表</th><th>代表</th></tr></thead>
          <tbody>
            <tr><td><b>BPE</b> (Byte-Pair Encoding)</td><td>合并高频字符对</td><td>30k-50k</td><td>GPT-2</td></tr>
            <tr><td><b>BBPE</b> (Byte-level BPE)</td><td>字节级合并，解决 UTF-8</td><td>50k-100k</td><td>GPT-3/4, LLaMA</td></tr>
            <tr><td>SentencePiece</td><td>子词分词 unigram</td><td>32k</td><td>T5, LLaMA-2</td></tr>
            <tr><td><b>Tiktoken</b></td><td>OpenAI 优化的 BBPE</td><td>100k+</td><td>GPT-3.5/4, GPT-4o</td></tr>
            <tr><td>SentencePiece-BBPE</td><td>结合两者</td><td>150k-200k</td><td>Qwen-2, Gemini</td></tr>
          </tbody>
        </table>
        ${MCH.info(`
          <b>中文 token 效率</b>：GPT-2 对中文每字约 2-3 tokens（效率低）；GPT-4o 约 1.1 tokens；Qwen 和 DeepSeek 的 tokenizer 针对中文优化，常约 0.6-0.8 tokens/字。
        `, "tip")}
      </div>

      <div class="section">
        <h2>7. 对齐阶段（Post-training）</h2>
        <div class="mermaid">
flowchart LR
    P[Pre-training<br/>万亿 tokens<br/>next-token prediction] --> S[SFT Supervised Fine-tuning<br/>高质量指令数据 10w-100w 对]
    S --> R[RLHF / DPO / RLAIF<br/>人类偏好对齐]
    R --> F[最终模型]
        </div>
        <table class="table mt-3">
          <thead><tr><th>方法</th><th>数据形式</th><th>成本</th><th>效果</th></tr></thead>
          <tbody>
            <tr><td>SFT</td><td>(prompt, response) 对</td><td>低</td><td>基础跟随能力</td></tr>
            <tr><td>RLHF (PPO)</td><td>(prompt, chosen, rejected)</td><td>高（训 reward model + PPO）</td><td>经典方法，质量高</td></tr>
            <tr><td><b>DPO</b></td><td>(prompt, chosen, rejected)</td><td>中（单阶段）</td><td>2024 主流，简化版 RLHF</td></tr>
            <tr><td>RLAIF</td><td>AI 自动生成偏好</td><td>低</td><td>降低标注成本</td></tr>
            <tr><td>ORPO / SimPO</td><td>-</td><td>中</td><td>更新的对齐算法</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>8. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("llm_foundation").pros, MCH.getById("llm_foundation").cons, MCH.getById("llm_foundation").best_for)}
      </div>
    `;
  },

  mount() {
    // Chinchilla 最优配比图
    const chinChart = MCH.echart(document.getElementById("chart-chinchilla"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 70, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "log", name: "参数量 N (B)", logBase: 10 },
      yAxis: { type: "log", name: "训练 tokens D (B)", logBase: 10 },
      series: [
        { name: "Chinchilla 最优 D = 20N", type: "line", smooth: true, showSymbol: false, color: "#ec4899", lineStyle: { width: 3 }, data: [] },
        { name: "等预算线", type: "line", smooth: true, showSymbol: false, color: "#4f46e5", lineStyle: { width: 2, type: "dashed" }, data: [] },
        { name: "历史模型", type: "scatter", data: [], symbolSize: 14 },
      ],
    });
    const famous = [
      { name: "GPT-3", N: 175, D: 300 },
      { name: "Gopher", N: 280, D: 300 },
      { name: "Chinchilla", N: 70, D: 1400 },
      { name: "LLaMA-2 7B", N: 7, D: 2000 },
      { name: "LLaMA-2 70B", N: 70, D: 2000 },
      { name: "LLaMA-3 70B", N: 70, D: 15000 },
      { name: "Qwen-2.5 72B", N: 72, D: 18000 },
      { name: "DeepSeek-V3", N: 37, D: 14800 },
    ];
    const updateC = () => {
      const logC = parseFloat(document.getElementById("cl-budget").value);
      const C = Math.pow(10, logC);
      // D = 20 · N; C = 6 · N · D = 120 · N² → N* = sqrt(C / 120), D* = 20 · N*
      const Nstar = Math.sqrt(C / 120) / 1e9;  // B
      const Dstar = 20 * Nstar;
      const Ns = MCH.linspace(-1, 4, 50).map(v => Math.pow(10, v));
      chinChart.setOption({
        series: [
          { data: Ns.map(n => [n, 20 * n]) },
          { data: Ns.map(n => { const d = C / (6 * n * 1e9) / 1e9; return d > 0 && d < 1e6 ? [n, d] : null; }).filter(Boolean) },
          {
            data: famous.map(f => ({ value: [f.N, f.D], label: { show: true, position: "right", formatter: f.name, fontSize: 10 } })),
            itemStyle: { color: "#10b981" },
          },
        ],
      });
      document.getElementById("cl-info").innerHTML = `
        <b>计算预算</b> C = 10^${logC.toFixed(1)} FLOPs<br/>
        <b>最优参数量</b> N* = <span style="color:#ec4899;font-weight:700;">${Nstar.toFixed(2)} B</span><br/>
        <b>最优训练 tokens</b> D* = <span style="color:#ec4899;font-weight:700;">${Dstar.toFixed(1)} B</span><br/>
        <b>最优配比</b>：每 1B 参数 ≈ 20B tokens<br/>
        <span style="color:#64748b;">GPT-3 (175B / 300B) 在现代视角明显"参数过剩、数据不足"。</span>
      `;
    };
    document.getElementById("cl-budget").addEventListener("input", (e) => {
      document.getElementById("cl-budget-val").textContent = "10^" + parseFloat(e.target.value).toFixed(1);
      updateC();
    });
    updateC();

    // MoE viz
    const moeChart = MCH.echart(document.getElementById("chart-moe"), {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 30 },
      xAxis: { type: "category", data: [] },
      yAxis: { type: "value", name: "比率", max: 1 },
      series: [
        { name: "实际激活比例", type: "bar", barWidth: 20, data: [], color: "#ec4899" },
        { name: "理想均匀", type: "line", showSymbol: false, color: "#94a3b8", lineStyle: { type: "dashed" }, data: [] },
      ],
    });
    const updateMoE = () => {
      const N = parseInt(document.getElementById("moe-experts").value);
      const k = Math.min(parseInt(document.getElementById("moe-topk").value), N);
      // 模拟路由概率分布（有些专家更热门）
      const loads = [];
      for (let i = 0; i < N; i++) {
        // 不均匀：某些专家被更多选中
        loads.push(0.5 + 0.8 * Math.exp(-Math.pow(i - N / 3, 2) / (N / 4)) + Math.random() * 0.2);
      }
      const total = loads.reduce((a, b) => a + b, 0);
      const normalized = loads.map(l => (l / total) * k);  // 每 token 贡献
      moeChart.setOption({
        xAxis: { data: [...Array(N).keys()].map(i => `E${i}`) },
        series: [
          { data: normalized.map(v => +v.toFixed(3)) },
          { data: [...Array(N).keys()].map(() => k / N) },
        ],
      });
    };
    ["moe-experts", "moe-topk"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateMoE();
      });
    });
    updateMoE();
  },
});
