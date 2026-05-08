/* 模块：LLM 采样参数 */
MCH.register("llm_params", {
  render() {
    const code = `# LLM 解码参数一图流（vLLM / Hugging Face transformers 接口）
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

inputs = tok("介绍一下 Transformer 的自注意力机制。", return_tensors="pt")
outputs = model.generate(
    **inputs,

    # === 确定性 vs 随机性 ===
    do_sample=True,              # False → greedy（每步选 argmax）
    temperature=0.7,             # 0.0~2.0；logits 除以 T 后 softmax
                                 #   T→0: 硬选最大；T→∞: 均匀随机
    top_p=0.9,                   # Nucleus Sampling: 保留累计概率 >= 0.9 的最小 token 集
    top_k=50,                    # 只从 top-k 个候选中采样（避免低概率 token）

    # === 长度控制 ===
    max_new_tokens=512,          # 最多生成多少新 token
    min_length=20,
    early_stopping=True,         # Beam Search 早停

    # === 重复控制 ===
    repetition_penalty=1.15,     # >1: 惩罚重复；=1 无效；<1 鼓励重复
    no_repeat_ngram_size=3,      # 禁止 3-gram 重复
    presence_penalty=0.0,        # OpenAI 风格：对已出现 token 减分（与 rep_penalty 不同）
    frequency_penalty=0.0,       # OpenAI 风格：按出现次数减分

    # === Beam Search（确定性多路径）===
    num_beams=1,                 # >1 启用 beam；通常 4~8；与 do_sample 不共存
    length_penalty=1.0,          # Beam 长度惩罚

    # === 约束 ===
    bad_words_ids=[[tok.convert_tokens_to_ids("[UNK]")]],
    forced_bos_token_id=None,
)`;

    return `
      ${MCH.hero({ icon: "🎲", name: "LLM 采样参数调控", en: "Sampling Parameters · Temperature / Top-p / Top-k", tags: ["Temperature", "Top-p", "Top-k", "Beam", "Penalty"], meta: ["◈ 零训练成本", "⚡ 立即生效"] })}

      ${MCH.versionSection("llm_params")}

      <div class="section">
        <h2>1. 核心公式 — 从 logits 到 token</h2>
        <p class="text-sm text-slate-600">LLM 每步输出词表上的一组 <b>logits</b>，经过一系列变换后采样一个 token：</p>
        <div class="formula-block">
          $$ z = \\text{logits} / T, \\quad p_i = \\frac{\\exp(z_i)}{\\sum_j \\exp(z_j)} $$
          <b>Top-k</b>：只保留 $p_i$ 中前 k 大的，其余设为 0 后重新归一化。<br/>
          <b>Top-p (nucleus)</b>：按概率从大到小累加，保留到累计 ≥ p 的最小集合。
        </div>
      </div>

      <div class="section">
        <h2>2. 代码 / 配置模板</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：Temperature + Top-p + Top-k 对分布的影响</h2>
        <p class="text-sm text-slate-600">假设原始 logits 如右图（10 个候选），调节参数观察 softmax 后的采样概率分布。</p>

        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "llm-T", label: "temperature T", min: 0.01, max: 2, step: 0.01, value: 0.7 })}
              ${MCH.slider({ id: "llm-topp", label: "top_p", min: 0.1, max: 1.0, step: 0.01, value: 0.9 })}
              ${MCH.slider({ id: "llm-topk", label: "top_k (0=disabled)", min: 0, max: 10, step: 1, value: 0 })}
            </div>
            <div id="llm-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-dist" style="height:360px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. Temperature × Top-p 建议配方</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>T</th><th>top_p</th><th>top_k</th><th>其他</th></tr></thead>
          <tbody>
            <tr><td>📝 代码生成（要严谨）</td><td>0.0 ~ 0.2</td><td>0.95</td><td>0 (disable)</td><td>或直接 greedy</td></tr>
            <tr><td>💼 商业邮件 / 摘要</td><td>0.3 ~ 0.5</td><td>0.9</td><td>0</td><td>repetition_penalty=1.1</td></tr>
            <tr><td>💬 聊天对话</td><td>0.7 ~ 0.8</td><td>0.9 ~ 0.95</td><td>40</td><td>presence_penalty=0.3</td></tr>
            <tr><td>🎨 创意写作 / 头脑风暴</td><td>0.9 ~ 1.2</td><td>0.95</td><td>0</td><td>给更多"思路泄露"</td></tr>
            <tr><td>🧮 数学 / 逻辑推理</td><td>0.0</td><td>1.0</td><td>1</td><td>greedy / beam=4 做 CoT 采样</td></tr>
            <tr><td>🎭 角色扮演 / RP</td><td>0.85 ~ 1.0</td><td>0.9</td><td>40</td><td>repetition_penalty=1.15</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. Beam Search vs Sampling</h2>
        <div class="grid-2">
          <div class="card">
            <h3 class="font-bold text-sm">🎯 Beam Search</h3>
            <p class="text-xs text-slate-600 mt-2">
              每步保留概率最高的 num_beams 条路径，最终选联合概率最大的。适合<b>翻译 / 摘要</b>等"正确答案唯一"的任务。<br/>
              <b>缺点</b>：对开放生成（聊天/故事）会退化为重复、无趣输出。
            </p>
          </div>
          <div class="card">
            <h3 class="font-bold text-sm">🎲 Sampling</h3>
            <p class="text-xs text-slate-600 mt-2">
              基于概率随机采样，配 Temp + Top-p 控制多样性。适合<b>开放式任务</b>。<br/>
              <b>缺点</b>：可能采到低概率的"不靠谱" token，需 top-p/top-k 限制。
            </p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("llm_params").pros,
          MCH.getById("llm_params").cons,
          MCH.getById("llm_params").best_for,
        )}
      </div>
    `;
  },

  mount() {
    // 原始 logits（模拟）
    const rawLogits = [5.2, 4.5, 4.1, 3.8, 2.9, 2.1, 1.5, 0.8, 0.2, -0.5];
    const tokens = ["the", "a", "an", "this", "that", "our", "my", "its", "no", "yet"];
    const chart = MCH.echart(document.getElementById("chart-dist"), {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", data: tokens },
      yAxis: { type: "value", name: "概率", max: 1 },
      series: [
        { name: "原始 (T=1.0)", type: "bar", data: [], color: "#94a3b8", barGap: 0.1 },
        { name: "应用参数后", type: "bar", data: [], color: "#ec4899" },
      ],
    });
    const update = () => {
      const T = parseFloat(document.getElementById("llm-T").value);
      const topP = parseFloat(document.getElementById("llm-topp").value);
      const topK = parseInt(document.getElementById("llm-topk").value);
      const softmax = (logits, temp) => {
        const z = logits.map(v => v / temp);
        const m = Math.max(...z);
        const e = z.map(v => Math.exp(v - m));
        const s = e.reduce((a, b) => a + b, 0);
        return e.map(v => v / s);
      };
      const original = softmax(rawLogits, 1.0);
      let probs = softmax(rawLogits, T);
      // Apply top-k
      if (topK > 0) {
        const sorted = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
        const keep = new Set(sorted.slice(0, topK).map(x => x.i));
        probs = probs.map((p, i) => keep.has(i) ? p : 0);
        const s = probs.reduce((a, b) => a + b, 0) || 1;
        probs = probs.map(p => p / s);
      }
      // Apply top-p
      if (topP < 1.0) {
        const sorted = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
        let acc = 0; const keep = new Set();
        for (const { p, i } of sorted) { keep.add(i); acc += p; if (acc >= topP) break; }
        probs = probs.map((p, i) => keep.has(i) ? p : 0);
        const s = probs.reduce((a, b) => a + b, 0) || 1;
        probs = probs.map(p => p / s);
      }
      chart.setOption({
        series: [
          { data: original.map(v => +v.toFixed(3)) },
          { data: probs.map(v => +v.toFixed(3)) },
        ],
      });
      const entropy = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log2(p) : 0), 0);
      const maxP = Math.max(...probs);
      const effective = probs.filter(p => p > 0.001).length;
      document.getElementById("llm-info").innerHTML = `
        <b>有效候选 token</b>：<span style="color:#ec4899;font-weight:700;">${effective} / 10</span><br/>
        <b>最大概率</b>：${maxP.toFixed(3)} · <b>熵 H</b>：${entropy.toFixed(3)} bits (max = ${Math.log2(10).toFixed(2)})<br/>
        <span style="color:#64748b;">熵越高越随机 / 越低越确定。</span>
      `;
    };
    ["llm-T", "llm-topp", "llm-topk"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
