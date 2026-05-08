/* 模块：LLM 推理部署架构 */
MCH.register("llm_serving", {
  render() {
    const code = `# vLLM PagedAttention 的核心思想（类比虚拟内存的分页）
# Kwon et al., SOSP 2023

# 问题：朴素 KV Cache 为每个请求预留 max_len 的连续显存
#       → 碎片严重，显存利用率只有 20-40%

# 方案：把 KV Cache 分成固定大小 block（通常 16 tokens/block）
#       每个请求按需分配 block，不要求连续，类似 OS 分页

class PagedKVCache:
    def __init__(self, block_size=16, num_blocks=10000, num_heads, d_head):
        # 物理显存：一个大 tensor 池
        self.blocks = torch.zeros(num_blocks, block_size, num_heads, d_head)
        self.free_blocks = deque(range(num_blocks))

    def allocate(self, req_id, n_tokens):
        """为请求分配若干 block（逻辑 → 物理映射）"""
        n_blocks = ceil(n_tokens / self.block_size)
        physical = [self.free_blocks.popleft() for _ in range(n_blocks)]
        self.block_table[req_id] = physical

    def release(self, req_id):
        for b in self.block_table[req_id]:
            self.free_blocks.append(b)
        del self.block_table[req_id]


# Continuous Batching（连续批处理）
# ------------------------------------------------------------------
# 问题：传统静态 batching 等所有请求完成才释放 → 长尾拖慢整体
# 方案：每步都可以加入新请求 / 释放已完成请求

# 时间线：
#  t=0  batch = [req1(len=3, gen=100), req2(len=5, gen=50)]
#  t=50 req2 完成 → release → 立即加入 req3(len=4, gen=80)
#  → 吞吐显著提升


# Speculative Decoding（投机解码）
# ------------------------------------------------------------------
# 问题：大模型每步只生成 1 token，GPU 利用率低
# 方案：用小模型连续生成 k 个 draft tokens → 大模型一次验证
#       接受前缀正确的部分，平均 2-3× 加速

def speculative_decode(big_model, small_model, prompt, k=4):
    tokens = list(prompt)
    while len(tokens) < max_len:
        # ① Small model 连续生成 k 个 drafts
        drafts = [small_model.next(tokens + drafts[:i]) for i in range(k)]
        # ② Big model 一次 forward 验证 k+1 个位置
        big_logits = big_model.forward(tokens + drafts)  # 并行！
        # ③ 从 drafts 前缀接受 (拒绝采样)
        accepted = 0
        for i in range(k):
            if accept(big_logits[len(tokens)+i], drafts[i]):
                accepted += 1
            else: break
        tokens.extend(drafts[:accepted])
        tokens.append(big_model.sample(big_logits[len(tokens)]))`;

    return `
      ${MCH.hero({
        icon: "🚢",
        name: "LLM 推理部署架构",
        en: "vLLM / PagedAttention / Continuous Batching / Speculative Decoding / Tensor Parallel",
        tags: ["vLLM", "PagedAttention", "Continuous Batching", "Speculative", "TP/PP/EP", "Flash Attn v3"],
        meta: ["◈ 显存利用 95%+", "⚡ 吞吐 10-20×", "◇ 延迟 2-3×"],
      })}

      ${MCH.versionSection("llm_serving")}

      <div class="section">
        <h2>1. LLM 推理的核心瓶颈</h2>
        <p class="text-sm text-slate-600">LLM 推理不同于训练，主要瓶颈在：</p>
        <div class="grid-3">
          <div class="card"><h3 class="font-bold text-indigo-700 text-sm">💾 显存带宽</h3><p class="text-xs text-slate-600 mt-2">自回归推理每步都要从 HBM 读取全部权重（"带宽受限"），而非算力受限。</p></div>
          <div class="card"><h3 class="font-bold text-indigo-700 text-sm">📦 KV Cache 容量</h3><p class="text-xs text-slate-600 mt-2">序列越长 KV Cache 越大；长上下文 + 高并发 → 显存爆炸。</p></div>
          <div class="card"><h3 class="font-bold text-indigo-700 text-sm">🐢 单步串行</h3><p class="text-xs text-slate-600 mt-2">生成每个 token 必须等上一个完成 → GPU 利用率低（尤其小 batch）。</p></div>
        </div>
      </div>

      <div class="section">
        <h2>2. 现代推理框架栈</h2>
        <table class="table">
          <thead><tr><th>框架</th><th>亮点</th><th>吞吐</th><th>场景</th></tr></thead>
          <tbody>
            <tr><td><b>vLLM</b></td><td>PagedAttention + Continuous Batching</td><td>10-20× HF</td><td>通用 Serving（主流）</td></tr>
            <tr><td><b>SGLang</b></td><td>RadixAttention + 结构化生成</td><td>20-30× HF</td><td>多轮 / 复杂 prompt</td></tr>
            <tr><td>TGI (HuggingFace)</td><td>稳定 + 多模型</td><td>5-10× HF</td><td>企业生产</td></tr>
            <tr><td>TensorRT-LLM</td><td>NVIDIA 深度优化 + 量化</td><td>最快（硬件厂商）</td><td>NVIDIA 独占生态</td></tr>
            <tr><td>lmdeploy</td><td>OpenMMLab，支持 W4A16</td><td>快</td><td>国产硬件友好</td></tr>
            <tr><td><b>llama.cpp</b></td><td>CPU / 端侧，GGUF 格式</td><td>-</td><td>个人电脑 / 移动</td></tr>
            <tr><td>Ollama</td><td>基于 llama.cpp 的本地管理器</td><td>-</td><td>个人开发</td></tr>
            <tr><td>MLC-LLM</td><td>WebGPU / 跨平台</td><td>中</td><td>浏览器推理</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 三大加速技术详解</h2>

        <h3>🔷 ① PagedAttention — KV Cache 分页（vLLM）</h3>
        <p class="text-sm text-slate-600">把 KV Cache 分成固定 block（16 tokens/block），动态分配，类似操作系统虚拟内存：</p>
        <div class="ctrl-panel">
          ${MCH.slider({ id: "pa-seqs", label: "并发请求数", min: 1, max: 32, step: 1, value: 8 })}
          ${MCH.slider({ id: "pa-len-var", label: "序列长度方差", min: 0, max: 1, step: 0.05, value: 0.7 })}
        </div>
        <div id="chart-paged" style="height:260px;margin-top:10px;"></div>

        <h3 style="margin-top:20px;">🔶 ② Continuous Batching — 连续批处理</h3>
        <p class="text-sm text-slate-600">不等 batch 里所有请求完成，而是每步动态替换：一个完成立刻加新的。</p>
        <div id="chart-cb" style="height:280px;margin-top:10px;"></div>

        <h3 style="margin-top:20px;">🚀 ③ Speculative Decoding — 投机解码</h3>
        <p class="text-sm text-slate-600">用小模型连续生成 k 个 draft token → 大模型一次 forward 并行验证。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "sd-k", label: "draft 长度 k", min: 1, max: 8, step: 1, value: 4 })}
              ${MCH.slider({ id: "sd-acc", label: "接受率 α", min: 0.2, max: 0.95, step: 0.01, value: 0.75 })}
              ${MCH.slider({ id: "sd-small-speed", label: "小模型速度比 (vs 大)", min: 2, max: 20, step: 1, value: 10 })}
            </div>
            <div id="sd-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-spec" style="height:300px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. 并行策略（TP / PP / EP）</h2>
        <table class="table">
          <thead><tr><th>策略</th><th>切分维度</th><th>通信开销</th><th>适合规模</th><th>延迟影响</th></tr></thead>
          <tbody>
            <tr><td><b>Tensor Parallel (TP)</b></td><td>矩阵按列/行切</td><td>每层 1 次 AllReduce</td><td>单节点多 GPU</td><td>小（高速 NVLink）</td></tr>
            <tr><td><b>Pipeline Parallel (PP)</b></td><td>层按阶段切</td><td>阶段间 P2P</td><td>多节点（NVLink 不足时）</td><td>大（需 warmup）</td></tr>
            <tr><td><b>Expert Parallel (EP)</b></td><td>MoE 专家分布</td><td>All-to-All</td><td>MoE 大模型</td><td>中</td></tr>
            <tr><td><b>Sequence Parallel (SP)</b></td><td>序列维度切</td><td>小</td><td>超长上下文</td><td>小</td></tr>
            <tr><td>Data Parallel (DP)</td><td>batch 切分</td><td>AllReduce 梯度（训练）</td><td>多卡训练</td><td>-（训练用）</td></tr>
          </tbody>
        </table>
        ${MCH.info(`
          <b>实战组合</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li>单机 8 卡推理 70B：<b>TP=8</b>（全部 NVLink 互联）</li>
            <li>多机推理 400B：<b>TP=8 × PP=4</b>（节点内 TP，跨节点 PP）</li>
            <li>MoE 671B：<b>TP=8 × EP=8</b>（专家跨 8 卡分布）</li>
            <li>1M context：<b>TP + SP</b>（序列并行避免激活爆炸）</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>5. 核心代码：PagedAttention + Speculative</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>6. 部署选型决策树</h2>
        <div class="mermaid">
flowchart TD
    S[需要部署 LLM?] --> Q1{规模?}
    Q1 -->|小于 3B 个人电脑| A1[llama.cpp / Ollama<br/>GGUF Q5_K_M]
    Q1 -->|3-70B 单机 GPU| Q2{高并发?}
    Q1 -->|超过 70B 多机| Q3{MoE?}
    Q2 -->|是 超过 100 QPS| A2[vLLM + PagedAttn<br/>+ TP]
    Q2 -->|否| A3[TGI / HF transformers<br/>+ Flash Attn]
    Q3 -->|是| A4[vLLM + TP + EP<br/>DeepSpeed-FastGen]
    Q3 -->|否| A5[vLLM + TP + PP<br/>TensorRT-LLM]
        </div>
      </div>

      <div class="section">
        <h2>7. 延迟/吞吐指标速查</h2>
        <table class="table">
          <thead><tr><th>指标</th><th>含义</th><th>典型值 (7B on A100)</th><th>优化</th></tr></thead>
          <tbody>
            <tr><td>TTFT (Time To First Token)</td><td>从请求到首 token</td><td>100-500 ms</td><td>KV Cache 预热、Prefill 优化</td></tr>
            <tr><td>TBT / TPOT (Time Per Output Token)</td><td>后续每 token 延迟</td><td>20-50 ms</td><td>Spec Decoding / 量化</td></tr>
            <tr><td>Throughput (tokens/sec)</td><td>系统总吞吐</td><td>1000-5000 (batch)</td><td>Continuous Batching</td></tr>
            <tr><td>Goodput</td><td>满足 SLA 的有效吞吐</td><td>70-90% 总吞吐</td><td>SLO-aware 调度</td></tr>
            <tr><td>GPU Utilization</td><td>MFU</td><td>vLLM: 50-70%</td><td>Flash Attn v3</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>8. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("llm_serving").pros, MCH.getById("llm_serving").cons, MCH.getById("llm_serving").best_for)}
      </div>
    `;
  },

  mount() {
    // PagedAttention 可视化
    const pgChart = MCH.echart(document.getElementById("chart-paged"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "category", data: [] },
      yAxis: { type: "value", name: "KV Cache 显存 (GB)" },
      series: [
        { name: "朴素预分配 (max_len 每请求)", type: "bar", barWidth: 16, color: "#ef4444", data: [] },
        { name: "实际使用", type: "bar", barWidth: 16, color: "#f59e0b", data: [] },
        { name: "PagedAttention (block 分配)", type: "bar", barWidth: 16, color: "#10b981", data: [] },
      ],
    });
    const updatePaged = () => {
      const S = parseInt(document.getElementById("pa-seqs").value);
      const v = parseFloat(document.getElementById("pa-len-var").value);
      const xs = [...Array(S).keys()].map(i => `Req ${i + 1}`);
      const maxLen = 2048;  // 预分配空间
      const actualLens = [...Array(S).keys()].map(i => {
        const base = 400 + Math.random() * (1 - v) * 1000;
        return Math.min(maxLen, base + Math.random() * v * 1500);
      });
      const naive = actualLens.map(() => maxLen * 2 * 32 * 4096 * 2 / 1024 / 1024 / 1024);  // GB per req
      const actual = actualLens.map(l => l * 2 * 32 * 4096 * 2 / 1024 / 1024 / 1024);
      const paged = actualLens.map(l => Math.ceil(l / 16) * 16 * 2 * 32 * 4096 * 2 / 1024 / 1024 / 1024);
      pgChart.setOption({
        xAxis: { data: xs },
        series: [
          { data: naive.map(v => v.toFixed(2)) },
          { data: actual.map(v => v.toFixed(2)) },
          { data: paged.map(v => v.toFixed(2)) },
        ],
      });
    };
    ["pa-seqs", "pa-len-var"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updatePaged();
      });
    });
    updatePaged();

    // Continuous Batching 时间线
    MCH.echart(document.getElementById("chart-cb"), {
      title: { text: "静态 vs 连续批处理时间线（模拟）", left: "center", textStyle: { fontSize: 12 } },
      tooltip: {},
      legend: { top: 30 },
      grid: { left: 80, right: 30, top: 60, bottom: 40 },
      xAxis: { type: "value", name: "时间步", min: 0, max: 100 },
      yAxis: { type: "category", data: ["静态 Req 5", "静态 Req 4", "静态 Req 3", "静态 Req 2", "静态 Req 1", "连续 Req 5", "连续 Req 4", "连续 Req 3", "连续 Req 2", "连续 Req 1"] },
      series: [
        {
          name: "静态批处理", type: "custom",
          data: [
            { value: [0, 0, 100], itemStyle: { color: "#fca5a5" } },      // Req 1 (最长)
            { value: [0, 1, 50], itemStyle: { color: "#fca5a5" } },
            { value: [0, 2, 80], itemStyle: { color: "#fca5a5" } },
            { value: [0, 3, 30], itemStyle: { color: "#fca5a5" } },
            { value: [0, 4, 60], itemStyle: { color: "#fca5a5" } },
            // Idle slots (红色)
            { value: [50, 1, 50], itemStyle: { color: "#fecaca", opacity: 0.5 } },
            { value: [80, 2, 20], itemStyle: { color: "#fecaca", opacity: 0.5 } },
            { value: [30, 3, 70], itemStyle: { color: "#fecaca", opacity: 0.5 } },
            { value: [60, 4, 40], itemStyle: { color: "#fecaca", opacity: 0.5 } },
          ],
          renderItem: (params, api) => {
            const start = api.value(0); const y = api.value(1); const dur = api.value(2);
            const p1 = api.coord([start, y]); const p2 = api.coord([start + dur, y]);
            return { type: "rect", shape: { x: p1[0], y: p1[1] - 10, width: p2[0] - p1[0], height: 20 }, style: api.style() };
          },
          encode: { x: [0, 2], y: 1 },
        },
        {
          name: "连续批处理", type: "custom",
          data: [
            { value: [0, 5, 100], itemStyle: { color: "#86efac" } },
            { value: [0, 6, 50], itemStyle: { color: "#86efac" } },
            { value: [50, 6, 50], itemStyle: { color: "#bbf7d0" } },     // 新请求填入
            { value: [0, 7, 80], itemStyle: { color: "#86efac" } },
            { value: [80, 7, 20], itemStyle: { color: "#bbf7d0" } },
            { value: [0, 8, 30], itemStyle: { color: "#86efac" } },
            { value: [30, 8, 30], itemStyle: { color: "#bbf7d0" } },
            { value: [60, 8, 40], itemStyle: { color: "#6ee7b7" } },
            { value: [0, 9, 60], itemStyle: { color: "#86efac" } },
            { value: [60, 9, 40], itemStyle: { color: "#bbf7d0" } },
          ],
          renderItem: (params, api) => {
            const start = api.value(0); const y = api.value(1); const dur = api.value(2);
            const p1 = api.coord([start, y]); const p2 = api.coord([start + dur, y]);
            return { type: "rect", shape: { x: p1[0], y: p1[1] - 10, width: p2[0] - p1[0], height: 20 }, style: api.style() };
          },
          encode: { x: [0, 2], y: 1 },
        },
      ],
    });

    // Spec decoding
    const sdChart = MCH.echart(document.getElementById("chart-spec"), {
      tooltip: { trigger: "axis" },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "draft 长度 k" },
      yAxis: { type: "value", name: "加速比 ×" },
      series: [{ type: "line", smooth: true, showSymbol: true, color: "#ec4899", lineStyle: { width: 3 }, data: [] }],
    });
    const updateSd = () => {
      const k = parseInt(document.getElementById("sd-k").value);
      const alpha = parseFloat(document.getElementById("sd-acc").value);
      const c = parseFloat(document.getElementById("sd-small-speed").value);
      // Expected accepted: α·(1-α^k) / (1-α) + 1 = 期望每轮生成的 token 数
      // Cost per round: 1 big forward + k small forward → 1 + k/c
      // speedup = expected_tokens / cost
      const ks = [...Array(8).keys()].map(i => i + 1);
      sdChart.setOption({
        series: [{
          data: ks.map(ki => {
            const expected = (1 - Math.pow(alpha, ki + 1)) / (1 - alpha);
            const cost = 1 + ki / c;
            return [ki, expected / cost];
          }),
        }],
      });
      const expected = (1 - Math.pow(alpha, k + 1)) / (1 - alpha);
      const cost = 1 + k / c;
      const speedup = expected / cost;
      document.getElementById("sd-info").innerHTML = `
        <b>每轮大模型 forward</b> 期望产出 <span style="color:#ec4899;">${expected.toFixed(2)}</span> tokens<br/>
        <b>每轮等效大模型调用</b>: ${cost.toFixed(2)}×<br/>
        <b>理论加速比</b>：<span style="color:#ec4899;font-size:18px;font-weight:700;">${speedup.toFixed(2)}×</span><br/>
        <span style="color:#64748b;">实际 Llama-3 70B + 7B draft 通常 α≈0.7, 加速 2-2.5×。</span>
      `;
    };
    ["sd-k", "sd-acc", "sd-small-speed"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateSd();
      });
    });
    updateSd();
  },
});
