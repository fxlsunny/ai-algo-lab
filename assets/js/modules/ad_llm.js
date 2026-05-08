/* 模块：LLM 异常检测 */
MCH.register("ad_llm", {
  render() {
    const code = `# LogGPT — 用 LLM 做日志异常检测（零样本）
# Li et al., 2023

import openai

def detect_log_anomaly(logs: list[str]) -> dict:
    """用 GPT-4 直接判断一批日志是否异常"""
    prompt = f'''你是资深运维工程师。下面是一批 Nginx 日志，判断是否存在异常。
如果存在异常，请指出具体哪条日志异常，并说明原因。

日志（共 {len(logs)} 条）：
{chr(10).join(logs)}

输出格式 (JSON):
{{"is_anomaly": bool, "anomaly_indices": [...], "reasons": [...]}}'''

    resp = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# ============================================================
# SigLLM (2024) — 时序异常 LLM 检测
# 核心：把时序量化成 token → LLM 预测"下一个 token" → 差异即异常
# ============================================================
def sigllm_anomaly(series: np.ndarray, llm):
    # 1) 量化时序为 token
    tokens = quantize_series_to_tokens(series, bins=128)
    # 2) LLM 推理下一个 token 的分布
    for i in range(len(tokens)):
        logits = llm.predict_next(tokens[:i])
        prob_actual = softmax(logits)[tokens[i]]
        # 概率低的 token = 异常
        if prob_actual < threshold:
            anomalies.append(i)
    return anomalies


# ============================================================
# Few-Shot 提示 + Chain-of-Thought（推荐）
# ============================================================
prompt = '''任务：检测支付交易是否异常。

示例 1 (正常)：金额 50，夜间 0，同商户连续 2 次。
思考：小额正常消费，无异常。→ 正常

示例 2 (异常)：金额 5000，夜间 1，同商户连续 20 次。
思考：大额夜间连续多次 → 疑似盗刷。→ 异常

示例 3 (异常)：金额 0.01，同一用户 1 分钟 100 次。
思考：小额高频，疑似撞库测试。→ 异常

现在判断：{transaction}
先分析，再结论。'''`;

    return `
      ${MCH.hero({
        icon: "🤖",
        name: "LLM 异常检测",
        en: "Large Language Model for Anomaly Detection",
        tags: ["零样本", "日志异常", "时序异常", "Prompting", "可解释"],
        meta: ["◈ 2023-24 新范式", "⚡ 无需训练 + 自然语言解释"],
      })}

      ${MCH.versionSection("ad_llm")}

      <div class="section">
        <h2>1. 为什么 LLM 也能做异常检测？</h2>
        <p class="text-sm text-slate-600">LLM 在海量文本上预训练，已经<b>隐式学习了世界中"什么是正常"</b>。所以可以做零样本异常检测：</p>
        <div class="mermaid">
flowchart LR
    I[输入数据<br/>日志/时序/JSON] --> P[Prompt 构造<br/>任务描述 + 数据 + 示例]
    P --> L[LLM 推理<br/>GPT-4/LLaMA/Qwen]
    L --> O[结构化输出<br/>is_anomaly + 原因解释]
    O --> A[运维/风控动作]
        </div>
        ${MCH.info(`
          <b>LLM 异常检测的独特优势</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>零样本</b>：不需要收集异常样本训练</li>
            <li><b>自然语言解释</b>：直接输出"为什么异常"（传统 AE 只给一个分数）</li>
            <li><b>多模态</b>：支持日志、JSON、代码、图像（多模态 LLM）混合输入</li>
            <li><b>快速迭代</b>：改 prompt 即可，不需要重训</li>
            <li><b>结合外部知识</b>：配合 RAG 查询历史异常知识库</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. LLM 异常检测的三种范式</h2>
        <table class="table">
          <thead>
            <tr><th>范式</th><th>做法</th><th>适合场景</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>① 直接分类</b></td>
              <td>Zero-shot：直接让 LLM 判断 is_anomaly<br/>Few-shot：给几个示例再判断</td>
              <td>小样本日志/业务异常</td>
            </tr>
            <tr>
              <td><b>② 序列建模</b></td>
              <td>把时序 tokenize（类似 Chronos）<br/>LLM 预测下一个 token，概率低 = 异常</td>
              <td>运维指标 / KPI 异常</td>
            </tr>
            <tr>
              <td><b>③ Agent + RAG</b></td>
              <td>LLM 编排多种工具：<br/>IsolationForest + LLM 解释<br/>历史异常知识库 retrieval</td>
              <td>🏆 复杂异常分析 (SRE/AIOps)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 代表性工作</h2>
        <table class="table">
          <thead><tr><th>工作</th><th>年份</th><th>核心贡献</th></tr></thead>
          <tbody>
            <tr><td><b>🏆 LogGPT</b></td><td>2023</td><td>首次用 GPT-4 做日志异常检测，超越传统模型</td></tr>
            <tr><td><b>LLMAD</b></td><td>2024</td><td>时序异常 + LLM 自然语言解释，结合规则和 CoT</td></tr>
            <tr><td><b>🏆 SigLLM</b></td><td>2024</td><td>时序 tokenize → LLM 预测，5 大工业数据集验证</td></tr>
            <tr><td><b>AnomalyLLM</b></td><td>2024</td><td>In-Context Learning，只需 1-5 条异常示例</td></tr>
            <tr><td><b>🆕 AIOps Agent</b></td><td>2024</td><td>LLM Agent 编排 Metrics + Trace + Log 多源异常</td></tr>
            <tr><td><b>🆕 Contextual AD</b> (Anthropic)</td><td>2024</td><td>Claude 3.5 长上下文 + RAG 历史异常知识</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>4. Prompt 模板示例</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 交互：简单场景下 LLM vs 传统方法对比（模拟）</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              <label class="text-xs block mb-2">选择场景：</label>
              <select id="allm-sc" class="text-xs border rounded p-1 w-full">
                <option value="log">Nginx 日志异常</option>
                <option value="ts">CPU 使用率时序</option>
                <option value="fraud">支付交易异常</option>
                <option value="api">API 错误模式</option>
              </select>
              ${MCH.slider({ id: "allm-complex", label: "异常复杂度", min: 1, max: 5, step: 1, value: 3 })}
            </div>
            <div id="allm-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-allm" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>6. LLM 异常检测的工程挑战与对策</h2>
        <table class="table">
          <thead><tr><th>挑战</th><th>对策</th></tr></thead>
          <tbody>
            <tr><td><b>推理成本</b>：每样本需要 1 次 LLM 调用</td><td>两阶段：先 IF 筛可疑，再 LLM 解释；或批量推理</td></tr>
            <tr><td><b>幻觉</b>：LLM 可能"编造"异常</td><td>Function Calling + 约束输出格式；加 self-check</td></tr>
            <tr><td><b>上下文长度</b>：日志量大易超长</td><td>长上下文 LLM (Claude 200k) 或滑窗摘要</td></tr>
            <tr><td><b>延迟</b>：API 调用 0.5-3s</td><td>异步 + 流式；本地部署 Qwen/LLaMA + vLLM</td></tr>
            <tr><td><b>成本</b>：GPT-4 按 token 计费</td><td>小模型（Qwen-2.5 7B）本地部署；蒸馏</td></tr>
            <tr><td><b>结果不稳定</b></td><td>temperature=0 + JSON Schema 约束</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ad_llm").pros, MCH.getById("ad_llm").cons, MCH.getById("ad_llm").best_for)}
      </div>
    `;
  },

  mount() {
    const scenarios = {
      log: { llm: [88, 92, 85, 90, 82], trad: [65, 70, 60, 68, 58], name: "Nginx 日志", xLabels: ["简单", "中等", "复杂", "组合", "罕见"] },
      ts: { llm: [75, 78, 72, 80, 70], trad: [80, 75, 68, 60, 50], name: "CPU 时序", xLabels: ["单点突起", "缓慢漂移", "周期破缺", "多模态", "新模式"] },
      fraud: { llm: [82, 88, 85, 90, 92], trad: [85, 80, 75, 65, 55], name: "支付异常", xLabels: ["金额异常", "时间异常", "组合模式", "对手方异常", "新型攻击"] },
      api: { llm: [85, 88, 82, 87, 80], trad: [70, 72, 65, 68, 55], name: "API 错误", xLabels: ["4xx 激增", "5xx 波动", "慢查询", "依赖链", "零日错误"] },
    };
    const chart = MCH.echart(document.getElementById("chart-allm"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", data: [] },
      yAxis: { type: "value", name: "F1 (%)", max: 100 },
      series: [
        { name: "传统方法 (IF/AE)", type: "bar", barWidth: 24, color: "#94a3b8", data: [] },
        { name: "LLM 零样本", type: "bar", barWidth: 24, color: "#ec4899", data: [] },
      ],
    });
    const update = () => {
      const sc = document.getElementById("allm-sc").value;
      const cx = parseInt(document.getElementById("allm-complex").value);
      const s = scenarios[sc];
      chart.setOption({
        xAxis: { data: s.xLabels },
        series: [{ data: s.trad }, { data: s.llm }],
      });
      const idx = Math.min(cx - 1, 4);
      document.getElementById("allm-info").innerHTML = `
        <b>场景</b>：${s.name}<br/>
        <b>复杂度 ${cx}/5</b>（${s.xLabels[idx]}）<br/>
        <b>传统 F1</b>：<span style="color:#94a3b8;font-weight:700;">${s.trad[idx]}%</span><br/>
        <b>LLM F1</b>：<span style="color:#ec4899;font-weight:700;">${s.llm[idx]}%</span><br/>
        <b>差距</b>：${s.llm[idx] > s.trad[idx] ? `<span style="color:#10b981;">LLM +${s.llm[idx] - s.trad[idx]}%</span>` : `<span style="color:#f59e0b;">传统 +${s.trad[idx] - s.llm[idx]}%</span>`}<br/>
        <span style="color:#64748b;">越复杂的异常模式，LLM 的优势越明显（利用世界知识）。</span>
      `;
    };
    document.getElementById("allm-sc").addEventListener("change", update);
    document.getElementById("allm-complex").addEventListener("input", (e) => {
      document.getElementById("allm-complex-val").textContent = e.target.value;
      update();
    });
    update();
  },
});
