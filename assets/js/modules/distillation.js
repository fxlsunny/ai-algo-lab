/* 模块：蒸馏 & 量化 */
MCH.register("distillation", {
  render() {
    const code = `# Teacher → Student 知识蒸馏 + INT8 量化
# 来源：src/inference/distill.py + quantize.py

def distill_loss(teacher_out, student_out, labels, temperature=4.0,
                 w_feat=1.0, w_logit=2.0, w_hard=1.0):
    """蒸馏 3 种信号融合：
       1) Feature KD  : unified_emb 的 L2 对齐（让 student 学 teacher 的 feature 空间）
       2) Logit KD    : 各任务头 logits 做 KL（T=4 软化）
       3) Hard Label  : student 对真实标签 CE（防止 teacher 的错误传染）"""

    # Feature KD：embedding 对齐
    loss_feat = F.mse_loss(student_out["unified_emb"],
                            teacher_out["unified_emb"].detach()) * w_feat

    # Logit KD：KL(softmax(s/T) || softmax(t/T)) * T²
    loss_logit = 0.
    for key in ["risk_cls_logits", "risk_ord_logits",
                 "trust_level_logits", "cat_l1_logits", "cat_l2_logits"]:
        t = teacher_out[key].detach() / temperature
        s = student_out[key] / temperature
        loss_logit += F.kl_div(F.log_softmax(s, -1),
                                F.softmax(t, -1),
                                reduction="batchmean") * (temperature ** 2)

    # Hard Label
    loss_hard = F.cross_entropy(student_out["risk_cls_logits"], labels["risk_label"])
    # ... + trust + cat

    return loss_feat * w_feat + loss_logit * w_logit + loss_hard * w_hard


def build_student_model(cfg):
    """Student 规格（cfg["distill"]["student"] 配置）："""
    st = cfg["distill"]["student"]
    # MacBERT base → MiniLM-L6（~60M → ~22M）
    cfg["model"]["text"]["backbone"] = st["text_backbone"]          # "sentence-transformers/paraphrase-MiniLM-L6-v2"
    cfg["model"]["text"]["use_lora"] = True
    # Seq Transformer 4 层 → 2 层
    cfg["model"]["sequence"]["num_layers"] = st["sequence_num_layers"]  # 2
    cfg["model"]["sequence"]["num_heads"] = st["sequence_num_heads"]    # 4
    # Perceiver 32 latent / 4 层 → 16 latent / 2 层
    cfg["model"]["fusion"]["num_layers"]  = st["fusion_num_layers"]     # 2
    cfg["model"]["fusion"]["num_latents"] = st["fusion_num_latents"]    # 16
    # MMoE 4 shared / 256 → 2 shared / 128
    cfg["model"]["mmoe"]["num_shared_experts"] = st["mmoe_num_shared_experts"]  # 2
    cfg["model"]["mmoe"]["expert_dim"] = st["mmoe_expert_dim"]                   # 128
    return build_model(cfg)


def int8_quantize(onnx_fp32_path, onnx_int8_path):
    """ONNX Runtime Dynamic INT8 量化（MatMul / Gemm 权重量化）"""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(onnx_fp32_path, onnx_int8_path,
                     weight_type=QuantType.QInt8,
                     per_channel=True, reduce_range=False)`;

    return `
      ${MCH.hero({
        icon: "K",
        name: "蒸馏 & 量化 — Teacher → Student + INT8",
        en: "Knowledge Distillation + ONNX INT8 Dynamic Quantization",
        tags: ["Teacher 320M FP16", "Student 25M INT8", "Feat + Logit KD", "ONNX Runtime", "日 1亿全量"],
        meta: ["◈ T=4 蒸馏温度", "⚡ Student 快 10×", "◇ P99 < 50ms"],
      })}

      <div class="section">
        <h2>1. 为什么需要蒸馏+量化？</h2>
        ${MCH.info(`
          线上服务两大硬约束：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>日级 1 亿商户 batch 推理</b> 要求每商户 &lt; 10ms；</li>
            <li><b>在线 5w QPS</b> 要求 P99 &lt; 50ms；</li>
          </ul>
          但 Teacher 完整模型 ≈ 320M（MacBERT + BST + HGT + Fusion + MMoE）GPU 推理 ≈ 80ms。
          于是走 <b>Teacher / Student 双模型 + INT8</b> 分层架构。
        `, "biz")}
      </div>

      <div class="section">
        <h2>2. Teacher / Student 规格对比</h2>
        <table class="table">
          <thead><tr><th>组件</th><th>Teacher (FP16)</th><th>Student (FP32)</th><th>Student (INT8)</th></tr></thead>
          <tbody>
            <tr><td>Text Backbone</td><td>MacBERT-base 110M</td><td>MiniLM-L6 22M</td><td>MiniLM-L6 22M (量化)</td></tr>
            <tr><td>Seq Transformer</td><td>4 层 × 8 头</td><td>2 层 × 4 头</td><td>2 层 × 4 头</td></tr>
            <tr><td>Perceiver-IO</td><td>K=32, 4 层</td><td>K=16, 2 层</td><td>K=16, 2 层</td></tr>
            <tr><td>PLE-MMoE</td><td>4 shared × 256-d</td><td>2 shared × 128-d</td><td>2 shared × 128-d</td></tr>
            <tr><td><b>总参数</b></td><td><b>320 M</b></td><td><b>65 M</b></td><td><b>25 M</b></td></tr>
            <tr><td><b>显存占用</b></td><td>~640 MB</td><td>~260 MB</td><td><b>~65 MB</b></td></tr>
            <tr><td><b>单商户推理</b></td><td>80 ms</td><td>22 ms</td><td><b>9 ms</b></td></tr>
            <tr><td><b>对外指标 AUC 回收率</b></td><td>100% 基线</td><td>98.5%</td><td><b>97.2%</b></td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 三路蒸馏信号</h2>
        <div class="mermaid">
flowchart LR
    T["🎓 Teacher Model<br/>FP16 · 320M<br/>(冻结)"]
    S["👶 Student Model<br/>FP32 · 65M"]
    Batch[训练 batch]

    Batch --> T
    Batch --> S

    T -- unified_emb --> FKD["Feature KD<br/>MSE(s_emb, t_emb.detach())"]
    S -- unified_emb --> FKD

    T -- logits softmax/T --> LKD["Logit KD<br/>KL(s/T, t/T) · T²"]
    S -- logits softmax/T --> LKD

    Batch -- labels --> HKD["Hard Label CE<br/>CE(s_logits, y)"]
    S -- logits --> HKD

    FKD --> Loss["Total = w_feat·L_feat + w_logit·L_logit + w_hard·L_hard"]
    LKD --> Loss
    HKD --> Loss
        </div>
      </div>

      <div class="section">
        <h2>4. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 交互：Logit KD 温度的影响</h2>
        <p class="text-sm text-slate-600">
          KL 蒸馏会把 teacher 的 logits 先除以 T 再 softmax，T 越大 softmax 越平，<b>"暗知识"</b>（非目标类的概率分布）越能被传递。
        </p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "kd-T", label: "温度 T", min: 1, max: 10, step: 0.1, value: 4 })}
              ${MCH.slider({ id: "kd-true-prob", label: "Teacher 正确类原始概率", min: 0.5, max: 0.99, step: 0.01, value: 0.85 })}
            </div>
            <p class="text-xs text-slate-500 mt-2">T=1 → 硬标签 one-hot；T>1 暗知识显现。<b>实际 loss 会乘 T²</b>修正梯度量级。</p>
          </div>
          <div id="chart-kd" style="height:300px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>6. 交互：INT8 量化的数值表示</h2>
        <p class="text-sm text-slate-600">
          Dynamic INT8 对权重做逐通道量化：<code>w_q = round(w / s)</code>，推理时还原 <code>w ≈ s · w_q</code>。
          精度损失可控但量化误差在激活分布尾部会被放大。
        </p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "q-scale", label: "Scale (per-channel)", min: 0.001, max: 0.05, step: 0.001, value: 0.01, format: (v) => parseFloat(v).toFixed(3) })}
              ${MCH.slider({ id: "q-bits", label: "量化位数", min: 2, max: 16, step: 1, value: 8 })}
            </div>
            <div id="q-stats" class="text-xs text-slate-600 mt-3"></div>
          </div>
          <div id="chart-quant" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>7. 上线分层策略（双模型协同）</h2>
        <div class="mermaid">
flowchart LR
    In[商户请求]
    Stu[Student INT8<br/>25M 9ms]
    HighRisk{{"风险分 超过 0.7?"}}
    Tea[Teacher FP16<br/>320M 80ms]
    Out1[输出结果]
    Out2[Teacher 复核输出]

    In --> Stu
    Stu --> HighRisk
    HighRisk -- 否 --> Out1
    HighRisk -- 是 --> Tea
    Tea --> Out2
        </div>
        ${MCH.info(`
          <b>业务细节：</b>
          <ul style="margin-top:6px;padding-left:20px;">
            <li>99% 请求只走 <b>Student</b>（<code>P99 &lt; 10ms</code>）；</li>
            <li>Student 风险分 &gt; 0.7 的 <b>高危 1%</b> 流量异步走 Teacher 复核（不阻塞主链路）；</li>
            <li>Teacher 结果回写覆盖 Student 的输出，保证<b>高危精度</b>不损失；</li>
            <li>Teacher 同时用于 <b>日级 Top 500 万商户的全链路复核</b>（Spark GPU 批量）。</li>
          </ul>
        `, "tip")}
      </div>
    `;
  },

  mount() {
    // Logit KD 温度图
    const kdChart = MCH.echart(document.getElementById("chart-kd"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "category", data: ["✓ 正确", "次优", "其他 1", "其他 2", "其他 3", "其他 4"] },
      yAxis: { type: "value", name: "概率", max: 1 },
      series: [
        { name: "T=1 (原始)", type: "bar", data: [], barGap: 0.1, color: "#94a3b8" },
        { name: "T 当前", type: "bar", data: [], color: "#4f46e5" },
      ],
    });
    const updateKd = () => {
      const T = parseFloat(document.getElementById("kd-T").value);
      const trueP = parseFloat(document.getElementById("kd-true-prob").value);
      // 构造一个 teacher logits 对应的概率分布
      const raw_probs = [trueP, (1 - trueP) * 0.45, (1 - trueP) * 0.25, (1 - trueP) * 0.15, (1 - trueP) * 0.10, (1 - trueP) * 0.05];
      const logits = raw_probs.map(p => Math.log(Math.max(p, 1e-6)));
      const softT = (z, T) => { const m = Math.max(...z); const ez = z.map(v => Math.exp((v - m) / T)); const s = ez.reduce((a, b) => a + b, 0); return ez.map(v => v / s); };
      const p1 = softT(logits, 1);
      const pt = softT(logits, T);
      kdChart.setOption({
        series: [
          { data: p1.map(v => v.toFixed(3)) },
          { data: pt.map(v => v.toFixed(3)) },
        ],
      });
    };
    ["kd-T", "kd-true-prob"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateKd();
      });
    });
    updateKd();

    // Quant viz
    const qChart = MCH.echart(document.getElementById("chart-quant"), {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "原始 FP32 权重" },
      yAxis: { type: "value", name: "量化后 INT8 还原" },
      series: [
        { name: "y = x (理想)", type: "line", showSymbol: false, data: [], color: "#94a3b8", lineStyle: { type: "dashed" } },
        { name: "INT8 量化还原", type: "line", showSymbol: false, data: [], color: "#4f46e5", step: "middle", lineStyle: { width: 2 } },
      ],
    });
    const updateQ = () => {
      const s = parseFloat(document.getElementById("q-scale").value);
      const bits = parseInt(document.getElementById("q-bits").value);
      const qmax = Math.pow(2, bits - 1) - 1;
      const xs = MCH.linspace(-0.2, 0.2, 500);
      const dequant = xs.map(x => {
        const q = Math.max(-qmax - 1, Math.min(qmax, Math.round(x / s)));
        return q * s;
      });
      qChart.setOption({
        series: [
          { data: xs.map(x => [x, x]) },
          { data: xs.map((x, i) => [x, dequant[i]]) },
        ],
      });
      const err = dequant.map((d, i) => Math.abs(d - xs[i]));
      const mean = err.reduce((a, b) => a + b, 0) / err.length;
      const max = Math.max(...err);
      document.getElementById("q-stats").innerHTML = `
        <b>量化误差统计</b><br/>
        平均 |err| = <b style="color:#4f46e5;">${mean.toExponential(2)}</b> · 最大 |err| = <b style="color:#ef4444;">${max.toExponential(2)}</b><br/>
        可表示范围 ≈ ±${(qmax * s).toFixed(3)} · 量化点数 = 2^${bits} = ${Math.pow(2, bits)} 档<br/>
        <span style="color:#64748b;">⚡ 实际 ONNX Runtime 用 per-channel scale，每通道独立找最优 s。</span>
      `;
    };
    ["q-scale", "q-bits"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        const v = id === "q-scale" ? parseFloat(e.target.value).toFixed(3) : e.target.value;
        document.getElementById(id + "-val").textContent = v;
        updateQ();
      });
    });
    updateQ();
  },
});
