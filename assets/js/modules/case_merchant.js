/* 模块：案例 · 商户多模态识别 */
MCH.register("case_merchant", {
  render() {
    return `
      ${MCH.hero({
        icon: "🛡",
        name: "案例 · 商户多模态多任务识别 MCH-Recog-v0",
        en: "Merchant Multi-Modal Multi-Task Recognition (Industry Case)",
        tags: ["多模态", "多任务", "长尾 1:10000", "日 1亿商户", "Perceiver-IO", "PLE-MMoE"],
        meta: ["◈ 5 模态融合", "⚡ Teacher 320M / Student 25M INT8", "◇ P99 < 50ms"],
      })}

      ${MCH.versionSection("case_merchant")}

      <div class="section">
        <h2>1. 业务场景</h2>
        <p class="text-sm text-slate-600">对支付平台每日 <b>1 亿+ 商户</b>做多模态多任务识别：</p>
        <div class="grid-2">
          <div class="card"><h4 class="font-semibold">🎯 核心任务</h4>
            <ul class="text-xs text-slate-600 mt-2 list-disc pl-5">
              <li><b>风险识别</b>：黑/灰/白 3 分类 + S/A/B/C/D 5 级</li>
              <li><b>可信评分</b>：0-1 连续分 + 5 级</li>
              <li><b>类目识别</b>：20 L1 × 2000 L2 叶子</li>
              <li><b>PluginHead</b>：套现/刷单/养商户/返现中介 (热插拔)</li>
            </ul>
          </div>
          <div class="card"><h4 class="font-semibold">⚠️ 核心挑战</h4>
            <ul class="text-xs text-slate-600 mt-2 list-disc pl-5">
              <li>黑样本:白样本 ≈ <b>1:10000</b> 极度长尾</li>
              <li>Top 1% 商户贡献 80%+ 交易额（业务量纲）</li>
              <li>5 路异构模态：文本/序列/图/图像/辅助</li>
              <li>P99 延迟 &lt; 50ms，批量 2h 跑完 1 亿</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 整体架构</h2>
        <div class="mermaid">
flowchart TB
    subgraph S0[数据源 Data Sources]
        D1[文本资料]
        D2[交易行为序列]
        D3[关系数据]
        D4[门头/场景图片]
        D5[辅助资料APP/URL/统计]
    end
    subgraph S1[模态 Encoders]
        E1[TextEncoder<br/>MacBERT + LoRA]
        E2[SeqEncoder<br/>BST + Time2Vec]
        E3[GraphEncoder<br/>precomputed/HGT]
        E4[ImageEncoder<br/>CLIP Attn Pool]
        E5[AuxEncoder<br/>Hash + Dense]
    end
    subgraph S2[跨模态融合]
        F[Perceiver-IO<br/>32 Latent × CrossAttn×4<br/>+ Modality Dropout p=0.15]
    end
    subgraph S3[多任务专家]
        M[PLE-MMoE<br/>共享 4 + 每任务私有 1]
        H1[Risk Head<br/>3-cls + 5级Ordinal]
        H2[Trust Head<br/>回归 + 5 级]
        H3[Category Head<br/>L1 + L2 余弦]
        H4[Plugin Heads<br/>热插拔]
    end
    subgraph S5[推理部署]
        T1[Teacher ~320M FP16]
        T2[Student ~25M INT8]
    end
    D1 --> E1
    D2 --> E2
    D3 --> E3
    D4 --> E4
    D5 --> E5
    E1 & E2 & E3 & E4 & E5 --> F --> M
    M --> H1 & H2 & H3 & H4
    H1 & H2 & H3 -->|KD| T1 --> T2
        </div>
      </div>

      <div class="section">
        <h2>3. 子模块导航（按技术归类）</h2>
        <div class="grid-3">
          <div class="card" style="border-top:3px solid #ec4899;">
            <h4 class="font-semibold text-pink-700">🤖 大模型分支</h4>
            <div class="mt-2 text-xs text-slate-600">
              <a href="#/text_encoder" class="text-pink-600 block">→ 文本 Encoder (MacBERT + LoRA)</a>
            </div>
          </div>
          <div class="card" style="border-top:3px solid #7c3aed;">
            <h4 class="font-semibold text-violet-700">🧠 神经网络通用技术</h4>
            <div class="mt-2 text-xs text-slate-600">
              <a href="#/seq_encoder" class="text-violet-600 block">→ 交易序列 (BST + Time2Vec)</a>
              <a href="#/image_encoder" class="text-violet-600 block">→ 图像 Attention Pool</a>
              <a href="#/fusion" class="text-violet-600 block">→ Perceiver-IO 融合</a>
              <a href="#/mmoe" class="text-violet-600 block">→ PLE-MMoE 多任务</a>
            </div>
          </div>
          <div class="card" style="border-top:3px solid #10b981;">
            <h4 class="font-semibold text-emerald-700">🕸️ 图算法分支</h4>
            <div class="mt-2 text-xs text-slate-600">
              <a href="#/graph_encoder" class="text-emerald-600 block">→ HGT 异构图 Encoder</a>
            </div>
          </div>
          <div class="card" style="border-top:3px solid #0ea5e9;">
            <h4 class="font-semibold text-sky-700">🛠 商户识别专属</h4>
            <div class="mt-2 text-xs text-slate-600">
              <a href="#/aux_encoder" class="text-sky-600 block">→ 辅助 Encoder (Hash+Dense)</a>
              <a href="#/heads" class="text-sky-600 block">→ 多任务 Heads</a>
              <a href="#/samplers" class="text-sky-600 block">→ CB+GMV 采样器</a>
            </div>
          </div>
          <div class="card" style="border-top:3px solid #f59e0b;">
            <h4 class="font-semibold text-amber-700">🔧 训练 &amp; 部署</h4>
            <div class="mt-2 text-xs text-slate-600">
              <a href="#/training" class="text-amber-600 block">→ Decoupled cRT 三阶段</a>
              <a href="#/distillation" class="text-amber-600 block">→ 蒸馏 &amp; INT8 量化</a>
            </div>
          </div>
          <div class="card" style="border-top:3px solid #ef4444;">
            <h4 class="font-semibold text-red-700">⚖️ 损失函数协同</h4>
            <div class="mt-2 text-xs text-slate-600">
              <a href="#/losses" class="text-red-600 block">→ Focal/LDAM/Seesaw/SupCon</a>
              <a href="#/loss_multitask" class="text-red-600 block">→ 多目标优化 (GradNorm/PCGrad)</a>
            </div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 设计目标 SLA</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>目标</th><th>技术支撑</th></tr></thead>
          <tbody>
            <tr><td><b>识别任务</b></td><td>风险 3cls + S/A/B/C/D；可信；类目 20→2000；+ Plugin</td><td>PLE-MMoE + 多 Head</td></tr>
            <tr><td><b>规模</b></td><td>日 1亿+；QPS 5w+</td><td>Student 25M INT8 + Triton</td></tr>
            <tr><td><b>不均衡</b></td><td>黑:白 ≈ 1:10,000；Top 1% 占 80% GMV</td><td>CB Sampler + GMV 加权 + LDAM/Seesaw/SupCon</td></tr>
            <tr><td><b>时延</b></td><td>批量 &lt; 2h / 1 亿；P99 &lt; 50ms</td><td>Graph/Image/Text 全 T-1 预计算</td></tr>
            <tr><td><b>成本</b></td><td>单日全量推理 &lt; ¥8,000</td><td>Student INT8 batch 推理</td></tr>
          </tbody>
        </table>
      </div>
    `;
  },
});
