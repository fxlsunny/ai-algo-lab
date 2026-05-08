/* 模块：Decoupled cRT 训练 */
MCH.register("training", {
  render() {
    const code = `# Decoupled cRT 三阶段训练 (Kang et al., ICLR'20)
# 来源：src/train/trainer.py

# Stage 1  表征学习：全网训练，uniform 采样 —— 让 backbone 学好的 feature
# Stage 2  分类器重训：冻结 backbone，CB 采样 —— 只调 heads + mmoe
# Stage 3  GradNorm 微调：解冻 backbone，小 lr —— 联合微调 + 动态任务权重

class Trainer:
    def __init__(self, cfg, train_records, eval_records, class_stats, stage=1):
        self.model = build_model(cfg).to(device)

        # ---- Decoupled cRT 冻结策略 ----
        if stage == 2:
            self._freeze_backbone(True)       # 冻 encoder + fusion
        elif stage == 3:
            self._freeze_backbone(False)      # 全网解冻

        # ---- Sampler 按 stage 自动切换 ----
        scfg = cfg["training"]["sampler"]
        if scfg["type"] == "class_balanced":
            sampler = ClassBalancedSampler(labels_risk, beta=0.9999)
        elif scfg["type"] == "gmv_weighted":
            sampler = GmvWeightedSampler(labels_risk, gmv, beta=0.9999, gmv_lambda=1.0)
        # Stage 2 若没配 → 默认 CB
        if stage == 2 and sampler is None:
            sampler = ClassBalancedSampler(labels_risk, beta=0.9999)

        # ---- Optimizer 分组：backbone 低 lr、head 高 lr ----
        groups = [
            {"params": backbone_params, "lr": cfg["training"]["lr_backbone"]},  # 1e-5
            {"params": head_params,     "lr": cfg["training"]["lr_head"]},     # 1e-4
        ]
        self.optim = torch.optim.AdamW(groups, weight_decay=0.01)

        # ---- 多任务 balancer（Stage 3 启用 GradNorm）----
        if bal_cfg["type"] == "gradnorm":
            self.balancer = GradNormBalancer(["risk", "trust", "category"], alpha=1.5)

        # ---- DRW Scheduler（Stage 2 起生效）----
        self.drw = DRWScheduler(class_stats["trust_level"], start_epoch=12)

    def train_one_epoch(self, epoch):
        # DRW：每 epoch 刷新 LDAM weight
        w = self.drw.get_weight(epoch)
        if w is not None:
            self.loss_fns["trust_level"].weight = w.to(self.device)

        for step, batch in enumerate(self.train_loader):
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out, losses = self._forward_loss(batch)
                loss = losses["total"] / grad_accum
            self.scaler.scale(loss).backward()
            if (step + 1) % grad_accum == 0:
                self.scaler.unscale_(self.optim)
                nn.utils.clip_grad_norm_(params, clip_norm)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)`;

    return `
      ${MCH.hero({
        icon: "C",
        name: "Decoupled cRT — 三阶段训练流程",
        en: "Decoupled Classifier Re-Training (Kang et al., ICLR'20)",
        tags: ["Stage 1 表征", "Stage 2 分类器重训", "Stage 3 联合微调", "DRW", "GradNorm", "AMP fp16"],
        meta: ["◈ 冻结/解冻切换", "⚡ AdamW 双 lr group", "◇ 来源 src/train/trainer.py"],
      })}

      <div class="section">
        <h2>1. 核心思想</h2>
        <p class="text-sm text-slate-600">
          ICLR'20 的 Decoupled cRT 论文揭示了长尾学习的一个关键事实：
        </p>
        ${MCH.info(`
          <b>表征学习</b>（如何把商户编码成 embedding）和 <b>分类器学习</b>（如何把 embedding 分到各类）
          应该用<b>不同的数据分布</b>！—— 前者用<b>真实分布</b>学到最泛化的 feature；后者用<b>重平衡分布</b>消除分类器偏置。
        `, "biz")}
        <p class="text-sm text-slate-600 mt-3">
          本方案在此基础上加入 Stage 3：用 <b>GradNorm</b> 联合微调，进一步平衡多任务损失量级。
        </p>
      </div>

      <div class="section">
        <h2>2. 三阶段流程图</h2>
        <div class="mermaid">
flowchart LR
    A["🎲 Stage 1<br/>表征学习<br/>Uniform 采样<br/>全网可训<br/>20 epoch"]
    B["🎯 Stage 2<br/>分类器重训<br/>CB 采样 β=0.9999<br/>❄️ 冻结 encoders<br/>仅训 heads+mmoe<br/>5 epoch"]
    C["🧬 Stage 3<br/>联合微调<br/>GMV-Weighted 采样<br/>🔓 解冻全网 小 lr<br/>GradNorm 动态权重<br/>3 epoch"]
    A --> B --> C
    D["💾 Teacher 最佳 Ckpt<br/>teacher_best_stage3.pt"]
    C --> D
        </div>
      </div>

      <div class="section">
        <h2>3. 交互：三阶段详情（点击切换）</h2>
        <div class="grid-3" id="stage-panel">
          <div class="stage-node active" data-stage="1">
            <div class="badge">Stage 1</div>
            <h4 class="font-semibold">🎲 表征学习</h4>
            <table class="text-xs mt-2">
              <tr><td class="text-slate-500 pr-3">采样</td><td><b>Uniform</b>（随机）</td></tr>
              <tr><td class="text-slate-500 pr-3">Backbone</td><td class="tag tag-green">🔥 可训</td></tr>
              <tr><td class="text-slate-500 pr-3">Heads</td><td class="tag tag-green">🔥 可训</td></tr>
              <tr><td class="text-slate-500 pr-3">lr_backbone</td><td>1e-4</td></tr>
              <tr><td class="text-slate-500 pr-3">lr_head</td><td>1e-3</td></tr>
              <tr><td class="text-slate-500 pr-3">Epochs</td><td>~20</td></tr>
              <tr><td class="text-slate-500 pr-3">Balance</td><td>Uniform λ_t=1</td></tr>
            </table>
            <p class="text-xs text-slate-600 mt-3">目标：backbone 学到通用的商户表征，分类器只是"附带品"。</p>
          </div>
          <div class="stage-node" data-stage="2">
            <div class="badge">Stage 2</div>
            <h4 class="font-semibold">🎯 分类器重训</h4>
            <table class="text-xs mt-2">
              <tr><td class="text-slate-500 pr-3">采样</td><td><b>Class-Balanced</b> β=0.9999</td></tr>
              <tr><td class="text-slate-500 pr-3">Backbone</td><td class="tag tag-red">❄️ 冻结</td></tr>
              <tr><td class="text-slate-500 pr-3">Heads</td><td class="tag tag-green">🔥 可训</td></tr>
              <tr><td class="text-slate-500 pr-3">lr_backbone</td><td>—</td></tr>
              <tr><td class="text-slate-500 pr-3">lr_head</td><td>1e-3</td></tr>
              <tr><td class="text-slate-500 pr-3">Epochs</td><td>~5</td></tr>
              <tr><td class="text-slate-500 pr-3">DRW</td><td>Epoch 12+ 激活 LDAM weight</td></tr>
            </table>
            <p class="text-xs text-slate-600 mt-3">目标：在已冻结的好表征上，重训分类头消除长尾偏置。</p>
          </div>
          <div class="stage-node" data-stage="3">
            <div class="badge">Stage 3</div>
            <h4 class="font-semibold">🧬 GradNorm 微调</h4>
            <table class="text-xs mt-2">
              <tr><td class="text-slate-500 pr-3">采样</td><td><b>GMV-Weighted</b> λ=1</td></tr>
              <tr><td class="text-slate-500 pr-3">Backbone</td><td class="tag tag-amber">🔓 小 lr 解冻</td></tr>
              <tr><td class="text-slate-500 pr-3">Heads</td><td class="tag tag-green">🔥 可训</td></tr>
              <tr><td class="text-slate-500 pr-3">lr_backbone</td><td>1e-5 ×0.1</td></tr>
              <tr><td class="text-slate-500 pr-3">lr_head</td><td>5e-5</td></tr>
              <tr><td class="text-slate-500 pr-3">Epochs</td><td>~3</td></tr>
              <tr><td class="text-slate-500 pr-3">Balance</td><td><b>GradNorm</b> α=1.5</td></tr>
            </table>
            <p class="text-xs text-slate-600 mt-3">目标：贴近业务损失分布（GMV 加权）+ 动态调节多任务权重。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 交互：三阶段 Loss / Metric 演化</h2>
        <p class="text-sm text-slate-600">下面是模拟的三阶段训练曲线（真实数据会在 trainer.run() 的 eval 里打印）。</p>
        <div class="grid-2">
          <div id="chart-loss" style="height:320px;"></div>
          <div id="chart-metric" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>6. AMP / 分布式 / 梯度累积</h2>
        <table class="table">
          <thead><tr><th>机制</th><th>配置</th><th>收益</th></tr></thead>
          <tbody>
            <tr><td>AMP bf16</td><td><code>torch.autocast(dtype=torch.bfloat16)</code></td><td>显存 ÷ 1.8，H100/A100 上吞吐 ×1.5~2</td></tr>
            <tr><td>Gradient Accumulation</td><td>accum=4 → 等效 batch 512</td><td>小卡也能跑大 batch</td></tr>
            <tr><td>Gradient Clipping</td><td>max_norm=1.0</td><td>防 LDAM / Cosine 训练发散</td></tr>
            <tr><td>DDP / ZeRO-2</td><td>torchrun --nproc_per_node=8</td><td>多机多卡，teacher 训练必需</td></tr>
            <tr><td>CheckPoint 策略</td><td>按 risk_auc 取 best</td><td>避免 eval 震荡误选</td></tr>
          </tbody>
        </table>
      </div>
    `;
  },

  mount() {
    // Stage 点击切换
    document.querySelectorAll(".stage-node").forEach((n) => {
      n.addEventListener("click", () => {
        document.querySelectorAll(".stage-node").forEach((x) => x.classList.remove("active"));
        n.classList.add("active");
      });
    });

    // Loss / Metric 演化
    const stage1 = { start: 0, end: 20 };
    const stage2 = { start: 20, end: 25 };
    const stage3 = { start: 25, end: 28 };
    const xs = MCH.linspace(0, 28, 280);
    const trainLoss = xs.map(e => {
      if (e < stage1.end) return 2.5 - 1.6 * (1 - Math.exp(-e / 5)) + 0.05 * Math.sin(e * 2);
      if (e < stage2.end) return 1.3 - 0.35 * (1 - Math.exp(-(e - stage1.end) / 1.5)) + 0.04 * Math.sin(e * 3);
      return 0.9 - 0.2 * (1 - Math.exp(-(e - stage2.end) / 1)) + 0.02 * Math.sin(e * 3);
    });
    const evalAuc = xs.map(e => {
      if (e < stage1.end) return 0.78 + 0.10 * (1 - Math.exp(-e / 5));
      if (e < stage2.end) return 0.88 + 0.02 * (1 - Math.exp(-(e - stage1.end) / 1.5));
      return 0.90 + 0.01 * (1 - Math.exp(-(e - stage2.end) / 1));
    });
    const evalCatAcc = xs.map(e => {
      if (e < stage1.end) return 0.55 + 0.17 * (1 - Math.exp(-e / 6));
      if (e < stage2.end) return 0.72 + 0.04 * (1 - Math.exp(-(e - stage1.end) / 1.5));
      return 0.76 + 0.02 * (1 - Math.exp(-(e - stage2.end) / 1));
    });

    const markArea = {
      silent: true,
      data: [
        [{ xAxis: stage1.start, itemStyle: { color: "rgba(79, 70, 229, 0.08)" } }, { xAxis: stage1.end }],
        [{ xAxis: stage2.start, itemStyle: { color: "rgba(245, 158, 11, 0.10)" } }, { xAxis: stage2.end }],
        [{ xAxis: stage3.start, itemStyle: { color: "rgba(16, 185, 129, 0.10)" } }, { xAxis: stage3.end }],
      ],
      label: { position: "insideTop", formatter: (p) => {
        const x = p.data.coord[0];
        return x === 0 ? "Stage 1" : (x === 20 ? "Stage 2" : "Stage 3");
      } },
    };

    MCH.echart(document.getElementById("chart-loss"), {
      title: { text: "Training Loss (总 loss)", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis" },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "Epoch", min: 0, max: 28 },
      yAxis: { type: "value", name: "loss", min: 0, max: 3 },
      series: [
        { type: "line", smooth: true, showSymbol: false, data: xs.map((x, i) => [x, trainLoss[i]]), color: "#4f46e5", lineStyle: { width: 2 }, markArea },
      ],
    });
    MCH.echart(document.getElementById("chart-metric"), {
      title: { text: "Eval Metric (risk AUC & cat L2 Acc)", left: "center", textStyle: { fontSize: 13 } },
      tooltip: { trigger: "axis" },
      legend: { top: 30 },
      grid: { left: 50, right: 30, top: 70, bottom: 40 },
      xAxis: { type: "value", name: "Epoch", min: 0, max: 28 },
      yAxis: { type: "value", min: 0.5, max: 1.0 },
      series: [
        { name: "risk AUC", type: "line", smooth: true, showSymbol: false, data: xs.map((x, i) => [x, evalAuc[i]]), color: "#ef4444", lineStyle: { width: 3 }, markArea },
        { name: "cat L2 Acc", type: "line", smooth: true, showSymbol: false, data: xs.map((x, i) => [x, evalCatAcc[i]]), color: "#10b981", lineStyle: { width: 3 } },
      ],
    });
  },
});
