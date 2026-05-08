/* 模块：多目标优化 (Pareto / MGDA / PCGrad / CAGrad) */
MCH.register("loss_multitask", {
  render() {
    const code = `# 四种多任务优化方法对比
import torch

# -------- 1. 简单加权（baseline，广泛使用） --------
def weighted_sum(losses, weights):
    return sum(w * l for w, l in zip(weights, losses))

# -------- 2. Uncertainty Weighting (Kendall et al., CVPR 2018) --------
# 可学习 log σ²，每任务自己学权重
class UncertaintyMTL(nn.Module):
    def __init__(self, n_tasks):
        self.log_vars = nn.Parameter(torch.zeros(n_tasks))
    def forward(self, losses):
        precisions = torch.exp(-self.log_vars)
        return sum(p * l + 0.5 * logv for p, l, logv in zip(precisions, losses, self.log_vars))


# -------- 3. GradNorm (Chen et al., ICML 2018) --------
# 按各任务梯度范数自适应调 weight，使各任务训练进度一致
class GradNorm:
    def __init__(self, n_tasks, alpha=1.5):
        self.w = nn.Parameter(torch.ones(n_tasks))
        self.alpha = alpha; self.init_losses = None
    def step(self, losses, shared_params):
        if self.init_losses is None: self.init_losses = [l.item() for l in losses]
        # 每任务 loss ratio (当前/初始)
        loss_ratios = [l.item() / l0 for l, l0 in zip(losses, self.init_losses)]
        r_avg = sum(loss_ratios) / len(loss_ratios)
        relative = [r / r_avg for r in loss_ratios]
        # 每任务梯度范数
        G = [torch.norm(torch.autograd.grad(l * w, shared_params, retain_graph=True)[0])
             for l, w in zip(losses, self.w)]
        G_avg = sum(G) / len(G)
        # target 梯度范数：让慢的任务梯度大一些
        target = [G_avg * (ri ** self.alpha) for ri in relative]
        grad_norm_loss = sum((Gi - ti.detach()).abs() for Gi, ti in zip(G, target))
        grad_norm_loss.backward()


# -------- 4. PCGrad (Yu et al., NeurIPS 2020) --------
# 检测任务梯度冲突，冲突时把冲突分量投影掉
def pcgrad(losses, shared_params):
    grads = [torch.autograd.grad(l, shared_params, retain_graph=True)[0].flatten() for l in losses]
    new_grads = [g.clone() for g in grads]
    for i in range(len(grads)):
        for j in range(len(grads)):
            if i == j: continue
            # 若 cos(g_i, g_j) < 0 → 冲突，减掉 g_i 在 g_j 上的投影
            if torch.dot(new_grads[i], grads[j]) < 0:
                proj = torch.dot(new_grads[i], grads[j]) / torch.dot(grads[j], grads[j]) * grads[j]
                new_grads[i] = new_grads[i] - proj
    # 聚合 pc 化后的梯度
    final_grad = sum(new_grads) / len(new_grads)
    # 应用到共享参数
    ...

# -------- 5. MGDA / Pareto front (Sener & Koltun, NeurIPS 2018) --------
# 求 min_α ||Σ α_i · g_i||² s.t. Σα_i=1, α_i≥0 (二次规划)
# 收敛到 Pareto 稳定点（任何方向改进某任务都会损害另一任务）

# -------- 6. CAGrad (Liu et al., NeurIPS 2021) --------
# min_g ||g - g_avg||²  s.t.  min_i g_i · g ≥ c · ||g_avg||²
# 在 PCGrad 思想上加约束：冲突消解 + 不远离平均梯度`;

    return `
      ${MCH.hero({ icon: "⚖", name: "多目标优化 — Pareto / MGDA / PCGrad / CAGrad", en: "Multi-Task &amp; Pareto Optimization", tags: ["梯度冲突", "Pareto 最优", "PCGrad", "CAGrad", "Nash-MTL"], meta: ["◈ 任务 ≥ 5 首选", "⚡ 从加权平均到梯度博弈"] })}

      ${MCH.versionSection("loss_multitask")}

      <div class="section">
        <h2>1. 多任务的本质冲突</h2>
        <p class="text-sm text-slate-600">多个任务共享 backbone 时，各任务的梯度可能<b>方向相反</b>。最朴素的"总 loss 加权求和"会让冲突任务的梯度相互抵消，出现<b>"雨天打伞互相伤"</b>的情况：</p>
        <div class="formula-block">
          $$ g_{\\text{shared}} = \\sum_{t=1}^{T} w_t \\cdot \\nabla L_t(\\theta), \\quad \\text{当 } \\cos(g_i, g_j) &lt; 0 \\text{ 时 → 冲突} $$
        </div>
        ${MCH.info(`
          <b>多目标算法的谱系</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>启发式加权</b>（Weighted Sum / Uncertainty / GradNorm）→ 通过调 weight 缓解</li>
            <li><b>几何修正</b>（PCGrad / GradDrop）→ 检测冲突并投影掉</li>
            <li><b>优化理论</b>（MGDA / CAGrad / Nash-MTL）→ 求解 Pareto 最优方向</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 方法对比表</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>论文</th><th>核心思想</th><th>额外开销</th><th>性能</th></tr></thead>
          <tbody>
            <tr><td>Weighted Sum</td><td>-</td><td>手动设 w_t</td><td>无</td><td>基线</td></tr>
            <tr><td>Uncertainty</td><td>Kendall CVPR'18</td><td>学 σ_t², 自适应 w</td><td>+ T 个参数</td><td>中等</td></tr>
            <tr><td>GradNorm</td><td>Chen ICML'18</td><td>按 grad norm 调 w</td><td>×1.5 训练时间</td><td>中高</td></tr>
            <tr><td><b>PCGrad</b></td><td>Yu NeurIPS'20</td><td>冲突梯度投影消除</td><td>×2 训练时间</td><td>高</td></tr>
            <tr><td><b>MGDA</b></td><td>Sener NeurIPS'18</td><td>二次规划求 Pareto 方向</td><td>×2-3 训练时间</td><td>Pareto 理论最优</td></tr>
            <tr><td><b>CAGrad</b></td><td>Liu NeurIPS'21</td><td>约束最小冲突方向</td><td>×2 训练时间</td><td>SOTA</td></tr>
            <tr><td>Nash-MTL</td><td>Navon ICML'22</td><td>博弈论 Nash 均衡</td><td>×2 训练时间</td><td>SOTA</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码（6 种方法）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：两任务梯度可视化</h2>
        <p class="text-sm text-slate-600">两个任务的梯度 g₁, g₂（2D 平面表示）。观察不同方法给出的最终更新方向：</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "mt-ang1", label: "任务 1 梯度角度 (°)", min: 0, max: 360, step: 5, value: 30 })}
              ${MCH.slider({ id: "mt-ang2", label: "任务 2 梯度角度 (°)", min: 0, max: 360, step: 5, value: 160 })}
              ${MCH.slider({ id: "mt-mag1", label: "任务 1 梯度模", min: 0.1, max: 2, step: 0.1, value: 1 })}
              ${MCH.slider({ id: "mt-mag2", label: "任务 2 梯度模", min: 0.1, max: 2, step: 0.1, value: 1.5 })}
            </div>
            <div id="mt-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-mt" style="height:400px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. Pareto Front 可视化</h2>
        <p class="text-sm text-slate-600">两个任务 loss 构成的 2D 空间，<b>Pareto Front</b> 是不可改进的前沿：任何向左或向下移动都会损害另一任务。</p>
        <div id="chart-pareto" style="height:360px;"></div>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("loss_multitask").pros, MCH.getById("loss_multitask").cons, MCH.getById("loss_multitask").best_for)}
      </div>
    `;
  },

  mount() {
    const chart = MCH.echart(document.getElementById("chart-mt"), {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 50, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", min: -2, max: 2, splitLine: { lineStyle: { color: "#f1f5f9" } } },
      yAxis: { type: "value", min: -2, max: 2, splitLine: { lineStyle: { color: "#f1f5f9" } } },
      series: [],
    });
    const update = () => {
      const a1 = parseFloat(document.getElementById("mt-ang1").value) * Math.PI / 180;
      const a2 = parseFloat(document.getElementById("mt-ang2").value) * Math.PI / 180;
      const m1 = parseFloat(document.getElementById("mt-mag1").value);
      const m2 = parseFloat(document.getElementById("mt-mag2").value);
      const g1 = [m1 * Math.cos(a1), m1 * Math.sin(a1)];
      const g2 = [m2 * Math.cos(a2), m2 * Math.sin(a2)];
      // 加权平均
      const avg = [(g1[0] + g2[0]) / 2, (g1[1] + g2[1]) / 2];
      // PCGrad: 若 dot<0，投影掉
      const dot = g1[0] * g2[0] + g1[1] * g2[1];
      let pc1 = [...g1], pc2 = [...g2];
      if (dot < 0) {
        const k12 = dot / (g2[0] ** 2 + g2[1] ** 2);
        pc1 = [g1[0] - k12 * g2[0], g1[1] - k12 * g2[1]];
        const k21 = dot / (g1[0] ** 2 + g1[1] ** 2);
        pc2 = [g2[0] - k21 * g1[0], g2[1] - k21 * g1[1]];
      }
      const pcg = [(pc1[0] + pc2[0]) / 2, (pc1[1] + pc2[1]) / 2];
      // MGDA: 沿两向量夹角二分，简化为归一化后平均
      const n1 = [g1[0] / m1, g1[1] / m1], n2 = [g2[0] / m2, g2[1] / m2];
      const mgda = [(n1[0] + n2[0]) / 2, (n1[1] + n2[1]) / 2];
      const mgdaLen = Math.hypot(...mgda);

      const arrow = (name, from, to, color, dash) => ({
        name, type: "lines", coordinateSystem: "cartesian2d",
        data: [{ coords: [from, to] }],
        lineStyle: { color, width: 3, opacity: 0.9, type: dash ? "dashed" : "solid" },
        effect: { show: true, symbol: "arrow", symbolSize: 10, color },
      });
      chart.setOption({
        series: [
          arrow(`任务 1 g₁`, [0, 0], g1, "#ef4444"),
          arrow(`任务 2 g₂`, [0, 0], g2, "#f59e0b"),
          arrow(`加权平均 (baseline)`, [0, 0], avg, "#94a3b8", true),
          arrow(`PCGrad`, [0, 0], pcg, "#4f46e5"),
          arrow(`MGDA`, [0, 0], [mgda[0] * 1.2, mgda[1] * 1.2], "#10b981"),
        ],
      });

      const cos = (g1[0] * g2[0] + g1[1] * g2[1]) / (m1 * m2);
      const conflict = cos < 0;
      document.getElementById("mt-info").innerHTML = `
        <b>g₁ · g₂ cosine</b>：<span style="color:${conflict ? '#ef4444' : '#10b981'};font-weight:700;">${cos.toFixed(3)}</span> ${conflict ? "⚠️ 冲突" : "✓ 协同"}<br/>
        <b>加权平均</b>：|g|=${Math.hypot(...avg).toFixed(3)} ${conflict ? "（冲突抵消削弱）" : ""}<br/>
        <b>PCGrad</b>：|g|=${Math.hypot(...pcg).toFixed(3)} ${conflict ? "（投影掉冲突分量）" : "（与 avg 相同）"}<br/>
        <b>MGDA</b>：找到平均方向上的 Pareto 稳定点<br/>
        <span style="color:#64748b;">调节两任务方向使其相反（180°），观察 PCGrad 和平均的差异。</span>
      `;
    };
    ["mt-ang1", "mt-ang2", "mt-mag1", "mt-mag2"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();

    // Pareto front
    const t = MCH.linspace(0, Math.PI / 2, 50);
    const front = t.map(a => [Math.cos(a) * 0.7 + 0.2, Math.sin(a) * 0.7 + 0.2]);
    const dominated = [];
    for (let i = 0; i < 80; i++) {
      const g = Math.random() * 0.4 + 0.5;
      const h = Math.random() * 0.4 + 0.5;
      if (g > 0.25 && h > 0.25) dominated.push([g, h]);
    }
    MCH.echart(document.getElementById("chart-pareto"), {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "Task 1 Loss", min: 0, max: 1.2 },
      yAxis: { type: "value", name: "Task 2 Loss", min: 0, max: 1.2 },
      series: [
        { name: "Pareto Front (最优前沿)", type: "line", smooth: true, showSymbol: false, color: "#4f46e5", lineStyle: { width: 4 }, data: front },
        { name: "被支配解", type: "scatter", data: dominated, itemStyle: { color: "#cbd5e1" }, symbolSize: 8 },
        { name: "加权平均收敛点", type: "scatter", data: [[0.45, 0.58]], itemStyle: { color: "#f59e0b" }, symbolSize: 18, label: { show: true, position: "right", formatter: "Weighted Sum", fontSize: 11 } },
        { name: "MGDA 收敛点", type: "scatter", data: [[0.35, 0.48]], itemStyle: { color: "#10b981" }, symbolSize: 18, label: { show: true, position: "right", formatter: "MGDA (Pareto)", fontSize: 11 } },
      ],
    });
  },
});
