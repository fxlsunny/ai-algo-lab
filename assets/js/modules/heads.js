/* 模块：多任务 Heads */
MCH.register("heads", {
  render() {
    const code = `# 多任务 Heads：风险 / 可信 / 类目 / Plugin
# 来源：src/models/heads.py

class CosineClassifier(nn.Module):
    """NormFace 风格：s * cos(θ)，长尾 + LDAM/Seesaw 效果好
        w̃ = W / ||W||;  x̃ = x / ||x||;  logits = s · (x̃·w̃ᵀ)"""
    def __init__(self, in_dim, num_classes, s=30.0):
        self.weight = nn.Parameter(torch.randn(num_classes, in_dim))
        self.s = s

    def forward(self, x):
        w = F.normalize(self.weight, dim=-1)
        x = F.normalize(x, dim=-1)
        return (x @ w.t()) * self.s        # (B, num_classes)


class RiskHead(nn.Module):
    """风险识别：3 分类 (黑/灰/白) + Ordinal 5 级 (S/A/B/C/D)"""
    def __init__(self, in_dim=256, num_classes=3, ordinal_levels=5):
        self.cls_head = _mlp(in_dim, num_classes)           # 标准 softmax
        self.ord_head = _mlp(in_dim, ordinal_levels - 1)    # K-1 累积二分类
    def forward(self, x):
        return {"risk_cls_logits": self.cls_head(x),
                "risk_ord_logits": self.ord_head(x)}


class CategoryHead(nn.Module):
    """Hierarchical：L1 20 类 softmax + L2 2000 类 余弦分类器
       L2 输入拼接 L1 概率，让二级类知道上一级 context。"""
    def __init__(self, in_dim=256, level1_num=20, level2_num=2000, cosine_s=30.0):
        self.l1 = _mlp(in_dim, level1_num)
        self.l2_feat = nn.Sequential(
            nn.Linear(in_dim + level1_num, 512), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(512, 256),
        )
        self.l2_cosine = CosineClassifier(256, level2_num, s=cosine_s)

    def forward(self, x):
        l1_logits = self.l1(x)
        l1_prob = torch.softmax(l1_logits, dim=-1)
        l2_feat = self.l2_feat(torch.cat([x, l1_prob], dim=-1))
        return {
            "cat_l1_logits": l1_logits,
            "cat_l2_logits": self.l2_cosine(l2_feat),
            "cat_l2_feat"  : l2_feat,      # 供 SupCon 对比学习用
        }


class TrustHead(nn.Module):
    """可信：连续 0-1 sigmoid 分 + 5 级离散等级"""
    def __init__(self, in_dim=256, num_levels=5):
        self.score_head = _mlp(in_dim, 1)
        self.level_head = _mlp(in_dim, num_levels)


class PluginHead(nn.Module):
    """通用热插拔 head：二/多分类 / 回归"""
    def __init__(self, in_dim=256, num_classes=2, mode="classification"):
        self.net = _mlp(in_dim, 1 if mode == "regression" else num_classes)`;

    return `
      ${MCH.hero({
        icon: "H",
        name: "多任务 Heads — Cosine / Ordinal / Hierarchical / Plugin",
        en: "Task Heads · Risk / Trust / Category / Plugin",
        tags: ["CosineClassifier(NormFace)", "OrdinalCE K-1 累积", "Hierarchical L1→L2", "热插拔 Plugin"],
        meta: ["◈ 4+N 任务头", "⚡ 余弦分类器长尾友好", "◇ supcon feature 对齐"],
      })}

      <div class="section">
        <h2>1. 四种 Head 设计</h2>
        <div class="grid-2">
          <div class="card">
            <div class="flex items-center gap-2"><span class="tag tag-red">⚠️ 风险</span></div>
            <h4 class="font-semibold mt-1">RiskHead — 双输出</h4>
            <p class="text-xs text-slate-600 mt-2"><code>3-cls</code> 黑/灰/白 + <code>5-级 Ordinal</code> S/A/B/C/D，分别配 Focal 和 OrdinalCE。</p>
          </div>
          <div class="card">
            <div class="flex items-center gap-2"><span class="tag tag-amber">🎯 可信</span></div>
            <h4 class="font-semibold mt-1">TrustHead — 回归 + 等级</h4>
            <p class="text-xs text-slate-600 mt-2">Sigmoid 连续分 ∈ [0,1] + 5 级 LDAM 离散等级，下游可融合/选一。</p>
          </div>
          <div class="card">
            <div class="flex items-center gap-2"><span class="tag">📂 类目</span></div>
            <h4 class="font-semibold mt-1">CategoryHead — Hierarchical</h4>
            <p class="text-xs text-slate-600 mt-2">L1 20 类 softmax + <b>L2 2000 类余弦分类器</b>（NormFace）。L2 输入拼接 L1 概率做"上下文调节"。</p>
          </div>
          <div class="card">
            <div class="flex items-center gap-2"><span class="tag tag-green">🔌 Plugin</span></div>
            <h4 class="font-semibold mt-1">PluginHead — 热插拔</h4>
            <p class="text-xs text-slate-600 mt-2">通用二/多分类 / 回归，<code>register_plugin("gambling")</code> 即时生效。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：CosineClassifier（NormFace）</h2>
        <p class="text-sm text-slate-600">
          标准 softmax 分类器用 <code>W x</code> 做 logits，对长尾类容易出现 "||W_c|| 小、||W_head|| 大" 导致头部支配。
          NormFace 先把 W 和 x 都 L2 归一化，只保留"方向信息"，再乘 scale <b>s</b>：
        </p>
        <div class="formula-block">
          $$ \\text{logit}_c = s \\cdot \\frac{W_c \\cdot x}{\\|W_c\\| \\|x\\|} = s \\cos(\\theta_c) $$
        </div>

        <div class="grid-2 mt-4">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "cos-s", label: "Scale s", min: 5, max: 60, step: 1, value: 30 })}
              ${MCH.slider({ id: "cos-theta", label: "正确类别角度 θ (度)", min: 0, max: 90, step: 1, value: 20 })}
              ${MCH.slider({ id: "cos-gap", label: "次优类别额外角度 Δθ (度)", min: 1, max: 60, step: 1, value: 15 })}
            </div>
          </div>
          <div>
            <div id="chart-cos" style="height:280px;"></div>
            <div id="cos-info" class="text-xs text-slate-600 mt-2"></div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 交互：Ordinal CE（K-1 累积二分类）</h2>
        <p class="text-sm text-slate-600">
          把 K 级标签 y ∈ {0,1,...,K-1} 展开成 K-1 个二分类：<code>y > 0 ?, y > 1 ?, ..., y > K-2 ?</code>。
          这样"预测 S 为 D"会被重复惩罚 K-1 次，比普通 CE 惩罚"跨级错"更重。
        </p>
        <div class="formula-block">
          $$ L_{\\text{ord}} = -\\frac{1}{K-1} \\sum_{k=0}^{K-2} \\Big[ \\mathbb{1}[y>k] \\log \\sigma(z_k) + \\mathbb{1}[y \\leq k] \\log(1-\\sigma(z_k)) \\Big] $$
        </div>

        <div class="grid-2 mt-4">
          <div>
            <h3>· 等级标签展开</h3>
            <div id="ord-matrix" class="card"></div>
          </div>
          <div>
            <h3>· 错判代价对比</h3>
            <div class="ctrl-panel" style="margin-bottom:10px;">
              <label class="text-xs text-slate-600">真实等级: <select id="ord-true" class="text-xs border rounded p-1"><option>S(0)</option><option>A(1)</option><option>B(2)</option><option>C(3)</option><option>D(4)</option></select></label>
            </div>
            <div id="chart-ord" style="height:280px;"></div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>5. Hierarchical Category（L1 → L2）</h2>
        <p class="text-sm text-slate-600">
          类目体系 20 大类 → 2000 叶子类。L2 特征由 <code>[x, l1_prob]</code> 拼接得到，
          相当于告诉 L2 分类器"上一级 softmax 觉得你偏向哪一大类"，从而<b>显著降低跨大类误分</b>。
        </p>

        <div class="mermaid">
flowchart LR
    x[unified emb 256-d] --> L1["L1 MLP → 20 类 softmax"]
    L1 --> P1[L1 Probability]
    x --> CAT[concat]
    P1 --> CAT
    CAT --> L2F["L2 Feat MLP → 256-d"]
    L2F --> COS["CosineClassifier<br/>(NormFace s=30)"]
    COS --> Out["2000 类 logits + L2 特征"]
    L2F -.-> SupCon["SupCon 对比学习"]
        </div>
      </div>
    `;
  },

  mount() {
    // CosineClassifier 可视化
    const cosChart = MCH.echart(document.getElementById("chart-cos"), {
      tooltip: {},
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "category", name: "类别", data: ["正确", "次优", "其他 1", "其他 2"] },
      yAxis: { type: "value", name: "logit = s·cos(θ)" },
      series: [{ type: "bar", barWidth: 40, data: [] }],
    });
    const updateCos = () => {
      const s = parseFloat(document.getElementById("cos-s").value);
      const th = parseFloat(document.getElementById("cos-theta").value) * Math.PI / 180;
      const gap = parseFloat(document.getElementById("cos-gap").value) * Math.PI / 180;
      const logits = [
        s * Math.cos(th),
        s * Math.cos(th + gap),
        s * Math.cos(th + gap + 0.5),
        s * Math.cos(th + gap + 1.0),
      ];
      cosChart.setOption({
        series: [{
          data: logits.map((v, i) => ({ value: v.toFixed(2), itemStyle: { color: i === 0 ? "#10b981" : ["#f59e0b", "#94a3b8", "#94a3b8"][i - 1] || "#94a3b8" } })),
          label: { show: true, position: "top", formatter: (p) => p.value },
        }],
      });
      const exps = logits.map(l => Math.exp(l));
      const Z = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(e => e / Z);
      document.getElementById("cos-info").innerHTML = `
        <b>softmax 概率</b>：正确类 = <span style="color:#10b981;font-weight:700;">${(probs[0] * 100).toFixed(1)}%</span> · 次优 = <span style="color:#f59e0b;">${(probs[1] * 100).toFixed(1)}%</span><br/>
        <b>s ↑</b> → 类别间 logit 差距放大 → softmax 更"尖锐"，但训练不稳定风险增加，通常 s=30 适中
      `;
    };
    ["cos-s", "cos-theta", "cos-gap"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateCos();
      });
    });
    updateCos();

    // Ordinal 展开矩阵
    const om = document.getElementById("ord-matrix");
    const K = 5;
    let mh = `<div class="text-xs text-slate-500 mb-2">等级标签展开为 K-1 = 4 个二分类（y > k ?）</div>`;
    mh += `<table class="table"><thead><tr><th>y (等级)</th><th>y > 0</th><th>y > 1</th><th>y > 2</th><th>y > 3</th></tr></thead><tbody>`;
    ["S(0)", "A(1)", "B(2)", "C(3)", "D(4)"].forEach((l, i) => {
      mh += `<tr><td><b>${l}</b></td>${[0, 1, 2, 3].map(k => `<td><span class="tag ${i > k ? 'tag-green' : 'tag-slate'}">${i > k ? 1 : 0}</span></td>`).join("")}</tr>`;
    });
    mh += `</tbody></table>`;
    om.innerHTML = mh;

    // Ordinal 错判代价
    const ordChart = MCH.echart(document.getElementById("chart-ord"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "category", name: "预测等级", data: ["S", "A", "B", "C", "D"] },
      yAxis: { type: "value", name: "损失 (归一化)" },
      series: [
        { name: "普通 CE (等代价)", type: "bar", data: [], barGap: 0.1, color: "#94a3b8" },
        { name: "Ordinal CE (跨级更重)", type: "bar", data: [], color: "#4f46e5" },
      ],
    });
    const updateOrd = () => {
      const trueIdx = parseInt(document.getElementById("ord-true").value.replace(/\D/g, ""));
      const ce = [0, 0, 0, 0, 0].map((_, i) => (i === trueIdx ? 0 : 1));
      const ord = [0, 1, 2, 3, 4].map(i => Math.abs(i - trueIdx) / 4);
      ordChart.setOption({
        series: [{ data: ce }, { data: ord }],
      });
    };
    const sel = document.getElementById("ord-true");
    sel.selectedIndex = 0;
    sel.addEventListener("change", updateOrd);
    updateOrd();
  },
});
