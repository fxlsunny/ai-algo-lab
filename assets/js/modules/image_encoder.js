/* 模块：图像 Encoder */
MCH.register("image_encoder", {
  render() {
    const code = `# 图像 Encoder：基于 T-1 预计算 CLIP-ViT Embedding 的 Attention Pooling
# 来源：src/models/image_encoder.py

class ImageEncoder(nn.Module):
    """生产侧：门头照 × 1 + 场景照 × 3 = 4 张；T-1 离线跑 CLIP-ViT-B/32，
       在线模型仅 Attention Pool + Proj，完全不加载 CLIP 主干（省 86M 显存）。"""
    def __init__(self, in_dim=512, hidden_dim=512, num_images=4):
        # Additive Attention：scalar score per image
        self.attn = nn.Sequential(nn.Linear(in_dim, 64), nn.Tanh(), nn.Linear(64, 1))
        self.proj = nn.Sequential(nn.Linear(in_dim, hidden_dim),
                                  nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.empty_token = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)

    def forward(self, image_embs, image_mask):
        # image_embs: (B, N, 512), image_mask: (B, N) 1=有效

        # 1) 逐图打分 → masked softmax
        logits = self.attn(image_embs).squeeze(-1)                  # (B, N)
        logits = logits.masked_fill(image_mask == 0, -1e9)          # 缺失图得 0 注意力
        weights = torch.softmax(logits, dim=-1).unsqueeze(-1)       # (B, N, 1)

        # 2) 加权求和 + 投影
        pooled = (image_embs * weights).sum(1)                       # (B, 512)
        h = self.proj(pooled)                                        # (B, hidden_dim)

        # 3) 整商户无图片兜底
        any_valid = image_mask.any(1, keepdim=True).float()
        return h * any_valid + self.empty_token * (1 - any_valid)`;

    return `
      ${MCH.hero({
        icon: "I",
        name: "图像 Encoder — Attention Pooling on CLIP-ViT Embedding",
        en: "Image Attention Pool · pre-computed CLIP embedding",
        tags: ["CLIP-ViT-B/32", "T-1 离线预计算", "Additive Attention", "N=4 图", "BLIP2 图生文闭环"],
        meta: ["◈ 输入 512-d × 4", "⚡ 在线 &lt; 0.5 ms", "◇ 输出 512-d"],
      })}

      <div class="section">
        <h2>1. 算法原理</h2>
        <p class="text-sm text-slate-600 leading-relaxed">
          商户"门头照 / 场景照"是判别<b>餐饮 / 零售 / 养发 / 美甲 / 赌博机房</b>等类目的强信号。但实时在线加载 CLIP-ViT（86M）成本过高，
          这里采用 <b>双路策略</b>：
        </p>
        <ol class="text-sm text-slate-700 list-decimal pl-6 mt-2 space-y-1">
          <li>离线 T-1 跑 <b>CLIP-ViT-B/32</b> → 每商户 4 张图 × 512-d embedding，落 Parquet / HBase；</li>
          <li>离线同时跑 <b>BLIP2</b> 图生文，生成 caption 送入 <b>TextEncoder</b>（跨模态闭环）；</li>
          <li>在线仅做 <b>Additive Attention Pool + Proj</b>，并处理整商户无图的兜底。</li>
        </ol>

        <h3>· Additive Attention 公式</h3>
        <div class="formula-block">
          $$ \\alpha_i = \\text{softmax}_i\\big( v^\\top \\tanh(W e_i + b) \\big) \\;\\; \\cdot \\;\\; \\text{mask}_i, \\quad\\; h = \\text{Proj}\\Big( \\sum_i \\alpha_i \\, e_i \\Big) $$
        </div>

        ${MCH.info(`
          <b>为什么不用 self-attention / mean pool？</b>
          <ul style="margin-top:6px;padding-left:20px;">
            <li>N=4 张图太短，self-attention 过杀；</li>
            <li>Mean pool 对"1 张清晰门头 + 3 张模糊场景"会稀释信息；</li>
            <li>Additive attention 让模型自己学"哪张最代表商户类目"。</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互可视化 — Attention 权重模拟</h2>
        <p class="text-sm text-slate-600">
          调节每张图的"图像质量分"（模拟 attention 打分），观察 softmax 归一后的 attention 权重如何分配。
          切换缺失图片观察 <b>masked_fill + softmax</b> 兜底行为。
        </p>

        <div class="grid-4 mt-4" id="image-cards">
          ${[0, 1, 2, 3].map(i => `
            <div class="card" id="img-card-${i}">
              <div style="height:110px;border-radius:8px;background:linear-gradient(135deg, ${["#cbd5e1","#a5b4fc","#fca5a5","#6ee7b7"][i]} 0%, ${["#64748b","#6366f1","#ef4444","#10b981"][i]} 100%);display:flex;align-items:center;justify-content:center;color:white;font-size:24px;">
                ${["🏢 门头", "🪑 场景 1", "🍽️ 场景 2", "🧾 收银"][i]}
              </div>
              <div style="margin-top:10px;">
                <div class="ctrl-label">
                  <span>质量分 (raw score)</span>
                  <span class="ctrl-val" id="img-score-${i}-val">${[1.8, 0.9, 1.2, 0.3][i]}</span>
                </div>
                <input type="range" id="img-score-${i}" min="-2" max="3" step="0.1" value="${[1.8, 0.9, 1.2, 0.3][i]}" />
                <label class="text-xs text-slate-600 flex items-center gap-2 mt-2">
                  <input type="checkbox" id="img-mask-${i}" ${i === 3 ? "" : "checked"} /> 图片存在（mask）
                </label>
                <div style="margin-top:8px;">
                  <div class="text-xs text-slate-500">softmax α_i</div>
                  <div style="height:20px;background:#f1f5f9;border-radius:4px;overflow:hidden;margin-top:4px;">
                    <div id="img-bar-${i}" style="height:100%;background:linear-gradient(90deg,#4f46e5,#7c3aed);width:0%;"></div>
                  </div>
                  <div class="text-xs text-indigo-600 mt-1"><span id="img-alpha-${i}">0%</span></div>
                </div>
              </div>
            </div>
          `).join("")}
        </div>

        <div class="section" style="background:#f8fafc;margin-top:16px;">
          <h3>▎ 池化向量贡献分解</h3>
          <div id="chart-pool" style="height:260px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. 业务洞察</h2>
        <div class="grid-3">
          <div class="card">
            <div class="tag tag-green">类目识别</div>
            <h4 class="font-semibold mt-2">餐饮识别率提升</h4>
            <p class="text-xs text-slate-600 mt-2">加入图像模态后，类目 L2 "餐饮-茶饮"类 Recall 从 71% → <b>84%</b>（纯文本易混淆"奶茶店"与"便利店"）。</p>
          </div>
          <div class="card">
            <div class="tag tag-amber">反欺诈</div>
            <h4 class="font-semibold mt-2">空壳门头照</h4>
            <p class="text-xs text-slate-600 mt-2">CLIP embedding 对"纯白墙 / 虚假招牌"有明显聚类，配合 SupCon 表征对齐可捕获这类造假。</p>
          </div>
          <div class="card">
            <div class="tag">图文闭环</div>
            <h4 class="font-semibold mt-2">BLIP2 Caption</h4>
            <p class="text-xs text-slate-600 mt-2">图像生文 "A shop front with green sign ..." 送入 TextEncoder 第 7 字段，形成<b>图 → 文 → 统一表征</b>闭环。</p>
          </div>
        </div>
      </div>
    `;
  },

  mount() {
    const chartEl = document.getElementById("chart-pool");
    const chart = MCH.echart(chartEl, {
      tooltip: { trigger: "item", formatter: "{b}<br/>贡献 ≈ α_i = {d}%" },
      legend: { top: 0 },
      series: [{
        type: "pie", radius: ["35%", "65%"], center: ["50%", "55%"],
        data: [],
        label: { formatter: "{b}\n{d}%" },
        itemStyle: { borderColor: "#fff", borderWidth: 2 },
      }],
    });

    const update = () => {
      const scores = [0, 1, 2, 3].map(i => {
        const mask = document.getElementById(`img-mask-${i}`).checked;
        const s = parseFloat(document.getElementById(`img-score-${i}`).value);
        document.getElementById(`img-score-${i}-val`).textContent = s.toFixed(1);
        return mask ? s : -1e9;
      });
      // softmax
      const max = Math.max(...scores);
      const exps = scores.map(s => Math.exp(s - max));
      const sum = exps.reduce((a, b) => a + b, 0) || 1;
      const alphas = exps.map(e => e / sum);
      const names = ["门头", "场景 1", "场景 2", "收银"];
      alphas.forEach((a, i) => {
        document.getElementById(`img-bar-${i}`).style.width = (a * 100).toFixed(1) + "%";
        document.getElementById(`img-alpha-${i}`).textContent = (a * 100).toFixed(1) + "%";
        const card = document.getElementById(`img-card-${i}`);
        if (card) card.style.opacity = document.getElementById(`img-mask-${i}`).checked ? 1 : 0.4;
      });
      chart.setOption({
        series: [{
          data: alphas.map((a, i) => ({
            name: names[i],
            value: (a * 100).toFixed(2),
            itemStyle: { color: ["#4f46e5", "#7c3aed", "#f59e0b", "#10b981"][i] },
          })),
        }],
      });
    };

    [0, 1, 2, 3].forEach(i => {
      document.getElementById(`img-score-${i}`).addEventListener("input", update);
      document.getElementById(`img-mask-${i}`).addEventListener("change", update);
    });
    update();
  },
});
