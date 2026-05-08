/* 模块：文本 Encoder — MacBERT + LoRA */
MCH.register("text_encoder", {
  render() {
    const code = `# 核心结构：MacBERT 冻结 + LoRA 低秩注入 + [CLS]/Mean 双池化
# 来源：src/models/text_encoder.py

TEXT_FIELDS = (
    "merchant_name",   # 商户名 fmerchantname
    "company_name",    # 公司名 fcompanyname
    "subject_type",    # 主体类型 fsubjecttype
    "category_name",   # 类目 new_categoryname
    "business_scope",  # 经营范围 fbusinessscope
    "archive_scope",   # 监管合规经营范围 farchivebusinessscope
    "image_caption",   # BLIP2 图生文 —— 跨模态闭环！
)

class TextEncoder(nn.Module):
    def __init__(self, backbone="hfl/chinese-macbert-base", hidden_dim=512,
                 freeze_backbone=True, use_lora=True, lora_r=8, lora_alpha=16):
        # 1) 加载 HuggingFace backbone
        self.tokenizer = AutoTokenizer.from_pretrained(backbone)
        self.backbone = AutoModel.from_pretrained(backbone)

        # 2) LoRA 注入：只在 Q/V 投影矩阵上加低秩 adapter
        #    冻结主干 110M 参数，LoRA 仅新增 ~0.3M 可训参数
        if freeze_backbone and use_lora:
            for p in self.backbone.parameters():
                p.requires_grad = False
            lora_cfg = LoraConfig(
                r=lora_r,                        # 低秩维度
                lora_alpha=lora_alpha,           # 缩放：α / r
                target_modules=["query", "value"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.backbone = get_peft_model(self.backbone, lora_cfg)

        # 3) 池化头：CLS + Mean 双路拼接
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

    @staticmethod
    def build_text(fields):
        """按固定顺序拼接，缺失 → '未知'。保证 token 位置信息稳定。"""
        return " [SEP] ".join(fields.get(f, "") or "未知" for f in TEXT_FIELDS)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        cls  = hidden[:, 0, :]                                    # (B, 768) [CLS]
        mean = (hidden * attention_mask.unsqueeze(-1)).sum(1)     # (B, 768) attention-weighted mean
        mean = mean / attention_mask.sum(1, keepdim=True).clamp(min=1)
        return self.proj(torch.cat([cls, mean], dim=-1))          # (B, hidden_dim)`;

    return `
      ${MCH.hero({
        icon: "T",
        name: "文本 Encoder — MacBERT + LoRA",
        en: "Text Encoder · MacBERT (frozen) + Low-Rank Adaptation",
        tags: ["MacBERT-base", "LoRA r=8", "CLS+Mean 池化", "7 字段拼接", "21128 中文词表"],
        meta: ["◈ backbone ≈ 102M（冻结）", "⚡ LoRA ≈ 0.3M（可训）", "◇ 输出 512-d"],
      })}

      <div class="section">
        <h2>1. 算法原理</h2>
        <p class="text-sm text-slate-600 leading-relaxed">
          文本侧承载商户的"静态身份画像"——商户名、公司名、主体类型、类目、经营范围、合规备案、门头图生文
          共 <b>7 个字段</b>，按固定顺序用 <code>[SEP]</code> 拼接后送入 <b>MacBERT</b>。模型策略采用
          <b>backbone 冻结 + LoRA 微调</b>：
        </p>

        <div class="formula-block">
          $$ h = \\text{MacBERT}(x) \\in \\mathbb{R}^{L \\times 768}, \\quad W_{\\text{lora}} = W_0 + \\frac{\\alpha}{r} B A, \\quad B \\in \\mathbb{R}^{d \\times r},\\; A \\in \\mathbb{R}^{r \\times d} $$
        </div>

        ${MCH.info(`
          <b>为什么冻结 + LoRA？</b>
          <ul style="margin-top:6px;padding-left:20px;">
            <li>102M MacBERT 全量微调 → 显存爆炸 / 过拟合小样本风险；</li>
            <li>LoRA 只新增 <b>2·r·d ≈ 0.3M</b> 可训参数（r=8, d=768），训练速度 <b>×5</b>、显存 <b>÷3</b>；</li>
            <li>不同业务场景可分别训练独立 LoRA adapter，<b>热切换</b> 零改 backbone。</li>
          </ul>
        `, "tip")}

        <h3>池化策略：[CLS] + Mean Pool 拼接</h3>
        <p class="text-sm text-slate-600">
          纯 <code>[CLS]</code> 池化易被开头 token 主导；纯 Mean 忽略句法结构。两者拼接后经 Linear 投到 <code>hidden_dim=512</code>，
          对"经营范围"等长文本字段能更稳定提取主题表征。
        </p>
        <div class="formula-block">
          $$ v = \\text{Proj}\\Big[\\; h_{[CLS]} \\;\\|\\; \\frac{\\sum_i m_i h_i}{\\sum_i m_i} \\;\\Big] $$
        </div>
      </div>

      <div class="section">
        <h2>2. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互可视化</h2>

        <div class="grid-2">
          <div>
            <h3>· LoRA 参数量 vs 秩 r</h3>
            <div class="ctrl-panel" style="margin-bottom:12px;">
              ${MCH.slider({ id: "lora-r", label: "秩 r", min: 1, max: 64, step: 1, value: 8 })}
              ${MCH.slider({ id: "lora-d", label: "隐层维度 d", min: 256, max: 1024, step: 128, value: 768 })}
              ${MCH.slider({ id: "lora-L", label: "LoRA 层数 L (Q,V × 层数)", min: 2, max: 48, step: 2, value: 24 })}
            </div>
            <div id="chart-lora" style="height:300px;"></div>
          </div>

          <div>
            <h3>· 7 字段拼接模拟器</h3>
            <div class="ctrl-panel" style="margin-bottom:12px;">
              <div style="display:grid;grid-template-columns:110px 1fr;gap:6px;font-size:12px;">
                <div style="color:#64748b;">商户名</div><input id="tf-0" class="text-xs p-1 border rounded" value="星巴克咖啡"/>
                <div style="color:#64748b;">公司名</div><input id="tf-1" class="text-xs p-1 border rounded" value="星巴克（中国）有限公司"/>
                <div style="color:#64748b;">主体类型</div><input id="tf-2" class="text-xs p-1 border rounded" value="有限责任公司"/>
                <div style="color:#64748b;">类目</div><input id="tf-3" class="text-xs p-1 border rounded" value="餐饮/咖啡厅"/>
                <div style="color:#64748b;">经营范围</div><input id="tf-4" class="text-xs p-1 border rounded" value="咖啡及饮料制作 餐饮服务"/>
                <div style="color:#64748b;">备案范围</div><input id="tf-5" class="text-xs p-1 border rounded" value=""/>
                <div style="color:#64748b;">图生文 caption</div><input id="tf-6" class="text-xs p-1 border rounded" value="店铺门头绿色 Logo 室外阳光"/>
              </div>
            </div>
            <h3>· 拼接后输入 BERT 的 token 流</h3>
            <pre class="text-[11px] bg-slate-900 text-slate-100 p-3 rounded-lg overflow-x-auto"><code id="tok-preview"></code></pre>
            <div id="tok-stats" class="text-xs text-slate-600 mt-2"></div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 生产部署要点</h2>
        <table class="table">
          <thead><tr><th>项</th><th>方案</th><th>动机</th></tr></thead>
          <tbody>
            <tr><td>backbone</td><td><code>hfl/chinese-macbert-base</code></td><td>中文 MLM + 基于原字替换的 Mask 方式，对商户名这种"专有短语"鲁棒</td></tr>
            <tr><td>max_len</td><td>256</td><td>7 字段拼接均值 ~ 180 字，256 可覆盖 P99</td></tr>
            <tr><td>缺失兜底</td><td>空字段 → "未知"</td><td>保证位置 id 稳定，便于 KV Cache / 特征仓查询</td></tr>
            <tr><td>T-1 缓存</td><td>Text Emb 落 HBase</td><td>线上走 <code>batch["text_emb"]</code> 分支，跳过 BERT 前向 → 省 80% 推理时间</td></tr>
            <tr><td>蒸馏</td><td>Student 用 MiniLM-L6</td><td>Student backbone ↓ 10×，配 Logit KD 精度回收 &gt;95%</td></tr>
          </tbody>
        </table>
      </div>
    `;
  },

  mount() {
    // LoRA 图表
    const el = document.getElementById("chart-lora");
    if (el) {
      const chart = MCH.echart(el, {
        tooltip: { trigger: "axis" },
        legend: { top: 0 },
        grid: { left: 60, right: 40, top: 40, bottom: 40 },
        xAxis: { type: "category", name: "r", data: [] },
        yAxis: { type: "log", name: "Params", logBase: 10 },
        series: [
          { name: "全参微调", type: "line", data: [], color: "#ef4444", smooth: true, lineStyle: { width: 3 } },
          { name: "LoRA 可训参数", type: "line", data: [], color: "#4f46e5", smooth: true, lineStyle: { width: 3 } },
        ],
      });
      const update = () => {
        const r = parseInt(document.getElementById("lora-r").value);
        const d = parseInt(document.getElementById("lora-d").value);
        const L = parseInt(document.getElementById("lora-L").value);
        const rs = [1, 2, 4, 8, 16, 32, 64];
        const loraParams = rs.map(rr => 2 * rr * d * L);  // B (d×r) + A (r×d) per proj
        const fullParams = rs.map(() => d * d * L * 2);
        chart.setOption({
          xAxis: { data: rs },
          series: [
            { data: fullParams },
            { data: loraParams, markPoint: { data: [{ coord: [rs.indexOf(r), 2 * r * d * L], symbol: "circle", symbolSize: 12, itemStyle: { color: "#f59e0b" }, label: { formatter: "当前" } }] } },
          ],
        });
      };
      ["lora-r", "lora-d", "lora-L"].forEach(id => document.getElementById(id).addEventListener("input", e => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      }));
      update();
    }

    // 文本拼接模拟
    const tokEl = document.getElementById("tok-preview");
    const statsEl = document.getElementById("tok-stats");
    const update2 = () => {
      const vals = Array.from({ length: 7 }, (_, i) => document.getElementById(`tf-${i}`).value || "未知");
      const joined = "[CLS] " + vals.join(" [SEP] ") + " [SEP]";
      const tokenCount = joined.split(/\s+/).length + Math.floor(joined.length * 0.3); // 估算
      tokEl.textContent = joined;
      statsEl.innerHTML = `估算 token ≈ <b>${tokenCount}</b> · 实际输入 <code>max_len=256</code> · 7 个字段用 <code>[SEP]</code> 分隔位置稳定`;
    };
    for (let i = 0; i < 7; i++) {
      const input = document.getElementById(`tf-${i}`);
      if (input) input.addEventListener("input", update2);
    }
    update2();
  },
});
