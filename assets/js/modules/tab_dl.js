/* 模块：表格深度学习 */
MCH.register("tab_dl", {
  render() {
    const code = `# FT-Transformer (Yandex 2021) — 表格数据的 Transformer
# 关键：数值和类别特征统一 token 化

class FTTransformer(nn.Module):
    def __init__(self, num_num_features, num_cat_categories, d_model=192, n_layers=3, n_heads=8):
        # 每个数值特征映射为 d_model 维 token（带 bias）
        self.num_embed = NumericalFeatureTokenizer(num_num_features, d_model)
        # 每个类别特征 → embedding lookup
        self.cat_embed = CategoricalFeatureTokenizer(num_cat_categories, d_model)
        # [CLS] token 用于最终分类
        self.cls = nn.Parameter(torch.randn(1, 1, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, n_heads, batch_first=True), n_layers)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x_num, x_cat):
        num_tokens = self.num_embed(x_num)                      # (B, F_num, D)
        cat_tokens = self.cat_embed(x_cat)                      # (B, F_cat, D)
        cls = self.cls.expand(x_num.size(0), -1, -1)
        tokens = torch.cat([cls, num_tokens, cat_tokens], dim=1)
        encoded = self.transformer(tokens)
        return self.head(encoded[:, 0])                         # 取 [CLS]


# TabPFN (2023) — 🏆 表格基座的 In-Context Learning
# 把整个训练集当 context，直接预测测试样本（不需要 fit）！
from tabpfn import TabPFNClassifier
clf = TabPFNClassifier(device="cuda")
clf.fit(X_train, y_train)           # 只是记录数据（0 秒）
preds = clf.predict(X_test)         # 实际推理
# 适用：样本 < 10k，特征 < 100，类别 < 10
# 原理：在合成的数百万张表上预训练 → 学会"表格识别通用规律"


# 实战：什么时候 DL 能打败 GBDT？
# 经验法则（Grinsztajn 2022 详细对比）：
# - 样本 < 10万:   GBDT (XGBoost / LightGBM) 几乎永远更好
# - 样本 10万-100万:  GBDT 仍领先，但 FT-Transformer 接近
# - 样本 > 100万:    DL 开始接近或反超
# - 特征 > 10000 或大量类别: DL 的 embedding 优势显现`;

    return `
      ${MCH.hero({
        icon: "🧮",
        name: "表格深度学习 — TabNet / FT-Transformer / TabPFN",
        en: "Deep Learning for Tabular Data",
        tags: ["TabNet", "FT-Transformer", "SAINT", "TabPFN", "NODE"],
        meta: ["◈ 挑战 XGBoost 的深度方案", "⚡ 2024 开始接近"],
      })}

      ${MCH.versionSection("tab_dl")}

      <div class="section">
        <h2>1. 表格 DL 的挑战与突破</h2>
        ${MCH.info(`
          <b>多年以来的"经验事实"</b>：
          在表格（结构化）数据上，GBDT (XGBoost / LightGBM) 几乎永远优于深度学习。
          直到 2020 之后，一系列突破开始改变这个局面：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>2019</b> TabNet - Sequential Attention + 稀疏选择</li>
            <li><b>2021</b> FT-Transformer - 数值/类别统一 token</li>
            <li><b>2023</b> TabPFN - 基座模型 + In-Context Learning</li>
            <li><b>2024</b> TabPFN v2 - 扩展到 10w 样本，精度逼近 XGB</li>
          </ul>
          <b>趋势</b>：在大数据 + 多类别特征 + 多任务场景下，DL 已开始反超；小数据 GBDT 仍是王。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 主流表格 DL 方法对比</h2>
        <table class="table">
          <thead>
            <tr><th>方法</th><th>核心机制</th><th>可解释性</th><th>特点</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>TabNet</b> (Google 2019)</td>
              <td>Sequential Attention<br/>稀疏特征选择</td>
              <td>✓ mask 可解释</td>
              <td>早期尝试，自注意力特征选择</td>
            </tr>
            <tr>
              <td><b>NODE</b> (Yandex 2019)</td>
              <td>Neural Oblivious Decision Ensembles</td>
              <td>-</td>
              <td>神经版的 GBDT</td>
            </tr>
            <tr>
              <td><b>🏆 FT-Transformer</b> (Yandex 2021)</td>
              <td>数值 + 类别统一 token<br/>标准 Transformer Encoder</td>
              <td>-</td>
              <td>简单有效，RTDL 基准强</td>
            </tr>
            <tr>
              <td><b>SAINT</b> (2021)</td>
              <td>Self-Attention + Intersample<br/>对比学习</td>
              <td>-</td>
              <td>样本间信息交互</td>
            </tr>
            <tr>
              <td><b>🏆 TabPFN</b> (2023)</td>
              <td>In-Context Learning<br/>在百万合成表上预训练</td>
              <td>🆕 零训练</td>
              <td>小数据 &lt; 10k 的革命</td>
            </tr>
            <tr>
              <td><b>TabR</b> (2023)</td>
              <td>Retrieval-Augmented<br/>查找相似样本</td>
              <td>✓</td>
              <td>结合了 kNN 思想</td>
            </tr>
            <tr>
              <td><b>🆕 TabPFN v2</b> (2024)</td>
              <td>扩展到 10w+ 样本</td>
              <td>🆕 零训练</td>
              <td>在多数据集上与 XGBoost 持平</td>
            </tr>
            <tr>
              <td><b>GANDALF</b> (2024)</td>
              <td>Gating Adaptive Tabular<br/>Deep Feature Learning</td>
              <td>-</td>
              <td>特征重要性动态门控</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：表格 DL vs GBDT 性能对比（模拟）</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "tab-n", label: "样本量 log10(N)", min: 3, max: 8, step: 0.2, value: 5 })}
              ${MCH.slider({ id: "tab-d", label: "特征数", min: 5, max: 500, step: 5, value: 50 })}
              ${MCH.slider({ id: "tab-cat-ratio", label: "类别特征占比", min: 0, max: 1, step: 0.05, value: 0.3 })}
            </div>
            <div id="tab-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-tab" style="height:360px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 选型决策树</h2>
        <div class="mermaid">
flowchart TD
    S[表格数据建模] --> Q1{样本量}
    Q1 -->|N 小于 10k| A1[🏆 TabPFN<br/>零训练，秒级]
    Q1 -->|10k - 100k| Q2{特征类别复杂度}
    Q1 -->|超过 100w| A2[FT-Transformer<br/>或 XGBoost]
    Q2 -->|简单数值为主| A3[🏆 XGBoost / LightGBM]
    Q2 -->|类别特征多| A4[FT-Transformer<br/>或 CatBoost]
    A2 --> Q3{需要多任务?}
    Q3 -->|是| A5[FT-Transformer + MTL]
    Q3 -->|否| A3
        </div>
      </div>

      <div class="section">
        <h2>6. 开源资源</h2>
        <table class="table">
          <thead><tr><th>资源</th><th>说明</th></tr></thead>
          <tbody>
            <tr><td><a href="https://github.com/yandex-research/rtdl-revisiting-models" target="_blank"><b>RTDL</b> (Yandex)</a></td><td>🏆 FT-Transformer + Revisiting Deep Learning 基准</td></tr>
            <tr><td><a href="https://github.com/PriorLabs/TabPFN" target="_blank"><b>TabPFN</b></a></td><td>🆕 v2 支持 10w 样本，开箱即用</td></tr>
            <tr><td><a href="https://github.com/dreamquark-ai/tabnet" target="_blank"><b>pytorch-tabnet</b></a></td><td>TabNet PyTorch 实现</td></tr>
            <tr><td><a href="https://github.com/yandex-research/tab-ddpm" target="_blank"><b>TabDDPM</b></a></td><td>表格数据扩散模型生成</td></tr>
            <tr><td><a href="https://github.com/manujosephv/pytorch_tabular" target="_blank"><b>PyTorch Tabular</b></a></td><td>统一框架，20+ 表格 DL 模型</td></tr>
            <tr><td><a href="https://arxiv.org/abs/2207.08815" target="_blank">Grinsztajn 2022 论文</a></td><td>表格 DL vs GBDT 的全面基准</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("tab_dl").pros, MCH.getById("tab_dl").cons, MCH.getById("tab_dl").best_for)}
      </div>
    `;
  },

  mount() {
    const chart = MCH.echart(document.getElementById("chart-tab"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "category", data: [] },
      yAxis: { type: "value", name: "模型精度 (相对)", min: 0.7, max: 1 },
      series: [
        { name: "🏛 LightGBM", type: "bar", barWidth: 16, color: "#94a3b8", data: [] },
        { name: "🏆 XGBoost", type: "bar", barWidth: 16, color: "#4f46e5", data: [] },
        { name: "🧠 FT-Transformer", type: "bar", barWidth: 16, color: "#7c3aed", data: [] },
        { name: "🆕 TabPFN v2", type: "bar", barWidth: 16, color: "#d946ef", data: [] },
      ],
    });
    const update = () => {
      const logN = parseFloat(document.getElementById("tab-n").value);
      const N = Math.pow(10, logN);
      const d = parseInt(document.getElementById("tab-d").value);
      const catR = parseFloat(document.getElementById("tab-cat-ratio").value);
      // 模拟精度随规模变化
      const small = N < 5000;
      const medium = N >= 5000 && N < 1e6;
      const large = N >= 1e6;
      const highDim = d > 200;
      const manyCats = catR > 0.5;

      const scenarios = ["N=" + N.toExponential(0), "类别比=" + (catR * 100).toFixed(0) + "%", "特征=" + d];
      const lgb = [small ? 0.82 : medium ? 0.90 : 0.91, manyCats ? 0.88 : 0.90, highDim ? 0.87 : 0.91];
      const xgb = [small ? 0.80 : medium ? 0.91 : 0.92, manyCats ? 0.87 : 0.92, highDim ? 0.86 : 0.92];
      const ft = [small ? 0.72 : medium ? 0.89 : 0.93, manyCats ? 0.92 : 0.88, highDim ? 0.91 : 0.89];
      const pfn = [small ? 0.91 : medium ? 0.87 : 0.80, manyCats ? 0.89 : 0.88, highDim ? 0.80 : 0.89];

      chart.setOption({
        xAxis: { data: scenarios },
        series: [
          { data: lgb.map(v => v.toFixed(3)) },
          { data: xgb.map(v => v.toFixed(3)) },
          { data: ft.map(v => v.toFixed(3)) },
          { data: pfn.map(v => v.toFixed(3)) },
        ],
      });
      const recommend = small ? "🏆 TabPFN"
        : large ? "🧠 FT-Transformer / XGBoost"
        : manyCats ? "🧠 FT-Transformer / CatBoost"
        : "🏆 LightGBM / XGBoost";
      document.getElementById("tab-info").innerHTML = `
        <b>N</b>=${N.toExponential(0)}, <b>特征</b>=${d}, <b>类别比</b>=${(catR * 100).toFixed(0)}%<br/>
        <b>规模分级</b>：${small ? "小" : large ? "大" : "中"}数据<br/>
        <b>推荐</b>：<span style="color:#d946ef;font-weight:700;">${recommend}</span><br/>
        <span style="color:#64748b;">参考 Grinsztajn NeurIPS'22：小数据 GBDT 几乎永远赢；大数据开始追平；多类别+多任务 DL 有优势。</span>
      `;
    };
    ["tab-n", "tab-d", "tab-cat-ratio"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
