/* 模块：案例 · 广告 CTR 预估 */
MCH.register("case_ads_ctr", {
  render() {
    return `
      ${MCH.hero({
        icon: "📢",
        name: "案例 · 广告 CTR 预估 Ad CTR Prediction",
        en: "Click-Through Rate Prediction for Display/Search/Feed Ads",
        tags: ["高维稀疏", "特征交叉", "DeepFM", "DCN-V2", "AutoInt", "在线学习"],
        meta: ["◈ 10 亿+ 特征", "⚡ QPS 100w+", "◇ ROI 直接收益"],
      })}

      ${MCH.versionSection("case_ads_ctr")}

      <div class="section">
        <h2>1. 业务场景</h2>
        <p class="text-sm text-slate-600">广告 CTR / CVR 预估是计算广告学的核心问题：
        在用户请求到达时，预测每个候选广告的<b>点击率</b>（或转化率），用于排序和 eCPM 计算。</p>
        <div class="formula-block">
          <b>eCPM 模型</b>：$\\text{eCPM} = \\text{bid} \\times \\text{pCTR} \\times 1000$<br/>
          <b>目标</b>：最大化平台收益 = $\\sum_{i} \\text{price}_i \\times \\text{pCTR}_i \\times \\mathbb{1}[\\text{rank}_i \\leq K]$
        </div>
        ${MCH.info(`
          <b>核心挑战</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>超高维稀疏</b>：user × item × context 特征交叉，10 亿+ 稀疏特征不罕见</li>
            <li><b>实时性</b>：QPS 100w+，单次预估 &lt; 10ms</li>
            <li><b>分布漂移</b>：新广告/新用户持续进入，分布变化快</li>
            <li><b>Calibration</b>：pCTR 不仅要排序对，绝对值也要对（影响价格）</li>
            <li><b>Position Bias</b>：用户点击率受展示位置影响</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. CTR 模型演进大图</h2>
        <div class="mermaid">
flowchart LR
    LR[LR 2007<br/>Google Search Ads<br/>FTRL 在线学习] --> POLY2[Poly2 2010<br/>显式二阶]
    POLY2 --> FM[FM 2010<br/>低秩分解交互]
    FM --> FFM[FFM 2016<br/>Field 感知]
    FM --> WD[Wide&Deep 2016<br/>Google]
    WD --> DeepFM[DeepFM 2017<br/>华为]
    WD --> DCN[DCN 2017<br/>Google]
    DCN --> DCNV2[DCN-V2 2021]
    DeepFM --> xDeepFM[xDeepFM 2018<br/>MSRA CIN]
    xDeepFM --> AutoInt[AutoInt 2019<br/>Attention 交叉]
    AutoInt --> FiBiNET[FiBiNET 2019<br/>SENet + 双线性]
    AutoInt --> MaskNet[MaskNet 2021<br/>特征门控]
    AutoInt --> FinalMLP[FinalMLP 2023<br/>回归 MLP]
        </div>
      </div>

      <div class="section">
        <h2>3. 传统方案 vs 前沿方案</h2>
        <table class="table">
          <thead>
            <tr><th>维度</th><th>🏛 传统 (2010-2016)</th><th>🚀 前沿 (2020+)</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>模型</b></td>
              <td>LR + FTRL<br/>FM / FFM</td>
              <td>DeepFM / DCN-V2 / AutoInt<br/>基于 Transformer 的序列模型</td>
            </tr>
            <tr>
              <td><b>特征工程</b></td>
              <td>手工一阶 + 二阶交叉<br/>hash embedding</td>
              <td>端到端 embedding 学习<br/>多模态（图文CTR）</td>
            </tr>
            <tr>
              <td><b>用户行为</b></td>
              <td>滑窗统计 (点击/曝光 / 近 7/30 天)</td>
              <td>DIN/DIEN 目标注意力<br/>SIM 超长行为序列</td>
            </tr>
            <tr>
              <td><b>训练</b></td>
              <td>LR + FTRL 在线学习</td>
              <td>小时级增量训练<br/>ODL (Online Deep Learning)</td>
            </tr>
            <tr>
              <td><b>多任务</b></td>
              <td>独立 CTR + CVR 模型</td>
              <td>ESMM (CVR 全样本)<br/>MMoE / PLE</td>
            </tr>
            <tr>
              <td><b>Calibration</b></td>
              <td>Platt Scaling</td>
              <td>Focal + Calibration Layer<br/>Distillation from Oracle</td>
            </tr>
            <tr>
              <td><b>Position Bias</b></td>
              <td>position feature</td>
              <td>Position-aware Deep Net<br/>PAL (Position-Aware Learning)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>4. 代表模型亮点</h2>
        <table class="table">
          <thead><tr><th>模型</th><th>年份</th><th>核心贡献</th><th>工业落地</th></tr></thead>
          <tbody>
            <tr><td><b>LR + FTRL</b></td><td>2013</td><td>McMahan 提出 FTRL-Proximal，稀疏更新 + L1</td><td>Google / 百度广告</td></tr>
            <tr><td><b>FM</b></td><td>2010</td><td>低秩分解二阶交互 $\\langle v_i, v_j \\rangle x_i x_j$</td><td>CTR baseline 标配</td></tr>
            <tr><td><b>FFM</b></td><td>2016</td><td>Field-aware FM，Criteo Kaggle 冠军</td><td>美团 / Criteo</td></tr>
            <tr><td><b>Wide &amp; Deep</b></td><td>2016</td><td>🏆 Google Play 应用商店，记忆+泛化</td><td>TensorFlow Recommenders</td></tr>
            <tr><td><b>DeepFM</b></td><td>2017</td><td>🏆 DNN + FM 共享 embedding</td><td>华为应用市场，开源 DeepCTR</td></tr>
            <tr><td><b>DCN-V2</b></td><td>2021</td><td>🏆 改良交叉层，低秩参数化，GPU 友好</td><td>Google Cloud Recommender</td></tr>
            <tr><td><b>AutoInt</b></td><td>2019</td><td>Multi-Head Attention 做特征交互</td><td>学术基准</td></tr>
            <tr><td><b>FiBiNET</b></td><td>2019</td><td>SENet 特征重要性 + 双线性交互</td><td>新浪微博</td></tr>
            <tr><td><b>DIN / DIEN / SIM</b></td><td>2018-20</td><td>🏆 用户行为序列建模</td><td>阿里妈妈</td></tr>
            <tr><td><b>ESMM</b> (Alibaba)</td><td>2018</td><td>CVR 在全空间建模，解决样本偏差</td><td>淘宝 CVR</td></tr>
            <tr><td><b>MaskNet</b></td><td>2021</td><td>微博，特征门控机制</td><td>新浪微博</td></tr>
            <tr><td><b>FinalMLP</b></td><td>2023</td><td>🆕 双分支 MLP 简单超越复杂模型</td><td>华为</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 相关算法模块</h2>
        <div class="grid-3">
          ${["ml_logistic","ml_lightgbm","nn_mlp","nn_attention","fusion","mmoe","seq_encoder","losses","loss_multitask"].map(id => {
            const a = MCH.getById(id);
            if (!a) return "";
            const c = MCH.catColors[a.category];
            return `<a href="#/${a.route}" class="card block" style="text-decoration:none;color:inherit;border-left:3px solid ${c};">
              <div class="font-semibold text-sm" style="color:${c};">${a.icon} ${a.name}</div>
              <div class="text-xs text-slate-600 mt-1">${(a.tags || []).slice(0, 3).join(" · ")}</div>
            </a>`;
          }).join("")}
        </div>
      </div>

      <div class="section">
        <h2>6. 开源框架与工具</h2>
        <table class="table">
          <thead><tr><th>工具</th><th>亮点</th><th>链接</th></tr></thead>
          <tbody>
            <tr><td><b>DeepCTR</b></td><td>20+ CTR 模型 (TF/PyTorch)</td><td><a href="https://github.com/shenweichen/DeepCTR" target="_blank">GitHub</a></td></tr>
            <tr><td><b>FuxiCTR</b></td><td>100+ CTR 模型，华为诺亚</td><td><a href="https://github.com/reczoo/FuxiCTR" target="_blank">GitHub</a></td></tr>
            <tr><td><b>TorchRec</b> (Meta)</td><td>Meta 工业级，支持 10B+ 特征</td><td><a href="https://github.com/pytorch/torchrec" target="_blank">GitHub</a></td></tr>
            <tr><td><b>TF Recommenders</b></td><td>TensorFlow 官方推荐框架</td><td><a href="https://www.tensorflow.org/recommenders" target="_blank">官网</a></td></tr>
            <tr><td><b>DLRM</b> (Meta)</td><td>Facebook 开源 CTR baseline</td><td><a href="https://github.com/facebookresearch/dlrm" target="_blank">GitHub</a></td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 选型建议</h2>
        ${MCH.prosCons(
          ["数据量巨大，模型能力上限高", "业务价值直接（CPM × CTR = 收入）", "算法 + 工程双轮驱动，技术栈成熟"],
          ["10 亿特征的工程复杂度很高（需 DLRM/TorchRec 级框架）", "模型更新频繁（小时/分钟级）", "Calibration / Debias 是长期难题"],
          ["搜索广告 / 信息流广告 / 电商广告", "千万 DAU 以上大平台", "有在线工程能力"],
        )}
      </div>
    `;
  },
});
