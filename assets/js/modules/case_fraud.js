/* 模块：案例 · 欺诈识别 */
MCH.register("case_fraud", {
  render() {
    return `
      ${MCH.hero({
        icon: "💳",
        name: "案例 · 欺诈识别 Fraud Detection",
        en: "Real-time Fraud Detection for Payment &amp; Credit",
        tags: ["1:10000 不均衡", "实时 <100ms", "对抗性", "GNN", "Isolation Forest"],
        meta: ["◈ 支付/信贷/账户安全", "⚡ 规则+GBDT+GNN+IF 多重防护"],
      })}

      ${MCH.versionSection("case_fraud")}

      <div class="section">
        <h2>1. 业务场景 &amp; 类型</h2>
        <div class="grid-3">
          <div class="card" style="border-top:3px solid #ef4444;">
            <h4 class="font-semibold text-red-700">💳 支付欺诈</h4>
            <div class="text-xs text-slate-600 mt-2">盗刷 / 套现 / 跑分 / 代付代收<br/>特征：金额/时间/IP/设备/商户</div>
          </div>
          <div class="card" style="border-top:3px solid #f59e0b;">
            <h4 class="font-semibold text-amber-700">🏦 信贷欺诈</h4>
            <div class="text-xs text-slate-600 mt-2">虚假资料 / 团伙骗贷 / 身份盗用<br/>特征：申请资料/关系图/设备指纹</div>
          </div>
          <div class="card" style="border-top:3px solid #7c3aed;">
            <h4 class="font-semibold text-violet-700">👤 账户安全</h4>
            <div class="text-xs text-slate-600 mt-2">撞库 / 撞号 / 账户接管 (ATO)<br/>特征：登录行为序列 + 设备</div>
          </div>
        </div>

        ${MCH.info(`
          <b>⚠️ 核心挑战</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>极度不均衡</b>：坏样本通常 &lt; 0.1%，极端情况 1:100,000</li>
            <li><b>对抗性</b>：坏人会研究规则、变招绕过 → 模型要持续迭代</li>
            <li><b>标签延迟</b>：盗刷可能 3 天后才投诉，信贷逾期可能 90+ 天</li>
            <li><b>低延迟</b>：在线决策 &lt; 100ms（否则影响支付体验）</li>
            <li><b>可解释</b>：拦截用户要能给出理由（客诉/合规）</li>
          </ul>
        `, "warn")}
      </div>

      <div class="section">
        <h2>2. 传统方案 vs 前沿方案</h2>
        <table class="table">
          <thead>
            <tr><th>组件</th><th>🏛 传统方案</th><th>🚀 前沿方案 (2024)</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>主模型</b></td>
              <td>规则引擎（阈值 + 黑白名单）<br/>评分卡（LR + WOE 编码）<br/>随机森林</td>
              <td>XGBoost / LightGBM 主力<br/>+ GNN (GraphSAGE/GAT) 团伙识别<br/>+ AutoEncoder 异常检测</td>
            </tr>
            <tr>
              <td><b>特征工程</b></td>
              <td>手工构造 500+ 特征<br/>滑窗统计 / RFM</td>
              <td>Graph Embedding (Node2Vec)<br/>序列 Embedding (BST)<br/>Graph + Tabular 融合</td>
            </tr>
            <tr>
              <td><b>不均衡处理</b></td>
              <td>上采样 (SMOTE)<br/>下采样 + 集成<br/>class_weight</td>
              <td>🏆 Focal Loss + LDAM<br/>CB Sampler (Effective Number)<br/>硬难例挖掘</td>
            </tr>
            <tr>
              <td><b>图/关系</b></td>
              <td>关联规则 / 社群分析<br/>简单 1-hop 统计</td>
              <td>🏆 HGT / GraphSAGE 异构图<br/>Louvain 社区挖掘团伙<br/>Label Propagation 风险扩散</td>
            </tr>
            <tr>
              <td><b>异常检测</b></td>
              <td>3σ / IQR 统计规则<br/>One-Class SVM</td>
              <td>🏆 Isolation Forest<br/>AutoEncoder + 重构误差<br/>LOF + HBOS 集成</td>
            </tr>
            <tr>
              <td><b>实时推理</b></td>
              <td>规则引擎（Drools）<br/>批处理 + 缓存</td>
              <td>Flink / Kafka 流式<br/>Redis 特征仓<br/>ONNX/TensorRT 推理服务</td>
            </tr>
            <tr>
              <td><b>可解释</b></td>
              <td>评分卡系数 (WOE × β)</td>
              <td>SHAP / LIME / Integrated Gradients<br/>GNNExplainer (图注意力)</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 现代欺诈识别架构</h2>
        <div class="mermaid">
flowchart LR
    subgraph D[数据层]
        D1[交易流水]
        D2[用户画像]
        D3[设备指纹]
        D4[关系图]
    end
    subgraph F[特征工程]
        F1[滑窗统计<br/>RFM]
        F2[序列 Embedding<br/>BST]
        F3[图 Embedding<br/>Node2Vec/GNN]
    end
    subgraph M[模型层]
        M1[规则引擎<br/>强拦截]
        M2[XGBoost<br/>主打分]
        M3[GNN<br/>团伙识别]
        M4[Isolation Forest<br/>未知异常]
    end
    subgraph S[决策层]
        S1[加权融合]
        S2[阈值决策]
        S3[可解释生成]
    end
    D1 --> F1
    D2 --> F1
    D3 --> F1
    D4 --> F3
    D1 --> F2
    F1 --> M2
    F2 --> M2
    F3 --> M3
    F1 --> M4
    M1 --> S1
    M2 --> S1
    M3 --> S1
    M4 --> S1
    S1 --> S2 --> S3
        </div>
      </div>

      <div class="section">
        <h2>4. 代表性前沿论文 &amp; 开源</h2>
        <table class="table">
          <thead><tr><th>方向</th><th>工作</th><th>亮点</th></tr></thead>
          <tbody>
            <tr><td>图欺诈检测</td><td><b>xFraud</b> (eBay, IJCAI 2021)</td><td>异构图 + 可解释 GNN</td></tr>
            <tr><td>图欺诈检测</td><td><b>APATE</b> (2018)</td><td>基于图扩散的反欺诈</td></tr>
            <tr><td>欺诈 GNN 框架</td><td><b>DGFraud</b> (SIGIR'20)</td><td>欺诈 GNN 统一框架</td></tr>
            <tr><td>异构图风控</td><td><b>MAHINDER</b> (Alibaba, WWW 2022)</td><td>阿里反洗钱</td></tr>
            <tr><td>图对抗鲁棒</td><td><b>GraphRfi</b> (2020)</td><td>对抗攻击下的图风控</td></tr>
            <tr><td>标签噪声</td><td><b>DMBE</b> (2023)</td><td>含噪标签下的欺诈识别</td></tr>
            <tr><td>开源框架</td><td><b>PyOD</b></td><td><a href="https://pyod.readthedocs.io/" target="_blank">30+ 异常检测算法</a></td></tr>
            <tr><td>开源框架</td><td><b>DGFraud-TF2</b></td><td><a href="https://github.com/safe-graph/DGFraud-TF2" target="_blank">图欺诈检测基准</a></td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 相关算法模块</h2>
        <div class="grid-3">
          ${["ml_xgboost","ml_lightgbm","ml_isolation_forest","graph_sage","graph_louvain","graph_label_prop","graph_node2vec","losses","samplers","ml_logistic"].map(id => {
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
        <h2>6. 选型建议</h2>
        ${MCH.prosCons(
          ["业务价值直接（拦截损失 = ROI）", "反馈数据明确（客诉 / chargeback）", "多种方法可组合（规则+GBDT+GNN+IF）"],
          ["对抗性：坏人持续变招，模型需持续迭代", "标签延迟/噪声大（盗刷 3 天后才知）", "极端不均衡 + 实时推理双重压力"],
          ["支付反欺诈（实时拦截）", "信贷反欺诈（审批侧）", "账户安全（登录 / 密码保护）", "反洗钱 AML", "商户准入风控"],
        )}
      </div>
    `;
  },
});
