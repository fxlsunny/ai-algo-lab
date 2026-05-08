/* 模块：案例 · 信用评分 */
MCH.register("case_credit_score", {
  render() {
    return `
      ${MCH.hero({
        icon: "📊",
        name: "案例 · 信用评分 Credit Scoring",
        en: "Credit Risk Scoring for Banking &amp; Lending",
        tags: ["评分卡", "WOE/IV", "逻辑回归", "XGBoost", "监管合规", "Shapley"],
        meta: ["◈ 银行/消金核心", "⚡ 监管友好", "◇ 标签延迟 6-12月"],
      })}

      ${MCH.versionSection("case_credit_score")}

      <div class="section">
        <h2>1. 业务场景</h2>
        <p class="text-sm text-slate-600">信用评分用于金融机构的<b>贷款审批、授信额度、风险定价</b>，典型任务：</p>
        <div class="grid-3">
          <div class="card"><h4 class="font-semibold text-emerald-700">A 卡 Application</h4><p class="text-xs text-slate-600 mt-1">申请评分：贷款<b>审批阶段</b>决定是否放贷。<br/>特征：申请资料 + 外部征信 + 反欺诈信号</p></div>
          <div class="card"><h4 class="font-semibold text-emerald-700">B 卡 Behavior</h4><p class="text-xs text-slate-600 mt-1">行为评分：贷后期管理，<b>调整额度/利率</b>。<br/>特征：还款行为 + 资金周转</p></div>
          <div class="card"><h4 class="font-semibold text-emerald-700">C 卡 Collection</h4><p class="text-xs text-slate-600 mt-1">催收评分：逾期后<b>预测回款概率</b>，决定催收策略。<br/>特征：逾期时长 + 历史联系记录</p></div>
        </div>

        ${MCH.info(`
          <b>核心约束</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>监管合规</b>：银保监/央行要求模型<b>可审计可解释</b>，不能纯黑盒</li>
            <li><b>标签延迟</b>：是否违约通常 6-12 月后才知，训练-上线滞后明显</li>
            <li><b>样本偏差</b>：幸存者偏差（拒绝的人看不到表现）→ 需要拒绝推断</li>
            <li><b>稳定性</b>：PSI / CSI 监控，模型漂移容忍度低</li>
            <li><b>公平性</b>：不能对特定人群（性别/种族/年龄）歧视</li>
          </ul>
        `, "warn")}
      </div>

      <div class="section">
        <h2>2. 传统方案：评分卡（Scorecard）</h2>
        <p class="text-sm text-slate-600">金融行业的"事实标准"，核心是 <b>WOE 编码 + 逻辑回归</b>：</p>
        <div class="mermaid">
flowchart LR
    D[原始数据] --> B[分箱 Binning<br/>等频/等宽/决策树<br/>或 chimerge 卡方]
    B --> W[WOE Weight of Evidence<br/>ln P good/bad]
    W --> I[IV Information Value<br/>筛选特征 IV 大于 0.02]
    I --> LR[LR 建模<br/>L1 + 交叉验证]
    LR --> S[转换为评分<br/>基础分+系数·WOE]
    S --> O[评分卡 Scorecard<br/>人肉可审]
        </div>
        <div class="formula-block">
          <b>WOE</b>：$\\text{WOE}_i = \\ln\\frac{P(x_i | y=1)}{P(x_i | y=0)} = \\ln\\frac{\\text{Bad Rate}_i / \\bar{\\text{Bad}}}{\\text{Good Rate}_i / \\bar{\\text{Good}}}$<br/>
          <b>IV</b>：$\\text{IV} = \\sum_i (P_i^{1} - P_i^{0}) \\cdot \\text{WOE}_i$ · 特征重要性度量<br/>
          <b>评分</b>：$\\text{Score} = A - B \\cdot (w^\\top x + b)$ · A, B 是缩放常数
        </div>
        <table class="table mt-3">
          <thead><tr><th>IV 范围</th><th>预测能力</th><th>建议</th></tr></thead>
          <tbody>
            <tr><td>&lt; 0.02</td><td>无</td><td>剔除</td></tr>
            <tr><td>0.02 - 0.1</td><td>弱</td><td>可选入模</td></tr>
            <tr><td>0.1 - 0.3</td><td>中等</td><td>✓ 优先保留</td></tr>
            <tr><td>0.3 - 0.5</td><td>强</td><td>✓✓ 强预测力</td></tr>
            <tr><td>&gt; 0.5</td><td>可疑</td><td>⚠ 检查是否穿越</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 前沿方案：GBDT + 可解释性</h2>
        <p class="text-sm text-slate-600">近年逐步引入 GBDT（XGBoost / LightGBM）+ 后验可解释工具：</p>
        <table class="table">
          <thead>
            <tr><th>维度</th><th>🏛 传统评分卡</th><th>🚀 现代 GBDT 方案</th></tr>
          </thead>
          <tbody>
            <tr><td><b>主模型</b></td><td>LR + WOE 分箱</td><td>XGBoost / LightGBM / CatBoost</td></tr>
            <tr><td><b>特征工程</b></td><td>手工分箱 + WOE</td><td>原始数值 + 自动特征交叉</td></tr>
            <tr><td><b>KS 指标</b></td><td>0.35 - 0.45 (常见)</td><td>0.45 - 0.55 (提升 5-15%)</td></tr>
            <tr><td><b>可解释性</b></td><td>系数直接可解（满分）</td><td>SHAP 值 / LIME / 规则抽取</td></tr>
            <tr><td><b>监管审批</b></td><td>✓ 完全接受</td><td>⚠ 需额外可解释报告</td></tr>
            <tr><td><b>稳定性</b></td><td>✓ 极佳</td><td>中等（需 PSI 监控）</td></tr>
            <tr><td><b>部署</b></td><td>评分公式 → 数据库 SQL</td><td>模型服务 / ONNX / 规则抽取</td></tr>
            <tr><td><b>迭代速度</b></td><td>季度/半年级</td><td>月级甚至周级（可自动）</td></tr>
          </tbody>
        </table>

        ${MCH.info(`
          <b>🎯 2024 业界共识</b>：<b>评分卡不死，但越来越多的银行/消金用 "评分卡（审批侧）+ GBDT（辅助决策/授信）" 组合</b>。
          GBDT 模型的 <b>KS ↑ 5-10%</b>，带来的坏账率降低可达 20%+，商业价值巨大。
          SHAP 值让模型具备"为何拒绝"的解释能力，通过了多数监管审查。
        `, "tip")}
      </div>

      <div class="section">
        <h2>4. 关键工程细节</h2>
        <table class="table">
          <thead><tr><th>环节</th><th>难点</th><th>工业最佳实践</th></tr></thead>
          <tbody>
            <tr><td><b>样本构造</b></td><td>幸存者偏差（拒绝样本无 y 标签）</td><td>Reject Inference：KNN / Parcelling / Fuzzy Augmentation</td></tr>
            <tr><td><b>时间窗口</b></td><td>观察期 / 表现期如何设</td><td>6-12 月观察期；vintage 分析</td></tr>
            <tr><td><b>特征筛选</b></td><td>共线性 / 单调性</td><td>IV 筛选 + VIF + 单调性约束</td></tr>
            <tr><td><b>模型监控</b></td><td>分布漂移 / 效果衰减</td><td>PSI / CSI / KS 月度监控</td></tr>
            <tr><td><b>拒绝推断</b></td><td>拒绝客户的真实坏率未知</td><td>增广样本 + 迭代训练</td></tr>
            <tr><td><b>公平性</b></td><td>不能隐含歧视</td><td>Fairness Audit (demographic parity/equal odds)</td></tr>
            <tr><td><b>部署</b></td><td>监管要求离线</td><td>评分公式 SQL 化 / 模型文件归档</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 相关算法模块</h2>
        <div class="grid-3">
          ${["ml_logistic","ml_xgboost","ml_lightgbm","ml_decision_tree","ml_random_forest","losses"].map(id => {
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
        <h2>6. 开源工具</h2>
        <table class="table">
          <thead><tr><th>工具</th><th>用途</th><th>链接</th></tr></thead>
          <tbody>
            <tr><td><b>toad</b> (众安)</td><td>🏆 评分卡全流程（分箱/WOE/IV/LR）</td><td><a href="https://github.com/amphibian-dev/toad" target="_blank">GitHub</a></td></tr>
            <tr><td><b>scorecardpy</b></td><td>Python 评分卡工具包</td><td><a href="https://github.com/ShichenXie/scorecardpy" target="_blank">GitHub</a></td></tr>
            <tr><td><b>optbinning</b></td><td>优化分箱 + 评分卡</td><td><a href="https://gnpalencia.org/optbinning/" target="_blank">文档</a></td></tr>
            <tr><td><b>SHAP</b></td><td>🏆 GBDT 可解释性事实标准</td><td><a href="https://shap.readthedocs.io/" target="_blank">文档</a></td></tr>
            <tr><td><b>Fairlearn</b></td><td>公平性评估 (Microsoft)</td><td><a href="https://fairlearn.org/" target="_blank">官网</a></td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 选型建议</h2>
        ${MCH.prosCons(
          ["监管接受度高（可审计）", "稳定性好，抗漂移", "解释性强（每个因子贡献清晰）"],
          ["精度上限低（线性模型）", "手工特征工程成本高", "迭代慢（季度/半年级）"],
          ["银行/持牌消金贷款审批（监管要求可解释）", "额度/定价/催收决策", "稳定性要求极高的场景"],
        )}
      </div>
    `;
  },
});
