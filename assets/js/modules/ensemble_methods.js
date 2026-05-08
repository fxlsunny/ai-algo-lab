/* 模块：融合/集成方法 */
MCH.register("ensemble_methods", {
  render() {
    const code = `# 融合方法五大类

# ① Voting (投票/平均)
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier([
    ("lr", LogisticRegression()),
    ("rf", RandomForestClassifier()),
    ("xgb", XGBClassifier()),
], voting="soft")   # soft=平均概率 / hard=多数决

# ② Bagging (Bootstrap Aggregating)
from sklearn.ensemble import BaggingClassifier
bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
                        n_estimators=100, max_samples=0.8)

# ③ Boosting 略（XGBoost/LightGBM）

# ④ Stacking (Kaggle 冠军标配)
from sklearn.ensemble import StackingClassifier
sc = StackingClassifier(
    estimators=[
        ("xgb", XGBClassifier()),
        ("lgbm", LGBMClassifier()),
        ("cat", CatBoostClassifier(verbose=False)),
    ],
    final_estimator=LogisticRegression(),   # meta-learner
    cv=5,                                    # 5 折 out-of-fold 预测
    passthrough=True,                        # 原始特征也传给 meta
)

# ⑤ Blending (Stacking 简化版 + 独立 holdout)
X_train_final, X_blend, y_train_final, y_blend = train_test_split(X_train, y_train, test_size=0.2)
# 基模型在 X_train_final 训练；用 X_blend 的预测构造元特征；meta-learner 在 X_blend 上训练

# ⑥ Weight Averaging (现代 DL 流行)
# Model Soup (2022):
#   最终权重 = (w_model1 + w_model2 + ... + w_modelN) / N
#   成本极低，常提升 0.5-2% 精度`;

    return `
      ${MCH.hero({
        icon: "🧩",
        name: "融合 / 集成 — Bagging / Boosting / Stacking / Blending",
        en: "Ensemble Methods",
        tags: ["Bagging", "Boosting", "Stacking", "Blending", "Voting", "Model Soup"],
        meta: ["◈ Kaggle 冠军标配", "⚡ 1%+ 精度提升"],
      })}

      ${MCH.versionSection("ensemble_methods")}

      <div class="section">
        <h2>1. 六大融合范式</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>并行性</th><th>偏差/方差</th><th>代表</th></tr></thead>
          <tbody>
            <tr><td><b>Bagging</b></td><td>✓ 并行</td><td>降方差</td><td>Random Forest</td></tr>
            <tr><td><b>Boosting</b></td><td>✗ 串行</td><td>降偏差</td><td>AdaBoost / GBDT / XGBoost</td></tr>
            <tr><td><b>Voting</b></td><td>✓ 并行</td><td>多数决 / 平均</td><td>soft / hard voting</td></tr>
            <tr><td><b>Stacking</b></td><td>部分并行</td><td>降两者</td><td>Kaggle 冠军标配</td></tr>
            <tr><td><b>Blending</b></td><td>部分并行</td><td>降两者</td><td>Stacking 简化，holdout 代替 CV</td></tr>
            <tr><td><b>🆕 Model Soup</b></td><td>✓ 并行</td><td>无</td><td>权重平均，训练后组合</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. Stacking 工作原理</h2>
        <div class="mermaid">
flowchart TB
    X[原始特征 X] --> B1[基模型 1<br/>XGBoost]
    X --> B2[基模型 2<br/>LightGBM]
    X --> B3[基模型 3<br/>Random Forest]
    X --> B4[基模型 N<br/>NN]
    B1 -->|Out-of-fold prediction| M[Meta-Learner<br/>Logistic Regression<br/>学习如何组合]
    B2 --> M
    B3 --> M
    B4 --> M
    X -.passthrough.-> M
    M --> Y[最终预测]
        </div>
        ${MCH.info(`
          <b>Stacking 成功的关键</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>基模型差异化</b>：异质模型（如 GBDT + NN + KNN）比同质更好</li>
            <li><b>Out-of-fold (OOF)</b>：必须用 CV 避免数据泄漏</li>
            <li><b>Meta-learner 简单</b>：通常 Logistic Regression / Ridge 即可，太复杂反而过拟合</li>
            <li><b>passthrough 可选</b>：把原特征也喂给 meta，有时能再提 0.1-0.5%</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：Bagging vs Boosting 降误差曲线</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "ens-n", label: "基模型数 N", min: 1, max: 300, step: 1, value: 50 })}
              ${MCH.slider({ id: "ens-bias", label: "单模型偏差", min: 0.1, max: 1, step: 0.05, value: 0.3 })}
              ${MCH.slider({ id: "ens-var", label: "单模型方差", min: 0.1, max: 1, step: 0.05, value: 0.5 })}
            </div>
            <div id="ens-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-ens" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. Kaggle 冠军方案常用配方</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>常见配方</th></tr></thead>
          <tbody>
            <tr><td>结构化数据二分类</td><td>XGBoost + LightGBM + CatBoost 加权 + LR 做 stacking</td></tr>
            <tr><td>图像分类</td><td>EfficientNet + ConvNeXt + ViT 投票 + snapshot ensemble</td></tr>
            <tr><td>NLP 分类</td><td>DeBERTa + RoBERTa + ERNIE + LR stacking</td></tr>
            <tr><td>时序回归</td><td>LightGBM + N-BEATS + TFT 加权 + 简单 Blending</td></tr>
            <tr><td>多类别长尾</td><td>多个 XGBoost（不同随机种子）+ Focal Loss DNN + LR</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 开源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://scikit-learn.org/stable/modules/ensemble.html" target="_blank">scikit-learn Ensemble</a> — VotingClassifier/StackingClassifier/BaggingClassifier</li>
          <li><a href="https://mlens.readthedocs.io/" target="_blank">mlens</a> — 专业 Ensemble 库</li>
          <li><a href="https://github.com/yzhao062/combo" target="_blank">combo</a> — 异常检测集成</li>
          <li><a href="https://github.com/mljar/mljar-supervised" target="_blank">AutoML mljar</a> — 自动 Stacking</li>
          <li>Kaggle Solutions 搜索 "stacking ensemble" 看真实获奖方案</li>
        </ul>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ensemble_methods").pros, MCH.getById("ensemble_methods").cons, MCH.getById("ensemble_methods").best_for)}
      </div>
    `;
  },

  mount() {
    const chart = MCH.echart(document.getElementById("chart-ens"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "基模型数 N" },
      yAxis: { type: "value", name: "误差" },
      series: [
        { name: "单个模型", type: "line", smooth: true, data: [], color: "#94a3b8", lineStyle: { type: "dashed" } },
        { name: "Bagging (降方差)", type: "line", smooth: true, data: [], color: "#4f46e5", lineStyle: { width: 3 } },
        { name: "Boosting (降偏差)", type: "line", smooth: true, data: [], color: "#f59e0b", lineStyle: { width: 3 } },
        { name: "Stacking", type: "line", smooth: true, data: [], color: "#10b981", lineStyle: { width: 3 } },
      ],
    });
    const update = () => {
      const N = parseInt(document.getElementById("ens-n").value);
      const bias = parseFloat(document.getElementById("ens-bias").value);
      const variance = parseFloat(document.getElementById("ens-var").value);
      const ns = [...Array(50).keys()].map(i => i * 6 + 1);
      const single = ns.map(() => (bias * bias + variance));
      const bagging = ns.map(n => bias * bias + variance / n);   // 独立假设
      const boosting = ns.map(n => Math.max(0.1 * bias * bias, bias * bias * Math.exp(-n / 20) + variance * 0.8));
      const stacking = ns.map(n => Math.max(0.08 * bias * bias, bias * bias * Math.exp(-n / 15) + variance / n / 1.2));
      chart.setOption({
        series: [
          { data: ns.map((n, i) => [n, single[i]]) },
          { data: ns.map((n, i) => [n, bagging[i]]), markPoint: { data: [{ coord: [N, bagging[Math.min(49, Math.floor(N / 6))]], itemStyle: { color: "#4f46e5" }, symbolSize: 12 }] } },
          { data: ns.map((n, i) => [n, boosting[i]]) },
          { data: ns.map((n, i) => [n, stacking[i]]) },
        ],
      });
      const idx = Math.min(49, Math.floor(N / 6));
      document.getElementById("ens-info").innerHTML = `
        <b>当前 N=${N} 个基模型</b><br/>
        <span class="text-slate-500">单模型</span>：${single[idx].toFixed(3)}<br/>
        <span style="color:#4f46e5;">Bagging</span>：${bagging[idx].toFixed(3)}<br/>
        <span style="color:#f59e0b;">Boosting</span>：${boosting[idx].toFixed(3)}<br/>
        <span style="color:#10b981;">Stacking</span>：${stacking[idx].toFixed(3)}<br/>
        <span style="color:#64748b;">Bagging 降方差 ∝ 1/N；Boosting 降偏差指数衰减；Stacking 同时利用二者。</span>
      `;
    };
    ["ens-n", "ens-bias", "ens-var"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
