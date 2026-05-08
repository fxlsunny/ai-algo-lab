/* 模块：工业案例总览 */
MCH.register("case_overview", {
  render() {
    const cases = [
      {
        id: "case_recommendation", icon: "🛒", name: "推荐算法", en: "Recommendation Systems",
        color: "#4f46e5",
        challenges: "冷启动 · 长尾 item · 多目标冲突 · Feedback Loop",
        traditional: "协同过滤 · MF · ItemCF",
        modern: "DIN → DIEN → SASRec → BERT4Rec → LLM4Rec",
        companies: "阿里·京东·字节·美团·小红书",
      },
      {
        id: "case_fraud", icon: "💳", name: "欺诈识别", en: "Fraud Detection",
        color: "#ef4444",
        challenges: "极度不均衡 1:10000 · 对抗性 · 标签延迟 · 实时性 &lt; 100ms",
        traditional: "规则引擎 · 评分卡 · 随机森林",
        modern: "XGBoost + GNN + Isolation Forest + Graph-based 聚合",
        companies: "支付宝·微信支付·PayPal·Visa",
      },
      {
        id: "case_ads_ctr", icon: "📢", name: "广告 CTR 预估", en: "Ad CTR Prediction",
        color: "#f59e0b",
        challenges: "10 亿+ 稀疏特征 · 实时更新 · Calibration · Position Bias",
        traditional: "LR + FM + FFM",
        modern: "Wide&Deep → DeepFM → DCN-V2 → AutoInt → 多模态 CTR",
        companies: "Google·Meta·字节·百度·腾讯",
      },
      {
        id: "case_credit_score", icon: "📊", name: "信用评分", en: "Credit Scoring",
        color: "#10b981",
        challenges: "监管合规 · 可审计 · 样本偏置 · 标签延迟 6-12 月",
        traditional: "评分卡 · WOE + LR",
        modern: "GBDT (XGBoost/LightGBM) + Shapley Value 可解释",
        companies: "银行 · 消金 · 蚂蚁芝麻分 · 腾讯财付通",
      },
      {
        id: "case_merchant", icon: "🛡", name: "商户多模态识别", en: "Merchant Multi-Modal Recognition",
        color: "#7c3aed",
        challenges: "5 模态融合 · 1:10000 长尾 · 日 1亿商户 · P99 &lt; 50ms",
        traditional: "特征工程 + XGBoost / LR",
        modern: "Perceiver-IO + PLE-MMoE + Decoupled cRT + 蒸馏",
        companies: "支付宝·微信支付商户风控",
      },
    ];

    return `
      <div class="module-hero" style="background:linear-gradient(135deg,#312e81 0%,#7e22ce 50%,#be185d 100%);">
        <div class="flex items-center gap-3 mb-2">
          <div class="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center text-2xl">📚</div>
          <div>
            <h1>工业案例总览 · Industry Case Studies</h1>
            <div class="text-sm text-slate-300 mt-1">5 个典型工业场景 · 每个都给出"传统 vs 前沿"方案对比 + 优缺点 + 开源实现参考</div>
          </div>
        </div>
        <div class="flex gap-6 mt-4 text-sm text-slate-300">
          <span>🛒 推荐</span><span>💳 风控</span><span>📢 广告</span><span>📊 信用</span><span>🛡 风险识别</span>
        </div>
      </div>

      <div class="section">
        <h2>🎯 案例对比一览</h2>
        <table class="table">
          <thead><tr><th>案例</th><th>核心挑战</th><th>传统方案</th><th>前沿方案</th><th>代表厂商</th></tr></thead>
          <tbody>
            ${cases.map(c => `
              <tr>
                <td>
                  <a href="#/${c.id}" style="color:${c.color};font-weight:700;text-decoration:none;">${c.icon} ${c.name}</a>
                  <div class="text-[10px] text-slate-500">${c.en}</div>
                </td>
                <td class="text-xs text-slate-600">${c.challenges}</td>
                <td class="text-xs text-slate-600"><code>${c.traditional}</code></td>
                <td class="text-xs text-slate-600"><code style="color:${c.color};">${c.modern}</code></td>
                <td class="text-xs text-slate-500">${c.companies}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>🗺️ 案例选择指引（按业务特征）</h2>
        <div class="grid-3">
          <div class="card" style="border-top:3px solid #4f46e5;">
            <h3 class="font-bold text-indigo-700">🎨 用户体验优化</h3>
            <p class="text-xs text-slate-600 mt-2">核心是<b>"猜你喜欢"</b>类问题：多样性 &amp; 个性化 &amp; 实时更新。</p>
            <div class="mt-3 text-xs">→ <a href="#/case_recommendation" class="text-indigo-600 font-semibold">推荐算法</a></div>
          </div>
          <div class="card" style="border-top:3px solid #ef4444;">
            <h3 class="font-bold text-red-700">🛡 风险控制</h3>
            <p class="text-xs text-slate-600 mt-2">核心是<b>"坏人识别"</b>：极度不均衡 + 对抗性 + 低延迟。</p>
            <div class="mt-3 text-xs">
              → <a href="#/case_fraud" class="text-red-600 font-semibold">欺诈识别</a> / 
              → <a href="#/case_merchant" class="text-red-600 font-semibold">商户识别</a>
            </div>
          </div>
          <div class="card" style="border-top:3px solid #f59e0b;">
            <h3 class="font-bold text-amber-700">💰 变现优化</h3>
            <p class="text-xs text-slate-600 mt-2">核心是<b>"CTR/CVR/eCPM"</b>：特征工程 + 精度 + 在线学习。</p>
            <div class="mt-3 text-xs">→ <a href="#/case_ads_ctr" class="text-amber-600 font-semibold">广告 CTR</a></div>
          </div>
          <div class="card" style="border-top:3px solid #10b981;">
            <h3 class="font-bold text-emerald-700">🏦 金融决策</h3>
            <p class="text-xs text-slate-600 mt-2">核心是<b>"信用评估"</b>：监管合规 + 可解释 + 稳定性。</p>
            <div class="mt-3 text-xs">→ <a href="#/case_credit_score" class="text-emerald-600 font-semibold">信用评分</a></div>
          </div>
          <div class="card" style="border-top:3px solid #7c3aed;">
            <h3 class="font-bold text-violet-700">🧬 复杂多模态</h3>
            <p class="text-xs text-slate-600 mt-2">文本 + 行为序列 + 关系图 + 图像 + 辅助特征融合。</p>
            <div class="mt-3 text-xs">→ <a href="#/case_merchant" class="text-violet-600 font-semibold">商户多模态识别</a>（深度工业案例）</div>
          </div>
          <div class="card" style="border-top:3px solid #64748b;">
            <h3 class="font-bold text-slate-700">🏷️ 快速 baseline</h3>
            <p class="text-xs text-slate-600 mt-2">小数据 / 快速验证：LightGBM / LR + 评分卡。</p>
            <div class="mt-3 text-xs">→ <a href="#/ml_lightgbm" class="text-slate-600 font-semibold">LightGBM</a> / <a href="#/ml_logistic" class="text-slate-600 font-semibold">LR</a></div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>📑 案例卡片详情</h2>
        <div class="grid-2">
          ${cases.map(c => `
            <a href="#/${c.id}" class="card block" style="text-decoration:none;color:inherit;border-left:4px solid ${c.color};">
              <div class="flex items-center gap-3 mb-2">
                <div style="font-size:28px;">${c.icon}</div>
                <div>
                  <div class="font-bold text-lg" style="color:${c.color};">${c.name}</div>
                  <div class="text-[11px] text-slate-500">${c.en}</div>
                </div>
              </div>
              <div class="text-xs text-slate-600 mt-2">
                <div><b>核心挑战</b>：${c.challenges}</div>
                <div class="mt-1"><b>🏛 传统</b>：${c.traditional}</div>
                <div class="mt-1"><b>🚀 前沿</b>：${c.modern}</div>
                <div class="mt-1 text-slate-500"><b>🏢 代表厂商</b>：${c.companies}</div>
              </div>
              <div class="mt-3 text-xs" style="color:${c.color};">深入查看 →</div>
            </a>
          `).join("")}
        </div>
      </div>

      <div class="section">
        <h2>💡 通用方法论：传统 → 前沿 的演进规律</h2>
        <div class="mermaid">
flowchart LR
    T1[特征工程时代<br/>手工构造 + LR/GBDT<br/>2010s 前]
    T2[浅度学习时代<br/>FM/FFM/DeepFM<br/>2015-2019]
    T3[序列建模时代<br/>DIN/BST/SASRec<br/>2018-2022]
    T4[多模态+大模型<br/>CLIP/LLM4Rec/GNN<br/>2023+]
    T1 --> T2 --> T3 --> T4
        </div>
        <p class="text-sm text-slate-600 mt-3">核心趋势：<b>少特征工程、多模型能力 · 浅层到端到端 · 单模态到多模态 · 离线到在线 · 专用到通用</b></p>
      </div>
    `;
  },
});
