/* 模块：朴素贝叶斯 */
MCH.register("ml_naive_bayes", {
  render() {
    const code = `# 朴素贝叶斯：从贝叶斯定理 + 独立假设
# P(y | x) ∝ P(y) · ∏ P(x_i | y)    ← 假设特征给定类别下相互独立

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

# ① MultinomialNB —— 文本分类经典（垃圾邮件 / 情感）
# 特征：词频向量（TF 或 TF-IDF）
# 参数：alpha (Laplace 平滑，默认 1.0)
clf = MultinomialNB(alpha=1.0)
clf.fit(X_tfidf, y)

# ② GaussianNB —— 连续特征
# 假设 P(x_i | y) ~ N(μ_{y,i}, σ²_{y,i})
# 对每个类别 y，每个特征 i，学一个均值和方差
clf = GaussianNB(var_smoothing=1e-9)

# ③ BernoulliNB —— 二值特征
# 特征：0/1（单词是否出现，比 TF 更鲁棒）

# ========================================================================
# Laplace 平滑：解决零概率问题
# ========================================================================
# 问题：测试时遇到训练未见的特征值 → P(x|y)=0 → 整个后验概率为 0
# 修正：P(x_i | y) = (count(x_i, y) + α) / (count(y) + α · V)
#       α = 1 是经典 Laplace 平滑；α < 1 更弱；α = 0 无平滑（危险）

# 预测时用 log 防溢出：
# log P(y | x) ∝ log P(y) + Σ log P(x_i | y)
def predict(x):
    return argmax(log_prior[y] + sum(log_likelihood[y, x_i] for x_i in x))`;

    return `
      ${MCH.hero({
        icon: "ℙ",
        name: "朴素贝叶斯 Naive Bayes",
        en: "Bayesian + Feature Independence Assumption",
        tags: ["贝叶斯定理", "文本分类", "极速训练", "O(n·d)"],
        meta: ["◈ 文本分类 baseline 王者", "⚡ 百万样本秒级训练"],
      })}

      ${MCH.versionSection("ml_naive_bayes")}

      <div class="section">
        <h2>1. 核心思想：贝叶斯定理 + 条件独立假设</h2>
        <div class="formula-block">
          <b>贝叶斯定理</b>：
          $$ P(y \\mid x) = \\frac{P(x \\mid y) P(y)}{P(x)} $$
          <b>朴素假设</b>：给定类别 y 时，各特征条件独立：
          $$ P(x \\mid y) = \\prod_{i=1}^{d} P(x_i \\mid y) $$
          <b>所以分类规则</b>（MAP）：
          $$ \\hat y = \\arg\\max_y \\Big[ \\log P(y) + \\sum_i \\log P(x_i \\mid y) \\Big] $$
        </div>
        ${MCH.info(`
          <b>"朴素"是什么意思？</b>
          假设特征相互独立 —— 这在现实中几乎总是不成立（例如"北京"和"天安门"明显相关）。
          但神奇的是，即便假设错误，模型对<b>分类结果</b>仍然相当鲁棒
          —— 因为我们只需要 argmax，不需要准确的概率估计。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 三种常见变体对比</h2>
        <table class="table">
          <thead><tr><th>变体</th><th>特征类型</th><th>P(x_i\\|y) 建模</th><th>经典应用</th></tr></thead>
          <tbody>
            <tr><td><b>MultinomialNB</b></td><td>离散计数（词频）</td><td>多项分布</td><td>文本分类、BoW / TF-IDF</td></tr>
            <tr><td>BernoulliNB</td><td>二值 (0/1)</td><td>伯努利分布</td><td>短文本、是/否特征</td></tr>
            <tr><td><b>GaussianNB</b></td><td>连续实数</td><td>正态分布</td><td>连续特征、sklearn 默认</td></tr>
            <tr><td>ComplementNB</td><td>离散 + 不均衡</td><td>反例似然</td><td>类不均衡文本（Google 邮件分类）</td></tr>
            <tr><td>CategoricalNB</td><td>离散（非计数）</td><td>类别分布</td><td>类目特征</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：Laplace 平滑的作用</h2>
        <div class="grid-2">
          <div>
            <p class="text-sm text-slate-600">假设一个词汇表 V=20，某类训练集词频表如下。观察不同 α 值下的 P(x|y)：</p>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "nb-alpha", label: "Laplace α", min: 0, max: 5, step: 0.1, value: 1 })}
              ${MCH.slider({ id: "nb-V", label: "词表大小 V", min: 10, max: 200, step: 10, value: 20 })}
            </div>
            <div id="nb-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-nb" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 为什么朴素贝叶斯在文本分类中如此强大？</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6 space-y-1">
          <li><b>训练极快</b>：一次扫描计数即可，百万文档分分钟；</li>
          <li><b>天然并行</b>：每个类别的参数独立；</li>
          <li><b>高维友好</b>：词汇表 10w+ 也无压力（vs SVM 的 O(n²)）；</li>
          <li><b>增量学习</b>：<code>partial_fit</code> 新样本即加即训；</li>
          <li><b>鲁棒</b>：即使独立假设不成立，argmax 仍然合理；</li>
          <li><b>可解释</b>：每个词对每类的贡献 (log P(x|y)) 一目了然。</li>
        </ul>
        ${MCH.info(`
          <b>现代地位</b>：在短文本 / 垃圾邮件 / 新闻分类等任务上，NB 仍然是极强 baseline。
          配合 TF-IDF 甚至能击败未精调的 BERT（小数据场景）。
          Google 的 Gmail 垃圾邮件早期就用 ComplementNB。
        `, "biz")}
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ml_naive_bayes").pros, MCH.getById("ml_naive_bayes").cons, MCH.getById("ml_naive_bayes").best_for)}
      </div>
    `;
  },

  mount() {
    // 词频平滑可视化
    const chart = MCH.echart(document.getElementById("chart-nb"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 60, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "category", data: [] },
      yAxis: { type: "value", name: "P(word | class)" },
      series: [
        { name: "无平滑 α=0 (⚠ 零概率)", type: "bar", color: "#ef4444", barWidth: 12, data: [] },
        { name: "当前 α", type: "bar", color: "#4f46e5", barWidth: 12, data: [] },
      ],
    });
    const update = () => {
      const alpha = parseFloat(document.getElementById("nb-alpha").value);
      const V = parseInt(document.getElementById("nb-V").value);
      // 模拟词频（包含未出现词）
      const counts = [10, 5, 3, 2, 0, 0, 0, 0, 0, 0];
      const labels = ["w1", "w2", "w3", "w4", "w5 (未见)", "w6 (未见)", "w7 (未见)", "w8 (未见)", "w9 (未见)", "w10 (未见)"];
      const N = counts.reduce((a, b) => a + b, 0);
      const noSmooth = counts.map(c => c / N);
      const smoothed = counts.map(c => (c + alpha) / (N + alpha * V));
      chart.setOption({
        xAxis: { data: labels },
        series: [{ data: noSmooth.map(v => +v.toFixed(3)) }, { data: smoothed.map(v => +v.toFixed(3)) }],
      });
      document.getElementById("nb-info").innerHTML = `
        <b>α=${alpha}</b> · V=${V}<br/>
        对未见词 w5 的 P：<br/>
        无平滑: <span style="color:#ef4444;">0</span> → <b>整个类别概率变成 0！</b><br/>
        当前 α: <span style="color:#4f46e5;">${(alpha / (N + alpha * V)).toFixed(4)}</span><br/>
        <span class="text-slate-500">α=1 是经典 Laplace 平滑（加 1 平滑）；α &lt; 1 更弱；α 过大会"稀释"真实频率。</span>
      `;
    };
    ["nb-alpha", "nb-V"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();
  },
});
