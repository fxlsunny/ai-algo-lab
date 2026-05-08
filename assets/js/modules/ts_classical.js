/* 模块：时序经典模型 */
MCH.register("ts_classical", {
  render() {
    const code = `# 经典时序模型三剑客

# -------- ① ARIMA / SARIMA ----------
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(series, order=(p, d, q), seasonal_order=(P, D, Q, s))
fit = model.fit()
forecast = fit.forecast(steps=30)
# p: 自回归阶; d: 差分阶（去趋势）; q: 移动平均阶

# -------- ② ETS (指数平滑状态空间) ----------
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model = ExponentialSmoothing(series, trend="add", seasonal="add", seasonal_periods=7)
fit = model.fit()

# -------- ③ Prophet (Facebook 2017) ----------
from prophet import Prophet
m = Prophet(
    yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
    changepoint_prior_scale=0.05,   # 关键：越大越灵活
    holidays=CN_holidays,           # 节假日支持
)
m.fit(df.rename(columns={"date": "ds", "value": "y"}))
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)        # 包含 yhat, yhat_lower, yhat_upper

# Prophet 模型公式：
#   y(t) = g(t) + s(t) + h(t) + ε
#          趋势   季节   节假日   噪声
# g(t): 分段线性或逻辑趋势（自动找 changepoint）
# s(t): 傅里叶级数拟合周期
# h(t): 节假日指示函数`;

    return `
      ${MCH.hero({
        icon: "📈",
        name: "时序经典模型 — ARIMA / ETS / Prophet",
        en: "Classical Time Series Forecasting",
        tags: ["ARIMA", "ETS", "Prophet", "季节分解", "统计方法"],
        meta: ["◈ 百年统计基础", "⚡ 可解释性满分"],
      })}

      ${MCH.versionSection("ts_classical")}

      <div class="section">
        <h2>1. 三大经典方法</h2>
        <div class="grid-3">
          <div class="card" style="border-top:3px solid #0ea5e9;">
            <h3 class="font-bold text-sky-700">📊 ARIMA / SARIMA</h3>
            <p class="text-xs text-slate-600 mt-2">
              Box &amp; Jenkins 1970 经典。<br/>
              <b>AR</b>：自回归 y_t 依赖 y_{t-1..p}<br/>
              <b>I</b>：差分去趋势<br/>
              <b>MA</b>：移动平均残差
            </p>
          </div>
          <div class="card" style="border-top:3px solid #10b981;">
            <h3 class="font-bold text-emerald-700">🎯 ETS (指数平滑)</h3>
            <p class="text-xs text-slate-600 mt-2">
              Hyndman 2008 统一框架。<br/>
              Error / Trend / Seasonality<br/>
              自动模型选择（AIC/BIC）<br/>
              简单稳健，小数据首选
            </p>
          </div>
          <div class="card" style="border-top:3px solid #f59e0b;">
            <h3 class="font-bold text-amber-700">🚀 Prophet (Meta)</h3>
            <p class="text-xs text-slate-600 mt-2">
              Taylor &amp; Letham 2017。<br/>
              加法模型 y = g + s + h + ε<br/>
              🏆 节假日 + 变点自动识别<br/>
              业务友好，不需要统计背景
            </p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 核心公式</h2>
        <div class="formula-block">
          <b>SARIMA(p,d,q)(P,D,Q)s</b>：
          $$ \\Phi(B^s) \\phi(B) (1-B)^d (1-B^s)^D y_t = \\Theta(B^s) \\theta(B) \\epsilon_t $$
          <b>Prophet 加法模型</b>：
          $$ y(t) = g(t) + s(t) + h(t) + \\epsilon_t $$
          趋势 g(t) 分段线性；季节 s(t) = $\\sum_{n=1}^{N} [a_n \\cos(\\frac{2\\pi nt}{P}) + b_n \\sin(\\frac{2\\pi nt}{P})]$（傅里叶）；节假日 h(t) = 一组指示函数
        </div>
      </div>

      <div class="section">
        <h2>3. 代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：调节 Prophet 参数</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "pr-change", label: "changepoint_prior_scale", min: 0.001, max: 0.5, step: 0.001, value: 0.05, format: (v) => parseFloat(v).toFixed(3) })}
              ${MCH.slider({ id: "pr-season", label: "seasonality_prior_scale", min: 0.01, max: 20, step: 0.1, value: 10 })}
              ${MCH.slider({ id: "pr-season-f", label: "fourier_order", min: 3, max: 30, step: 1, value: 10 })}
            </div>
            <div id="pr-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-prophet" style="height:340px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 何时用经典 vs 深度时序</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>首选</th><th>理由</th></tr></thead>
          <tbody>
            <tr><td>单序列 &lt; 1000 点</td><td><b>ARIMA / ETS</b></td><td>统计方法小数据更稳</td></tr>
            <tr><td>业务有明显节假日 / 活动</td><td><b>Prophet</b></td><td>内建节假日支持</td></tr>
            <tr><td>多序列联合训练 (&gt;1000 序列)</td><td>深度时序（DeepAR/PatchTST）</td><td>共享参数，冷启友好</td></tr>
            <tr><td>多变量 + 复杂协变量</td><td>深度时序（TFT）</td><td>非线性交互</td></tr>
            <tr><td>高频/分钟级长序列</td><td>深度 + 基座</td><td>长上下文必须</td></tr>
            <tr><td>需要快速 baseline</td><td><b>Prophet</b> 或 NaiveForecaster</td><td>1 分钟出结果</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ts_classical").pros, MCH.getById("ts_classical").cons, MCH.getById("ts_classical").best_for)}
      </div>
    `;
  },

  mount() {
    const chart = MCH.echart(document.getElementById("chart-prophet"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "Day" },
      yAxis: { type: "value" },
      series: [
        { name: "真实值", type: "line", smooth: true, showSymbol: false, color: "#94a3b8", data: [] },
        { name: "Prophet 预测", type: "line", smooth: true, showSymbol: false, color: "#f59e0b", lineStyle: { width: 3 }, data: [] },
        { name: "预测区间", type: "line", smooth: true, showSymbol: false, color: "#fef3c7", areaStyle: { color: "#fef3c7", opacity: 0.5 }, data: [] },
      ],
    });
    const update = () => {
      const cp = parseFloat(document.getElementById("pr-change").value);
      const sp = parseFloat(document.getElementById("pr-season").value);
      const fo = parseInt(document.getElementById("pr-season-f").value);
      const N = 120;
      const xs = [...Array(N).keys()];
      // 真实：趋势 + 周周期 + 节日突起
      const truth = xs.map(t => {
        const trend = 50 + 0.5 * t;
        const weekly = 10 * Math.sin(2 * Math.PI * t / 7);
        const yearly = 15 * Math.sin(2 * Math.PI * t / 30);
        const holiday = (t === 30 || t === 60 || t === 90) ? 20 : 0;
        const noise = MCH.randn(1, t + 1)[0] * 2;
        return trend + weekly + yearly + holiday + noise;
      });
      // Prophet 拟合：季节 fourier 数量控制细节；changepoint scale 控制趋势灵活度
      const fitted = xs.map(t => {
        const trend = 50 + 0.5 * t * (1 + (cp - 0.05) * 0.5);
        // 傅里叶 fo 阶拟合周期（模拟）
        let weekly = 0;
        for (let k = 1; k <= Math.min(fo, 7); k++) weekly += (10 / k) * Math.sin(2 * Math.PI * k * t / 7);
        let yearly = 0;
        for (let k = 1; k <= Math.min(fo, 5); k++) yearly += (15 / k) * Math.sin(2 * Math.PI * k * t / 30);
        const holiday = (t === 30 || t === 60 || t === 90) ? 18 * Math.min(1, sp) : 0;
        return trend + weekly * Math.min(1, sp / 10) + yearly * Math.min(1, sp / 10) + holiday;
      });
      chart.setOption({
        series: [
          { data: xs.map((t, i) => [t, truth[i]]) },
          { data: xs.map((t, i) => [t, fitted[i]]) },
          { data: xs.map((t, i) => [t, fitted[i] + 5]) },
        ],
      });
      const mse = truth.reduce((s, v, i) => s + Math.pow(v - fitted[i], 2), 0) / N;
      document.getElementById("pr-info").innerHTML = `
        <b>MSE</b> = <span style="color:#f59e0b;font-weight:700;">${mse.toFixed(2)}</span><br/>
        changepoint_prior_scale=${cp.toFixed(3)}: ${cp > 0.2 ? '<span style="color:#ef4444;">过大 → 过拟合趋势</span>' : cp < 0.01 ? '<span style="color:#f59e0b;">过小 → 欠拟合</span>' : '<span style="color:#10b981;">合适</span>'}<br/>
        seasonality_prior_scale=${sp}: 控制周期强度（默认 10）<br/>
        <span style="color:#64748b;">Prophet 对 changepoint 超参非常敏感；节假日周期建议从默认开始。</span>
      `;
    };
    ["pr-change", "pr-season", "pr-season-f"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        const v = e.target.value;
        document.getElementById(id + "-val").textContent = id === "pr-change" ? parseFloat(v).toFixed(3) : v;
        update();
      });
    });
    update();
  },
});
