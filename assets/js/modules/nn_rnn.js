/* 模块：RNN / LSTM / GRU */
MCH.register("nn_rnn", {
  render() {
    const code = `# RNN / LSTM / GRU 三兄弟实现对比
import torch, torch.nn as nn

# ---------- 原始 RNN ----------
class VanillaRNN(nn.Module):
    """h_t = tanh(W_xh · x_t + W_hh · h_{t-1} + b)
       梯度问题：反向传播时 ∂L/∂h_t 通过 W_hh^t 累积，
       当 W_hh 最大特征值 >1 → 爆炸，<1 → 消失"""
    def __init__(self, in_dim, hid_dim):
        self.W_xh = nn.Linear(in_dim, hid_dim, bias=False)
        self.W_hh = nn.Linear(hid_dim, hid_dim, bias=True)
    def forward(self, x):          # x: (B, T, in_dim)
        B, T, _ = x.shape
        h = torch.zeros(B, self.W_hh.out_features, device=x.device)
        outs = []
        for t in range(T):
            h = torch.tanh(self.W_xh(x[:, t]) + self.W_hh(h))
            outs.append(h)
        return torch.stack(outs, dim=1)  # (B, T, hid_dim)


# ---------- LSTM：3 个门 + cell state ----------
# Hochreiter & Schmidhuber 1997
class LSTMCell(nn.Module):
    """i_t = σ(W_i · [x, h])   # input gate: 多少新信息写入 cell
       f_t = σ(W_f · [x, h])   # forget gate: 多少旧信息保留
       o_t = σ(W_o · [x, h])   # output gate: 多少 cell 信息输出
       g_t = tanh(W_g · [x, h]) # 候选记忆
       c_t = f_t * c_{t-1} + i_t * g_t   # 关键：线性累加 + 选择性遗忘
       h_t = o_t * tanh(c_t)"""
    def __init__(self, in_dim, hid):
        self.gates = nn.Linear(in_dim + hid, 4 * hid)     # 一次算 4 个门
    def forward(self, x, state):
        h, c = state
        g = self.gates(torch.cat([x, h], -1))
        i, f, o, cand = g.chunk(4, dim=-1)
        i, f, o = [torch.sigmoid(t) for t in (i, f, o)]
        cand = torch.tanh(cand)
        c = f * c + i * cand          # cell state 是梯度高速公路
        h = o * torch.tanh(c)
        return h, (h, c)


# ---------- GRU：简化版 LSTM，2 个门 ----------
# Cho et al. 2014
class GRUCell(nn.Module):
    """r_t = σ(W_r · [x, h])        # reset gate
       z_t = σ(W_z · [x, h])        # update gate (融合 input 和 forget)
       h'_t = tanh(W · [x, r_t * h])
       h_t = (1 - z_t) * h + z_t * h'_t"""
    def __init__(self, in_dim, hid):
        self.x2h = nn.Linear(in_dim, 3 * hid)
        self.h2h = nn.Linear(hid, 3 * hid)
    def forward(self, x, h):
        gx = self.x2h(x); gh = self.h2h(h)
        i_r, i_z, i_n = gx.chunk(3, -1)
        h_r, h_z, h_n = gh.chunk(3, -1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)       # r gate 控制是否用过去 hidden
        return (1 - z) * n + z * h


# 实际使用：PyTorch 内置的已经 cuDNN 加速
lstm = nn.LSTM(input_size=128, hidden_size=256, num_layers=2,
                bidirectional=True, batch_first=True, dropout=0.1)
out, (h_n, c_n) = lstm(x)        # out: (B, T, 2·H)`;

    return `
      ${MCH.hero({ icon: "↻", name: "RNN / LSTM / GRU — 循环神经网络家族", en: "Recurrent Neural Networks · Hochreiter 1997", tags: ["时序", "门控机制", "Cell State", "Bidirectional"], meta: ["◈ 变长序列", "⚡ 参数共享", "◇ Transformer 之前的王者"] })}

      ${MCH.versionSection("nn_rnn")}

      <div class="section">
        <h2>1. 为什么需要"循环"？</h2>
        <p class="text-sm text-slate-600">普通 MLP 只能处理<b>定长向量</b>。对于文本 / 语音 / 交易序列这类 <b>变长时序</b>数据，我们希望有一个"记忆"随序列传递。RNN 通过引入<b>隐状态 h_t</b> 实现这一点：</p>
        <div class="formula-block">
          $$ h_t = f(x_t, h_{t-1}), \\quad h_0 = \\mathbf{0} $$
          每个 time step 的输出依赖于<b>当前输入</b>和<b>之前所有输入</b>（通过 h 传递）。
        </div>
        ${MCH.info(`
          <b>核心痛点：梯度消失 / 爆炸。</b>
          反向传播穿越 T 步时，梯度要乘 W_hh 的 T 次方。
          若 W_hh 特征值 &lt; 1 → 指数衰减（消失）；&gt; 1 → 指数增长（爆炸）。
          LSTM 通过<b>加法累加的 cell state</b>解决此问题；GRU 是 LSTM 的轻量化版本。
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 三者对比</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>Vanilla RNN</th><th>LSTM</th><th>GRU</th></tr></thead>
          <tbody>
            <tr><td>参数量（per cell）</td><td>1× WH²</td><td>4× WH²</td><td>3× WH²</td></tr>
            <tr><td>门结构</td><td>-</td><td>input / forget / output</td><td>reset / update</td></tr>
            <tr><td>Cell State</td><td>-</td><td>✓ c_t (长期记忆)</td><td>-（与 h 合并）</td></tr>
            <tr><td>梯度稳定性</td><td>❌ 消失/爆炸</td><td>✓ 很好</td><td>✓ 较好</td></tr>
            <tr><td>长程依赖（100+）</td><td>❌</td><td>✓</td><td>⚠ 稍逊</td></tr>
            <tr><td>训练速度</td><td>最快</td><td>最慢</td><td>中等</td></tr>
            <tr><td>精度（小/中数据）</td><td>低</td><td>高</td><td>接近 LSTM</td></tr>
            <tr><td>现代地位</td><td>基本不用</td><td>时序 baseline</td><td>LSTM 轻量版</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码对比</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：梯度消失 / 爆炸可视化</h2>
        <div class="grid-2">
          <div>
            <p class="text-sm text-slate-600">反向传播梯度通过 T 步 W_hh 传递 → 梯度范数 = spectral_radius^T</p>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "rnn-T", label: "序列长度 T", min: 10, max: 500, step: 10, value: 100 })}
              ${MCH.slider({ id: "rnn-sr", label: "W_hh 最大特征值 λ", min: 0.5, max: 1.5, step: 0.01, value: 1.0 })}
              ${MCH.slider({ id: "rnn-clip", label: "梯度裁剪阈值", min: 0, max: 10, step: 0.1, value: 0 })}
            </div>
            <div id="rnn-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-grad" style="height:340px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 交互：LSTM 门控机制动态</h2>
        <p class="text-sm text-slate-600">模拟一个时序信号（正弦 + 突变），观察 LSTM 各个门的开关状态如何跟随输入变化。</p>
        <div id="chart-lstm" style="height:400px;"></div>
      </div>

      <div class="section">
        <h2>6. 使用建议</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>推荐</th><th>备注</th></tr></thead>
          <tbody>
            <tr><td>通用时序分类 / 回归</td><td><b>LSTM / GRU</b></td><td>bidirectional + dropout=0.2</td></tr>
            <tr><td>短序列（&lt; 30 步）</td><td>GRU</td><td>比 LSTM 快 30%</td></tr>
            <tr><td>长程依赖（&gt; 100 步）</td><td>LSTM</td><td>GRU 对长程信息保留差些</td></tr>
            <tr><td>流式 / 在线推理</td><td>GRU / LSTM</td><td>可维护 hidden state 增量更新</td></tr>
            <tr><td>大规模 NLP</td><td>❌ RNN 家族</td><td>用 Transformer</td></tr>
            <tr><td>音频 / 语音</td><td>LSTM / Conv1D</td><td>仍是语音合成常用模块</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("nn_rnn").pros, MCH.getById("nn_rnn").cons, MCH.getById("nn_rnn").best_for)}
      </div>
    `;
  },

  mount() {
    // Gradient viz
    const gChart = MCH.echart(document.getElementById("chart-grad"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0 },
      grid: { left: 70, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "反向传播步数" },
      yAxis: { type: "log", name: "梯度范数", logBase: 10 },
      series: [
        { name: "Vanilla RNN", type: "line", smooth: true, showSymbol: false, color: "#ef4444", lineStyle: { width: 3 }, data: [] },
        { name: "LSTM (cell state 保护)", type: "line", smooth: true, showSymbol: false, color: "#10b981", lineStyle: { width: 3 }, data: [] },
        { name: "裁剪后 RNN", type: "line", smooth: true, showSymbol: false, color: "#f59e0b", lineStyle: { width: 2, type: "dashed" }, data: [] },
      ],
    });
    const update = () => {
      const T = parseInt(document.getElementById("rnn-T").value);
      const lam = parseFloat(document.getElementById("rnn-sr").value);
      const clip = parseFloat(document.getElementById("rnn-clip").value);
      const ts = MCH.linspace(0, T, 120);
      const rnn = ts.map(t => Math.pow(lam, t));
      const lstm = ts.map(t => Math.pow(Math.max(0.995, Math.min(1.005, lam)), t));
      const clipped = ts.map(t => clip > 0 ? Math.min(clip, Math.pow(lam, t)) : Math.pow(lam, t));
      gChart.setOption({
        series: [
          { data: ts.map((t, i) => [t, Math.max(1e-20, rnn[i])]) },
          { data: ts.map((t, i) => [t, lstm[i]]) },
          { data: ts.map((t, i) => [t, Math.max(1e-20, clipped[i])]) },
        ],
      });
      const status = lam < 0.99 ? '<span style="color:#ef4444;">⚠️ 梯度消失</span>' : (lam > 1.01 ? '<span style="color:#ef4444;">💥 梯度爆炸</span>' : '<span style="color:#10b981;">✓ 稳定</span>');
      document.getElementById("rnn-info").innerHTML = `
        T=${T}, λ=${lam.toFixed(2)}: ${status}<br/>
        T 步后 RNN 梯度 ≈ <b>${Math.pow(lam, T).toExponential(2)}</b><br/>
        LSTM cell state 近似线性累加，实际 ≈ 1.0（高速通道）<br/>
        <span style="color:#64748b;">梯度裁剪只能缓解爆炸，不能解决消失 → 这就是 LSTM 诞生的意义。</span>
      `;
    };
    ["rnn-T", "rnn-sr", "rnn-clip"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();

    // LSTM gates viz
    const T = 100;
    const x = MCH.linspace(0, 10, T);
    const signal = x.map(t => Math.sin(t) + (t > 4 && t < 6 ? 2 : 0));  // 一个突变信号
    const cellState = [0];
    const forgetG = [], inputG = [], outputG = [];
    for (let i = 0; i < T; i++) {
      const inp = signal[i];
      // 启发式：突变时 forget 关闭，input 打开
      const isSpike = Math.abs(inp - (i > 0 ? signal[i - 1] : 0)) > 0.5;
      const f = isSpike ? 0.3 : 0.95;
      const g = Math.tanh(inp);
      const igate = 0.5 + 0.4 * Math.tanh(inp);
      const c = f * cellState[cellState.length - 1] + igate * g;
      cellState.push(c);
      const o = 0.5 + 0.4 * Math.abs(Math.tanh(c));
      forgetG.push(f); inputG.push(igate); outputG.push(o);
    }
    cellState.shift();
    MCH.echart(document.getElementById("chart-lstm"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0, data: ["输入 x_t", "Forget Gate f_t", "Input Gate i_t", "Output Gate o_t", "Cell State c_t"] },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "time" },
      yAxis: [
        { type: "value", name: "信号 / 状态", position: "left" },
        { type: "value", name: "门开关值 [0,1]", position: "right", min: 0, max: 1 },
      ],
      series: [
        { name: "输入 x_t", type: "line", smooth: true, showSymbol: false, yAxisIndex: 0, data: x.map((t, i) => [t, signal[i]]), color: "#94a3b8", lineStyle: { width: 2 } },
        { name: "Cell State c_t", type: "line", smooth: true, showSymbol: false, yAxisIndex: 0, data: x.map((t, i) => [t, cellState[i]]), color: "#7c3aed", lineStyle: { width: 3 } },
        { name: "Forget Gate f_t", type: "line", smooth: true, showSymbol: false, yAxisIndex: 1, data: x.map((t, i) => [t, forgetG[i]]), color: "#ef4444", lineStyle: { width: 2 } },
        { name: "Input Gate i_t", type: "line", smooth: true, showSymbol: false, yAxisIndex: 1, data: x.map((t, i) => [t, inputG[i]]), color: "#10b981", lineStyle: { width: 2 } },
        { name: "Output Gate o_t", type: "line", smooth: true, showSymbol: false, yAxisIndex: 1, data: x.map((t, i) => [t, outputG[i]]), color: "#f59e0b", lineStyle: { width: 2 } },
      ],
    });
  },
});
