/* 模块：MLP / DNN */
MCH.register("nn_mlp", {
  render() {
    const code = `# MLP / DNN：层与层之间的全连接 + 激活函数
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation="gelu", dropout=0.1):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.LayerNorm(h),              # 稳定训练（比 BN 更适合小 batch）
                self._act(activation),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def _act(self, name):
        return {"relu": nn.ReLU(), "gelu": nn.GELU(), "swish": nn.SiLU(),
                "tanh": nn.Tanh(), "leaky": nn.LeakyReLU(0.1)}[name]

    def forward(self, x):
        return self.net(x)


# 反向传播 = 链式求导
# ∂L/∂W_l = (δ_l) · h_{l-1}^T, 其中 δ_l = ∂L/∂z_l
# PyTorch 的 autograd 已自动处理这一切`;

    return `
      ${MCH.hero({ icon: "▦", name: "MLP / DNN — 多层感知机", en: "Multi-Layer Perceptron · Dense Network", tags: ["全连接", "反向传播", "激活函数", "万能近似"], meta: ["◈ W·x + b 堆叠", "⚡ GPU 矩阵乘法"] })}

      ${MCH.versionSection("nn_mlp")}

      <div class="section">
        <h2>1. 为什么 MLP 能学非线性？</h2>
        <p class="text-sm text-slate-600">单层 MLP 是线性模型 $y = Wx + b$，无法分离异或 XOR。关键在于<b>非线性激活函数</b>：层与层之间必须插入非线性，否则堆多少层仍然是线性。</p>
        <div class="formula-block">
          $$ h_l = \\sigma(W_l h_{l-1} + b_l), \\quad h_0 = x, \\quad y = W_L h_{L-1} $$
          <b>万能近似定理 (Cybenko, 1989)</b>：单隐层 + sigmoid MLP 可任意逼近紧集上的连续函数；深层网络更参数高效。
        </div>
      </div>

      <div class="section">
        <h2>2. 代码解读</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互：激活函数对比</h2>
        <div class="grid-2">
          <div>
            <h3>· 常见激活函数</h3>
            <div id="chart-act" style="height:300px;"></div>
          </div>
          <div>
            <h3>· 导数对比（反向传播关键）</h3>
            <div id="chart-act-d" style="height:300px;"></div>
            <p class="text-xs text-slate-500 mt-2">注意：Sigmoid / Tanh 两端梯度→0 导致"梯度消失"；ReLU 系激活函数在负半轴死亡（但正半轴梯度恒为 1，解决了消失问题）。</p>
          </div>
        </div>

        <h3 style="margin-top:18px;">· 网络宽度 / 深度 vs 表达力</h3>
        <div class="ctrl-panel">
          <div class="grid-3">
            <div>${MCH.slider({ id: "mlp-L", label: "层数 L", min: 1, max: 10, step: 1, value: 3 })}</div>
            <div>${MCH.slider({ id: "mlp-W", label: "宽度 W", min: 8, max: 512, step: 8, value: 64 })}</div>
            <div>${MCH.slider({ id: "mlp-in", label: "输入维度 d", min: 2, max: 1024, step: 2, value: 128 })}</div>
          </div>
        </div>
        <div id="mlp-info" class="card mt-3"></div>
      </div>

      <div class="section">
        <h2>4. 初始化 & 正则技巧</h2>
        <table class="table">
          <thead><tr><th>技巧</th><th>作用</th><th>推荐场景</th></tr></thead>
          <tbody>
            <tr><td>Kaiming / He Init</td><td>ReLU 网络的方差保持初始化</td><td>默认：<code>kaiming_uniform</code></td></tr>
            <tr><td>Xavier Init</td><td>tanh/sigmoid 的方差保持</td><td>tanh 激活时</td></tr>
            <tr><td>BatchNorm / LayerNorm</td><td>稳定中间分布</td><td>LN 对小 batch 友好（Transformer 标配）</td></tr>
            <tr><td>Dropout (p=0.1-0.5)</td><td>随机屏蔽神经元</td><td>过拟合时</td></tr>
            <tr><td>Weight Decay (AdamW)</td><td>L2 正则的解耦版</td><td>几乎所有大模型都用</td></tr>
            <tr><td>Gradient Clipping</td><td>梯度范数截断</td><td>RNN/Transformer 防爆炸</td></tr>
            <tr><td>残差连接 (ResNet)</td><td>y = x + F(x) 训练深层网络</td><td>&gt; 10 层必用</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("nn_mlp").pros,
          MCH.getById("nn_mlp").cons,
          MCH.getById("nn_mlp").best_for,
        )}
      </div>
    `;
  },

  mount() {
    const xs = MCH.linspace(-5, 5, 200);
    const actFns = {
      "Sigmoid": x => 1 / (1 + Math.exp(-x)),
      "Tanh": x => Math.tanh(x),
      "ReLU": x => Math.max(0, x),
      "LeakyReLU (0.1)": x => x > 0 ? x : 0.1 * x,
      "GELU": x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
      "Swish / SiLU": x => x / (1 + Math.exp(-x)),
    };
    const dFns = {
      "Sigmoid": x => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); },
      "Tanh": x => 1 - Math.tanh(x) ** 2,
      "ReLU": x => x > 0 ? 1 : 0,
      "LeakyReLU (0.1)": x => x > 0 ? 1 : 0.1,
      "GELU": x => { const h = 0.044715 * x ** 3; const u = Math.sqrt(2 / Math.PI) * (x + h); const tanhu = Math.tanh(u); return 0.5 * (1 + tanhu) + 0.5 * x * (1 - tanhu ** 2) * Math.sqrt(2 / Math.PI) * (1 + 3 * 0.044715 * x ** 2); },
      "Swish / SiLU": x => { const s = 1 / (1 + Math.exp(-x)); return s + x * s * (1 - s); },
    };
    const colors = ["#4f46e5", "#7c3aed", "#10b981", "#f59e0b", "#ef4444", "#06b6d4"];
    const buildSeries = (fns) => Object.entries(fns).map(([name, fn], i) => ({ name, type: "line", showSymbol: false, smooth: true, color: colors[i], lineStyle: { width: 2 }, data: xs.map(x => [x, fn(x)]) }));
    MCH.echart(document.getElementById("chart-act"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0, type: "scroll" },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "x" },
      yAxis: { type: "value", name: "σ(x)" },
      series: buildSeries(actFns),
    });
    MCH.echart(document.getElementById("chart-act-d"), {
      tooltip: { trigger: "axis" },
      legend: { top: 0, type: "scroll" },
      grid: { left: 50, right: 20, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "x" },
      yAxis: { type: "value", name: "σ'(x)" },
      series: buildSeries(dFns),
    });

    // Param count info
    const updateInfo = () => {
      const L = parseInt(document.getElementById("mlp-L").value);
      const W = parseInt(document.getElementById("mlp-W").value);
      const d = parseInt(document.getElementById("mlp-in").value);
      // Input layer d -> W, hidden L-1 W -> W, output W -> 1
      let params = d * W + W;
      for (let i = 0; i < L - 1; i++) params += W * W + W;
      params += W * 1 + 1;
      const flops = params; // approximate
      document.getElementById("mlp-info").innerHTML = `
        <div class="grid-4">
          <div><div class="text-xs text-slate-500">参数总数</div><div class="text-xl font-bold text-indigo-700">${(params / 1e3).toFixed(1)} K</div></div>
          <div><div class="text-xs text-slate-500">Forward FLOPs/sample</div><div class="text-xl font-bold text-indigo-700">${(flops / 1e3).toFixed(1)} K</div></div>
          <div><div class="text-xs text-slate-500">内存（FP32）</div><div class="text-xl font-bold text-indigo-700">${(params * 4 / 1024 / 1024).toFixed(2)} MB</div></div>
          <div><div class="text-xs text-slate-500">理论表达力</div><div class="text-xl font-bold text-indigo-700">${L === 1 ? "线性映射" : "万能近似"}</div></div>
        </div>
        <p class="text-xs text-slate-500 mt-2">注：宽度 W 决定表达力上限；深度 L 决定特征抽象层级。通常"深而窄"比"浅而宽"更高效。</p>
      `;
    };
    ["mlp-L", "mlp-W", "mlp-in"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        updateInfo();
      });
    });
    updateInfo();
  },
});
