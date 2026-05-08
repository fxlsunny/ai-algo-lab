/* 模块：CNN */
MCH.register("nn_cnn", {
  render() {
    const code = `# CNN 核心：卷积 + 池化 + 激活
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        # Conv2d(in, out, kernel, stride, padding)
        #   权重形状 (out_c, in_c, kH, kW) —— 参数共享核心
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)    # 224 → 224
        self.pool1 = nn.MaxPool2d(2)                                 # 224 → 112
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)   # 112 → 112
        self.pool2 = nn.MaxPool2d(2)                                 #  → 56
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 → (B, 128, 1, 1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x)); x = self.pool1(x)
        x = F.relu(self.conv2(x)); x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.gap(x).flatten(1)
        return self.fc(x)

# 关键概念：
# - Kernel (核)：一个小矩阵在图像上滑动做点积，学习局部模式
# - Stride (步长)：滑动步长，stride=2 下采样 2×
# - Padding (填充)：边缘补 0 保持尺寸
# - Channels (通道)：输入/输出特征图数 = 学习的模式数
# - 感受野 Receptive Field：随深度指数增长`;

    return `
      ${MCH.hero({ icon: "◰", name: "CNN — 卷积神经网络", en: "Convolutional Neural Network", tags: ["参数共享", "局部连接", "平移不变", "感受野层级"], meta: ["◈ LeNet/AlexNet/ResNet 基石", "⚡ 图像 SOTA 核心"] })}

      ${MCH.versionSection("nn_cnn")}

      <div class="section">
        <h2>1. 为什么是卷积而不是全连接？</h2>
        <p class="text-sm text-slate-600">224×224×3 的图像直接接全连接到 1024 单元 → <b>1.5 亿参数</b>，且无视平移不变性。卷积的三大先验利用了图像的结构：</p>
        <div class="grid-3 mt-3">
          <div class="card"><h3 class="text-violet-700 font-bold text-sm">① 局部连接</h3><p class="text-xs text-slate-600 mt-1">每个神经元只关注局部 k×k 窗口，贴合"物体的局部特征（边缘/纹理）"先于整体出现。</p></div>
          <div class="card"><h3 class="text-violet-700 font-bold text-sm">② 参数共享</h3><p class="text-xs text-slate-600 mt-1">同一 kernel 在整幅图上滑动，<b>参数量 ÷ H·W</b>。利用了平移不变性。</p></div>
          <div class="card"><h3 class="text-violet-700 font-bold text-sm">③ 层级感受野</h3><p class="text-xs text-slate-600 mt-1">浅层学边缘 → 中层学纹理 → 深层学部件 → 更深层学物体。</p></div>
        </div>
      </div>

      <div class="section">
        <h2>2. 代码解读</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互可视化</h2>

        <h3>· 卷积输出尺寸计算</h3>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "cnn-in", label: "输入 H (px)", min: 32, max: 512, step: 8, value: 224 })}
              ${MCH.slider({ id: "cnn-k", label: "kernel k", min: 1, max: 11, step: 2, value: 3 })}
              ${MCH.slider({ id: "cnn-s", label: "stride s", min: 1, max: 4, step: 1, value: 1 })}
              ${MCH.slider({ id: "cnn-p", label: "padding p", min: 0, max: 5, step: 1, value: 1 })}
              ${MCH.slider({ id: "cnn-cin", label: "输入通道 Cin", min: 1, max: 512, step: 1, value: 3 })}
              ${MCH.slider({ id: "cnn-cout", label: "输出通道 Cout", min: 1, max: 1024, step: 1, value: 64 })}
            </div>
            <div id="cnn-info" class="card mt-3"></div>
          </div>
          <div id="chart-rf" style="height:380px;"></div>
        </div>

        <h3 style="margin-top:18px;">· 常见卷积核（边缘 / 模糊 / 锐化）可视化</h3>
        <p class="text-xs text-slate-500">3×3 kernel 直接应用到一张模拟图片上。观察不同核学到的模式。</p>
        <div class="grid-4 mt-3" id="kernel-gallery"></div>
      </div>

      <div class="section">
        <h2>4. 经典 CNN 架构演进</h2>
        <table class="table">
          <thead><tr><th>年份</th><th>模型</th><th>关键创新</th><th>ImageNet Top-5 误差</th></tr></thead>
          <tbody>
            <tr><td>1998</td><td>LeNet-5</td><td>首个真正的 CNN（手写数字）</td><td>-</td></tr>
            <tr><td>2012</td><td>AlexNet</td><td>ReLU + Dropout + GPU 训练，开启深度学习</td><td>15.3%</td></tr>
            <tr><td>2014</td><td>VGG</td><td>堆叠 3×3 小 kernel，统一架构</td><td>7.3%</td></tr>
            <tr><td>2014</td><td>GoogLeNet / Inception</td><td>并行多尺度 kernel，1×1 卷积降维</td><td>6.7%</td></tr>
            <tr><td><b>2015</b></td><td><b>ResNet</b></td><td><b>残差连接</b>，训练到 152 层不退化</td><td><b>3.57%</b></td></tr>
            <tr><td>2017</td><td>DenseNet</td><td>层与层之间全连接，特征复用</td><td>3.46%</td></tr>
            <tr><td>2019</td><td>EfficientNet</td><td>复合缩放（深度/宽度/分辨率一起）</td><td>2.5%</td></tr>
            <tr><td>2020</td><td>ViT (Vision Transformer)</td><td>用 Attention 取代卷积，大数据时超越 CNN</td><td>2.3%</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("nn_cnn").pros,
          MCH.getById("nn_cnn").cons,
          MCH.getById("nn_cnn").best_for,
        )}
      </div>
    `;
  },

  mount() {
    // Output size calc
    const update = () => {
      const H = parseInt(document.getElementById("cnn-in").value);
      const k = parseInt(document.getElementById("cnn-k").value);
      const s = parseInt(document.getElementById("cnn-s").value);
      const p = parseInt(document.getElementById("cnn-p").value);
      const cin = parseInt(document.getElementById("cnn-cin").value);
      const cout = parseInt(document.getElementById("cnn-cout").value);
      const Hout = Math.floor((H + 2 * p - k) / s) + 1;
      const params = cout * cin * k * k + cout;
      const flops = Hout * Hout * cout * cin * k * k;
      document.getElementById("cnn-info").innerHTML = `
        <div class="text-xs text-slate-500">输出尺寸公式：H_out = ⌊(H + 2p - k) / s⌋ + 1</div>
        <div class="grid-3 mt-3">
          <div><div class="text-xs text-slate-500">输出 H×W</div><div class="text-xl font-bold text-violet-700">${Hout} × ${Hout}</div></div>
          <div><div class="text-xs text-slate-500">参数量</div><div class="text-xl font-bold text-violet-700">${(params / 1e3).toFixed(1)} K</div></div>
          <div><div class="text-xs text-slate-500">单张 FLOPs</div><div class="text-xl font-bold text-violet-700">${(flops / 1e6).toFixed(2)} M</div></div>
        </div>
        <div class="text-xs text-slate-500 mt-2">对比全连接：FC ${H}×${H}×${cin} → ${cout} 需要 <b>${(H * H * cin * cout / 1e6).toFixed(1)} M</b> 参数 = Conv 的 <b>${(H * H / (k * k)).toFixed(0)}×</b> — 这就是参数共享的力量。</div>
      `;
    };
    ["cnn-in", "cnn-k", "cnn-s", "cnn-p", "cnn-cin", "cnn-cout"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        update();
      });
    });
    update();

    // Receptive field growth
    MCH.echart(document.getElementById("chart-rf"), {
      title: { text: "感受野随深度指数增长", left: "center", textStyle: { fontSize: 12 } },
      tooltip: { trigger: "axis" },
      legend: { top: 30 },
      grid: { left: 60, right: 30, top: 70, bottom: 40 },
      xAxis: { type: "value", name: "层数" },
      yAxis: { type: "value", name: "有效感受野 (px)" },
      series: [
        { name: "3×3 conv + stride 1", type: "line", smooth: true, showSymbol: true, color: "#7c3aed", data: [...Array(20).keys()].map(i => [i + 1, 1 + 2 * i]) },
        { name: "3×3 + pool×2 每 2 层", type: "line", smooth: true, showSymbol: true, color: "#f59e0b", data: [...Array(20).keys()].map(i => [i + 1, Math.round(Math.pow(2, Math.floor(i / 2)) * 3)]) },
        { name: "Dilated conv (rate 2^l)", type: "line", smooth: true, showSymbol: true, color: "#10b981", data: [...Array(20).keys()].map(i => [i + 1, Math.pow(2, i + 1) - 1]) },
      ],
    });

    // Kernel gallery
    const gallery = document.getElementById("kernel-gallery");
    const kernels = {
      "原图": [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
      "水平边缘": [[-1, -1, -1], [0, 0, 0], [1, 1, 1]],
      "垂直边缘": [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]],
      "锐化": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
      "模糊": [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
      "Sobel-X": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
      "Laplacian": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
      "浮雕": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    };
    // generate a test image: 圆 + 线
    const N = 24;
    const img = [];
    for (let i = 0; i < N; i++) {
      const row = [];
      for (let j = 0; j < N; j++) {
        const cx = N / 2, cy = N / 2;
        const inCircle = ((i - cx) ** 2 + (j - cy) ** 2) < 60 ? 0.8 : 0;
        const line = (i === 5 || i === 18) ? 0.7 : 0;
        row.push(Math.min(1, inCircle + line));
      }
      img.push(row);
    }
    Object.entries(kernels).forEach(([name, k]) => {
      const div = document.createElement("div");
      div.className = "card";
      div.style.padding = "10px";
      div.innerHTML = `<div class="text-xs font-semibold text-slate-700 mb-2">${name}</div><div id="kc-${name}" style="height:140px;"></div>`;
      gallery.appendChild(div);
      // apply conv
      const out = [];
      for (let i = 1; i < N - 1; i++) {
        for (let j = 1; j < N - 1; j++) {
          let s = 0;
          for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) s += img[i + di][j + dj] * k[di + 1][dj + 1];
          out.push([j, i, s]);
        }
      }
      const vals = out.map(p => p[2]);
      const min = Math.min(...vals), max = Math.max(...vals);
      MCH.echart(document.getElementById(`kc-${name}`), {
        grid: { left: 5, right: 5, top: 5, bottom: 5 },
        xAxis: { show: false, type: "category", data: [...Array(N - 2).keys()] },
        yAxis: { show: false, type: "category", data: [...Array(N - 2).keys()], inverse: true },
        visualMap: { show: false, min, max, inRange: { color: ["#f1f5f9", "#4f46e5", "#1e1b4b"] } },
        series: [{ type: "heatmap", data: out, progressive: 0 }],
      });
    });
  },
});
