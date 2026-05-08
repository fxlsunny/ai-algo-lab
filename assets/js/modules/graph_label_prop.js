/* 模块：Label Propagation 标签传播 */
MCH.register("graph_label_prop", {
  render() {
    const code = `# 标签传播 (Raghavan et al., 2007)
def label_propagation(G, max_iter=20, seed=42):
    # 1) 初始化：每个节点一个独立标签
    labels = {v: v for v in G.nodes}

    for it in range(max_iter):
        # 2) 按随机顺序遍历节点
        order = np.random.permutation(list(G.nodes))
        changed = 0

        for v in order:
            # 3) 统计邻居的标签频率
            counts = Counter(labels[u] for u in G.neighbors(v))
            if not counts: continue

            # 4) 取最高频标签（冲突时随机选）
            max_freq = max(counts.values())
            top = [lab for lab, f in counts.items() if f == max_freq]
            new_label = random.choice(top)

            if labels[v] != new_label:
                labels[v] = new_label
                changed += 1

        if changed == 0:
            break                        # 稳定 → 收敛

    return labels

# 半监督版本：种子节点的标签不更新
def semi_supervised_lp(G, seed_labels, max_iter=20):
    labels = {v: seed_labels.get(v, v) for v in G.nodes}
    for it in range(max_iter):
        for v in G.nodes:
            if v in seed_labels: continue     # 锁定种子
            counts = Counter(labels[u] for u in G.neighbors(v))
            labels[v] = counts.most_common(1)[0][0]
    return labels`;

    return `
      ${MCH.hero({ icon: "⟹", name: "Label Propagation — 标签传播", en: "Label Propagation Algorithm (LPA)", tags: ["半监督", "消息传递", "O(E·iter)", "无参数"], meta: ["◈ Raghavan 2007", "⚡ TB 级图可用"] })}

      ${MCH.versionSection("graph_label_prop")}

      <div class="section">
        <h2>1. 核心思想：用邻居投票</h2>
        <p class="text-sm text-slate-600">Label Propagation 的规则极其简洁：<b>"每个节点每轮都取邻居出现频率最高的标签"</b>。经过若干轮，相似连接密的节点会趋同 → 自然形成社区。</p>
        <div class="formula-block">
          $$ \\text{label}(v)^{(t+1)} = \\text{argmax}_c \\Big| \\{ u \\in N(v) : \\text{label}(u)^{(t)} = c \\} \\Big| $$
        </div>
        ${MCH.info(`
          <b>LPA vs Louvain</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li>LPA：<b>极简</b>，每轮 O(E)，无参数，但结果有随机性，可能振荡；</li>
            <li>Louvain：<b>基于模块度优化</b>，收敛稳定，但复杂度较高；</li>
            <li>LPA 常作为超大图的 <b>"快速初版"</b>，Louvain 用于<b>精细划分</b>。</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 半监督 LPA — 标签从种子扩散</h2>
        <p class="text-sm text-slate-600">给一小部分节点打上真实标签（"种子"），让标签沿着边扩散到未标注节点。这是<b>图上的半监督学习</b>经典方法。</p>
        <div class="mermaid">
flowchart LR
    S1[种子节点<br/>已知标签 A] -.扩散.-> U1[未标注邻居]
    U1 -.扩散.-> U2[2-hop 邻居]
    S2[种子节点<br/>已知标签 B] -.扩散.-> U3[未标注邻居]
    U2 -. 冲突点<br/>按多数投票 .- U3
        </div>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：LPA 实时传播</h2>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "lp-iter", label: "当前迭代轮 t", min: 0, max: 15, step: 1, value: 0 })}
              <label class="text-xs mt-2 flex items-center gap-2">
                <input type="checkbox" id="lp-semi" /> 半监督模式（锁定 2 个种子节点）
              </label>
              <div class="flex gap-2 mt-3">
                <button id="lp-step" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 下一步</button>
                <button id="lp-play" class="text-xs px-3 py-1 bg-indigo-600 text-white rounded">▶▶ 播放</button>
                <button id="lp-reset" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">↺ 重置</button>
              </div>
            </div>
            <div id="lp-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-lp" style="height:420px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. 应用场景</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>具体做法</th></tr></thead>
          <tbody>
            <tr><td>黑产传染</td><td>种子 = 已判定黑商户；通过同主体/同超管/同地址图扩散 → 标记风险邻居</td></tr>
            <tr><td>推荐 Topic 扩散</td><td>种子 = 热门 Topic 用户；扩散到相似兴趣用户</td></tr>
            <tr><td>新类别快速标注</td><td>小样本标一批，用 LPA 生成伪标签供模型训练</td></tr>
            <tr><td>大图初步聚类</td><td>TB 级图用 LPA 分桶，再对每桶精细分析</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("graph_label_prop").pros, MCH.getById("graph_label_prop").cons, MCH.getById("graph_label_prop").best_for)}
      </div>
    `;
  },

  mount() {
    // 3 个 cluster 的图 (18 nodes)
    const N = 18;
    const pos = [];
    // cluster 1: 左下
    for (let i = 0; i < 6; i++) pos.push([-80 + (i % 3) * 40 - 20, -80 + Math.floor(i / 3) * 40]);
    // cluster 2: 右上
    for (let i = 0; i < 6; i++) pos.push([60 + (i % 3) * 40, -80 + Math.floor(i / 3) * 40]);
    // cluster 3: 中下
    for (let i = 0; i < 6; i++) pos.push([-30 + (i % 3) * 40, 60 + Math.floor(i / 3) * 40]);
    const edges = [
      [0,1],[0,2],[1,3],[2,3],[3,4],[4,5],[2,5],[1,4], // c1
      [6,7],[6,8],[7,9],[8,9],[9,10],[10,11],[8,11],[7,10],// c2
      [12,13],[13,14],[12,15],[14,16],[15,17],[13,17],[14,15], // c3
      [5,6],[11,12],[4,17], // inter-cluster
    ];
    const palette = ["#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#06b6d4", "#ec4899", "#84cc16"];

    let labels = [...Array(N).keys()];

    const step = (semi) => {
      const seedIds = new Set([0, 9]);  // 种子
      const newLabels = [...labels];
      // 随机顺序
      const order = [...Array(N).keys()].sort(() => Math.random() - 0.5);
      for (const v of order) {
        if (semi && seedIds.has(v)) continue;
        const neigh = edges.filter(([u1, u2]) => u1 === v || u2 === v).map(e => e[0] === v ? e[1] : e[0]);
        if (neigh.length === 0) continue;
        const cnt = {};
        neigh.forEach(n => { cnt[newLabels[n]] = (cnt[newLabels[n]] || 0) + 1; });
        const max = Math.max(...Object.values(cnt));
        const tops = Object.entries(cnt).filter(([, c]) => c === max).map(([k]) => +k);
        newLabels[v] = tops[Math.floor(Math.random() * tops.length)];
      }
      labels = newLabels;
    };

    const render = () => {
      const semi = document.getElementById("lp-semi").checked;
      const unique = [...new Set(labels)];
      const nodes = pos.map((p, i) => ({
        name: `N${i}`,
        value: p,
        symbolSize: (semi && (i === 0 || i === 9)) ? 30 : 22,
        itemStyle: {
          color: palette[unique.indexOf(labels[i]) % palette.length],
          borderColor: (semi && (i === 0 || i === 9)) ? "#000" : "transparent",
          borderWidth: (semi && (i === 0 || i === 9)) ? 2 : 0,
        },
        label: { show: true, formatter: `${i}`, color: "white", fontWeight: 700, fontSize: 10 },
      }));
      const links = edges.map(([u, v]) => ({
        source: `N${u}`, target: `N${v}`,
        lineStyle: { color: labels[u] === labels[v] ? palette[unique.indexOf(labels[u]) % palette.length] : "#cbd5e1", width: labels[u] === labels[v] ? 2 : 1, opacity: 0.6 },
      }));
      MCH.echart(document.getElementById("chart-lp"), {
        tooltip: {},
        xAxis: { show: false, min: -150, max: 150 },
        yAxis: { show: false, min: -150, max: 150 },
        grid: { left: 20, right: 20, top: 20, bottom: 20 },
        series: [{ type: "graph", coordinateSystem: "cartesian2d", data: nodes, links }],
      });
      document.getElementById("lp-info").innerHTML = `
        <b>迭代轮 t = ${document.getElementById("lp-iter").value}</b><br/>
        <b>当前独立标签数</b>：${unique.length}<br/>
        ${semi ? '<span style="color:#4f46e5;">半监督模式：节点 0 和 9 为种子（黑框）</span>' : '<span style="color:#64748b;">无监督模式：所有标签初始化为自身 id</span>'}<br/>
        <span style="color:#64748b;">理想收敛：3 个 cluster 对应 3 种不同颜色。</span>
      `;
    };
    const doStep = () => {
      const el = document.getElementById("lp-iter");
      const t = parseInt(el.value) + 1;
      if (t > 15) return;
      el.value = t; document.getElementById("lp-iter-val").textContent = t;
      step(document.getElementById("lp-semi").checked);
      render();
    };
    document.getElementById("lp-step").addEventListener("click", doStep);
    let timer = null;
    document.getElementById("lp-play").addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; return; }
      timer = setInterval(() => { if (parseInt(document.getElementById("lp-iter").value) >= 15) { clearInterval(timer); timer = null; } else doStep(); }, 600);
    });
    document.getElementById("lp-reset").addEventListener("click", () => {
      labels = [...Array(N).keys()];
      document.getElementById("lp-iter").value = 0;
      document.getElementById("lp-iter-val").textContent = "0";
      render();
    });
    document.getElementById("lp-semi").addEventListener("change", () => {
      labels = [...Array(N).keys()];
      document.getElementById("lp-iter").value = 0;
      document.getElementById("lp-iter-val").textContent = "0";
      render();
    });
    document.getElementById("lp-iter").addEventListener("input", (e) => {
      document.getElementById("lp-iter-val").textContent = e.target.value;
    });
    render();
  },
});
