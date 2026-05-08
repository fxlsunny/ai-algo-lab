/* 模块：Node2Vec / DeepWalk */
MCH.register("graph_node2vec", {
  render() {
    const code = `# Node2Vec 算法（Grover & Leskovec, 2016）
# 两阶段：① 生成偏置随机游走序列  ② 用 Skip-gram（Word2Vec）学 embedding

def node2vec(G, d=128, walk_length=80, num_walks=10, p=1.0, q=1.0,
              window_size=10, epochs=5):
    # Step 1: 为每个节点启动 num_walks 条随机游走
    walks = []
    for _ in range(num_walks):
        for v in G.nodes:
            walks.append(biased_random_walk(G, v, walk_length, p, q))

    # Step 2: 把游走序列当做"句子"，节点当做"词"，训练 Word2Vec Skip-gram
    model = Word2Vec(walks, vector_size=d, window=window_size,
                      sg=1, workers=8, epochs=epochs)
    return {v: model.wv[str(v)] for v in G.nodes}


def biased_random_walk(G, start, length, p, q):
    """
    二阶随机游走：当前在 v，上一步在 t
    选择下一个节点 x 的概率：
    - 若 x = t         : 权重 1/p  (return，回到来的地方)
    - 若 x ∈ N(t)     : 权重 1    (stay close，BFS 式探索)
    - 若 x ∉ N(t)     : 权重 1/q  (向外走，DFS 式探索)

    p 小 → 游走偏向返回（BFS，学结构等价性 → 类似角色）
    q 小 → 游走偏向外推（DFS，学社区同质性 → 类似社群）
    """
    walk = [start]
    while len(walk) < length:
        cur = walk[-1]
        neighbors = list(G.neighbors(cur))
        if not neighbors: break
        if len(walk) == 1:
            walk.append(random.choice(neighbors))
        else:
            prev = walk[-2]
            prev_neigh = set(G.neighbors(prev))
            weights = []
            for x in neighbors:
                if x == prev:             weights.append(1.0 / p)
                elif x in prev_neigh:     weights.append(1.0)
                else:                      weights.append(1.0 / q)
            walk.append(weighted_choice(neighbors, weights))
    return walk


# DeepWalk 就是 p=q=1 的 Node2Vec（纯均匀随机游走）`;

    return `
      ${MCH.hero({ icon: "◈", name: "Node2Vec / DeepWalk — 随机游走图嵌入", en: "Random-Walk based Graph Embedding", tags: ["随机游走", "Skip-gram", "BFS/DFS", "无监督"], meta: ["◈ 不需要节点特征", "⚡ 离线训练 + 在线查嵌入"] })}

      ${MCH.versionSection("graph_node2vec")}

      <div class="section">
        <h2>1. 核心直觉：把图转换成文本问题</h2>
        <p class="text-sm text-slate-600">
          <b>Word2Vec</b> 能从文本中学出 "king - man + woman ≈ queen"，因为它利用了"相似上下文中的词含义相近"。
          <b>Node2Vec</b> 用随机游走生成"节点序列"当作"句子"，同样套用 Skip-gram：
        </p>
        <div class="formula-block">
          <b>Skip-gram 目标</b>：给定序列中的节点 $v$，预测它的上下文邻居 $N(v)$：
          $$ \\max_\\phi \\sum_{v \\in V} \\sum_{u \\in N(v)} \\log P(u \\mid v;\\phi), \\quad P(u|v) = \\frac{\\exp(\\phi_u \\cdot \\phi_v)}{\\sum_w \\exp(\\phi_w \\cdot \\phi_v)} $$
        </div>
      </div>

      <div class="section">
        <h2>2. p / q 参数 —— BFS vs DFS 的平衡</h2>
        <p class="text-sm text-slate-600">Node2Vec 的精华在 <b>二阶偏置随机游走</b>。通过两个参数 p、q 控制探索偏好：</p>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3 class="font-bold text-sm text-indigo-700">🔄 p 小（易返回） → BFS 偏好</h3>
            <p class="text-xs text-slate-600 mt-2">游走在<b>局部小范围</b>内来回。学到的 embedding 反映节点的<b>结构角色</b>（如"都是中心节点"）。</p>
          </div>
          <div class="card" style="border-left:3px solid #10b981;">
            <h3 class="font-bold text-sm text-emerald-700">🚀 q 小（易外推） → DFS 偏好</h3>
            <p class="text-xs text-slate-600 mt-2">游走向外深入探索。学到的 embedding 反映<b>社区归属</b>（如"同一社群"）。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 交互：随机游走模拟</h2>
        <p class="text-sm text-slate-600">观察同一起点，不同 (p, q) 组合下游走路径的差异。</p>
        <div class="grid-2">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "n2v-p", label: "p (return bias)", min: 0.1, max: 4, step: 0.1, value: 1 })}
              ${MCH.slider({ id: "n2v-q", label: "q (in-out bias)", min: 0.1, max: 4, step: 0.1, value: 1 })}
              ${MCH.slider({ id: "n2v-L", label: "walk length", min: 5, max: 40, step: 1, value: 15 })}
              <button id="n2v-resample" class="mt-3 text-xs px-3 py-1 bg-indigo-600 text-white rounded">🎲 重新采样游走</button>
            </div>
            <div id="n2v-info" class="card mt-3 text-xs"></div>
          </div>
          <div id="chart-walk" style="height:400px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>5. Node2Vec vs 其他图嵌入方法</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>思路</th><th>需特征</th><th>Inductive</th><th>特点</th></tr></thead>
          <tbody>
            <tr><td>DeepWalk</td><td>均匀随机游走 + Skip-gram</td><td>❌</td><td>❌</td><td>Node2Vec 的特例 (p=q=1)</td></tr>
            <tr><td><b>Node2Vec</b></td><td>偏置随机游走 + Skip-gram</td><td>❌</td><td>❌</td><td>BFS/DFS 可控</td></tr>
            <tr><td>LINE</td><td>1/2 阶邻接概率</td><td>❌</td><td>❌</td><td>大规模有向图友好</td></tr>
            <tr><td>struc2vec</td><td>结构相似图 + 游走</td><td>❌</td><td>❌</td><td>专攻结构等价性</td></tr>
            <tr><td>GCN</td><td>图卷积</td><td>✓ 必须</td><td>❌</td><td>Transductive，需全图</td></tr>
            <tr><td>GraphSAGE</td><td>邻居采样 + 聚合</td><td>✓ 必须</td><td>✓</td><td>大规模 / 新节点友好</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("graph_node2vec").pros, MCH.getById("graph_node2vec").cons, MCH.getById("graph_node2vec").best_for)}
      </div>
    `;
  },

  mount() {
    // 图结构：3 个社区 × 6 节点
    const N = 18;
    const pos = [];
    for (let c = 0; c < 3; c++) {
      const cx = Math.cos(c * 2 * Math.PI / 3) * 110;
      const cy = Math.sin(c * 2 * Math.PI / 3) * 110;
      for (let i = 0; i < 6; i++) {
        pos.push([cx + Math.cos(i * Math.PI / 3) * 40, cy + Math.sin(i * Math.PI / 3) * 40]);
      }
    }
    // 社区内部全连接 + 社区间少量
    const edges = [];
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < 6; i++) for (let j = i + 1; j < 6; j++) edges.push([c * 6 + i, c * 6 + j]);
    }
    edges.push([5, 6], [11, 12], [17, 0]);
    const adj = {};
    edges.forEach(([u, v]) => { (adj[u] = adj[u] || []).push(v); (adj[v] = adj[v] || []).push(u); });

    const walk = (start, len, p, q) => {
      const w = [start];
      let prev = null;
      while (w.length < len) {
        const cur = w[w.length - 1];
        const nbrs = adj[cur] || [];
        if (nbrs.length === 0) break;
        if (prev === null) {
          w.push(nbrs[Math.floor(Math.random() * nbrs.length)]);
        } else {
          const prevSet = new Set(adj[prev] || []);
          const weights = nbrs.map(x => x === prev ? 1 / p : (prevSet.has(x) ? 1 : 1 / q));
          const total = weights.reduce((a, b) => a + b, 0);
          let r = Math.random() * total, picked = nbrs[0];
          for (let i = 0; i < nbrs.length; i++) {
            r -= weights[i];
            if (r <= 0) { picked = nbrs[i]; break; }
          }
          w.push(picked);
        }
        prev = w[w.length - 2];
      }
      return w;
    };

    const render = () => {
      const p = parseFloat(document.getElementById("n2v-p").value);
      const q = parseFloat(document.getElementById("n2v-q").value);
      const L = parseInt(document.getElementById("n2v-L").value);
      const path = walk(0, L, p, q);
      // Color by community
      const commColor = ["#4f46e5", "#10b981", "#f59e0b"];
      const nodes = pos.map((p0, i) => ({
        name: `N${i}`,
        value: p0,
        symbolSize: path.includes(i) ? 26 : 18,
        itemStyle: {
          color: commColor[Math.floor(i / 6)],
          opacity: path.includes(i) ? 1 : 0.4,
        },
        label: { show: true, formatter: `${i}`, color: "white", fontSize: 10, fontWeight: 700 },
      }));
      const baseLinks = edges.map(([u, v]) => ({ source: `N${u}`, target: `N${v}`, lineStyle: { color: "#cbd5e1", width: 1, opacity: 0.3 } }));
      // 游走路径
      const pathLinks = [];
      for (let i = 0; i < path.length - 1; i++) {
        pathLinks.push({ source: `N${path[i]}`, target: `N${path[i + 1]}`, lineStyle: { color: "#ef4444", width: 2.5, opacity: 0.85, curveness: 0.15 }, symbol: ["none", "arrow"], symbolSize: 8 });
      }
      MCH.echart(document.getElementById("chart-walk"), {
        tooltip: {},
        xAxis: { show: false, min: -180, max: 180 },
        yAxis: { show: false, min: -180, max: 180 },
        grid: { left: 20, right: 20, top: 20, bottom: 20 },
        series: [{ type: "graph", coordinateSystem: "cartesian2d", data: nodes, links: [...baseLinks, ...pathLinks], edgeSymbol: ["none", "arrow"] }],
      });
      // stats
      const uniq = new Set(path);
      const commVisit = [0, 0, 0];
      path.forEach(n => commVisit[Math.floor(n / 6)]++);
      document.getElementById("n2v-info").innerHTML = `
        <b>从节点 0 出发的游走路径</b>（长度 ${path.length}）：<br/>
        <code style="font-size:11px;color:#ef4444;">${path.join(" → ")}</code><br/>
        <b>覆盖节点</b>：${uniq.size} / ${N}<br/>
        <b>社区访问分布</b>：
        <span style="color:${commColor[0]};">C1: ${commVisit[0]}</span>
        <span style="color:${commColor[1]};">C2: ${commVisit[1]}</span>
        <span style="color:${commColor[2]};">C3: ${commVisit[2]}</span><br/>
        <span style="color:#64748b;">提示：p 小游走更局部；q 小游走更发散。</span>
      `;
    };
    ["n2v-p", "n2v-q", "n2v-L"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        document.getElementById(id + "-val").textContent = e.target.value;
        render();
      });
    });
    document.getElementById("n2v-resample").addEventListener("click", render);
    render();
  },
});
