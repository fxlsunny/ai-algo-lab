/* 模块：BFS / DFS / 拓扑排序
 * 交互：网格迷宫 BFS/DFS 访问顺序动画
 * 修复 v6.1: 用 custom series + renderItem 绘制网格，不用 heatmap（避免 visualMap pieces bug）
 */
MCH.register("coding_graph_traversal", {
  render() {
    const code = `# ========== 图遍历完整模板（邻接表表示）==========
from collections import deque

# ---------- 1. BFS 层次遍历 ----------
def bfs(graph, start):
    visited, order = {start}, []
    q = deque([start])
    while q:
        node = q.popleft(); order.append(node)
        for nei in graph[node]:
            if nei not in visited:
                visited.add(nei); q.append(nei)
    return order

# ---------- 2. DFS 递归 / 迭代 ----------
def dfs_recur(graph, node, visited=None, order=None):
    if visited is None: visited, order = set(), []
    visited.add(node); order.append(node)
    for nei in graph[node]:
        if nei not in visited: dfs_recur(graph, nei, visited, order)
    return order

def dfs_iter(graph, start):
    visited, order, stack = set(), [], [start]
    while stack:
        node = stack.pop()
        if node in visited: continue
        visited.add(node); order.append(node)
        for nei in reversed(graph[node]):
            if nei not in visited: stack.append(nei)
    return order

# ---------- 3. 网格 DFS：岛屿数量（LC200）----------
def num_islands(grid):
    if not grid: return 0
    R, C = len(grid), len(grid[0])
    def dfs(r, c):
        if r < 0 or r >= R or c < 0 or c >= C or grid[r][c] != '1': return
        grid[r][c] = '0'
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]: dfs(r+dr, c+dc)
    count = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == '1': count += 1; dfs(r, c)
    return count

# ---------- 4. Kahn 拓扑排序（LC210）----------
def topo_kahn(num, prereqs):
    indeg = [0] * num
    g = [[] for _ in range(num)]
    for a, b in prereqs: g[b].append(a); indeg[a] += 1
    q = deque([i for i in range(num) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0: q.append(v)
    return order if len(order) == num else []

# ---------- 5. Tarjan 强连通分量 ----------
def tarjan_scc(graph):
    index, lowlink = {}, {}
    stack, on_stack, result = [], set(), []
    idx = [0]
    def strongconnect(v):
        index[v] = lowlink[v] = idx[0]; idx[0] += 1
        stack.append(v); on_stack.add(v)
        for w in graph.get(v, []):
            if w not in index:
                strongconnect(w); lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], index[w])
        if lowlink[v] == index[v]:
            component = []
            while True:
                w = stack.pop(); on_stack.remove(w); component.append(w)
                if w == v: break
            result.append(component)
    for v in graph:
        if v not in index: strongconnect(v)
    return result

# ---------- 6. 双向 BFS（LC127 单词接龙）----------
def bidirectional_bfs(graph, src, dst):
    if src == dst: return 0
    front, back = {src}, {dst}
    seen = {src, dst}; step = 0
    while front and back:
        if len(front) > len(back): front, back = back, front
        step += 1
        new_front = set()
        for u in front:
            for v in graph.get(u, []):
                if v in back: return step
                if v not in seen: seen.add(v); new_front.add(v)
        front = new_front
    return -1

# ---------- 7. 二分图判定（BFS 染色，LC785）----------
def is_bipartite(graph):
    color = {}
    for start in range(len(graph)):
        if start in color: continue
        color[start] = 0; q = deque([start])
        while q:
            u = q.popleft()
            for v in graph[u]:
                if v not in color:
                    color[v] = 1 - color[u]; q.append(v)
                elif color[v] == color[u]: return False
    return True`;

    return `
      ${MCH.hero({ icon: "🕸️", name: "BFS / DFS / 拓扑排序", en: "Graph Traversal", tags: ["广度优先", "深度优先", "拓扑"], meta: ["📚 CLRS 第 22 章", "⚡ O(V+E)"] })}

      ${MCH.versionSection("coding_graph_traversal")}

      <div class="section">
        <h2>1. BFS vs DFS 对比</h2>
        <table class="table">
          <thead><tr><th></th><th>BFS（队列）</th><th>DFS（栈/递归）</th></tr></thead>
          <tbody>
            <tr><td><b>数据结构</b></td><td>队列 Queue</td><td>栈 Stack / 递归</td></tr>
            <tr><td><b>访问顺序</b></td><td>按层次（距离由近到远）</td><td>先深后回溯</td></tr>
            <tr><td><b>空间</b></td><td>O(w)，w=最大宽度</td><td>O(h)，h=最大深度</td></tr>
            <tr><td><b>擅长</b></td><td>无权图最短路径；连通分量</td><td>路径搜索；拓扑排序；强连通</td></tr>
            <tr><td><b>最短路保证</b></td><td>✅（第一次到达即最优）</td><td>❌（需比较所有路径）</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. 🎮 交互：网格迷宫遍历</h2>
        <p class="text-sm text-slate-600 mb-2">从左上角出发遍历网格，黑色是墙。BFS 像水波扩散（层次清晰）；DFS 像蛇钻进一条路。</p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <label class="text-xs font-semibold">算法：</label>
            <select id="gt-algo" class="px-2 py-1 text-xs border rounded">
              <option value="bfs">BFS 广度优先</option>
              <option value="dfs">DFS 深度优先</option>
            </select>
            <button id="gt-regen" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">🎲 新迷宫</button>
            <button id="gt-run" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 开始遍历</button>
            <button id="gt-stop" class="text-xs px-3 py-1 bg-red-500 text-white rounded">⏹ 停止</button>
            <span id="gt-stats" class="text-xs text-slate-700"></span>
          </div>
          <div class="grid-2 mt-3">
            ${MCH.slider({ id: "gt-rows", label: "行数", min: 6, max: 18, step: 1, value: 10 })}
            ${MCH.slider({ id: "gt-cols", label: "列数", min: 6, max: 24, step: 1, value: 14 })}
            ${MCH.slider({ id: "gt-wall", label: "墙密度", min: 0, max: 50, step: 5, value: 25, format: v => `${v}%` })}
            ${MCH.slider({ id: "gt-speed", label: "动画 ms", min: 10, max: 300, step: 10, value: 60 })}
          </div>
        </div>

        <div id="chart-gt" style="height:380px;"></div>
      </div>

      <div class="section">
        <h2>3. 代码参考</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 💡 LeetCode 高频题</h2>
        <table class="table">
          <thead><tr><th>题号</th><th>题目</th><th>技巧</th></tr></thead>
          <tbody>
            <tr><td><a href="https://leetcode.cn/problems/number-of-islands/" target="_blank">200</a></td><td>岛屿数量</td><td>⭐ DFS/BFS 连通分量</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/word-ladder/" target="_blank">127</a></td><td>单词接龙</td><td>⭐ 双向 BFS</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/rotting-oranges/" target="_blank">994</a></td><td>腐烂的橘子</td><td>多源 BFS</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/course-schedule/" target="_blank">207</a></td><td>课程表</td><td>⭐ 拓扑排序判环</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/course-schedule-ii/" target="_blank">210</a></td><td>课程表 II</td><td>⭐ Kahn 拓扑</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/is-graph-bipartite/" target="_blank">785</a></td><td>判断二分图</td><td>BFS 染色</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("coding_graph_traversal").pros,
          MCH.getById("coding_graph_traversal").cons,
          MCH.getById("coding_graph_traversal").best_for,
        )}
      </div>
    `;
  },

  mount() {
    const el = document.getElementById("chart-gt");
    if (!el) return;

    let rows = 10, cols = 14, wallP = 0.25;
    let grid = [];   // 0=空 1=墙
    let visitOrder = [];  // [[r,c], ...]
    let isRunning = false, stopFlag = false;
    const statsEl = document.getElementById("gt-stats");

    // 颜色映射：0=空 #f1f5f9, 1=墙 #1e293b, 2=起点 #10b981, 3=终点 #ef4444, N>=4=已访问 #fbbf24(渐变)
    const CELL = { EMPTY: 0, WALL: 1, START: 2, END: 3, VISITED_BASE: 4 };
    const cellColor = (v) => {
      if (v === CELL.EMPTY) return "#f1f5f9";
      if (v === CELL.WALL) return "#1e293b";
      if (v === CELL.START) return "#10b981";
      if (v === CELL.END) return "#ef4444";
      // v >= CELL.VISITED_BASE: 访问顺序着色，越后访问颜色越深（橙黄）
      const idx = v - CELL.VISITED_BASE;
      const t = Math.min(idx / 30, 1); // 最多 30 步渐变
      const r = Math.round(251 + (t * (251 - 251)));  // #fbbf24
      const g = Math.round(191 + (t * (100 - 191)));
      const b = Math.round(36 + (t * (0 - 36)));
      return `rgb(${r},${g},${b})`;
    };

    // 用 custom series renderItem 绘制网格矩形（比 heatmap 更可靠）
    const chart = echarts.init(el, null, { renderer: "canvas" });
    MCH.charts.push(chart);

    const cellSize = () => {
      const w = el.clientWidth || 700;
      const h = el.clientHeight || 380;
      return {
        cw: Math.max(8, Math.min(40, (w - 40) / cols)),
        ch: Math.max(8, Math.min(40, (h - 40) / rows)),
      };
    };

    function genMaze() {
      rows = parseInt(document.getElementById("gt-rows").value);
      cols = parseInt(document.getElementById("gt-cols").value);
      wallP = parseInt(document.getElementById("gt-wall").value) / 100;
      grid = [];
      for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) {
          const isProtected = (r === 0 && c === 0) || (r === rows - 1 && c === cols - 1);
          row.push((Math.random() < wallP && !isProtected) ? 1 : 0);
        }
        grid.push(row);
      }
      visitOrder = [];
      drawChart();
    }

    function drawChart() {
      const { cw, ch } = cellSize();
      const pad = 10;
      const totalW = cols * cw + pad * 2;
      const totalH = rows * ch + pad * 2;

      const items = [];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          let val = grid[r][c] === 1 ? CELL.WALL : CELL.EMPTY;
          // 标记起点终点
          if (r === 0 && c === 0) val = CELL.START;
          else if (r === rows - 1 && c === cols - 1) val = CELL.END;
          else {
            // 已访问格子上色
            const idx = visitOrder.findIndex(([vr, vc]) => vr === r && vc === c);
            if (idx >= 0) val = CELL.VISITED_BASE + idx;
          }
          items.push({
            type: "rect",
            shape: { x: pad + c * cw, y: pad + r * ch, width: cw - 1, height: ch - 1 },
            style: { fill: cellColor(val), stroke: "rgba(0,0,0,0.08)", lineWidth: 0.5 },
            styleEmphasis: { shadowBlur: 8, shadowColor: "rgba(0,0,0,0.3)" },
          });
          // 访问顺序数字标注（仅已访问格子）
          const vIdx = visitOrder.findIndex(([vr, vc]) => vr === r && vc === c);
          if (vIdx >= 0) {
            items.push({
              type: "text",
              shape: { x: pad + c * cw + cw / 2, y: pad + r * ch + ch / 2 },
              style: { text: String(vIdx + 1), x: pad + c * cw + cw / 2, y: pad + r * ch + ch / 2 + 5,
                textAlign: "center", fontSize: Math.max(7, Math.min(cw, ch) * 0.38),
                fontWeight: "bold", fill: "#1e293b" },
            });
          }
        }
      }

      const option = {
        animation: false,
        xAxis: { show: false, min: 0, max: totalW, data: [] },
        yAxis: { show: false, min: 0, max: totalH, data: [] },
        tooltip: {
          formatter: (p) => {
            const item = items[p.dataIndex];
            if (!item || item.type !== "rect") return "";
            const r = Math.floor((p.dataIndex) / cols);
            const c = (p.dataIndex) % cols;
            const val = grid[r][c] === 1 ? "墙" : (r === 0 && c === 0 ? "起点" :
                      (r === rows - 1 && c === cols - 1 ? "终点" : "空"));
            const vIdx = visitOrder.findIndex(([vr, vc]) => vr === r && vc === c);
            return `(${r},${c}) ${val}` + (vIdx >= 0 ? ` | 访问 #${vIdx + 1}` : "");
          }
        },
        series: [{
          type: "custom",
          data: items,
          renderItem: (params, api) => {
            const item = items[params.dataIndex];
            if (!item) return null;
            if (item.type === "rect") {
              return {
                type: "rect",
                shape: item.shape,
                style: item.style,
                styleEmphasis: item.styleEmphasis,
              };
            }
            if (item.type === "text") {
              return {
                type: "text",
                style: item.style,
              };
            }
            return null;
          },
        }],
      };
      try { chart.setOption(option, true); } catch (e) { console.warn("[GT] draw error:", e); }
    }

    async function run() {
      if (isRunning) return;
      isRunning = true; stopFlag = false;
      visitOrder = [];
      const algo = document.getElementById("gt-algo").value;
      const speed = parseInt(document.getElementById("gt-speed").value);
      const visited = Array.from({ length: rows }, () => Array(cols).fill(false));
      const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
      const inBound = (r, c) => r >= 0 && r < rows && c >= 0 && c < cols;
      let reached = false;

      if (algo === "bfs") {
        const q = [[0, 0]]; visited[0][0] = true;
        while (q.length) {
          if (stopFlag) break;
          const cur = q.shift();
          const r = cur[0], c = cur[1];
          visitOrder.push([r, c]);
          statsEl.textContent = `已访问 ${visitOrder.length} 格 / 队列剩 ${q.length}`;
          drawChart();
          if (r === rows - 1 && c === cols - 1) { reached = true; break; }
          await new Promise(resolve => setTimeout(resolve, speed));
          for (const [dr, dc] of dirs) {
            const nr = r + dr, nc = c + dc;
            if (inBound(nr, nc) && !visited[nr][nc] && grid[nr][nc] !== 1) {
              visited[nr][nc] = true; q.push([nr, nc]);
            }
          }
        }
      } else {
        const stack = [[0, 0]]; visited[0][0] = true;
        while (stack.length) {
          if (stopFlag) break;
          const cur = stack.pop();
          const r = cur[0], c = cur[1];
          visitOrder.push([r, c]);
          statsEl.textContent = `已访问 ${visitOrder.length} 格 / 栈剩 ${stack.length}`;
          drawChart();
          if (r === rows - 1 && c === cols - 1) { reached = true; break; }
          await new Promise(resolve => setTimeout(resolve, speed));
          for (const [dr, dc] of dirs) {
            const nr = r + dr, nc = c + dc;
            if (inBound(nr, nc) && !visited[nr][nc] && grid[nr][nc] !== 1) {
              visited[nr][nc] = true; stack.push([nr, nc]);
            }
          }
        }
      }
      if (!reached && !stopFlag) statsEl.textContent += " · ❌ 被墙挡住，未到达终点";
      else if (reached) statsEl.textContent += " · ✅ 到达终点！";
      isRunning = false;
    }

    document.getElementById("gt-regen").addEventListener("click", () => {
      stopFlag = true; setTimeout(genMaze, 80);
    });
    document.getElementById("gt-run").addEventListener("click", run);
    document.getElementById("gt-stop").addEventListener("click", () => { stopFlag = true; });

    MCH.bindSlider("gt-rows", () => { if (!isRunning) genMaze(); });
    MCH.bindSlider("gt-cols", () => { if (!isRunning) genMaze(); });
    MCH.bindSlider("gt-wall", () => { if (!isRunning) genMaze(); });

    window.addEventListener("resize", () => { if (!isRunning) drawChart(); });
    genMaze();
    console.log("[GT] mounted, chart inited");
  },
});
