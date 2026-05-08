/* 模块：最短路径 Dijkstra / Floyd / A*
 * 交互：网格 Dijkstra 扩散动画 + 最短路回溯
 * 修复 v6.1: custom series + renderItem 替代 heatmap（避免 visualMap pieces bug）
 */
MCH.register("coding_graph_shortest", {
  render() {
    const code = `# ========== 最短路径 · 完整模板 ==========
import heapq
from collections import deque

# ---------- 1. Dijkstra + 二叉堆 ----------
def dijkstra(graph, start):
    dist = {v: float('inf') for v in graph}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]: dist[v] = nd; heapq.heappush(pq, (nd, v))
    return dist

def dijkstra_with_path(graph, start, end):
    dist, prev, pq = {v: float('inf') for v in graph}, {}, [(0, start)]
    dist[start] = 0
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]: dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    if end not in prev: return None
    path, cur = [], end
    while cur in prev: path.append(cur); cur = prev[cur]
    return dist[end], path[::-1]

# ---------- 2. SPFA（Bellman-Ford 队列优化）----------
def spfa(graph, start):
    dist = {v: float('inf') for v in graph}
    dist[start] = 0; q, inq = deque([start]), {start}
    while q:
        u = q.popleft(); inq.discard(u)
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if v not in inq: q.append(v); inq.add(v)
    return dist

# ---------- 3. Floyd-Warshall（全源最短路 O(V³)）----------
def floyd(dist):
    n = len(dist)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist

# ---------- 4. A* 启发式搜索 ----------
def astar(start, goal, neighbors, h):
    g_score = {start: 0}
    pq = [(h(start), start)]
    came_from = {}
    while pq:
        _, u = heapq.heappop(pq)
        if u == goal:
            path = [u]
            while u in came_from: u = came_from[u]; path.append(u)
            return path[::-1]
        for v, w in neighbors(u):
            tentative = g_score[u] + w
            if tentative < g_score.get(v, float('inf')):
                came_from[v] = u; g_score[v] = tentative
                heapq.heappush(pq, (tentative + h(v), v))
    return None

# ---------- 5. 网格 A* ----------
def astar_grid(grid, start, goal):
    R, C = len(grid), len(grid[0])
    def h(p): return abs(p[0]-goal[0]) + abs(p[1]-goal[1])
    def neighbors(p):
        r, c = p
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < R and 0 <= nc < C and grid[nr][nc] == 0:
                yield (nr, nc), 1
    return astar(start, goal, neighbors, h)`;

    return `
      ${MCH.hero({ icon: "🧭", name: "最短路径 Dijkstra / Floyd / A*", en: "Shortest Path", tags: ["Dijkstra", "A*", "最短路"], meta: ["📚 CLRS 第 24/25 章", "⚡ O((V+E) log V)"] })}

      ${MCH.versionSection("coding_graph_shortest")}

      <div class="section">
        <h2>1. 算法对比</h2>
        <table class="table">
          <thead><tr><th>算法</th><th>时间复杂度</th><th>支持负权</th><th>典型场景</th></tr></thead>
          <tbody>
            <tr><td>Dijkstra + 堆</td><td>O((V+E) log V)</td><td>❌</td><td>导航/路由（单源非负）</td></tr>
            <tr><td>SPFA</td><td>均摊 O(kE)</td><td>✅</td><td>稀疏图负权</td></tr>
            <tr><td>Bellman-Ford</td><td>O(VE)</td><td>✅</td><td>检测负环</td></tr>
            <tr><td>Floyd-Warshall</td><td>O(V³)</td><td>✅</td><td>全源最短路（≤500 点）</td></tr>
            <tr><td>A*</td><td>取决于启发</td><td>❌</td><td>游戏寻路/地图导航</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. 🎮 交互：网格 Dijkstra vs A*</h2>
        <p class="text-sm text-slate-600 mb-2">
          左上=起点，右下=终点，黑色=墙。<b>A*</b> 有启发函数引导，路径笔直朝目标走（扩展节点少）；<b>Dijkstra</b> 像水波扩散，先探索所有同等距离的点（扩展节点多）。
        </p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <label class="text-xs font-semibold">算法：</label>
            <select id="sp-algo" class="px-2 py-1 text-xs border rounded">
              <option value="astar" selected>A*（曼哈顿启发）</option>
              <option value="dijkstra">Dijkstra（无启发）</option>
            </select>
            <button id="sp-regen" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">🎲 新地图</button>
            <button id="sp-run" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 开始</button>
            <button id="sp-stop" class="text-xs px-3 py-1 bg-red-500 text-white rounded">⏹ 停止</button>
            <span id="sp-stats" class="text-xs text-slate-700"></span>
          </div>
          <div class="grid-2 mt-3">
            ${MCH.slider({ id: "sp-rows", label: "行数", min: 8, max: 18, step: 1, value: 12 })}
            ${MCH.slider({ id: "sp-cols", label: "列数", min: 10, max: 28, step: 1, value: 18 })}
            ${MCH.slider({ id: "sp-wall", label: "墙密度", min: 10, max: 40, step: 5, value: 25, format: v => `${v}%` })}
            ${MCH.slider({ id: "sp-speed", label: "延时 ms", min: 10, max: 120, step: 10, value: 30 })}
          </div>
        </div>

        <div id="chart-sp" style="height:400px;"></div>
      </div>

      <div class="section">
        <h2>3. 代码参考</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 💡 LeetCode 高频题</h2>
        <table class="table">
          <thead><tr><th>题号</th><th>题目</th><th>算法</th></tr></thead>
          <tbody>
            <tr><td><a href="https://leetcode.cn/problems/network-delay-time/" target="_blank">743</a></td><td>网络延迟时间</td><td>⭐ Dijkstra 模板题</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/cheapest-flights-within-k-stops/" target="_blank">787</a></td><td>K 站中转最便宜航班</td><td>⭐ 有限步 Dijkstra</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/path-with-maximum-probability/" target="_blank">1514</a></td><td>概率最大路径</td><td>Dijkstra 变式</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/swim-in-rising-water/" target="_blank">778</a></td><td>水位上升的泳池</td><td>Dijkstra + 二分</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("coding_graph_shortest").pros,
          MCH.getById("coding_graph_shortest").cons,
          MCH.getById("coding_graph_shortest").best_for,
        )}
      </div>
    `;
  },

  mount() {
    const el = document.getElementById("chart-sp");
    if (!el) return;

    let rows = 12, cols = 18, wallP = 0.25;
    let grid = [];
    let visitedSet = new Set(); // Set of "r,c"
    let pathSet = new Set();    // Set of "r,c" in shortest path
    let isRunning = false, stopFlag = false;

    // 用 custom series renderItem 绘制网格（比 heatmap 可靠）
    const chart = echarts.init(el, null, { renderer: "canvas" });
    MCH.charts.push(chart);

    function cellSize() {
      const w = el.clientWidth || 750;
      const h = el.clientHeight || 400;
      return {
        cw: Math.max(8, Math.min(40, (w - 40) / cols)),
        ch: Math.max(8, Math.min(40, (h - 40) / rows)),
      };
    }

    function gen() {
      rows = parseInt(document.getElementById("sp-rows").value);
      cols = parseInt(document.getElementById("sp-cols").value);
      wallP = parseInt(document.getElementById("sp-wall").value) / 100;
      grid = [];
      for (let r = 0; r < rows; r++) {
        const row = [];
        for (let c = 0; c < cols; c++) {
          const protect = (r === 0 && c === 0) || (r === rows - 1 && c === cols - 1);
          row.push((Math.random() < wallP && !protect) ? 1 : 0);
        }
        grid.push(row);
      }
      visitedSet = new Set();
      pathSet = new Set();
      drawChart();
    }

    // 颜色：空=f1f5f9, 墙=1e293b, 起点=10b981, 终点=ef4444, 已访问=#86efac~#22c55e(梯度),
    //       最短路径=#f59e0b
    function cellFill(r, c) {
      if (pathSet.has(`${r},${c}`)) return "#f59e0b";
      if (r === 0 && c === 0) return "#10b981";
      if (r === rows - 1 && c === cols - 1) return "#ef4444";
      if (grid[r][c] === 1) return "#1e293b";
      if (visitedSet.has(`${r},${c}`)) {
        // 访问顺序越靠后越深绿
        const visitedCount = visitedSet.size;
        const t = Math.min(visitedSet.size / 40, 1);
        // 从浅绿 #86efac 到深绿 #16a34a
        const R = Math.round(134 + t * (22 - 134));
        const G = Math.round(239 + t * (166 - 239));
        const B = Math.round(172 + t * (74 - 172));
        return `rgb(${R},${G},${B})`;
      }
      return "#f1f5f9";
    }

    function drawChart() {
      const { cw, ch } = cellSize();
      const pad = 10;
      const items = [];
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          items.push({
            type: "rect",
            shape: { x: pad + c * cw, y: pad + r * ch, width: cw - 1, height: ch - 1 },
            style: { fill: cellFill(r, c), stroke: "rgba(0,0,0,0.08)", lineWidth: 0.5 },
          });
          // 数字标注
          if (pathSet.has(`${r},${c}`)) {
            const pathIdx = [...pathSet].indexOf(`${r},${c}`);
            items.push({
              type: "text",
              style: {
                text: pathIdx + 1,
                x: pad + c * cw + cw / 2,
                y: pad + r * ch + ch / 2 + 5,
                textAlign: "center",
                fontSize: Math.max(7, Math.min(cw, ch) * 0.38),
                fontWeight: "bold",
                fill: "#1e293b",
              },
            });
          } else if (visitedSet.has(`${r},${c}`)) {
            // 显示扩展序号（太密则不显示）
            if (visitedSet.size <= 60) {
              const idx = [...visitedSet].indexOf(`${r},${c}`);
              items.push({
                type: "text",
                style: {
                  text: idx + 1,
                  x: pad + c * cw + cw / 2,
                  y: pad + r * ch + ch / 2 + 5,
                  textAlign: "center",
                  fontSize: Math.max(7, Math.min(cw, ch) * 0.35),
                  fontWeight: "normal",
                  fill: "#1e293b",
                  opacity: 0.7,
                },
              });
            }
          }
        }
      }

      try {
        chart.setOption({
          animation: false,
          xAxis: { show: false, data: [] },
          yAxis: { show: false, data: [] },
          tooltip: {
            formatter: (p) => {
              const idx = p.dataIndex;
              const r = Math.floor(idx / cols);
              const c = idx % cols;
              if (grid[r][c] === 1) return `(${r},${c}) 墙`;
              if (pathSet.has(`${r},${c}`)) return `(${r},${c}) 最短路径`;
              if (visitedSet.has(`${r},${c}`)) return `(${r},${c}) 已扩展`;
              if (r === 0 && c === 0) return `(${r},${c}) 起点`;
              if (r === rows - 1 && c === cols - 1) return `(${r},${c}) 终点`;
              return `(${r},${c}) 空`;
            }
          },
          series: [{
            type: "custom",
            data: items,
            renderItem: (params, api) => {
              const item = items[params.dataIndex];
              if (!item) return null;
              if (item.type === "rect") {
                return api.style({
                  type: "rect",
                  shape: item.shape,
                  style: item.style,
                });
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
        }, true);
      } catch (e) { console.warn("[SP] draw error:", e); }
    }

    async function run() {
      if (isRunning) return;
      isRunning = true; stopFlag = false;
      visitedSet = new Set();
      pathSet = new Set();
      const algo = document.getElementById("sp-algo").value;
      const speed = parseInt(document.getElementById("sp-speed").value);
      const statsEl = document.getElementById("sp-stats");
      const manhattan = (r, c) => Math.abs(r - (rows - 1)) + Math.abs(c - (cols - 1));

      const dist = Array.from({ length: rows }, () => Array(cols).fill(Infinity));
      const parent = Array.from({ length: rows }, () => Array(cols).fill(null));
      dist[0][0] = 0;
      // 简易堆（数组 + sort，简化版不影响理解）
      const heap = [[0, 0, 0]]; // [priority, r, c]

      while (heap.length) {
        if (stopFlag) break;
        heap.sort((a, b) => a[0] - b[0]);
        const [, r, c] = heap.shift();
        const key = `${r},${c}`;
        if (visitedSet.has(key)) continue;
        visitedSet.add(key);
        statsEl.textContent = `已扩展 ${visitedSet.size} 格 · 堆剩 ${heap.length}`;
        drawChart();
        await new Promise(res => setTimeout(res, speed));
        if (r === rows - 1 && c === cols - 1) {
          // 回溯最短路径
          const pathCoords = [];
          let [pr, pc] = [r, c];
          while (pr !== null) {
            pathCoords.unshift([pr, pc]);
            const p = parent[pr][pc];
            if (!p) break;
            [pr, pc] = p;
          }
          pathCoords.forEach(([pr2, pc2]) => pathSet.add(`${pr2},${pc2}`));
          statsEl.textContent += ` · ✅ 最短距离=${dist[r][c]}，路径长度=${pathCoords.length}`;
          drawChart();
          break;
        }
        const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
        for (const [dr, dc] of dirs) {
          const nr = r + dr, nc = c + dc;
          if (nr < 0 || nr >= rows || nc < 0 || nc >= cols) continue;
          if (grid[nr][nc] === 1) continue;
          const nd = dist[r][c] + 1;
          if (nd < dist[nr][nc]) {
            dist[nr][nc] = nd;
            parent[nr][nc] = [r, c];
            const priority = (algo === "astar") ? nd + manhattan(nr, nc) : nd;
            heap.push([priority, nr, nc]);
          }
        }
      }
      if (!pathSet.size && !stopFlag) statsEl.textContent += " · ❌ 无法到达终点";
      isRunning = false;
    }

    document.getElementById("sp-regen").addEventListener("click", () => {
      stopFlag = true; setTimeout(gen, 80);
    });
    document.getElementById("sp-run").addEventListener("click", run);
    document.getElementById("sp-stop").addEventListener("click", () => { stopFlag = true; });
    MCH.bindSlider("sp-rows", () => { if (!isRunning) gen(); });
    MCH.bindSlider("sp-cols", () => { if (!isRunning) gen(); });
    MCH.bindSlider("sp-wall", () => { if (!isRunning) gen(); });
    window.addEventListener("resize", () => { if (!isRunning) drawChart(); });
    gen();
    console.log("[SP] mounted");
  },
});
