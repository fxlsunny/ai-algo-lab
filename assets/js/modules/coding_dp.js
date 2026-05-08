/* 模块：动态规划 DP
 * 交互：0-1 背包 DP 表格填充动画 + LIS 折线
 */
MCH.register("coding_dp", {
  render() {
    const code = `# ========== 动态规划 · 完整模板 ==========

# ---------- 1. 线性 DP ----------
def climb(n):
    """LC70 爬楼梯：dp[i] = dp[i-1] + dp[i-2]"""
    if n <= 2: return n
    a, b = 1, 2
    for _ in range(3, n + 1): a, b = b, a + b
    return b

def rob(nums):
    """LC198 打家劫舍：dp[i] = max(dp[i-1], dp[i-2] + nums[i])"""
    prev, cur = 0, 0
    for x in nums: prev, cur = cur, max(cur, prev + x)
    return cur

def max_subarray(nums):
    """LC53 最大子数组和（Kadane 算法）"""
    cur, best = nums[0], nums[0]
    for x in nums[1:]:
        cur = max(x, cur + x)
        best = max(best, cur)
    return best

# ---------- 2. 最长上升子序列 LIS ----------
def lis_dp(nums):
    """O(n²) DP：dp[i] = 以 nums[i] 结尾的 LIS 长度"""
    if not nums: return 0
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]: dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

def lis_patience(nums):
    """O(n log n) 贪心+二分（Patience Sorting 耐心排序）"""
    import bisect
    piles = []
    for x in nums:
        i = bisect.bisect_left(piles, x)
        if i == len(piles): piles.append(x)
        else: piles[i] = x
    return len(piles)

# ---------- 3. 0-1 背包 ----------
def knapsack_01(weights, values, W):
    """二维 O(n*W) 空间"""
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            dp[i][w] = dp[i - 1][w]
            if w >= weights[i - 1]:
                dp[i][w] = max(dp[i][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
    return dp[n][W]

def knapsack_01_rolling(weights, values, W):
    """滚动数组优化：空间 O(W)，逆序遍历防止重复使用"""
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(W, weights[i] - 1, -1):       # 必须逆序
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]

# ---------- 4. 完全背包（每个物品无限次）----------
def knapsack_complete(weights, values, W):
    """正序遍历：允许重复使用"""
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for w in range(weights[i], W + 1):            # 正序
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[W]

def coin_change(coins, amount):
    """LC322 零钱兑换：完全背包求最小数"""
    INF = float('inf')
    dp = [0] + [INF] * amount
    for x in range(1, amount + 1):
        for c in coins:
            if x >= c: dp[x] = min(dp[x], dp[x - c] + 1)
    return dp[amount] if dp[amount] != INF else -1

# ---------- 5. 最长公共子序列 LCS ----------
def lcs(s, t):
    """LC1143：dp[i][j] = s[:i] 和 t[:j] 的 LCS 长度"""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]: dp[i][j] = dp[i - 1][j - 1] + 1
            else: dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

# ---------- 6. 编辑距离（LC72 经典）----------
def edit_distance(s, t):
    """dp[i][j] = s[:i] 转 t[:j] 需要的最少操作（插入/删除/替换）"""
    m, n = len(s), len(t)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s[i - 1] == t[j - 1]: dp[i][j] = dp[i - 1][j - 1]
            else: dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

# ---------- 7. 区间 DP（LC312 戳气球）----------
def max_coins(nums):
    """在 nums 两端补 1，dp[i][j] = 戳破开区间 (i, j) 能得到的最大硬币数
       枚举最后戳破的 k ∈ (i, j)"""
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):                    # 区间长度从小到大
        for i in range(n - length):
            j = i + length
            for k in range(i + 1, j):
                dp[i][j] = max(dp[i][j],
                    nums[i] * nums[k] * nums[j] + dp[i][k] + dp[k][j])
    return dp[0][n - 1]

# ---------- 8. 树形 DP（没有上司的舞会）----------
def tree_rob(root):
    """LC337 打家劫舍 III：每个节点返回 (偷, 不偷) 两个状态"""
    def dfs(node):
        if not node: return (0, 0)
        l_rob, l_not = dfs(node.left)
        r_rob, r_not = dfs(node.right)
        rob = node.val + l_not + r_not            # 偷当前
        not_rob = max(l_rob, l_not) + max(r_rob, r_not)
        return (rob, not_rob)
    return max(dfs(root))

# ---------- 9. 状压 DP（子集枚举）----------
def tsp(graph):
    """旅行商问题 Bitmask DP：O(n²·2ⁿ)"""
    n = len(graph)
    INF = float('inf')
    dp = [[INF] * n for _ in range(1 << n)]
    dp[1][0] = 0                                   # 从 0 出发
    for mask in range(1 << n):
        for u in range(n):
            if not (mask >> u) & 1 or dp[mask][u] == INF: continue
            for v in range(n):
                if (mask >> v) & 1: continue        # v 已访问
                dp[mask | (1 << v)][v] = min(
                    dp[mask | (1 << v)][v], dp[mask][u] + graph[u][v])
    return min(dp[(1 << n) - 1][u] + graph[u][0] for u in range(1, n))

# ---------- 10. 记忆化搜索（Top-down DP）----------
from functools import lru_cache

@lru_cache(None)
def fib(n):
    """等价于线性 DP，但写法更直观"""
    if n < 2: return n
    return fib(n - 1) + fib(n - 2)`;

    return `
      ${MCH.hero({ icon: "🧩", name: "动态规划 DP", en: "Dynamic Programming", tags: ["最优子结构", "状态转移", "滚动数组"], meta: ["📚 Bellman 1957", "⚡ 指数→多项式"] })}

      ${MCH.versionSection("coding_dp")}

      <div class="section">
        <h2>1. DP 三要素</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #14b8a6;">
            <div class="font-semibold text-teal-700 text-sm mb-2">① 状态定义</div>
            <p class="text-xs text-slate-600">用哪个变量表达子问题？<br/>
            <b>爬楼梯</b>: dp[i] = 到第 i 层的方案数<br/>
            <b>0-1 背包</b>: dp[i][w] = 前 i 个物品在容量 w 下的最大价值<br/>
            <b>LIS</b>: dp[i] = 以 i 结尾的最长上升子序列长度</p>
          </div>
          <div class="card" style="border-left:3px solid #0d9488;">
            <div class="font-semibold text-teal-700 text-sm mb-2">② 状态转移方程</div>
            <p class="text-xs text-slate-600">dp[i] 如何由 dp[&lt;i] 推出？<br/>
            <b>爬楼梯</b>: dp[i] = dp[i-1] + dp[i-2]<br/>
            <b>0-1 背包</b>: dp[i][w] = max(dp[i-1][w], dp[i-1][w-w_i]+v_i)<br/>
            <b>LCS</b>: 相等取对角+1，不等取左/上最大</p>
          </div>
          <div class="card" style="border-left:3px solid #0f766e;">
            <div class="font-semibold text-teal-700 text-sm mb-2">③ 初值 & 边界</div>
            <p class="text-xs text-slate-600">dp[0] 等于什么？空集怎么处理？<br/>
            <b>爬楼梯</b>: dp[1]=1, dp[2]=2<br/>
            <b>背包</b>: dp[0][w]=0（无物品价值 0）<br/>
            <b>LIS</b>: dp[i]=1（自己单独成子序列）</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 🎮 交互：0-1 背包 DP 表格填充</h2>
        <p class="text-sm text-slate-600">逐行填 dp 表。<b>黄色</b>是正在计算的格子，<b>绿色阴影</b>越深表示价值越高。右侧是回溯出的最优装包方案。</p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <label class="text-xs font-semibold">物品 (weight,value)：</label>
            <input id="kp-items" type="text" value="2,3;3,4;4,5;5,6" class="px-2 py-1 text-xs border rounded flex-1" />
            <label class="text-xs font-semibold">容量 W：</label>
            <input id="kp-W" type="number" value="8" min="3" max="20" class="px-2 py-1 text-xs border rounded w-16" />
            <button id="kp-run" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 填表</button>
            <button id="kp-reset" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">↺ 重置</button>
          </div>
          ${MCH.slider({ id: "kp-speed", label: "填表延时 ms", min: 20, max: 300, step: 20, value: 100 })}
        </div>

        <div id="kp-table" style="overflow-x:auto;"></div>
        <div id="kp-result" class="mt-3 text-sm text-slate-700"></div>
      </div>

      <div class="section">
        <h2>3. 🎮 交互：最长上升子序列 LIS</h2>
        <p class="text-sm text-slate-600">输入一串数字，观察 DP 从左到右推进每个位置的 dp 值。</p>
        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <input id="lis-arr" type="text" value="10,9,2,5,3,7,101,18,4,8" class="px-2 py-1 text-xs border rounded flex-1" />
            <button id="lis-run" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 计算 LIS</button>
          </div>
        </div>
        <div id="chart-lis" style="height:280px;"></div>
        <div id="lis-res" class="mt-2 text-sm text-slate-700"></div>
      </div>

      <div class="section">
        <h2>4. 代码参考</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 💡 LeetCode 高频题</h2>
        <div class="grid-2">
          <div class="card">
            <div class="font-semibold text-sm mb-2 text-teal-700">入门线性 DP</div>
            <ul class="text-xs text-slate-600" style="list-style:disc inside;line-height:1.7;">
              <li><a href="https://leetcode.cn/problems/climbing-stairs/" target="_blank">LC 70</a> · 爬楼梯</li>
              <li><a href="https://leetcode.cn/problems/house-robber/" target="_blank">LC 198</a> · 打家劫舍</li>
              <li><a href="https://leetcode.cn/problems/longest-increasing-subsequence/" target="_blank">LC 300</a> · LIS</li>
              <li><a href="https://leetcode.cn/problems/maximum-subarray/" target="_blank">LC 53</a> · 最大子数组</li>
            </ul>
          </div>
          <div class="card">
            <div class="font-semibold text-sm mb-2 text-teal-700">二维 DP / 背包</div>
            <ul class="text-xs text-slate-600" style="list-style:disc inside;line-height:1.7;">
              <li><a href="https://leetcode.cn/problems/partition-equal-subset-sum/" target="_blank">LC 416</a> · 分割等和子集（0-1 背包）</li>
              <li><a href="https://leetcode.cn/problems/coin-change/" target="_blank">LC 322</a> · 零钱兑换（完全背包）</li>
              <li><a href="https://leetcode.cn/problems/longest-common-subsequence/" target="_blank">LC 1143</a> · LCS</li>
              <li><a href="https://leetcode.cn/problems/edit-distance/" target="_blank">LC 72</a> · ⭐ 编辑距离</li>
              <li><a href="https://leetcode.cn/problems/burst-balloons/" target="_blank">LC 312</a> · ⭐⭐ 戳气球（区间 DP）</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("coding_dp").pros,
          MCH.getById("coding_dp").cons,
          MCH.getById("coding_dp").best_for,
        )}
      </div>
    `;
  },

  mount() {
    /* ======== 0-1 背包 DP 表 ======== */
    async function runKnapsack() {
      const raw = document.getElementById("kp-items").value;
      const items = raw.split(";").map(s => s.split(",").map(Number)).filter(x => x.length === 2 && !isNaN(x[0]));
      const W = parseInt(document.getElementById("kp-W").value);
      const speed = parseInt(document.getElementById("kp-speed").value);
      const n = items.length;
      const dp = Array.from({ length: n + 1 }, () => Array(W + 1).fill(0));
      const tableEl = document.getElementById("kp-table");

      function render(hi) {
        let html = '<table class="table" style="font-size:11px;min-width:100%;"><thead><tr><th>i＼w</th>';
        for (let w = 0; w <= W; w++) html += `<th>${w}</th>`;
        html += "</tr></thead><tbody>";
        for (let i = 0; i <= n; i++) {
          html += `<tr><td><b>${i === 0 ? "∅" : `物${i}(${items[i - 1][0]},${items[i - 1][1]})`}</b></td>`;
          for (let w = 0; w <= W; w++) {
            const v = dp[i][w];
            const maxV = dp[n][W] || 1;
            const ratio = v / Math.max(maxV, 1);
            const bg = hi && hi[0] === i && hi[1] === w ? "#fbbf24" :
                       `rgba(20,184,166,${(ratio * 0.6).toFixed(2)})`;
            html += `<td style="background:${bg};text-align:center;">${v}</td>`;
          }
          html += "</tr>";
        }
        html += "</tbody></table>";
        tableEl.innerHTML = html;
      }

      render();
      for (let i = 1; i <= n; i++) {
        for (let w = 0; w <= W; w++) {
          dp[i][w] = dp[i - 1][w];
          if (w >= items[i - 1][0]) {
            dp[i][w] = Math.max(dp[i][w], dp[i - 1][w - items[i - 1][0]] + items[i - 1][1]);
          }
          render([i, w]);
          await new Promise(r => setTimeout(r, speed));
        }
      }
      // 回溯最优方案
      const chosen = [];
      let ii = n, ww = W;
      while (ii > 0) {
        if (dp[ii][ww] !== dp[ii - 1][ww]) {
          chosen.push(ii);
          ww -= items[ii - 1][0];
        }
        ii--;
      }
      document.getElementById("kp-result").innerHTML =
        `✅ <b>最大价值 = ${dp[n][W]}</b>；选择物品：${chosen.reverse().map(i => `物${i}(w=${items[i - 1][0]},v=${items[i - 1][1]})`).join(" + ") || "无"}`;
      render();
    }
    document.getElementById("kp-run").onclick = runKnapsack;
    document.getElementById("kp-reset").onclick = () => {
      document.getElementById("kp-table").innerHTML = "";
      document.getElementById("kp-result").innerHTML = "";
    };
    MCH.bindSlider("kp-speed", () => {});

    /* ======== LIS ======== */
    const lisEl = document.getElementById("chart-lis");
    const lisChart = MCH.echart(lisEl, {
      xAxis: { type: "category", data: [] },
      yAxis: { type: "value" },
      series: [
        { name: "原数组", type: "line", data: [], lineStyle: { color: "#94a3b8" }, symbolSize: 10 },
        { name: "dp[i]", type: "bar", data: [], yAxisIndex: 0, itemStyle: { color: "#14b8a6" }, label: { show: true, position: "top", fontSize: 10 } },
      ],
      legend: { top: 0 },
      tooltip: { trigger: "axis" },
    });
    document.getElementById("lis-run").onclick = () => {
      const arr = document.getElementById("lis-arr").value.split(/[,\s]+/).map(Number).filter(x => !isNaN(x));
      const n = arr.length;
      const dp = Array(n).fill(1);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < i; j++)
          if (arr[j] < arr[i]) dp[i] = Math.max(dp[i], dp[j] + 1);
      const maxLen = Math.max(...dp);
      lisChart.setOption({
        xAxis: { data: arr.map((v, i) => `${i}:${v}`) },
        series: [
          { data: arr },
          { data: dp },
        ],
      }, false);
      document.getElementById("lis-res").innerHTML = `✅ <b>LIS 长度 = ${maxLen}</b>（O(n²) DP），dp 数组 = [${dp.join(", ")}]`;
    };
    document.getElementById("lis-run").click();
  },
});
