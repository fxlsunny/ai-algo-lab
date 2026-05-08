/* 模块：栈 / 队列 / 单调栈 */
MCH.register("coding_ds_stack_queue", {
  render() {
    const stackCode = `# ========== 栈 / 队列 / 单调栈 完整模板 ==========

# ---------- 1. 栈基础操作 ----------
class Stack:
    def __init__(self): self.data = []
    def push(self, x): self.data.append(x)
    def pop(self): return self.data.pop() if self.data else None
    def peek(self): return self.data[-1] if self.data else None
    def is_empty(self): return not self.data
    def size(self): return len(self.data)

# ---------- 2. 括号匹配（LC20 有效的括号）----------
def is_valid(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for c in s:
        if c in '([{': stack.append(c)
        elif not stack or stack.pop() != pairs[c]: return False
    return not stack

# ---------- 3. 最小栈（LC155，O(1) 查询 min）----------
class MinStack:
    """辅助栈法：主栈存数据，辅助栈存到当前位置为止的最小值"""
    def __init__(self):
        self.stack, self.min_stack = [], []

    def push(self, x):
        self.stack.append(x)
        if not self.min_stack or x <= self.min_stack[-1]:
            self.min_stack.append(x)

    def pop(self):
        x = self.stack.pop()
        if x == self.min_stack[-1]: self.min_stack.pop()
        return x

    def get_min(self):
        return self.min_stack[-1] if self.min_stack else None

# ---------- 4. 中缀转后缀（Shunting-yard）----------
def infix_to_postfix(tokens):
    """例：3+4*2 → 3 4 2 * +"""
    prec = {'+': 1, '-': 1, '*': 2, '/': 2}
    out, stack = [], []
    for t in tokens:
        if t.isdigit(): out.append(t)
        elif t == '(': stack.append(t)
        elif t == ')':
            while stack and stack[-1] != '(': out.append(stack.pop())
            stack.pop()                   # 弹出 '('
        else:
            while stack and stack[-1] != '(' and prec.get(stack[-1], 0) >= prec[t]:
                out.append(stack.pop())
            stack.append(t)
    while stack: out.append(stack.pop())
    return out

# ---------- 5. 队列（循环队列 / Ring Buffer）----------
class CircularQueue:
    """LC622：固定容量的 FIFO"""
    def __init__(self, cap):
        self.buf = [0] * cap
        self.head = self.tail = self.size = 0
        self.cap = cap

    def enqueue(self, x):
        if self.size == self.cap: return False
        self.buf[self.tail] = x
        self.tail = (self.tail + 1) % self.cap
        self.size += 1
        return True

    def dequeue(self):
        if self.size == 0: return None
        x = self.buf[self.head]
        self.head = (self.head + 1) % self.cap
        self.size -= 1
        return x

# ---------- 6. 单调栈：下一个更大元素（LC496）----------
def next_greater(nums):
    n = len(nums); res = [-1] * n; stk = []
    for i in range(n):
        while stk and nums[stk[-1]] < nums[i]:
            res[stk.pop()] = nums[i]
        stk.append(i)
    return res

# ---------- 7. 单调栈经典：柱状图最大矩形（LC84）----------
def largest_rectangle(heights):
    """时间 O(n)；技巧：首尾加 0 哨兵避免边界判断"""
    stk, max_area = [], 0
    heights = [0] + heights + [0]
    for i, h in enumerate(heights):
        while stk and heights[stk[-1]] > h:
            top = stk.pop()
            width = i - stk[-1] - 1
            max_area = max(max_area, heights[top] * width)
        stk.append(i)
    return max_area

# ---------- 8. 单调栈：接雨水（LC42）----------
def trap(heights):
    stk, water = [], 0
    for i, h in enumerate(heights):
        while stk and heights[stk[-1]] < h:
            bottom = heights[stk.pop()]
            if not stk: break
            left = stk[-1]
            water += (min(heights[left], h) - bottom) * (i - left - 1)
        stk.append(i)
    return water

# ---------- 9. 单调队列：滑动窗口最大值（LC239）----------
from collections import deque
def max_sliding_window(nums, k):
    dq, res = deque(), []
    for i, x in enumerate(nums):
        # 1) 窗口外的元素出队
        while dq and dq[0] <= i - k: dq.popleft()
        # 2) 维持单调递减：新元素比尾部大就把尾部踢掉
        while dq and nums[dq[-1]] < x: dq.pop()
        dq.append(i)
        # 3) 窗口形成后记录队首（即当前最大值的下标）
        if i >= k - 1: res.append(nums[dq[0]])
    return res

# ---------- 10. 每日温度（LC739，单调栈变式）----------
def daily_temperatures(T):
    res = [0] * len(T); stk = []
    for i, t in enumerate(T):
        while stk and T[stk[-1]] < t:
            j = stk.pop(); res[j] = i - j
        stk.append(i)
    return res`;

    return `
      ${MCH.hero({ icon: "🥞", name: "栈 / 队列 / 单调栈", en: "Stack · Queue · Monotonic", tags: ["LIFO", "FIFO", "O(n) 滑窗"], meta: ["📚 LC 单调栈专题", "⚡ O(1) 均摊"] })}

      ${MCH.versionSection("coding_ds_stack_queue")}

      <div class="section">
        <h2>1. 栈：LIFO 的魔力</h2>
        <p class="text-sm text-slate-600">栈（Stack）是<b>后进先出</b>的线性结构，push/pop 都 O(1)。在计算机科学里，<b>函数调用栈</b>、
        <b>表达式求值</b>、<b>括号匹配</b>、<b>DFS 递归展开</b>、<b>浏览器前进后退</b>背后都是栈。</p>
        <div class="grid-2">
          <div class="card">
            <div class="font-semibold text-sm mb-2 text-teal-700">典型应用</div>
            <ul class="text-xs text-slate-600" style="list-style:disc inside;line-height:1.7;">
              <li>括号匹配 (LC20)</li>
              <li>中缀转后缀（Shunting-yard）</li>
              <li>最小栈 / 队列（辅助栈）</li>
              <li>DFS 迭代实现</li>
            </ul>
          </div>
          <div class="card">
            <div class="font-semibold text-sm mb-2 text-teal-700">队列 FIFO</div>
            <ul class="text-xs text-slate-600" style="list-style:disc inside;line-height:1.7;">
              <li>BFS 层次遍历</li>
              <li>生产者-消费者模型</li>
              <li>任务调度</li>
              <li>循环队列 ring buffer</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 🎮 交互：单调栈求"下一个更大元素"</h2>
        <p class="text-sm text-slate-600">数组中每个元素找右边第一个比它大的，单调栈把暴力 O(n²) 优化到 <b>O(n)</b>。
        每个元素最多入栈出栈一次，均摊 O(1)。橙色是当前栈顶，红色是正在被"顶出"的元素。</p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <input id="ms-arr" type="text" value="2,1,5,3,8,4,6,7" class="px-2 py-1 text-xs border rounded flex-1" />
            <button id="ms-run" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 自动演示</button>
            <button id="ms-reset" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">↺ 重置</button>
          </div>
          <div id="ms-log" class="text-xs text-slate-700 mt-2" style="min-height:1.5em;"></div>
        </div>

        <div id="chart-ms" style="height:300px;"></div>
        <div class="mt-3">
          <b class="text-sm text-teal-700">结果 res[i] = 右边第一个更大元素：</b>
          <div id="ms-res" class="text-xs mt-1 font-mono text-slate-700"></div>
        </div>
      </div>

      <div class="section">
        <h2>3. 代码模板</h2>
        ${MCH.code(stackCode, "python")}
      </div>

      <div class="section">
        <h2>4. 💡 LeetCode 高频题</h2>
        <table class="table">
          <thead><tr><th>题号</th><th>题目</th><th>技巧</th></tr></thead>
          <tbody>
            <tr><td><a href="https://leetcode.cn/problems/valid-parentheses/" target="_blank">20</a></td><td>有效的括号</td><td>栈基础</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/min-stack/" target="_blank">155</a></td><td>最小栈</td><td>辅助栈 O(1) 查询 min</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/next-greater-element-i/" target="_blank">496</a></td><td>下一个更大元素</td><td>⭐ 单调栈模板</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/trapping-rain-water/" target="_blank">42</a></td><td>接雨水</td><td>⭐ 单调栈 / 双指针</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/largest-rectangle-in-histogram/" target="_blank">84</a></td><td>柱状图最大矩形</td><td>⭐⭐ 单调栈 + 哨兵</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/sliding-window-maximum/" target="_blank">239</a></td><td>滑动窗口最大值</td><td>⭐ 单调队列</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/daily-temperatures/" target="_blank">739</a></td><td>每日温度</td><td>单调栈变式</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("coding_ds_stack_queue").pros,
          MCH.getById("coding_ds_stack_queue").cons,
          MCH.getById("coding_ds_stack_queue").best_for,
        )}
      </div>
    `;
  },

  mount() {
    const el = document.getElementById("chart-ms");
    if (!el) return;
    const chart = MCH.echart(el, { xAxis: { type: "category" }, yAxis: { type: "value", max: 12 }, series: [{ type: "bar", data: [] }] });

    function parseArr() {
      const raw = document.getElementById("ms-arr").value;
      return raw.split(/[,\s]+/).map(x => parseInt(x)).filter(x => !isNaN(x));
    }
    function draw(arr, stk = [], curr = -1, popping = -1, res = []) {
      const data = arr.map((v, i) => {
        let color = "#cbd5e1";
        if (i === popping) color = "#ef4444";
        else if (i === curr) color = "#14b8a6";
        else if (stk.includes(i)) color = "#f59e0b";
        else if (res[i] !== undefined && res[i] !== -1) color = "#10b981";
        return { value: v, itemStyle: { color } };
      });
      chart.setOption({ series: [{ data }] }, false);
    }
    function renderRes(res) {
      document.getElementById("ms-res").textContent = "res = [" + res.map(r => r === -1 ? "-1" : r).join(", ") + "]";
    }
    async function run() {
      const arr = parseArr();
      const n = arr.length;
      const res = new Array(n).fill(-1);
      const stk = [];
      const log = document.getElementById("ms-log");
      draw(arr);
      renderRes(res);
      for (let i = 0; i < n; i++) {
        log.textContent = `i=${i}, nums[${i}]=${arr[i]}，扫描中...`;
        draw(arr, [...stk], i, -1, res);
        await new Promise(r => setTimeout(r, 500));
        while (stk.length && arr[stk[stk.length - 1]] < arr[i]) {
          const top = stk.pop();
          res[top] = arr[i];
          log.textContent = `nums[${i}]=${arr[i]} > 栈顶 nums[${top}]=${arr[top]}，弹出并设置 res[${top}]=${arr[i]}`;
          draw(arr, [...stk], i, top, res);
          renderRes(res);
          await new Promise(r => setTimeout(r, 700));
        }
        stk.push(i);
        log.textContent = `将 i=${i}(nums=${arr[i]}) 入栈，当前栈 = [${stk.map(k => arr[k]).join(",")}]`;
        draw(arr, [...stk], -1, -1, res);
        await new Promise(r => setTimeout(r, 400));
      }
      log.textContent = `✅ 完成，栈剩余元素对应的 res 都是 -1`;
      draw(arr, [], -1, -1, res);
      renderRes(res);
    }
    document.getElementById("ms-run").onclick = run;
    document.getElementById("ms-reset").onclick = () => { const arr = parseArr(); draw(arr); renderRes(new Array(arr.length).fill(-1)); document.getElementById("ms-log").textContent = ""; };
    draw(parseArr());
    renderRes(new Array(parseArr().length).fill(-1));
  },
});
