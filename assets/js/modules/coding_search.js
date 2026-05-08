/* 模块：查找算法 (二分 / 哈希) */
MCH.register("coding_search", {
  render() {
    const binaryCode = `# ========== 二分查找 · 三大模板 ==========

# 模板 1：闭区间 [l, r]，找精确等于
def binary_search(nums, target):
    l, r = 0, len(nums) - 1
    while l <= r:
        m = l + (r - l) // 2        # 防 int 溢出
        if nums[m] == target: return m
        if nums[m] < target: l = m + 1
        else: r = m - 1
    return -1

# 模板 2：左闭右开 [l, r)，找第一个 >= target 的下标
def lower_bound(nums, target):
    l, r = 0, len(nums)
    while l < r:
        m = (l + r) // 2
        if nums[m] < target: l = m + 1
        else: r = m
    return l

# 找第一个 > target 的下标
def upper_bound(nums, target):
    l, r = 0, len(nums)
    while l < r:
        m = (l + r) // 2
        if nums[m] <= target: l = m + 1
        else: r = m
    return l

# 出现次数 = upper_bound - lower_bound
def count(nums, target):
    return upper_bound(nums, target) - lower_bound(nums, target)

# ========== 答案二分（Parametric Search）==========
# 模式：把 "能否在 X 条件下完成" 写成单调函数 check(X)，在解空间二分

def min_eating_speed(piles, h):
    """LC875：爱吃香蕉的珂珂。速度 k 越大越快吃完，找最小可行 k"""
    def can_finish(k):
        return sum((p + k - 1) // k for p in piles) <= h
    l, r = 1, max(piles)
    while l < r:
        m = (l + r) // 2
        if can_finish(m): r = m          # 能完成，尝试更小
        else: l = m + 1                   # 不能完成，加速
    return l

def ship_within_days(weights, days):
    """LC1011：D 天运送包裹的最小载重"""
    def can_ship(cap):
        d, cur = 1, 0
        for w in weights:
            if cur + w > cap: d += 1; cur = 0
            cur += w
        return d <= days
    l, r = max(weights), sum(weights)
    while l < r:
        m = (l + r) // 2
        if can_ship(m): r = m
        else: l = m + 1
    return l

# ========== 浮点二分（精度型）==========
def sqrt_float(x, eps=1e-9):
    """牛顿迭代更快，但二分也能 O(log(1/eps))"""
    l, r = 0, max(1, x)
    while r - l > eps:
        m = (l + r) / 2
        if m * m < x: l = m
        else: r = m
    return l`;

    const hashCode = `# ========== 哈希表 · 完整实现 ==========

# 1. 链地址法（Java HashMap 默认）
class HashMap:
    def __init__(self, cap=16):
        self.cap = cap
        self.buckets = [[] for _ in range(cap)]
        self.size = 0

    def _h(self, key):
        return hash(key) & (self.cap - 1)   # 用位运算代替 %，要求 cap 是 2 的幂

    def put(self, k, v):
        bucket = self.buckets[self._h(k)]
        for i, (kk, _) in enumerate(bucket):
            if kk == k:
                bucket[i] = (k, v); return
        bucket.append((k, v)); self.size += 1
        # 负载因子 > 0.75 扩容
        if self.size > self.cap * 0.75:
            self._resize()

    def get(self, k, default=None):
        for kk, vv in self.buckets[self._h(k)]:
            if kk == k: return vv
        return default

    def remove(self, k):
        bucket = self.buckets[self._h(k)]
        for i, (kk, _) in enumerate(bucket):
            if kk == k:
                bucket.pop(i); self.size -= 1; return True
        return False

    def _resize(self):
        old = self.buckets
        self.cap *= 2
        self.buckets = [[] for _ in range(self.cap)]
        self.size = 0
        for bucket in old:
            for k, v in bucket: self.put(k, v)

# 2. 开放寻址（Python dict / Robin Hood Hashing 思想）
class OpenAddressingMap:
    EMPTY, TOMBSTONE = object(), object()

    def __init__(self, cap=16):
        self.cap = cap
        self.keys = [self.EMPTY] * cap
        self.vals = [None] * cap
        self.size = 0

    def _probe(self, k):
        i = hash(k) & (self.cap - 1)
        while self.keys[i] is not self.EMPTY:
            if self.keys[i] == k: return i
            i = (i + 1) & (self.cap - 1)          # 线性探测
        return i

    def put(self, k, v):
        i = self._probe(k)
        if self.keys[i] is self.EMPTY: self.size += 1
        self.keys[i], self.vals[i] = k, v

    def get(self, k):
        i = self._probe(k)
        return self.vals[i] if self.keys[i] is not self.EMPTY else None

# ========== Bloom Filter（概率型存在性查询）==========
import hashlib
class BloomFilter:
    """空间极省，O(1) 查询，存在假阳但无假阴。适合缓存前置去重"""
    def __init__(self, size=1 << 20, k=7):
        self.bits = bytearray(size // 8)
        self.size = size; self.k = k

    def _hashes(self, item):
        # 双哈希构造 k 个独立哈希（Kirsch-Mitzenmacher）
        h1 = int(hashlib.md5(str(item).encode()).hexdigest(), 16)
        h2 = int(hashlib.sha1(str(item).encode()).hexdigest(), 16)
        return [(h1 + i * h2) % self.size for i in range(self.k)]

    def add(self, item):
        for h in self._hashes(item):
            self.bits[h // 8] |= 1 << (h % 8)

    def contains(self, item):
        return all(self.bits[h // 8] & (1 << (h % 8)) for h in self._hashes(item))`;

    return `
      ${MCH.hero({ icon: "🔍", name: "查找算法 · 二分 / 哈希", en: "Searching Algorithms", tags: ["O(log n)", "O(1)", "二分答案"], meta: ["⚡ 海量数据首选", "📚 LeetCode 二分专题"] })}

      ${MCH.versionSection("coding_search")}

      <div class="section">
        <h2>1. 二分查找核心</h2>
        <p class="text-sm text-slate-600">二分的本质是<b>在单调序列上对半折叠</b>，每次把搜索空间减半，T(n) = T(n/2) + O(1) → <b>O(log n)</b>。
        虽然一行代码能写完，但<b>边界条件</b>让无数工程师翻车——<b>闭区间 [l, r] 还是左闭右开 [l, r) 要选一种用到底</b>。</p>
        ${MCH.info(`<b>进阶用法：答案二分（Parametric Search）</b><br/>当题目"最大化最小值/最小化最大值"时，把答案 x 是否可行当作单调函数 check(x)，在解空间 [L, R] 上二分。典型：分割数组最大值、运送包裹、珂珂吃香蕉。`, "tip")}
        ${MCH.code(binaryCode, "python")}
      </div>

      <div class="section">
        <h2>2. 🎮 交互：二分查找动画</h2>
        <p class="text-sm text-slate-600">在一个升序数组里用二分查找目标值，蓝色是当前搜索区间 <code>[l, r]</code>，黄色是 <code>mid</code>，红色是排除的部分，绿色是找到的位置。</p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <label class="text-xs font-semibold">目标值：</label>
            <input id="bs-target" type="number" value="47" class="px-2 py-1 text-xs border rounded w-20" />
            <button id="bs-new" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">🎲 新数组</button>
            <button id="bs-step" class="text-xs px-3 py-1 bg-indigo-600 text-white rounded">▶ 单步</button>
            <button id="bs-auto" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶▶ 自动</button>
            <button id="bs-reset" class="text-xs px-3 py-1 bg-red-500 text-white rounded">↺ 重置</button>
            <span id="bs-status" class="text-xs text-slate-700 ml-2"></span>
          </div>
          ${MCH.slider({ id: "bs-n", label: "数组长度", min: 10, max: 50, step: 1, value: 20 })}
        </div>

        <div id="chart-bs" style="height:300px;"></div>
      </div>

      <div class="section">
        <h2>3. 哈希表实现</h2>
        <p class="text-sm text-slate-600">哈希表把 key 通过哈希函数 h(k) 映射到桶，冲突解决有两大流派：</p>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #14b8a6;">
            <div class="font-semibold text-sm mb-2">🔗 链地址法 (Separate Chaining)</div>
            <ul class="text-xs text-slate-600" style="list-style:disc inside;">
              <li>每个桶是个链表/红黑树（Java 8 后当链表长度 &gt; 8 树化）</li>
              <li>实现简单，负载因子可 &gt; 1</li>
              <li>缓存不友好，有指针开销</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <div class="font-semibold text-sm mb-2">➡️ 开放寻址 (Open Addressing)</div>
            <ul class="text-xs text-slate-600" style="list-style:disc inside;">
              <li>冲突时线性/二次探测下一个桶</li>
              <li>Python dict / Google Swiss Table</li>
              <li>缓存友好，但负载因子需控在 0.75 以下</li>
            </ul>
          </div>
        </div>
        ${MCH.code(hashCode, "python")}
      </div>

      <div class="section">
        <h2>4. 💡 LeetCode 高频题</h2>
        <table class="table">
          <thead><tr><th>题号</th><th>题目</th><th>思想</th></tr></thead>
          <tbody>
            <tr><td><a href="https://leetcode.cn/problems/binary-search/" target="_blank">704</a></td><td>二分查找</td><td>基础模板</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/find-first-and-last-position-of-element-in-sorted-array/" target="_blank">34</a></td><td>有序数组首末位置</td><td>lower_bound + upper_bound</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/find-peak-element/" target="_blank">162</a></td><td>寻找峰值</td><td>二分非有序数组，比较 mid 与 mid+1</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/koko-eating-bananas/" target="_blank">875</a></td><td>爱吃香蕉的珂珂</td><td>⭐ 答案二分</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/capacity-to-ship-packages-within-d-days/" target="_blank">1011</a></td><td>D 天运送包裹</td><td>⭐ 答案二分</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/two-sum/" target="_blank">1</a></td><td>两数之和</td><td>哈希表 O(n)</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/longest-consecutive-sequence/" target="_blank">128</a></td><td>最长连续序列</td><td>哈希 + 技巧性扫描 O(n)</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("coding_search").pros,
          MCH.getById("coding_search").cons,
          MCH.getById("coding_search").best_for,
        )}
      </div>
    `;
  },

  mount() {
    const el = document.getElementById("chart-bs");
    if (!el) return;
    const chart = MCH.echart(el, { xAxis: { type: "category" }, yAxis: { type: "value", max: 110 }, series: [{ type: "bar", data: [] }] });
    let arr = [], l = 0, r = 0, foundIdx = -1, finished = false;

    function genArr() {
      const n = parseInt(document.getElementById("bs-n").value);
      const set = new Set();
      while (set.size < n) set.add(Math.floor(Math.random() * 100) + 1);
      arr = [...set].sort((a, b) => a - b);
      reset();
    }
    function reset() {
      l = 0; r = arr.length - 1; foundIdx = -1; finished = false;
      updateStatus(`区间 [${l}, ${r}]，待搜索`);
      draw();
    }
    function draw() {
      const target = parseInt(document.getElementById("bs-target").value);
      const data = arr.map((v, i) => {
        let color = "#cbd5e1";
        if (finished) {
          if (i === foundIdx) color = "#10b981";
        } else if (i >= l && i <= r) {
          color = "#14b8a6";
          if (i === ((l + r) >> 1)) color = "#f59e0b";
        } else {
          color = "#fca5a5";
        }
        return { value: v, itemStyle: { color }, label: v === target ? { show: true, position: "top", formatter: "🎯" } : { show: false } };
      });
      chart.setOption({ series: [{ data }] }, false);
    }
    function step() {
      if (finished) return;
      const target = parseInt(document.getElementById("bs-target").value);
      if (l > r) { finished = true; updateStatus(`❌ 未找到 ${target}`); draw(); return; }
      const m = (l + r) >> 1;
      if (arr[m] === target) {
        foundIdx = m; finished = true;
        updateStatus(`✅ 在 index=${m} 找到 ${target}（用了 ${countSteps()} 步）`);
        draw(); return;
      }
      if (arr[m] < target) { updateStatus(`arr[${m}]=${arr[m]} < ${target}，向右搜索`); l = m + 1; }
      else { updateStatus(`arr[${m}]=${arr[m]} > ${target}，向左搜索`); r = m - 1; }
      draw();
    }
    let autoTimer = null;
    function autoRun() {
      if (autoTimer) { clearInterval(autoTimer); autoTimer = null; return; }
      autoTimer = setInterval(() => { if (finished) { clearInterval(autoTimer); autoTimer = null; return; } step(); }, 600);
    }
    function countSteps() { return Math.ceil(Math.log2(arr.length + 1)); }
    function updateStatus(msg) { document.getElementById("bs-status").textContent = msg; }

    document.getElementById("bs-new").onclick = genArr;
    document.getElementById("bs-step").onclick = step;
    document.getElementById("bs-auto").onclick = autoRun;
    document.getElementById("bs-reset").onclick = reset;
    document.getElementById("bs-target").oninput = () => { if (!finished) draw(); };
    MCH.bindSlider("bs-n", () => genArr());
    genArr();
  },
});
