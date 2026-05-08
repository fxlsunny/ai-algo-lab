/* 模块：排序算法全家族
 * 核心交互：柱状图实时动画 + 速度滑块 + 算法切换
 */
MCH.register("coding_sort", {
  render() {
    const pyCode = `# ========== 比较排序 8 大家族完整 Python 实现 ==========

def bubble_sort(a):
    """冒泡 O(n²)：相邻比较 + 交换（稳定）"""
    n = len(a)
    for i in range(n):
        swapped = False
        for j in range(n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
                swapped = True
        if not swapped:          # 提前终止优化：已有序
            break
    return a

def insertion_sort(a):
    """插入 O(n²)，近乎有序时 O(n)（稳定）"""
    for i in range(1, len(a)):
        key, j = a[i], i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]; j -= 1
        a[j + 1] = key
    return a

def selection_sort(a):
    """选择 O(n²)：每轮挑最小放前面（不稳定，但交换次数 O(n) 最少）"""
    n = len(a)
    for i in range(n - 1):
        min_i = i
        for j in range(i + 1, n):
            if a[j] < a[min_i]: min_i = j
        a[i], a[min_i] = a[min_i], a[i]
    return a

def quick_sort(a, lo=0, hi=None):
    """快排 Hoare 划分 + 中位数枢轴（平均 O(n log n)）"""
    if hi is None: hi = len(a) - 1
    if lo >= hi: return a
    # 三数取中防止极端输入退化到 O(n²)
    mid = (lo + hi) // 2
    if a[lo] > a[mid]: a[lo], a[mid] = a[mid], a[lo]
    if a[lo] > a[hi]:  a[lo], a[hi]  = a[hi],  a[lo]
    if a[mid] > a[hi]: a[mid], a[hi] = a[hi], a[mid]
    pivot = a[mid]
    i, j = lo, hi
    while i <= j:
        while a[i] < pivot: i += 1
        while a[j] > pivot: j -= 1
        if i <= j:
            a[i], a[j] = a[j], a[i]
            i += 1; j -= 1
    quick_sort(a, lo, j)
    quick_sort(a, i, hi)
    return a

def quick_sort_3way(a, lo=0, hi=None):
    """三路快排（荷兰国旗）：处理大量重复值效率极高，Python 标准库前身"""
    if hi is None: hi = len(a) - 1
    if lo >= hi: return a
    lt, gt, i = lo, hi, lo + 1
    pivot = a[lo]
    while i <= gt:
        if a[i] < pivot:
            a[lt], a[i] = a[i], a[lt]; lt += 1; i += 1
        elif a[i] > pivot:
            a[gt], a[i] = a[i], a[gt]; gt -= 1
        else:
            i += 1
    quick_sort_3way(a, lo, lt - 1)
    quick_sort_3way(a, gt + 1, hi)
    return a

def merge_sort(a):
    """归并 O(n log n)，稳定；外部排序基石（big data）"""
    if len(a) <= 1: return a
    m = len(a) // 2
    L, R = merge_sort(a[:m]), merge_sort(a[m:])
    out, i, j = [], 0, 0
    while i < len(L) and j < len(R):
        if L[i] <= R[j]: out.append(L[i]); i += 1
        else:            out.append(R[j]); j += 1
    out.extend(L[i:]); out.extend(R[j:])
    return out

def heap_sort(a):
    """堆排 O(n log n)，原地；嵌入式/空间受限场景首选"""
    def sift_down(a, start, end):
        root = start
        while root * 2 + 1 <= end:
            child = root * 2 + 1
            if child + 1 <= end and a[child] < a[child + 1]: child += 1
            if a[root] < a[child]:
                a[root], a[child] = a[child], a[root]; root = child
            else: return
    n = len(a)
    for i in range((n - 2) // 2, -1, -1): sift_down(a, i, n - 1)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end - 1)
    return a

# ========== 非比较排序（突破 O(n log n) 下界）==========

def counting_sort(a, max_val):
    """计数 O(n+k)：值域 k 不能太大。稳定"""
    count = [0] * (max_val + 1)
    for x in a: count[x] += 1
    idx = 0
    for v in range(max_val + 1):
        for _ in range(count[v]):
            a[idx] = v; idx += 1
    return a

def radix_sort(a):
    """基数 O(d·(n+k))：按位依次计数排序，稳定"""
    if not a: return a
    max_val = max(a); exp = 1
    while max_val // exp:
        # 计数排序按当前位
        count = [0] * 10
        out = [0] * len(a)
        for x in a: count[(x // exp) % 10] += 1
        for i in range(1, 10): count[i] += count[i - 1]
        for x in reversed(a):              # 反向保证稳定
            d = (x // exp) % 10
            count[d] -= 1
            out[count[d]] = x
        a = out; exp *= 10
    return a

# ========== TimSort (Python sorted() 背后的黑科技) ==========
# Python 内建 sorted() / list.sort() 使用 TimSort：
#   1. 识别已有有序块（run），run 长度 < minrun 时插入排序延伸
#   2. 合并栈 runs，保证栈不变量 len[i-2] > len[i-1] + len[i]
#   3. 合并采用 galloping mode 快速跳过
# 效果：已部分有序时 O(n)，最差 O(n log n)，稳定
# 参考: https://en.wikipedia.org/wiki/Timsort
`;

    return `
      ${MCH.hero({ icon: "🔀", name: "排序算法全家族", en: "Sorting Algorithms", tags: ["冒泡", "插入", "快排", "归并", "堆"], meta: ["📚 LeetCode 912", "⚡ O(n log n) 下界"] })}

      ${MCH.versionSection("coding_sort")}

      <div class="section">
        <h2>1. 核心思想</h2>
        <p class="text-sm text-slate-600">排序的本质：<b>把 n 个元素按给定比较关系重排成单调序列</b>。经过 70 年的算法演进，比较排序的时间下界被证明为 <b>Ω(n log n)</b>，突破这个下界只能依赖<b>非比较方式</b>（计数/基数/桶）。下面是一张全家族时间复杂度对比：</p>

        <table class="table">
          <thead><tr><th>算法</th><th>平均</th><th>最坏</th><th>最好</th><th>空间</th><th>稳定性</th><th>一句话</th></tr></thead>
          <tbody>
            <tr><td><b>冒泡 Bubble</b></td><td>O(n²)</td><td>O(n²)</td><td>O(n)</td><td>O(1)</td><td>✅</td><td>教学用，从不用于工程</td></tr>
            <tr><td><b>插入 Insertion</b></td><td>O(n²)</td><td>O(n²)</td><td>O(n)</td><td>O(1)</td><td>✅</td><td>小数组&lt;16 最快，TimSort 内部</td></tr>
            <tr><td><b>选择 Selection</b></td><td>O(n²)</td><td>O(n²)</td><td>O(n²)</td><td>O(1)</td><td>❌</td><td>最少交换，适合写操作昂贵</td></tr>
            <tr><td><b>快速 Quick</b></td><td>O(n log n)</td><td>O(n²)</td><td>O(n log n)</td><td>O(log n)</td><td>❌</td><td>工程最常用（STL std::sort）</td></tr>
            <tr><td><b>归并 Merge</b></td><td>O(n log n)</td><td>O(n log n)</td><td>O(n log n)</td><td>O(n)</td><td>✅</td><td>外部排序基石，稳定</td></tr>
            <tr><td><b>堆 Heap</b></td><td>O(n log n)</td><td>O(n log n)</td><td>O(n log n)</td><td>O(1)</td><td>❌</td><td>原地，嵌入式首选</td></tr>
            <tr><td><b>计数 Counting</b></td><td>O(n+k)</td><td>O(n+k)</td><td>O(n+k)</td><td>O(k)</td><td>✅</td><td>值域小整数；基数排序底层</td></tr>
            <tr><td><b>基数 Radix</b></td><td>O(d·(n+k))</td><td>O(d·(n+k))</td><td>O(d·(n+k))</td><td>O(n+k)</td><td>✅</td><td>大数量整数/字符串</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. 代码参考</h2>
        ${MCH.code(pyCode, "python")}
      </div>

      <div class="section">
        <h2>3. 🎮 交互：排序动画可视化</h2>
        <p class="text-sm text-slate-600">选择算法 + 调节速度和数据量，观察每次<b>比较 (黄色)</b> 与 <b>交换 (红色)</b>，排序完成后变绿。</p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <label class="text-xs font-semibold">算法：</label>
            <select id="sort-algo" class="px-2 py-1 text-xs border rounded">
              <option value="bubble">冒泡 Bubble</option>
              <option value="insertion">插入 Insertion</option>
              <option value="selection">选择 Selection</option>
              <option value="quick" selected>快速 Quick</option>
              <option value="merge">归并 Merge</option>
              <option value="heap">堆排 Heap</option>
            </select>
            <button id="sort-shuffle" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">🎲 随机洗牌</button>
            <button id="sort-start" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 开始</button>
            <button id="sort-stop" class="text-xs px-3 py-1 bg-red-500 text-white rounded">⏹ 停止</button>
            <span id="sort-stats" class="text-xs text-slate-600 ml-3"></span>
          </div>
          <div class="grid-2 mt-3">
            ${MCH.slider({ id: "sort-n", label: "数据量 n", min: 10, max: 80, step: 2, value: 40 })}
            ${MCH.slider({ id: "sort-speed", label: "动画延时 (ms)", min: 5, max: 200, step: 5, value: 40 })}
          </div>
        </div>

        <div id="chart-sort" style="height:360px;"></div>
      </div>

      <div class="section">
        <h2>4. 💡 LeetCode 高频题</h2>
        <div class="grid-2">
          <div class="card">
            <div class="font-semibold text-sm mb-2 text-teal-700">基础排序实现</div>
            <ul class="text-xs text-slate-600" style="list-style:disc inside;line-height:1.7;">
              <li><a href="https://leetcode.cn/problems/sort-an-array/" target="_blank">LC 912</a> · 手写排序</li>
              <li><a href="https://leetcode.cn/problems/sort-colors/" target="_blank">LC 75</a> · 三路快排（荷兰国旗）</li>
              <li><a href="https://leetcode.cn/problems/sort-list/" target="_blank">LC 148</a> · 链表归并排序</li>
              <li><a href="https://leetcode.cn/problems/merge-intervals/" target="_blank">LC 56</a> · 区间合并（先排序）</li>
            </ul>
          </div>
          <div class="card">
            <div class="font-semibold text-sm mb-2 text-teal-700">排序思想应用</div>
            <ul class="text-xs text-slate-600" style="list-style:disc inside;line-height:1.7;">
              <li><a href="https://leetcode.cn/problems/kth-largest-element-in-an-array/" target="_blank">LC 215</a> · 第 K 大（快排 partition）</li>
              <li><a href="https://leetcode.cn/problems/reverse-pairs/" target="_blank">LC 493</a> · 逆序对（归并计数）</li>
              <li><a href="https://leetcode.cn/problems/h-index/" target="_blank">LC 274</a> · H 指数（排序/计数）</li>
              <li><a href="https://leetcode.cn/problems/wiggle-sort-ii/" target="_blank">LC 324</a> · 摆动排序</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>5. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("coding_sort").pros,
          MCH.getById("coding_sort").cons,
          MCH.getById("coding_sort").best_for,
        )}
      </div>
    `;
  },

  mount() {
    const chartEl = document.getElementById("chart-sort");
    if (!chartEl) return;
    const chart = MCH.echart(chartEl, { xAxis: { type: "category" }, yAxis: { type: "value" }, series: [{ type: "bar", data: [] }] });

    let arr = [];
    let isRunning = false;
    let stopFlag = false;
    let stats = { comparisons: 0, swaps: 0 };

    const getN = () => parseInt(document.getElementById("sort-n").value);
    const getSpeed = () => parseInt(document.getElementById("sort-speed").value);
    const statsEl = document.getElementById("sort-stats");

    function updateStats() {
      statsEl.textContent = `比较：${stats.comparisons} · 交换/写入：${stats.swaps}`;
    }

    function shuffle() {
      const n = getN();
      arr = [];
      for (let i = 0; i < n; i++) arr.push(Math.floor(Math.random() * 90) + 10);
      stats = { comparisons: 0, swaps: 0 };
      updateStats();
      draw({});
    }

    function draw({ highlight = {}, finished = false }) {
      const data = arr.map((v, i) => {
        let color = "#14b8a6";
        if (finished) color = "#10b981";
        else if (highlight.swap && highlight.swap.includes(i)) color = "#ef4444";
        else if (highlight.compare && highlight.compare.includes(i)) color = "#f59e0b";
        else if (highlight.done && i < highlight.done) color = "#059669";
        else if (highlight.pivot === i) color = "#8b5cf6";
        return { value: v, itemStyle: { color } };
      });
      chart.setOption({ series: [{ data }] }, false);
    }

    const sleep = () => new Promise(r => setTimeout(r, getSpeed()));

    async function bubbleSort() {
      const n = arr.length;
      for (let i = 0; i < n - 1; i++) {
        for (let j = 0; j < n - i - 1; j++) {
          if (stopFlag) return;
          stats.comparisons++;
          draw({ highlight: { compare: [j, j + 1], done: n - i } });
          await sleep();
          if (arr[j] > arr[j + 1]) {
            [arr[j], arr[j + 1]] = [arr[j + 1], arr[j]];
            stats.swaps++;
            draw({ highlight: { swap: [j, j + 1], done: n - i } });
            await sleep();
          }
          updateStats();
        }
      }
    }

    async function insertionSort() {
      for (let i = 1; i < arr.length; i++) {
        let key = arr[i], j = i - 1;
        while (j >= 0 && arr[j] > key) {
          if (stopFlag) return;
          stats.comparisons++;
          stats.swaps++;
          arr[j + 1] = arr[j];
          draw({ highlight: { compare: [j, j + 1] } });
          await sleep();
          j--;
          updateStats();
        }
        arr[j + 1] = key;
      }
    }

    async function selectionSort() {
      const n = arr.length;
      for (let i = 0; i < n - 1; i++) {
        let minIdx = i;
        for (let j = i + 1; j < n; j++) {
          if (stopFlag) return;
          stats.comparisons++;
          draw({ highlight: { compare: [minIdx, j], done: i } });
          await sleep();
          if (arr[j] < arr[minIdx]) minIdx = j;
          updateStats();
        }
        if (minIdx !== i) {
          [arr[i], arr[minIdx]] = [arr[minIdx], arr[i]];
          stats.swaps++;
          draw({ highlight: { swap: [i, minIdx] } });
          await sleep();
        }
      }
    }

    async function quickSort(lo, hi) {
      if (lo >= hi) return;
      const pivot = arr[Math.floor((lo + hi) / 2)];
      let i = lo, j = hi;
      while (i <= j) {
        if (stopFlag) return;
        while (arr[i] < pivot) { stats.comparisons++; i++; }
        while (arr[j] > pivot) { stats.comparisons++; j--; }
        if (i <= j) {
          [arr[i], arr[j]] = [arr[j], arr[i]];
          stats.swaps++;
          draw({ highlight: { swap: [i, j] } });
          await sleep();
          i++; j--;
          updateStats();
        }
      }
      await quickSort(lo, j);
      await quickSort(i, hi);
    }

    async function mergeSort(lo, hi) {
      if (lo >= hi) return;
      const mid = (lo + hi) >> 1;
      await mergeSort(lo, mid);
      await mergeSort(mid + 1, hi);
      const tmp = [];
      let i = lo, j = mid + 1;
      while (i <= mid && j <= hi) {
        if (stopFlag) return;
        stats.comparisons++;
        draw({ highlight: { compare: [i, j] } });
        await sleep();
        if (arr[i] <= arr[j]) tmp.push(arr[i++]);
        else tmp.push(arr[j++]);
      }
      while (i <= mid) tmp.push(arr[i++]);
      while (j <= hi) tmp.push(arr[j++]);
      for (let k = 0; k < tmp.length; k++) {
        arr[lo + k] = tmp[k];
        stats.swaps++;
        draw({ highlight: { swap: [lo + k] } });
        await sleep();
        updateStats();
      }
    }

    async function heapSort() {
      const n = arr.length;
      const siftDown = async (start, end) => {
        let root = start;
        while (root * 2 + 1 <= end) {
          let child = root * 2 + 1;
          if (child + 1 <= end && arr[child] < arr[child + 1]) child++;
          stats.comparisons++;
          if (arr[root] < arr[child]) {
            [arr[root], arr[child]] = [arr[child], arr[root]];
            stats.swaps++;
            draw({ highlight: { swap: [root, child] } });
            await sleep();
            root = child;
            updateStats();
          } else break;
        }
      };
      for (let i = (n >> 1) - 1; i >= 0; i--) { await siftDown(i, n - 1); if (stopFlag) return; }
      for (let end = n - 1; end > 0; end--) {
        [arr[0], arr[end]] = [arr[end], arr[0]];
        stats.swaps++;
        draw({ highlight: { swap: [0, end], done: n - end } });
        await sleep();
        await siftDown(0, end - 1);
        if (stopFlag) return;
      }
    }

    async function run() {
      if (isRunning) return;
      isRunning = true; stopFlag = false;
      const algo = document.getElementById("sort-algo").value;
      const map = { bubble: bubbleSort, insertion: insertionSort, selection: selectionSort, quick: () => quickSort(0, arr.length - 1), merge: () => mergeSort(0, arr.length - 1), heap: heapSort };
      try { await map[algo](); } catch (e) {}
      if (!stopFlag) draw({ finished: true });
      isRunning = false;
    }

    document.getElementById("sort-shuffle").onclick = () => { stopFlag = true; setTimeout(shuffle, 50); };
    document.getElementById("sort-start").onclick = run;
    document.getElementById("sort-stop").onclick = () => { stopFlag = true; };
    MCH.bindSlider("sort-n", () => { if (!isRunning) shuffle(); });
    MCH.bindSlider("sort-speed", () => {});
    shuffle();
  },
});
