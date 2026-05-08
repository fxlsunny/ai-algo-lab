/* 模块：链表 / 二叉树 / 堆
 * 交互：BST 动态插入 + 堆 push/pop 动画
 */
MCH.register("coding_ds_linked_tree", {
  render() {
    const code = `# ========== 1. 单链表：反转（迭代 & 递归） ==========
class ListNode:
    def __init__(self, val=0, next=None):
        self.val, self.next = val, next

def reverse_iter(head):
    """O(n) 时间，O(1) 空间"""
    prev, cur = None, head
    while cur:
        nxt = cur.next        # 暂存后继
        cur.next = prev       # 反向指针
        prev, cur = cur, nxt  # 前进
    return prev

def reverse_recur(head):
    """递归实现，空间 O(n)"""
    if not head or not head.next: return head
    new_head = reverse_recur(head.next)
    head.next.next = head
    head.next = None
    return new_head

# ========== 2. 链表快慢指针：检测环（Floyd 龟兔算法） ==========
def has_cycle(head):
    """LC141：龟兔相遇则有环"""
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast: return True
    return False

def find_cycle_start(head):
    """LC142：找环的入口（数学推导：相遇后从头再走一步步相遇点）"""
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:
            p = head
            while p is not slow:
                p, slow = p.next, slow.next
            return p
    return None

# ========== 3. 二叉搜索树 BST ==========
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

class BST:
    def __init__(self): self.root = None

    def insert(self, v):
        def _ins(node, v):
            if not node: return TreeNode(v)
            if v < node.val:   node.left  = _ins(node.left,  v)
            elif v > node.val: node.right = _ins(node.right, v)
            return node
        self.root = _ins(self.root, v)

    def search(self, v):
        node = self.root
        while node:
            if v == node.val: return node
            node = node.left if v < node.val else node.right
        return None

    def inorder(self):
        """中序遍历：BST 中序必然升序"""
        res, stack, node = [], [], self.root
        while stack or node:
            while node: stack.append(node); node = node.left
            node = stack.pop(); res.append(node.val); node = node.right
        return res

    def delete(self, v):
        """BST 删除三种情况：叶子/单子/双子（找右子树最小后继替换）"""
        def _del(node, v):
            if not node: return None
            if   v < node.val: node.left  = _del(node.left,  v)
            elif v > node.val: node.right = _del(node.right, v)
            else:
                if not node.left:  return node.right
                if not node.right: return node.left
                succ = node.right
                while succ.left: succ = succ.left
                node.val = succ.val
                node.right = _del(node.right, succ.val)
            return node
        self.root = _del(self.root, v)

# ========== 4. 二叉树遍历（递归 + 迭代） ==========
def inorder_iter(root):
    """中序迭代（面试高频：不用递归）"""
    res, stack, node = [], [], root
    while stack or node:
        while node: stack.append(node); node = node.left
        node = stack.pop(); res.append(node.val); node = node.right
    return res

def level_order(root):
    """层序 BFS（LC102）"""
    from collections import deque
    if not root: return []
    res, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            n = q.popleft(); level.append(n.val)
            if n.left:  q.append(n.left)
            if n.right: q.append(n.right)
        res.append(level)
    return res

# ========== 5. 二叉堆（小顶堆 / MinHeap） ==========
import heapq
h = []
for x in [5, 1, 3, 7, 2]:
    heapq.heappush(h, x)     # O(log n)
print(heapq.heappop(h))       # → 1 (O(log n))
print(heapq.nsmallest(3, h))  # → 最小 3 个 (O(k log n))

# 自己实现小顶堆（面试常考）
class MinHeap:
    def __init__(self): self.a = []

    def push(self, v):
        self.a.append(v)
        self._sift_up(len(self.a) - 1)

    def pop(self):
        if not self.a: raise IndexError
        top = self.a[0]
        last = self.a.pop()
        if self.a:
            self.a[0] = last
            self._sift_down(0)
        return top

    def _sift_up(self, i):
        while i > 0:
            p = (i - 1) // 2
            if self.a[p] > self.a[i]:
                self.a[p], self.a[i] = self.a[i], self.a[p]
                i = p
            else: break

    def _sift_down(self, i):
        n = len(self.a)
        while 2 * i + 1 < n:
            c = 2 * i + 1
            if c + 1 < n and self.a[c + 1] < self.a[c]: c += 1
            if self.a[i] > self.a[c]:
                self.a[i], self.a[c] = self.a[c], self.a[i]
                i = c
            else: break

# ========== 6. 堆的工程应用：Top-K ==========
def top_k_largest(nums, k):
    """O(n log k)：维护大小为 k 的最小堆"""
    import heapq
    return heapq.nlargest(k, nums)

def kth_largest(nums, k):
    """LC215：第 K 大"""
    import heapq
    h = nums[:k]; heapq.heapify(h)
    for x in nums[k:]:
        if x > h[0]: heapq.heapreplace(h, x)
    return h[0]

# ========== 7. Trie 字典树 ==========
class Trie:
    def __init__(self):
        self.child = {}; self.end = False

    def insert(self, word):
        node = self
        for c in word:
            node = node.child.setdefault(c, Trie())
        node.end = True

    def search(self, word):
        node = self._walk(word)
        return node is not None and node.end

    def starts_with(self, prefix):
        return self._walk(prefix) is not None

    def _walk(self, s):
        node = self
        for c in s:
            if c not in node.child: return None
            node = node.child[c]
        return node`;

    return `
      ${MCH.hero({ icon: "🌳", name: "链表 / 二叉树 / 堆", en: "Linked List · Tree · Heap", tags: ["指针", "递归", "BST", "堆"], meta: ["📚 CLRS 第 6/12/13 章", "⚡ O(log n)"] })}

      ${MCH.versionSection("coding_ds_linked_tree")}

      <div class="section">
        <h2>1. 三大基础非线性结构</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #14b8a6;">
            <div class="font-semibold text-sm mb-2 text-teal-700">🔗 链表</div>
            <p class="text-xs text-slate-600">指针串联节点，O(1) 插入删除，O(n) 查找。Linux kernel 到处都是双向循环链表（list.h）。</p>
            <ul class="text-[11px] text-slate-500 mt-2" style="list-style:disc inside;">
              <li>单向/双向/循环</li>
              <li>快慢指针判环（Floyd）</li>
              <li>头插 vs 尾插</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #0d9488;">
            <div class="font-semibold text-sm mb-2 text-teal-700">🌳 二叉树 / BST</div>
            <p class="text-xs text-slate-600">左子树 &lt; 根 &lt; 右子树。中序遍历天然有序。不平衡时退化为链表，需 AVL/红黑树保证 O(log n)。</p>
            <ul class="text-[11px] text-slate-500 mt-2" style="list-style:disc inside;">
              <li>前中后序 / 层序</li>
              <li>AVL / 红黑树</li>
              <li>数据库索引底层</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #0f766e;">
            <div class="font-semibold text-sm mb-2 text-teal-700">⛰ 二叉堆</div>
            <p class="text-xs text-slate-600">完全二叉树，父节点 ≤ 左右孩子（小顶堆）。数组实现，父 i 的孩子是 2i+1, 2i+2。</p>
            <ul class="text-[11px] text-slate-500 mt-2" style="list-style:disc inside;">
              <li>push/pop O(log n)</li>
              <li>build O(n)</li>
              <li>Top-K / 优先队列</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 🎮 交互：BST 动态插入</h2>
        <p class="text-sm text-slate-600">输入一串数字，逐个插入 BST 并可视化树形结构。观察<b>相同序列但插入顺序不同</b>会产生完全不同的树形——这就是为什么需要自平衡树。</p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <input id="bst-nums" type="text" value="50,30,70,20,40,60,80,10,35,65" class="px-2 py-1 text-xs border rounded flex-1" />
            <button id="bst-build" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">🌱 构建 BST</button>
            <button id="bst-sorted" class="text-xs px-3 py-1 bg-amber-600 text-white rounded">🪜 按顺序插入（退化）</button>
          </div>
          <div id="bst-order" class="text-xs text-slate-700 mt-2"></div>
        </div>

        <div id="chart-bst" style="height:380px;"></div>
      </div>

      <div class="section">
        <h2>3. 🎮 交互：二叉堆 push / pop 动画</h2>
        <p class="text-sm text-slate-600">小顶堆用数组实现。push 时在数组末尾添加后向上"sift-up"；pop 时把根换成末尾元素再向下"sift-down"。</p>

        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <input id="heap-val" type="number" value="5" class="px-2 py-1 text-xs border rounded w-20" />
            <button id="heap-push" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">↑ push</button>
            <button id="heap-pop" class="text-xs px-3 py-1 bg-red-500 text-white rounded">↓ pop (min)</button>
            <button id="heap-rand" class="text-xs px-3 py-1 bg-indigo-600 text-white rounded">🎲 随机 push 一个</button>
            <button id="heap-clear" class="text-xs px-3 py-1 bg-slate-500 text-white rounded">🗑 清空</button>
            <span id="heap-log" class="text-xs text-slate-700 ml-3"></span>
          </div>
        </div>

        <div id="chart-heap" style="height:340px;"></div>
        <div class="text-xs text-slate-600 mt-2 font-mono" id="heap-arr"></div>
      </div>

      <div class="section">
        <h2>4. 完整代码参考</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 💡 LeetCode 高频题</h2>
        <table class="table">
          <thead><tr><th>题号</th><th>题目</th><th>数据结构</th></tr></thead>
          <tbody>
            <tr><td><a href="https://leetcode.cn/problems/reverse-linked-list/" target="_blank">206</a></td><td>反转链表</td><td>链表</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/linked-list-cycle/" target="_blank">141</a></td><td>环形链表</td><td>⭐ 快慢指针</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/linked-list-cycle-ii/" target="_blank">142</a></td><td>环形链表 II（入口）</td><td>⭐ Floyd 算法</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/merge-k-sorted-lists/" target="_blank">23</a></td><td>合并 K 个升序链表</td><td>⭐ 最小堆</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/binary-tree-level-order-traversal/" target="_blank">102</a></td><td>二叉树层序遍历</td><td>BFS + 队列</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/validate-binary-search-tree/" target="_blank">98</a></td><td>验证 BST</td><td>中序 / 递归上下界</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/kth-largest-element-in-an-array/" target="_blank">215</a></td><td>第 K 大</td><td>⭐ 最小堆 / 快排 partition</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/top-k-frequent-elements/" target="_blank">347</a></td><td>前 K 高频元素</td><td>⭐ 堆 / 桶排序</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/find-median-from-data-stream/" target="_blank">295</a></td><td>数据流的中位数</td><td>⭐⭐ 对顶堆</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/implement-trie-prefix-tree/" target="_blank">208</a></td><td>实现 Trie</td><td>Trie 基础</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("coding_ds_linked_tree").pros,
          MCH.getById("coding_ds_linked_tree").cons,
          MCH.getById("coding_ds_linked_tree").best_for,
        )}
      </div>
    `;
  },

  mount() {
    /* ======== BST 可视化 ======== */
    const bstEl = document.getElementById("chart-bst");
    if (bstEl) {
      let bstChart = null;
      try {
        bstChart = echarts.init(bstEl, null, { renderer: "canvas" });
        MCH.charts.push(bstChart);
      } catch (e) {
        console.error("[BST] init error:", e);
        return;
      }

      // BST 用简单的类代替嵌套 children，最后转 ECharts tree 数据
      function buildBST(nums) {
        let root = null;
        function ins(node, v) {
          if (!node) return { val: v, left: null, right: null };
          if (v < node.val) node.left = ins(node.left, v);
          else if (v > node.val) node.right = ins(node.right, v);
          return node;
        }
        nums.forEach(v => { root = ins(root, v); });
        return root;
      }

      // 移除 placeholder 方案：直接返回 children，空则 []
      function toEChartsTree(node) {
        if (!node) return null;
        const children = [];
        if (node.left) { const l = toEChartsTree(node.left); if (l) children.push(l); }
        if (node.right) { const r = toEChartsTree(node.right); if (r) children.push(r); }
        return { name: String(node.val), value: 1, children };
      }

      function inorder(node) {
        const res = [];
        (function dfs(n) { if (!n) return; dfs(n.left); res.push(n.val); dfs(n.right); })(node);
        return res;
      }

      function height(node) {
        if (!node) return 0;
        return 1 + Math.max(height(node.left), height(node.right));
      }

      function build(input) {
        try {
          const nums = input.split(/[,\s]+/).map(x => parseInt(x)).filter(x => !isNaN(x));
          if (!nums.length) return;
          const root = buildBST(nums);
          const treeData = root ? [toEChartsTree(root)] : [];
          const option = {
            tooltip: { trigger: "item" },
            series: [{
              type: "tree",
              data: treeData,
              left: "5%", right: "5%", top: "5%", bottom: "5%",
              symbolSize: 36,
              orient: "vertical",
              initialTreeDepth: -1,
              expandAndCollapse: false,
              label: { position: "inside", color: "#fff", fontWeight: 700, fontSize: 13 },
              itemStyle: { color: "#14b8a6" },
              lineStyle: { color: "#94a3b8", width: 1.5, curveness: 0.3 },
              animationDuration: 600,
              animationEasing: "elasticOut",
            }],
          };
          bstChart.setOption(option, true);
          const ord = inorder(root);
          const h = height(root);
          const ideal = Math.ceil(Math.log2(nums.length + 1));
          document.getElementById("bst-order").innerHTML =
            `中序遍历（升序）：<b class="font-mono text-teal-700">${ord.join(" → ")}</b> · 树高 = <b>${h}</b>（理想 ${ideal}）`;
        } catch (e) {
          console.error("[BST] build error:", e);
        }
      }

      const btnBuild = document.getElementById("bst-build");
      const btnSorted = document.getElementById("bst-sorted");
      if (btnBuild) btnBuild.addEventListener("click", () => build(document.getElementById("bst-nums").value));
      if (btnSorted) btnSorted.addEventListener("click", () => {
        document.getElementById("bst-nums").value = "10,20,30,40,50,60,70,80";
        build(document.getElementById("bst-nums").value);
      });
      window.addEventListener("resize", () => { if (bstChart) bstChart.resize(); });
      build(document.getElementById("bst-nums").value);
    }

    /* ======== 二叉堆可视化 ======== */
    const heapEl = document.getElementById("chart-heap");
    if (heapEl) {
      let heapChart = null;
      try {
        heapChart = echarts.init(heapEl, null, { renderer: "canvas" });
        MCH.charts.push(heapChart);
      } catch (e) {
        console.error("[Heap] init error:", e);
        return;
      }

      const heap = [];
      const arrEl = document.getElementById("heap-arr");
      const log = document.getElementById("heap-log");

      function arrToTree(i = 0) {
        if (i >= heap.length) return null;
        const node = { name: String(heap[i]), value: 1, children: [] };
        const l = arrToTree(2 * i + 1);
        const r = arrToTree(2 * i + 2);
        if (l) node.children.push(l);
        if (r) node.children.push(r);
        return node;
      }

      function draw() {
        try {
          const tree = arrToTree();
          const option = {
            tooltip: { trigger: "item", formatter: (p) => {
              if (p.data.idx !== undefined) return `heap[${p.data.idx}] = ${p.data.name}`;
              return p.data.name;
            }},
            series: [{
              type: "tree",
              data: tree ? [tree] : [],
              left: "5%", right: "5%", top: "5%", bottom: "5%",
              symbolSize: 40,
              orient: "vertical",
              initialTreeDepth: -1,
              expandAndCollapse: false,
              label: { position: "inside", color: "#fff", fontWeight: 700, fontSize: 14 },
              itemStyle: { color: "#f59e0b" },
              lineStyle: { color: "#94a3b8", width: 1.5 },
              animationDuration: 400,
              animationEasing: "elasticOut",
            }],
          };
          heapChart.setOption(option, true);
          arrEl.textContent = `arr = [${heap.join(", ")}]    长度 = ${heap.length}`;
        } catch (e) {
          console.error("[Heap] draw error:", e);
          if (arrEl) arrEl.textContent = `[渲染错误: ${e.message}]`;
        }
      }

      function siftUp(i) {
        while (i > 0) {
          const p = (i - 1) >> 1;
          if (heap[p] > heap[i]) {
            [heap[p], heap[i]] = [heap[i], heap[p]];
            i = p;
          } else break;
        }
      }

      function siftDown(i) {
        const n = heap.length;
        while (2 * i + 1 < n) {
          let child = 2 * i + 1;
          if (child + 1 < n && heap[child + 1] < heap[child]) child++;
          if (heap[i] > heap[child]) {
            [heap[i], heap[child]] = [heap[child], heap[i]];
            i = child;
          } else break;
        }
      }

      const pushBtn = document.getElementById("heap-push");
      const popBtn = document.getElementById("heap-pop");
      const randBtn = document.getElementById("heap-rand");
      const clearBtn = document.getElementById("heap-clear");
      const valInput = document.getElementById("heap-val");

      if (pushBtn) pushBtn.addEventListener("click", () => {
        const v = parseInt(valInput.value);
        if (isNaN(v)) { if (log) log.textContent = "请输入数字"; return; }
        heap.push(v);
        siftUp(heap.length - 1);
        if (log) log.textContent = `push(${v}) → root=${heap[0]}（堆大小 ${heap.length}）`;
        draw();
      });
      if (popBtn) popBtn.addEventListener("click", () => {
        if (!heap.length) { if (log) log.textContent = "堆空，无法 pop"; return; }
        const min = heap[0];
        const last = heap.pop();
        if (heap.length) { heap[0] = last; siftDown(0); }
        if (log) log.textContent = `pop → ${min}（新 root=${heap[0] ?? "空"}，大小 ${heap.length}）`;
        draw();
      });
      if (randBtn) randBtn.addEventListener("click", () => {
        const v = Math.floor(Math.random() * 99) + 1;
        valInput.value = v;
        heap.push(v);
        siftUp(heap.length - 1);
        if (log) log.textContent = `随机 push(${v}) → root=${heap[0]}（堆大小 ${heap.length}）`;
        draw();
      });
      if (clearBtn) clearBtn.addEventListener("click", () => {
        heap.length = 0;
        if (log) log.textContent = "已清空";
        draw();
      });

      // 初始化
      [8, 3, 5, 1, 9, 6, 2].forEach(v => { heap.push(v); siftUp(heap.length - 1); });
      if (log) log.textContent = `初始数据 [8,3,5,1,9,6,2]`;
      window.addEventListener("resize", () => { if (heapChart) heapChart.resize(); });
      draw();
      console.log("[Heap] mounted, init", heap.length, "items");
    }
  },
});
