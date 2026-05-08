/* 模块：字符串 & 数学算法 */
MCH.register("coding_string_math", {
  render() {
    const code = `# ========== 字符串匹配 · 完整模板 ==========

# ---------- 1. KMP 算法（最经典 O(n+m)） ----------
def kmp(text, pat):
    if not pat: return 0
    # 构造 next 数组（失配指针）
    nxt = [0] * len(pat); k = 0
    for i in range(1, len(pat)):
        while k > 0 and pat[k] != pat[i]: k = nxt[k - 1]
        if pat[k] == pat[i]: k += 1
        nxt[i] = k
    # 主串扫描
    j = 0
    for i in range(len(text)):
        while j > 0 and pat[j] != text[i]: j = nxt[j - 1]
        if pat[j] == text[i]: j += 1
        if j == len(pat): return i - j + 1
    return -1

# ---------- 2. Rabin-Karp 滚动哈希 ----------
def rabin_karp(text, pat, base=256, mod=10**9+7):
    """多模式匹配场景下比 KMP 快"""
    n, m = len(text), len(pat)
    if m > n: return -1
    h = pow(base, m - 1, mod)
    pat_hash = text_hash = 0
    for i in range(m):
        pat_hash = (pat_hash * base + ord(pat[i])) % mod
        text_hash = (text_hash * base + ord(text[i])) % mod
    for i in range(n - m + 1):
        if pat_hash == text_hash and text[i:i+m] == pat: return i
        if i < n - m:
            text_hash = ((text_hash - ord(text[i]) * h) * base + ord(text[i+m])) % mod
            text_hash = (text_hash + mod) % mod                # 防负数
    return -1

# ---------- 3. Manacher 算法（最长回文子串 O(n)）----------
def manacher(s):
    """插 '#' 统一奇偶回文"""
    t = "#" + "#".join(s) + "#"
    n = len(t); p = [0] * n
    center = right = 0
    for i in range(n):
        mirror = 2 * center - i
        if i < right: p[i] = min(right - i, p[mirror])
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        if i + p[i] > right: center, right = i, i + p[i]
    max_i = p.index(max(p))
    return s[(max_i - p[max_i]) // 2 : (max_i + p[max_i]) // 2]

# ---------- 4. Z 函数（扩展 KMP）----------
def z_function(s):
    """z[i] = 以 s[i] 开始与 s 的最长公共前缀长度"""
    n = len(s); z = [0] * n; z[0] = n
    l = r = 0
    for i in range(1, n):
        if i < r: z[i] = min(r - i, z[i - l])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]: z[i] += 1
        if i + z[i] > r: l, r = i, i + z[i]
    return z

# ---------- 5. Trie + AC 自动机（多模式匹配）----------
from collections import deque
class AhoCorasick:
    """工业应用：敏感词过滤、病毒特征扫描"""
    def __init__(self):
        self.child = [{}]; self.fail = [0]; self.end = [[]]

    def insert(self, word, idx):
        cur = 0
        for c in word:
            if c not in self.child[cur]:
                self.child.append({}); self.fail.append(0); self.end.append([])
                self.child[cur][c] = len(self.child) - 1
            cur = self.child[cur][c]
        self.end[cur].append(idx)

    def build(self):
        """BFS 构建 fail 指针"""
        q = deque()
        for c, nxt in self.child[0].items(): q.append(nxt)
        while q:
            u = q.popleft()
            for c, v in self.child[u].items():
                self.fail[v] = self.child[self.fail[u]].get(c, 0)
                self.end[v].extend(self.end[self.fail[v]])
                q.append(v)

    def find_all(self, text):
        cur = 0; hits = []
        for i, c in enumerate(text):
            while cur and c not in self.child[cur]: cur = self.fail[cur]
            if c in self.child[cur]: cur = self.child[cur][c]
            for idx in self.end[cur]: hits.append((i, idx))
        return hits

# ========== 数学算法 ==========

# ---------- 6. 快速幂（二进制拆幂 O(log n)）----------
def fast_pow(a, n, mod=None):
    res = 1
    while n > 0:
        if n & 1:
            res = res * a
            if mod: res %= mod
        a = a * a
        if mod: a %= mod
        n >>= 1
    return res

# ---------- 7. 矩阵快速幂（求斐波那契 O(log n)）----------
def mat_mul(A, B, mod=None):
    n = len(A); C = [[0]*n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            if A[i][k]:
                for j in range(n):
                    C[i][j] += A[i][k] * B[k][j]
                    if mod: C[i][j] %= mod
    return C

def fib(n, mod=10**9+7):
    """矩阵 [[1,1],[1,0]]^n [0][1] = F(n)"""
    if n <= 1: return n
    M, R = [[1,1],[1,0]], [[1,0],[0,1]]            # 单位阵
    while n:
        if n & 1: R = mat_mul(R, M, mod)
        M = mat_mul(M, M, mod)
        n >>= 1
    return R[0][1]

# ---------- 8. 欧几里得 gcd / 扩展欧几里得 ----------
def gcd(a, b):
    while b: a, b = b, a % b
    return a

def ext_gcd(a, b):
    """返回 (g, x, y) 使 a*x + b*y = g"""
    if b == 0: return a, 1, 0
    g, x1, y1 = ext_gcd(b, a % b)
    return g, y1, x1 - (a // b) * y1

def mod_inverse(a, m):
    """乘法逆元：a * x ≡ 1 (mod m)"""
    g, x, _ = ext_gcd(a, m)
    return x % m if g == 1 else None

# ---------- 9. 埃氏筛 + 欧拉筛（线性筛素数）----------
def sieve_euler(n):
    """欧拉线性筛 O(n)：每个合数只被其最小质因子筛掉"""
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    primes = []
    for i in range(2, n + 1):
        if is_prime[i]: primes.append(i)
        for p in primes:
            if i * p > n: break
            is_prime[i * p] = False
            if i % p == 0: break                   # 关键：保证只被最小质因子筛
    return primes`;

    return `
      ${MCH.hero({ icon: "🔤", name: "字符串 & 数学算法", en: "String & Math", tags: ["KMP", "Manacher", "Trie", "快速幂", "gcd"], meta: ["⚡ O(n+m)", "📚 CP-Algorithms"] })}

      ${MCH.versionSection("coding_string_math")}

      <div class="section">
        <h2>1. KMP 的核心：next 数组</h2>
        <p class="text-sm text-slate-600">暴力字符串匹配是 O(n·m)，每次失配都要回退到 <code>i-j+1</code> 重新比。<b>KMP 的洞察</b>：
        利用模式串本身的<b>前后缀重复</b>信息，失配时只回退模式串，文本指针永不回退，达到 <b>O(n+m)</b>。</p>
        ${MCH.info(`<b>next[i]</b> 定义为：模式串 pat[0..i] 的<b>最长真前缀</b>（既是前缀也是后缀）的长度。例如 pat="ababc" 的 next=[0,0,1,2,0]。`, "tip")}
      </div>

      <div class="section">
        <h2>2. 🎮 交互：KMP 字符串匹配动画</h2>
        <p class="text-sm text-slate-600">橙色是当前比较位，红色是失配，绿色表示已匹配。观察<b>失配时模式串如何"跳"而文本指针不回退</b>。</p>
        <div class="ctrl-panel" style="margin-bottom:12px;">
          <div class="flex items-center gap-3 flex-wrap">
            <label class="text-xs font-semibold">文本：</label>
            <input id="kmp-text" type="text" value="ABABDABACDABABCABAB" class="px-2 py-1 text-xs border rounded flex-1" />
            <label class="text-xs font-semibold">模式：</label>
            <input id="kmp-pat" type="text" value="ABABCABAB" class="px-2 py-1 text-xs border rounded w-40" />
            <button id="kmp-run" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 运行 KMP</button>
          </div>
          ${MCH.slider({ id: "kmp-speed", label: "动画延时 ms", min: 100, max: 1000, step: 50, value: 400 })}
        </div>
        <div id="kmp-next" class="text-xs mt-2 font-mono text-slate-600"></div>
        <div id="kmp-viz" class="mt-2 font-mono" style="font-size:15px;line-height:1.8;letter-spacing:2px;"></div>
        <div id="kmp-log" class="text-xs text-slate-700 mt-2"></div>
      </div>

      <div class="section">
        <h2>3. 🎮 交互：快速幂（二进制拆幂）</h2>
        <p class="text-sm text-slate-600">把指数 n 写成二进制，每一位对应是否把当前 base 乘入结果。n=13 的二进制 = 1101，所以 a^13 = a^8 · a^4 · a^1。</p>
        <div class="ctrl-panel" style="margin-bottom:10px;">
          <div class="flex items-center gap-3 flex-wrap">
            <label class="text-xs">底数 a：</label>
            <input id="fp-a" type="number" value="2" class="px-2 py-1 text-xs border rounded w-16" />
            <label class="text-xs">指数 n：</label>
            <input id="fp-n" type="number" value="13" class="px-2 py-1 text-xs border rounded w-20" />
            <button id="fp-run" class="text-xs px-3 py-1 bg-emerald-600 text-white rounded">▶ 计算</button>
          </div>
        </div>
        <div id="fp-out" class="text-xs font-mono text-slate-700"></div>
      </div>

      <div class="section">
        <h2>4. 代码参考</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 💡 LeetCode 高频题</h2>
        <table class="table">
          <thead><tr><th>题号</th><th>题目</th><th>算法</th></tr></thead>
          <tbody>
            <tr><td><a href="https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/" target="_blank">28</a></td><td>找出字符串第一个匹配项下标</td><td>⭐ KMP 模板题</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/longest-palindromic-substring/" target="_blank">5</a></td><td>最长回文子串</td><td>⭐ Manacher / DP / 中心扩展</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/implement-trie-prefix-tree/" target="_blank">208</a></td><td>实现 Trie</td><td>Trie 基础</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/powx-n/" target="_blank">50</a></td><td>Pow(x, n)</td><td>⭐ 快速幂</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/divide-two-integers/" target="_blank">29</a></td><td>两数相除（不用除法）</td><td>位运算 + 快速翻倍</td></tr>
            <tr><td><a href="https://leetcode.cn/problems/word-search-ii/" target="_blank">212</a></td><td>单词搜索 II</td><td>Trie + DFS 剪枝</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 优缺点与场景</h2>
        ${MCH.prosCons(
          MCH.getById("coding_string_math").pros,
          MCH.getById("coding_string_math").cons,
          MCH.getById("coding_string_math").best_for,
        )}
      </div>
    `;
  },

  mount() {
    /* ======== KMP 动画 ======== */
    function buildNext(p) {
      const nxt = new Array(p.length).fill(0);
      let k = 0;
      for (let i = 1; i < p.length; i++) {
        while (k > 0 && p[k] !== p[i]) k = nxt[k - 1];
        if (p[k] === p[i]) k++;
        nxt[i] = k;
      }
      return nxt;
    }
    function renderKMP(text, pat, i, j, matchStart, status) {
      const colorText = text.split("").map((c, idx) => {
        let bg = "transparent";
        if (status === "matched" && idx >= matchStart && idx < matchStart + pat.length) bg = "#10b981";
        else if (idx === i) bg = status === "mismatch" ? "#ef4444" : "#f59e0b";
        return `<span style="padding:2px 4px;margin-right:1px;background:${bg};color:${bg === "transparent" ? "#334155" : "white"};border-radius:3px;">${c}</span>`;
      }).join("");

      const colorPat = pat.split("").map((c, idx) => {
        let bg = "transparent";
        if (idx === j) bg = status === "mismatch" ? "#ef4444" : "#f59e0b";
        else if (idx < j) bg = "#86efac";
        return `<span style="padding:2px 4px;margin-right:1px;background:${bg};color:${bg === "transparent" ? "#334155" : "white"};border-radius:3px;">${c}</span>`;
      }).join("");

      const padSpaces = "\u00A0".repeat(Math.max(0, i - j) * 3);
      document.getElementById("kmp-viz").innerHTML =
        `<div>text:  ${colorText}</div><div style="margin-top:4px;">pat:&nbsp;&nbsp;${padSpaces}${colorPat}</div>`;
    }
    async function runKMP() {
      const text = document.getElementById("kmp-text").value;
      const pat = document.getElementById("kmp-pat").value;
      const speed = parseInt(document.getElementById("kmp-speed").value);
      const log = document.getElementById("kmp-log");
      if (!pat) { log.textContent = "模式串为空"; return; }

      const nxt = buildNext(pat);
      document.getElementById("kmp-next").innerHTML =
        `next 数组 = [${nxt.join(", ")}] （pat[${pat}]）`;

      let j = 0;
      for (let i = 0; i < text.length; i++) {
        renderKMP(text, pat, i, j, -1, "compare");
        await new Promise(r => setTimeout(r, speed));
        while (j > 0 && pat[j] !== text[i]) {
          log.textContent = `text[${i}]='${text[i]}' ≠ pat[${j}]='${pat[j]}'，模式串跳到 next[${j - 1}]=${nxt[j - 1]}`;
          renderKMP(text, pat, i, j, -1, "mismatch");
          await new Promise(r => setTimeout(r, speed));
          j = nxt[j - 1];
          renderKMP(text, pat, i, j, -1, "compare");
          await new Promise(r => setTimeout(r, speed / 2));
        }
        if (pat[j] === text[i]) { j++; log.textContent = `匹配 '${text[i]}' ✓，j→${j}`; }
        else { log.textContent = `j=0 且失配，i 向前走`; }
        if (j === pat.length) {
          const start = i - j + 1;
          log.textContent = `✅ 找到匹配 起始下标 = ${start}`;
          renderKMP(text, pat, i, j - 1, start, "matched");
          return;
        }
      }
      log.textContent = "❌ 未找到匹配";
    }
    document.getElementById("kmp-run").onclick = runKMP;
    MCH.bindSlider("kmp-speed", () => {});
    renderKMP(document.getElementById("kmp-text").value, document.getElementById("kmp-pat").value, -1, 0, -1, "init");
    document.getElementById("kmp-next").innerHTML =
      `next 数组（点击"运行 KMP"计算）`;

    /* ======== 快速幂 ======== */
    document.getElementById("fp-run").onclick = () => {
      let a = parseInt(document.getElementById("fp-a").value);
      let n = parseInt(document.getElementById("fp-n").value);
      const steps = [];
      const binN = n.toString(2);
      let res = 1, base = a, pos = 0;
      while (n > 0) {
        const bit = n & 1;
        steps.push({
          pos,
          bit,
          base,
          res,
          action: bit ? `第 ${pos} 位=1，res *= ${base} → res=${res * base}` : `第 ${pos} 位=0，跳过`,
        });
        if (bit) res *= base;
        base *= base;
        n >>= 1;
        pos++;
      }
      const out = document.getElementById("fp-out");
      out.innerHTML = `指数二进制 = <b>${binN}</b>（从低位到高位）<br/>`
        + steps.map(s => `<div style="margin:4px 0;">${s.action}</div>`).join("")
        + `<div style="margin-top:8px;"><b class="text-teal-700">最终结果 = ${res}</b>（共 ${steps.length} 步乘法，暴力需 ${parseInt(document.getElementById("fp-n").value)} 步）</div>`;
    };
    document.getElementById("fp-run").click();
  },
});
