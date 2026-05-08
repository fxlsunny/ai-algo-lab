/* 模块：基础编程算法 · 总览
 * 作为 "coding" 大类的入口，展示 8 个子模块、对标资源和典型面试题路线
 */
MCH.register("coding_overview", {
  render() {
    const codingAlgos = (MCH.registry || []).filter(a => a.category === "coding");
    const cards = codingAlgos.map(a => `
      <a href="#/${a.route}" class="card block" style="text-decoration:none;color:inherit;border-top:3px solid #14b8a6;">
        <div class="flex items-center gap-2 mb-1">
          <div style="width:30px;height:30px;border-radius:6px;background:#14b8a61a;color:#14b8a6;display:inline-flex;align-items:center;justify-content:center;font-size:18px;">${a.icon}</div>
          <div class="text-[13px] font-bold text-slate-800">${a.name}</div>
        </div>
        <div class="text-[11px] text-slate-500">${a.en}</div>
        <div class="flex flex-wrap gap-1 mt-2">
          ${(a.tags || []).slice(0,3).map(t => `<span class="tag" style="background:#14b8a614;color:#14b8a6;">${t}</span>`).join("")}
        </div>
        <div class="text-[10px] text-slate-400 mt-2">→ 进入研究</div>
      </a>`).join("");

    return `
      ${MCH.hero({
        icon: "🧮",
        name: "基础编程算法 · 总览",
        en: "Classical Coding Algorithms · Overview",
        tags: ["数据结构", "算法导论", "LeetCode", "VisuAlgo 对标"],
        meta: [`🔢 ${codingAlgos.length} 大模块`, "◈ 互动可视化", "⚡ 零依赖"],
      })}

      <div class="section" style="background:linear-gradient(135deg,#ecfeff 0%,#f0fdfa 100%);border:1px solid #99f6e4;">
        <h2 style="color:#0f766e;border:none;padding:0;margin:0 0 10px 0;">为什么把编程算法放进 AI 算法平台？</h2>
        <p class="text-sm text-slate-700 leading-relaxed">
          任何工程化的 AI 系统最终都要落到<b>代码</b>——<b>数据加载</b>用到排序/哈希，<b>特征工程</b>依赖栈/队列/前缀和，
          <b>离线训练</b>频繁使用优先队列（Top-K 采样）、图遍历（RAG 邻居召回）、动态规划（Beam Search）。
          本模块不重复《算法导论》的教科书推导，而是用<b>可交互的动画</b>把最核心的
          <b>排序、查找、线性结构、树/图遍历、最短路径、动态规划、字符串匹配</b>串起来，
          让你在一个页面里一边看动画一边复习 LeetCode 高频题。
        </p>
      </div>

      <div class="section">
        <h2>🗂 子模块导航</h2>
        <div class="grid-3">${cards}</div>
      </div>

      <div class="section">
        <h2>🌐 对标资源</h2>
        <table class="table">
          <thead><tr><th style="width:220px;">资源</th><th>覆盖范围</th><th>对应本平台模块</th></tr></thead>
          <tbody>
            <tr><td><a href="https://visualgo.net/zh" target="_blank"><b>VisuAlgo (visualgo.net)</b></a></td><td>25+ 基础算法可视化（新加坡国立大学 Steven Halim）</td><td>全部 8 个编程模块对标</td></tr>
            <tr><td><b>《算法导论》CLRS</b></td><td>算法经典教材，覆盖排序/图/DP/NP</td><td>排序 / BFSDFS / 最短路 / DP</td></tr>
            <tr><td><b>《数据结构》严蔚敏</b></td><td>中文经典，线性结构+树+图</td><td>栈队列 / 链树堆 / 图遍历</td></tr>
            <tr><td><a href="https://leetcode.cn/" target="_blank"><b>LeetCode 中国版</b></a></td><td>3000+ 算法题；题目分类 tag</td><td>每个模块末尾列出 10-20 道高频题</td></tr>
            <tr><td><a href="https://oi-wiki.org/" target="_blank"><b>OI Wiki</b></a></td><td>中文竞赛算法百科</td><td>字符串/数学模块参考</td></tr>
            <tr><td><a href="https://cp-algorithms.com/" target="_blank"><b>CP-Algorithms</b></a></td><td>英文竞赛算法合集</td><td>KMP/Manacher/Z 函数</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>🛣 推荐学习路径</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #14b8a6;">
            <div class="font-semibold text-teal-700 text-sm mb-2">🎓 入门 (面试 3 个月)</div>
            <ol class="text-xs text-slate-600" style="list-style:decimal inside;line-height:1.8;">
              <li>排序（快排/归并/堆）</li>
              <li>二分 & 哈希</li>
              <li>栈 / 队列 / 单调栈</li>
              <li>链表 / 二叉树 / 堆</li>
              <li>BFS / DFS</li>
              <li>线性 DP（爬楼梯/打家劫舍）</li>
            </ol>
          </div>
          <div class="card" style="border-left:3px solid #0d9488;">
            <div class="font-semibold text-teal-700 text-sm mb-2">💼 进阶 (ACM 风格 / 社招)</div>
            <ol class="text-xs text-slate-600" style="list-style:decimal inside;line-height:1.8;">
              <li>单调栈/队列应用</li>
              <li>红黑树 / 平衡树</li>
              <li>Dijkstra / A*</li>
              <li>拓扑排序 / 强连通</li>
              <li>背包 DP / 区间 DP</li>
              <li>KMP / Manacher</li>
            </ol>
          </div>
          <div class="card" style="border-left:3px solid #0f766e;">
            <div class="font-semibold text-teal-700 text-sm mb-2">🚀 工程落地 (AI 系统应用)</div>
            <ol class="text-xs text-slate-600" style="list-style:decimal inside;line-height:1.8;">
              <li>Top-K → 堆（推荐系统召回）</li>
              <li>Trie → AC 自动机（敏感词）</li>
              <li>图遍历 → RAG 邻居召回</li>
              <li>Dijkstra → 物流路径规划</li>
              <li>DP → 序列标注 / Beam Search</li>
              <li>Bloom Filter → 去重缓存</li>
            </ol>
          </div>
        </div>
      </div>

      <div class="section" style="background:#f8fafc;border:1px solid #e2e8f0;">
        <h2 style="color:#334155;border:none;padding:0;margin:0 0 8px 0;">🤝 与 AI 算法的协同（在 AI 助手中自动推荐）</h2>
        <p class="text-xs text-slate-600 leading-relaxed">
          AI 算法选型助手在规则引擎里会根据你的场景自动追加编程算法的<b>工程组件</b>推荐：<br/>
          • 推荐系统 → <b>堆 (Top-K)</b> + <b>Trie (前缀搜索)</b><br/>
          • 风控图 → <b>BFS/DFS</b> + <b>并查集/最短路径</b><br/>
          • 文本/日志处理 → <b>KMP/Trie/AC 自动机</b><br/>
          • 序列标注 / Beam Search → <b>动态规划</b><br/>
          • 实时流处理 → <b>单调队列</b> + <b>滑动窗口</b>
        </p>
      </div>
    `;
  },
});
