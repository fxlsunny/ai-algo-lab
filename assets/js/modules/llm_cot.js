/* 模块：Chain-of-Thought 思维链 */
MCH.register("llm_cot", {
  render() {
    const code = `# ========================================================================
# Chain-of-Thought (CoT) 思维链 — 让 LLM "先思考再回答"
# 原始论文：Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning..."
# ========================================================================

# ① Zero-shot CoT —— 一句话魔法
#    只需在 prompt 末尾加 "Let's think step by step"
prompt_zero = """Q: 小明有 23 个苹果，吃掉 7 个后又买了 12 个，现在有几个？
Let's think step by step."""

# ② Few-shot CoT —— 示例引导（最稳定）
prompt_few = """Q: 罗杰有 5 个网球，买了 2 罐，每罐 3 个。总共多少个？
A: 罗杰一开始有 5 个。2 罐 × 3 个 = 6 个。5 + 6 = 11。答案是 11。

Q: 食堂有 23 个苹果，做午饭用了 20 个，又买了 6 个，现在多少个？
A: """  # 模型会主动展开推理链

# ③ Self-Consistency CoT —— 多路径 + 投票，准确率飙升
def self_consistency(prompt, llm, n=10, temperature=0.7):
    answers = []
    for _ in range(n):
        resp = llm(prompt, temperature=temperature)  # 高温采样多样路径
        ans = extract_final_answer(resp)
        answers.append(ans)
    return Counter(answers).most_common(1)[0][0]     # 多数投票

# ④ Tree-of-Thoughts (ToT) —— 搜索式推理
#    把推理当作树搜索：每步生成多个分支 → LLM 自评 → BFS/DFS 扩展最优
class TreeOfThoughts:
    def solve(self, problem, depth=3, breadth=5):
        root = Node(problem, score=1.0)
        frontier = [root]
        for d in range(depth):
            candidates = []
            for node in frontier:
                thoughts = llm.generate_thoughts(node, n=breadth)
                scores   = llm.value(thoughts)  # 用 LLM 给每个思路打分 0-1
                for t, s in zip(thoughts, scores):
                    candidates.append(Node(t, parent=node, score=s))
            frontier = sorted(candidates, key=lambda x: -x.score)[:breadth]
        return best_leaf(frontier)

# ⑤ Least-to-Most Prompting —— 分而治之
#    先让 LLM 把问题拆解成更简单的子问题，再逐个求解
split = llm("把下面问题分解成更简单的 3-5 个子问题：\\n" + complex_q)
for sub_q in split.items:
    ans = llm(sub_q + "\\n前文已知：" + prev_ans)

# ========================================================================
# 何时用 CoT？—— 经验法则
# ========================================================================
# ✅ 数学 / 逻辑 / 多步推理 / 代码合成：必用
# ✅ 模型 ≥ 10B 参数：CoT 带来巨大提升（涌现能力）
# ⚠ 模型 < 10B：CoT 可能无效甚至下降 → 用 Distilled CoT（蒸馏到小模型）
# ❌ 简单分类 / 事实查询：CoT 浪费 token，且可能幻觉
# ❌ 对延迟敏感的场景：CoT 输出长 5-10 倍，TTFT 和 TPS 都受影响`;

    return `
      ${MCH.hero({
        icon: "💭",
        name: "Chain-of-Thought 思维链",
        en: "CoT · Self-Consistency · Tree-of-Thoughts · Least-to-Most",
        tags: ["Zero-shot CoT", "Few-shot CoT", "Self-Consistency", "ToT", "Least-to-Most"],
        meta: ["◈ Wei et al. 2022 · Wang et al. 2023 · Yao et al. 2023", "⚡ 让大模型像人一样拆解问题"],
      })}

      ${MCH.versionSection ? MCH.versionSection("llm_cot") : ""}

      <div class="section">
        <h2>1. 五种 CoT 变体对比</h2>
        <table class="table">
          <thead><tr><th>方法</th><th>核心思想</th><th>GSM8K 准确率</th><th>适用场景</th></tr></thead>
          <tbody>
            <tr><td><b>标准 Prompt</b></td><td>直接问答</td><td>17.9%</td><td>简单 QA</td></tr>
            <tr><td><b>Zero-shot CoT</b></td><td>加 "Let's think step by step"</td><td>40.7%</td><td>快速试用，无示例</td></tr>
            <tr><td><b>Few-shot CoT</b></td><td>给几个带推理的示例</td><td>56.9%</td><td>稳定产品级应用</td></tr>
            <tr><td><b>Self-Consistency</b></td><td>10 路采样 + 投票</td><td><b>74.4%</b></td><td>准确率优先，可承受 10x 成本</td></tr>
            <tr><td><b>Tree-of-Thoughts</b></td><td>树搜索 + LLM 自评</td><td>74%+（24 点游戏提升 100×）</td><td>需多步规划的复杂任务</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>2. 代码实现（五种模式）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. Prompt 模板速查</h2>
        <div class="grid-2">
          <div class="card">
            <h3>🎯 数学推理模板</h3>
            <pre style="white-space:pre-wrap;font-size:12px;background:var(--bg-tertiary);padding:12px;border-radius:6px">【角色】资深数学老师
【任务】请按下列格式回答：
1. 题目理解：[用自己的话复述]
2. 已知条件：[列清楚]
3. 解题步骤：[一步一算式]
4. 最终答案：[只写结果]</pre>
          </div>
          <div class="card">
            <h3>🔧 代码生成模板</h3>
            <pre style="white-space:pre-wrap;font-size:12px;background:var(--bg-tertiary);padding:12px;border-radius:6px">【任务】实现 {功能}
【输出结构】
1. 🎯 需求分析
2. 📋 关键数据结构
3. 🧮 算法思路
4. 💻 完整代码（含注释）
5. ✅ 测试用例
6. ⚡ 复杂度分析</pre>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 陷阱与误区</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #ef4444">
            <h3>❌ 常见错误</h3>
            <ul style="margin:8px 0;padding-left:20px;line-height:1.7">
              <li>小模型（&lt;10B）用 CoT → 往往更差</li>
              <li>Self-Consistency 用 temperature=0 → 10 路完全一样，浪费</li>
              <li>把 CoT 用在分类任务 → 慢 + 幻觉</li>
              <li>推理链露给用户 → 不专业、影响 UX</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #10b981">
            <h3>✅ 最佳实践</h3>
            <ul style="margin:8px 0;padding-left:20px;line-height:1.7">
              <li>用 &lt;think&gt;...&lt;/think&gt; 标签包裹推理链，输出时隐藏</li>
              <li>Self-Consistency 时 temperature = 0.6-0.8</li>
              <li>CoT + Tool Use 组合拳（算术用计算器）</li>
              <li>DeepSeek-R1 / o1 已把 CoT 烘焙进权重（RL 训练）</li>
            </ul>
          </div>
        </div>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
