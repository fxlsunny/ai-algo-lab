/* 模块：ReAct 推理-行动框架 */
MCH.register("llm_react", {
  render() {
    const code = `# ========================================================================
# ReAct (Reasoning + Acting) — Yao et al. 2022
# 核心思想：让 LLM 交替执行 "推理(Thought)" 和 "行动(Action)"
# 每步循环：Thought → Action → Observation → Thought → ...
# ========================================================================

REACT_PROMPT = """你是能调用工具的 AI 助手。遵循以下格式：

Thought: 我需要思考下一步做什么
Action: 工具名[参数]
Observation: 工具返回结果
Thought: 基于观察结果继续思考...
Action: ...
...
Final Answer: 最终答案

可用工具：
- search[query]: 网络搜索
- calculator[expr]: 数学计算
- sql[query]: 查询数据库
- python[code]: 执行 Python

现在回答问题：{question}
"""

# ─── 核心循环 ────────────────────────────────────────────
def react_loop(question, max_steps=10):
    history = REACT_PROMPT.format(question=question)
    for step in range(max_steps):
        # ① Thought + Action：让 LLM 生成，遇到 "Observation:" 就停
        resp = llm(history, stop=["Observation:"])
        history += resp

        if "Final Answer:" in resp:
            return resp.split("Final Answer:")[-1].strip()

        # ② 解析 Action
        action_match = re.search(r"Action:\\s*(\\w+)\\[(.*?)\\]", resp)
        if not action_match:
            history += "\\nObservation: 无法解析 Action 格式，请用 ToolName[args] 格式\\n"
            continue

        tool_name, tool_args = action_match.groups()

        # ③ 调用工具 → 得到 Observation
        try:
            obs = TOOLS[tool_name](tool_args)
        except KeyError:
            obs = f"未知工具 {tool_name}，可用：{list(TOOLS.keys())}"
        except Exception as e:
            obs = f"工具执行失败：{e}"

        history += f"\\nObservation: {obs}\\n"

    return "超过最大步数仍未解决"


# ─── 工具注册 ────────────────────────────────────────────
TOOLS = {
    "search":     lambda q: duckduckgo.search(q)[:3],
    "calculator": lambda e: eval(e, {"__builtins__": {}}, math.__dict__),
    "sql":        lambda q: db.execute(q).fetchall()[:20],
    "python":     lambda c: sandbox.run(c, timeout=5),
    "weather":    lambda city: requests.get(f"https://api.weather.com/{city}").json(),
}

# ─── 实战示例：ReAct 解 HotpotQA ──────────────────────────
# Q: "2023 年图灵奖获得者是谁的学生？"
# Thought: 先查 2023 年图灵奖得主
# Action: search[2023 Turing Award winner]
# Observation: Bob Metcalfe (以太网发明人)
# Thought: 再查他的博士导师
# Action: search[Bob Metcalfe PhD advisor]
# Observation: J.C.R. Licklider & Jon Postel
# Final Answer: Bob Metcalfe 的博士导师是 J.C.R. Licklider

# ========================================================================
# ReAct vs CoT vs Tool-use — 三者关系
# ========================================================================
#   CoT:         只"想"，不"做" —— 纯推理（数学）
#   Tool-use:    只"做"，无"想" —— 单次工具调用（天气 API）
#   ReAct:       "想 + 做" 循环 —— 适合多步探索（搜索、研究）
#   Plan+Execute: 先规划整个流程，再一次性执行 —— 适合确定性任务

# ========================================================================
# 常见陷阱 & 解决方案
# ========================================================================
# 1) 无限循环：LLM 总在重复查同一信息
#    → 加 visited set，重复 query 直接告知"已查过"
# 2) 工具参数格式错：search[日期="2024"] vs search["2024 年某事"]
#    → 用 JSON-mode 或 function calling 强约束
# 3) Observation 过长：一次检索返回 10KB
#    → 截断 + 让 LLM 自己调用 summarize 工具
# 4) 幻觉调用：调用不存在的工具 get_bitcoin_price
#    → prompt 里列 TOOLS 白名单 + 严格拒绝未知工具`;

    return `
      ${MCH.hero({
        icon: "🔄",
        name: "ReAct 推理-行动框架",
        en: "Reasoning + Acting in Language Models",
        tags: ["Thought-Action-Observation", "Tool Use", "HotpotQA", "WebShop"],
        meta: ["◈ Yao et al. 2022 · ICLR 2023", "⚡ LangChain/AutoGPT 的思想源头"],
      })}

      ${MCH.versionSection ? MCH.versionSection("llm_react") : ""}

      <div class="section">
        <h2>1. ReAct 核心公式</h2>
        <div class="formula-block" style="font-family:monospace;font-size:14px;line-height:2">
          π(a<sub>t</sub> | x, τ<sub>t</sub>) = LLM( 前文(Thought<sub>1..t-1</sub>, Action<sub>1..t-1</sub>, Obs<sub>1..t-1</sub>), x )<br>
          τ<sub>t</sub> = τ<sub>t-1</sub> + [Thought<sub>t</sub>, Action<sub>t</sub>, Obs<sub>t</sub>]
        </div>
        <p style="margin-top:10px;color:var(--text-secondary)">
          对比纯 CoT：CoT 只有 Thought，无 Action/Obs — 无法获取外界信息。<br>
          对比 Tool-Use：Tool-Use 只有 Action，无 Thought — 无法多步规划。
        </p>
      </div>

      <div class="section">
        <h2>2. 核心循环代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. ReAct vs 其他范式</h2>
        <table class="table">
          <thead><tr><th>范式</th><th>有 Think?</th><th>有 Action?</th><th>循环?</th><th>典型场景</th></tr></thead>
          <tbody>
            <tr><td>Standard Prompt</td><td>❌</td><td>❌</td><td>❌</td><td>分类/翻译</td></tr>
            <tr><td>CoT</td><td>✅</td><td>❌</td><td>❌</td><td>数学/逻辑</td></tr>
            <tr><td>Tool-Use / Function Calling</td><td>❌</td><td>✅</td><td>❌</td><td>单次 API 调用</td></tr>
            <tr><td><b>ReAct</b></td><td>✅</td><td>✅</td><td>✅</td><td>多步搜索/研究</td></tr>
            <tr><td>Plan-and-Execute</td><td>✅ 一次性</td><td>✅</td><td>部分</td><td>确定性任务流</td></tr>
            <tr><td>Reflexion</td><td>✅ + 自省</td><td>✅</td><td>✅</td><td>带反馈修正</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>4. 落地框架</h2>
        <div class="grid-2">
          <div class="card">
            <h3>🦜 LangChain AgentExecutor</h3>
            <pre style="font-size:12px;background:var(--bg-tertiary);padding:10px;border-radius:6px;white-space:pre-wrap">from langchain.agents import create_react_agent
agent = create_react_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)
executor.invoke({"input": "..."})</pre>
          </div>
          <div class="card">
            <h3>🦙 LlamaIndex ReActAgent</h3>
            <pre style="font-size:12px;background:var(--bg-tertiary);padding:10px;border-radius:6px;white-space:pre-wrap">from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)
response = agent.chat("...")</pre>
          </div>
        </div>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
