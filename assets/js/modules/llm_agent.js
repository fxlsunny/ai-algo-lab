/* 模块：LLM Agent 智能体 */
MCH.register("llm_agent", {
  render() {
    const code = `# ========================================================================
# 大模型智能体（LLM Agent）核心架构
# ========================================================================
# 一个完整 Agent = LLM + Memory + Planning + Tools + Environment

class Agent:
    def __init__(self, llm, tools, memory=None, planner=None):
        self.llm     = llm                    # 大脑（决策）
        self.tools   = tools                  # 手（调用 API/代码）
        self.memory  = memory or ShortMemory()# 记忆（对话/长期）
        self.planner = planner or ReActLoop() # 规划（任务拆解）

    def run(self, task):
        plan = self.planner.decompose(task, self.llm)  # 拆解
        ctx  = self.memory.retrieve(task)              # 取记忆
        for step in plan:
            obs = self.executor(step, ctx)             # 执行
            self.memory.write(step, obs)
            ctx = self.memory.retrieve(task)
        return self.synthesize(ctx)

# ─── 1. 记忆系统（三级层级）────────────────────────────────
class HierarchicalMemory:
    def __init__(self):
        self.working = deque(maxlen=20)       # 工作记忆：最近对话
        self.episodic = []                    # 情景记忆：完整 trace
        self.semantic = VectorDB()            # 语义记忆：向量化知识

    def retrieve(self, query, k=5):
        # 分层检索：近期对话 + 相似历史 trace + 相关知识
        return {
            "recent": list(self.working)[-5:],
            "similar_traces": self.semantic.search(query, k=k, filter={"type":"trace"}),
            "knowledge": self.semantic.search(query, k=k, filter={"type":"doc"}),
        }

    def write(self, action, observation):
        self.working.append({"a": action, "o": observation})
        if action["type"] == "complete_task":
            # 任务完成后，整个 trace 总结入长期记忆
            summary = self.llm(f"总结任务轨迹：{self.working}")
            self.semantic.add(summary, meta={"type":"trace"})

# ─── 2. 规划器（三种范式）─────────────────────────────────
# (a) ReAct: 边想边做，一次一步（前面模块已详述）
# (b) Plan-and-Execute: 先出完整计划再执行
class PlanExecute:
    def decompose(self, task, llm):
        plan = llm(f"""请把任务拆分成 3-7 个子任务，输出 JSON List:
任务：{task}
示例输出：[{{"id":1,"step":"...","depends_on":[]}}, ...]""")
        return json.loads(plan)

# (c) Tree-of-Agents: 分支探索多个解法
class TreeOfAgents:
    def run(self, task, branching=3, depth=3):
        root = Node(task, score=1.0)
        beams = [root]
        for d in range(depth):
            next_beams = []
            for node in beams:
                branches = self.llm.sample_n(f"解决 {node.task} 的方法：", n=branching)
                for b in branches:
                    score = self.llm(f"评价此方案质量 0-1：{b}")
                    next_beams.append(Node(b, parent=node, score=float(score)))
            beams = sorted(next_beams, key=lambda x: -x.score)[:branching]
        return best_leaf(beams)

# ─── 3. 多 Agent 协作（MetaGPT 风格）─────────────────────
class MultiAgentTeam:
    def __init__(self):
        self.roles = {
            "PM":       Agent(llm, tools=[write_doc],    system="你是产品经理，负责写 PRD"),
            "Architect":Agent(llm, tools=[draw_diagram], system="你是架构师，负责技术设计"),
            "Engineer": Agent(llm, tools=[code, test],   system="你是工程师，负责实现"),
            "QA":       Agent(llm, tools=[test, review], system="你是 QA，负责测试"),
        }

    def build_product(self, requirement):
        prd     = self.roles["PM"].run(f"为需求写 PRD：{requirement}")
        design  = self.roles["Architect"].run(f"根据 PRD 设计架构：{prd}")
        code    = self.roles["Engineer"].run(f"根据设计写代码：{design}")
        report  = self.roles["QA"].run(f"测试代码：{code}")
        return {"prd": prd, "design": design, "code": code, "qa": report}`;

    const table = `# ========================================================================
# 主流 Agent 框架对比（2024 工业界）
# ========================================================================
# LangChain Agent    —— 生态最全，上手快，坑也多
# LlamaIndex Agent   —— RAG 场景首选
# AutoGPT            —— 第一个开源自治 Agent（deprecated）
# MetaGPT            —— 多 Agent 协作（软件开发场景标杆）
# AutoGen (Microsoft)—— 对话式多 Agent，灵活度高
# CrewAI             —— 角色驱动多 Agent，易上手
# OpenAI Swarm       —— 2024 官方轻量多 Agent
# Dify / Coze        —— 可视化低代码平台`;

    return `
      ${MCH.hero({
        icon: "🤖",
        name: "LLM Agent 智能体",
        en: "LLM-powered Autonomous Agents",
        tags: ["Memory", "Planning", "Tools", "Multi-Agent", "AutoGPT", "MetaGPT"],
        meta: ["◈ 2023 AutoGPT 引爆 · 2024 Agent Year", "⚡ 从 Chatbot 到 Autonomous Worker"],
      })}

      ${MCH.versionSection ? MCH.versionSection("llm_agent") : ""}

      <div class="section">
        <h2>1. Agent 四大核心组件</h2>
        <div class="grid-4">
          <div class="card">
            <h3>🧠 LLM Brain</h3>
            <p style="font-size:13px;line-height:1.6">决策中心。输入环境状态 → 输出下一步动作。用 GPT-4o / Claude 3.5 / DeepSeek 等</p>
          </div>
          <div class="card">
            <h3>💾 Memory</h3>
            <p style="font-size:13px;line-height:1.6">工作记忆（短期）+ 情景记忆（过往 trace）+ 语义记忆（向量库知识）</p>
          </div>
          <div class="card">
            <h3>📋 Planning</h3>
            <p style="font-size:13px;line-height:1.6">ReAct / Plan-Execute / ToT / Reflexion —— 把复杂任务拆成可执行步骤</p>
          </div>
          <div class="card">
            <h3>🔧 Tools</h3>
            <p style="font-size:13px;line-height:1.6">搜索、计算器、代码执行、数据库、第三方 API，所有能与外界交互的接口</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 完整 Agent 实现（含多 Agent 协作）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. Agent 架构对比</h2>
        <table class="table">
          <thead><tr><th>架构</th><th>核心特点</th><th>适用场景</th><th>代表产品</th></tr></thead>
          <tbody>
            <tr><td><b>单 Agent · ReAct</b></td><td>边想边做，一步一 observation</td><td>搜索、研究、客服</td><td>LangChain ReAct</td></tr>
            <tr><td><b>单 Agent · Plan-Execute</b></td><td>先全盘规划再执行</td><td>确定性任务流</td><td>BabyAGI</td></tr>
            <tr><td><b>多 Agent · 角色协作</b></td><td>PM/架构/工程/QA 分工</td><td>软件开发、复杂项目</td><td>MetaGPT, CrewAI</td></tr>
            <tr><td><b>多 Agent · 对话协作</b></td><td>Agent 之间互相讨论迭代</td><td>头脑风暴、代码 review</td><td>AutoGen, OpenAI Swarm</td></tr>
            <tr><td><b>GUI Agent</b></td><td>直接操作浏览器/桌面</td><td>自动化办公</td><td>Claude Computer Use</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>4. 主流框架速查</h2>
        ${MCH.code(table, "python")}
      </div>

      <div class="section">
        <h2>5. 工程陷阱</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #ef4444">
            <h3>❌ 翻车常见原因</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>无限工具调用循环 → 加 max_iterations</li>
              <li>上下文爆炸 → 用 summary buffer 压缩历史</li>
              <li>幻觉调用不存在工具 → function calling 强约束</li>
              <li>多 Agent 无限互聊 → 加 turn limit + 终止条件</li>
              <li>权限失控 → 危险工具加 human-in-the-loop</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #10b981">
            <h3>✅ 生产级 Checklist</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>所有工具有 timeout（默认 30s）</li>
              <li>可追溯：每步 Thought/Action/Obs 落库</li>
              <li>成本监控：token 预算 + 提前熔断</li>
              <li>灰度发布：先 10% 流量，异常率达阈值自动回滚</li>
              <li>持续评估：LangSmith / Langfuse 做 trace 分析</li>
            </ul>
          </div>
        </div>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
