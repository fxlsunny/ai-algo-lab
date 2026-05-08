/* 模块：Function Calling / Tool Use */
MCH.register("llm_tool_use", {
  render() {
    const code = `# ========================================================================
# OpenAI / DeepSeek / 混元 通用 Function Calling 协议（2024 行业标准）
# ========================================================================

import json
from openai import OpenAI
client = OpenAI()  # 兼容 DeepSeek / 混元 / Groq 等 OpenAI 协议厂商

# ─── 1. 声明工具（JSON Schema）─────────────────────────────
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称（中文或英文）"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": "查询公司数据库",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                },
                "required": ["sql"]
            }
        }
    },
]

# ─── 2. 工具实现（Python 真实函数）─────────────────────────
def get_weather(city, unit="celsius"):
    # 真实调 API
    resp = requests.get(f"https://api.weather.com/{city}", params={"unit": unit})
    return resp.json()

def query_database(sql):
    return db.execute(sql).fetchall()[:100]  # 限制返回行数

TOOL_IMPL = {"get_weather": get_weather, "query_database": query_database}

# ─── 3. 多轮交互循环 ─────────────────────────────────────
def chat_with_tools(user_msg, max_rounds=5):
    messages = [{"role": "user", "content": user_msg}]

    for round in range(max_rounds):
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto",  # "auto" | "required" | {"type":"function","function":{"name":"..."}}
        )
        msg = resp.choices[0].message
        messages.append(msg)

        # 没有工具调用 → 结束
        if not msg.tool_calls:
            return msg.content

        # 有工具调用 → 逐个执行
        for call in msg.tool_calls:
            fn_name = call.function.name
            args = json.loads(call.function.arguments)
            try:
                result = TOOL_IMPL[fn_name](**args)
            except Exception as e:
                result = {"error": str(e)}

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    return "超过最大轮次"


# ─── 4. 并行工具调用（GPT-4o 支持）─────────────────────────
# 一次返回多个 tool_calls，可并发执行（asyncio.gather）
async def parallel_execute(tool_calls):
    async def run_one(call):
        fn = TOOL_IMPL[call.function.name]
        args = json.loads(call.function.arguments)
        return await asyncio.to_thread(fn, **args)
    results = await asyncio.gather(*[run_one(c) for c in tool_calls])
    return results

# ─── 5. 结构化输出 mode（2024 新）──────────────────────────
# response_format={"type":"json_schema","json_schema":{...}} 强制保证 JSON 合规
response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    messages=[...],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "recipe",
            "schema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "ingredients": {"type": "array", "items": {"type": "string"}},
                    "steps": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "ingredients", "steps"],
                "additionalProperties": False,
            },
            "strict": True,  # 100% 保证 schema 合规
        }
    }
)`;

    const mcp = `# ========================================================================
# MCP (Model Context Protocol) — Anthropic 2024 · 工具生态标准化
# ========================================================================
# 一次接入，所有 Agent/IDE/应用都能用你的工具
# 类比：USB-C 之于数据线 ← MCP 之于 AI 工具

# MCP Server（你提供工具）
from mcp.server import Server
server = Server("my-tools")

@server.tool()
async def get_user_profile(user_id: str) -> dict:
    """查询用户画像"""
    return await db.users.find_one({"id": user_id})

@server.tool()
async def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件"""
    return mailer.send(to, subject, body)

# MCP Client（Claude Desktop / Cursor / WorkBuddy 自动接入）
# 用户在 IDE 里装你的 server，所有 MCP 兼容应用都能调用你的工具
# 已经有 1000+ MCP Server：GitHub / Notion / Slack / Google Drive / ...`;

    return `
      ${MCH.hero({
        icon: "🔧",
        name: "Function Calling / Tool Use",
        en: "让 LLM 调用外部工具的标准协议",
        tags: ["OpenAI Tools", "JSON Schema", "Parallel Calls", "Structured Output", "MCP"],
        meta: ["◈ OpenAI 2023/06 首发 · Anthropic MCP 2024", "⚡ Agent 落地的技术基座"],
      })}

      ${MCH.versionSection ? MCH.versionSection("llm_tool_use") : ""}

      <div class="section">
        <h2>1. 为什么需要 Function Calling？</h2>
        <p style="line-height:1.8;color:var(--text-secondary)">
          纯文本 Prompt 输出很难稳定解析（正则、YAML、JSON 都会出错）。<br>
          <b>Function Calling = 结构化 Action</b>：模型直接产出合法 JSON 参数，框架负责解析+调用+回填。<br>
          对比 ReAct 的纯文本 Action，Function Calling 成功率从 ~70% 提升到 ~99%（GPT-4o 实测）。
        </p>
      </div>

      <div class="section">
        <h2>2. 标准协议代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. MCP — 工具生态的未来</h2>
        ${MCH.code(mcp, "python")}
      </div>

      <div class="section">
        <h2>4. 主流模型支持对比</h2>
        <table class="table">
          <thead><tr><th>模型</th><th>Function Calling</th><th>并行调用</th><th>结构化输出</th><th>MCP</th></tr></thead>
          <tbody>
            <tr><td>GPT-4o / GPT-4-turbo</td><td>✅</td><td>✅</td><td>✅ strict</td><td>—</td></tr>
            <tr><td>Claude 3.5 Sonnet</td><td>✅</td><td>✅</td><td>✅</td><td>✅ 原生</td></tr>
            <tr><td>DeepSeek-V3</td><td>✅</td><td>✅</td><td>✅</td><td>—</td></tr>
            <tr><td>腾讯混元 turbos</td><td>✅</td><td>—</td><td>partial</td><td>—</td></tr>
            <tr><td>通义 Qwen 2.5</td><td>✅</td><td>✅</td><td>✅</td><td>—</td></tr>
            <tr><td>Llama 3.1-70B</td><td>✅</td><td>✅</td><td>partial</td><td>—</td></tr>
            <tr><td>Gemini 1.5 Pro</td><td>✅</td><td>✅</td><td>✅</td><td>—</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 工程实践要点</h2>
        <ul style="line-height:1.8;padding-left:20px">
          <li>🔒 <b>权限分层</b>：只读工具自动调用，写操作 human-in-the-loop 二次确认</li>
          <li>⏱ <b>Timeout + Retry</b>：每个工具设独立 timeout（5-60s）</li>
          <li>📏 <b>输出截断</b>：工具返回 &gt;10KB 先 summary，避免 context 爆炸</li>
          <li>🐛 <b>错误传回 LLM</b>：失败时把 error message 作为 tool result 返回，让模型自我纠错</li>
          <li>💰 <b>缓存幂等</b>：相同参数的 read-only 工具调用走缓存（LRU + TTL）</li>
          <li>📊 <b>Trace 完整记录</b>：每次 tool_call 的 input/output/耗时入库，便于回溯</li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
