/* 模块：Agent 最新进展 — Skills / OpenClaw / Hermes 三大派 */
MCH.register("llm_agent_advanced", {
  render() {
    const skill_md = `# ============================================================================
# Claude Skills · 渐进式披露的"能力包"（Anthropic 2025）
# ============================================================================
# 一个 Skill = 一个文件夹，根目录必有 SKILL.md，描述能力 + 可选附带：
#   ├── SKILL.md          # 元数据 + 简短指令（始终装入上下文）
#   ├── reference/        # 详细文档（按需被 Claude 调用 grep / read）
#   ├── scripts/          # 可执行脚本（py / sh / ts），Claude 写代码时调用
#   ├── examples/         # 示例输入输出
#   └── data/             # 模板 / 字典 / 参考数据

# SKILL.md 头部 (YAML frontmatter) 关键字段：
# ---
# name: invoice-processing
# description: 解析 PDF 发票、提取字段、生成 ERP 导入 csv。
#              Use when user mentions invoice / 发票 / 报销 / OCR.
# allowed-tools: Read, Write, Bash(python:*)
# ---

# 渐进式披露 (Progressive Disclosure)：
# Level 1：开机时只载入 (name + description) ≈ 50 token，告诉 Claude "我有什么能力"
# Level 2：Claude 判断需要时才 read SKILL.md 全文 ≈ 500-2000 token
# Level 3：仅在执行子任务时才 read reference/*.md 或 run scripts/*.py
#
# 核心收益：
#  · 上下文窗口不被 100 个能力描述塞爆（典型节约 90%+ token）
#  · 能力可由用户/团队"即插即用"打包分发（zip / git）
#  · 跨会话复用、版本可控、可审计、可分级权限
#  · 与 MCP 互补：MCP 提供"工具/数据接口"；Skills 提供"工作流知识 + 脚本"

# ─── 真实 SKILL.md 例子（精简版） ────────────────────────────
# ---
# name: pdf-form-fill
# description: Use when the user asks to fill a PDF form. Reads field names
#              from the PDF, prompts user for values, writes filled output.
# allowed-tools: Read, Write, Bash
# ---
# # PDF Form Fill
# 1. Use scripts/list_fields.py to enumerate all form fields
# 2. Ask the user for any missing values
# 3. Use scripts/fill.py to write the filled PDF to ./out/
# 4. Verify with scripts/verify.py and report number of filled fields
#
# References: reference/pdf_field_types.md, reference/encoding_quirks.md`;

    const openclaw_code = `# ============================================================================
# OpenClaw — 持续在线的开源 Agent 框架（基于 pi-agent，2025）
# ============================================================================
# 定位：一个"始终在线的数字员工"，跨多通道接收任务、自主规划、调用工具、留下记忆
# Hermes Agent 文档原话："OpenClaw 与 Claude Code (Anthropic)、Codex (OpenAI)、Hermes
#   同属一类：用工具调用与系统交互的自治编程 / 任务执行 Agent"

# ─── 1. 核心运行环（Goal → Plan → Act → Observe → Evaluate → Repeat）─
class OpenClawLoop:
    def step(self, session):
        plan      = self.llm.plan(session.goal, session.history, session.memory)
        action    = plan.next_action()
        result    = self.tools.execute(action)            # exec / fs / browser / web_fetch
        eval_     = self.llm.evaluate(action, result)
        if eval_.done: return result
        session.history.append((action, result, eval_))
        return self.step(session)                          # 反馈回环，降幻觉

# ─── 2. Markdown 配置三件套（零代码定制人格/规则/用户画像）─────
# ~/.openclaw/SOUL.md     人格、说话风格、价值观、口头禅、拒答边界
# ~/.openclaw/AGENTS.md   工作规则：可调工具白名单、危险动作策略、产出格式
# ~/.openclaw/USER.md     用户画像：身份、偏好、长期目标、敏感信息
#
# 例：~/.openclaw/AGENTS.md
# ---
# tools_allow: [exec, read, write, edit, web_fetch, browser]
# tools_deny:  [shell.rm_-rf_/, payment, send_email_external]
# require_human_approval_on: [push_to_main, deploy_prod, schema_migration]
# output: { format: "markdown_with_frontmatter", language: "zh_CN" }
# ---

# ─── 3. Gateway：跨多平台触达 ──────────────────────────────────
# OpenClaw Gateway 把同一个 Agent 暴露到 Telegram / WhatsApp / Discord / Slack /
# iMessage / Signal / web chat / 邮件 / cron 调度等多种入口，统一会话状态。

# ─── 4. 子 Agent 派生（Sub-Agent Spawning）─────────────────────
# 父 Agent 可派生子 Agent，独立工作区 + 独立权限：
#
parent.spawn(
    name="db-migrator",
    goal="把 user 表从 Postgres 迁到 ClickHouse，并校验数据一致性",
    workspace="./_sub/db-migrator",     # 隔离磁盘
    tools=["read", "exec", "psql", "ch-cli"],
    deny_tools=["payment", "send_email_external"],
    timeout="2h",
    on_done=parent.callback,
)

# ─── 5. 零代码 Skill 引擎 ─────────────────────────────────────
# 直接把脚本丢进 ~/.openclaw/skills/<name>/ 即可被加载：
#   skills/seo-audit/SKILL.md
#   skills/seo-audit/scripts/lighthouse.sh
#   skills/seo-audit/reference/checklist.md
# 与 Claude Skills 大体兼容（设计理念互相借鉴），但运行时是开源、自托管的。

# ─── 6. 持久会话 + 长期记忆 ───────────────────────────────────
sessions = openclaw.list_sessions()              # 历史所有会话
session  = openclaw.resume(sessions[-1].id)       # 跨天继续做同一件事
session.memory.write("note", "用户偏好凌晨发布")  # 短期 + 长期向量库
`;

    const hermes_code = `# ============================================================================
# Hermes 4 + Hermes Agent — Nous Research 开源旗舰（2025.08）
# ============================================================================
# Hermes 4 是 Nous Research 推出的"混合推理"开源模型族：
#   ╭───────────┬──────────┬─────────────────────────────────╮
#   │ Hermes 4  │ 14B/70B/405B │ <think>...</think> 切换思考模式 │
#   ╰───────────┴──────────┴─────────────────────────────────╯
# 训练数据较 Hermes 3 扩大 50×，由 Atropos 框架合成（强推理 + 创意写作）。
# 已收入 OpenRouter / Together / Nous API，OAuth 一键登录。

# Hermes Agent 是其同名 Agent 框架，跑在终端 / IDE / 消息平台：
#   · 68 个内置工具，可分组 toolset 启用/禁用
#   · MCP 集成：作为 Client 接 1000+ 社区 Server，亦可 hermes mcp serve 反向暴露
#   · Skills 系统：Agent "做完一遍，把流程沉淀成 SKILL.md"，下次直接复用 → 越用越聪明
#   · 跨平台 Gateway：Telegram / Discord / Slack / WhatsApp / Signal / Matrix / Email + 10+
#   · Provider-Agnostic：OpenRouter / Anthropic / OpenAI / DeepSeek / Nous / 本地（Ollama / vLLM / llama.cpp）+ 15+
#   · Profiles：多套独立配置 / sessions / skills / memory（个人 vs 工作 vs 客户隔离）
#   · 工作树 (--worktree)：开多个 git worktree 并行跑多个 Agent
#   · 内置插件：Honcho / Mem0 等记忆后端 · cron / webhook / 浏览器 (CDP)

# 关键 CLI（节选）：
#   hermes                                    # 启动交互 CLI（默认 Sonnet 4 / GPT-5 / Hermes 4）
#   hermes -r <session_id>                    # 恢复历史会话
#   hermes --worktree -s seo,db-migrate       # 隔离 worktree + 预载技能
#   hermes mcp add browser --command "...."   # 加 MCP server
#   hermes skills install <hub_id>            # 装社区技能
#   hermes claw migrate                       # 🆕 从 OpenClaw 迁移到 Hermes（官方提供脚本）
#   hermes gateway run                        # 启动多平台网关

# 在会话内的 Slash Commands（节选 — 体现 Agent 控制粒度）：
#   /goal "持续优化推荐系统 NDCG 直到 0.4+"   # 跨多轮长目标
#   /reasoning xhigh                            # 切到极致思考预算
#   /steer  "下一轮请只改后端不要碰前端"        # 工具调用之间注入指令不打断
#   /background "跑完 etl 后通知我"            # 后台任务
#   /branch                                    # 当前会话分叉，并行尝试两条路
#   /rollback 3                                # 把文件系统回滚到 3 个 checkpoint 之前
#   /yolo                                      # 关闭危险动作人工审批（高风险）

# Hermes Function Calling 模板（开源生态事实标准之一，被 Llama-cpp / vLLM 直接支持）：
# <tools>[{"type":"function","function":{"name":"get_weather","parameters":{...}}}]</tools>
# <|im_start|>assistant
# <tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>
# <|im_end|>`;

    return `
      ${MCH.hero({
        icon: "🦾",
        name: "Agent 最新进展 · Skills / OpenClaw / Hermes",
        en: "Frontier Agents · Claude Skills + OpenClaw + Hermes 4 + Hermes Agent + MCP",
        tags: ["Skills", "OpenClaw", "Hermes 4", "Hermes Agent", "MCP", "Computer Use", "OpenHands", "Cline"],
        meta: ["◈ 2024-26 Agent 三大派对比 + 全景生态", "⚡ 闭源专属 (Anthropic) ↔ 开源自治 (OpenClaw / Hermes)"],
      })}

      <div class="section" style="background:linear-gradient(135deg,#ede9fe 0%,#ddd6fe 100%);border:1px solid #c4b5fd;">
        <h2 style="color:#5b21b6;border:none;padding:0;margin:0 0 6px 0;">🆕 Agent 三大新范式：(1) <b>Skills</b>（Anthropic 渐进披露能力包）· (2) <b>OpenClaw</b>（开源持续在线 Agent · pi-agent 系）· (3) <b>Hermes 4 + Hermes Agent</b>（Nous Research 开源旗舰 · 自演化 Skills）。三家都用 <b>MCP</b> 当工具协议、都把"Skills 文件夹"当能力分发载体 —— 这是 2025-2026 行业新共识。</h2>
      </div>

      <div class="section">
        <h2>1. 2024-2026 Agent 演进时间线</h2>
        <div class="mermaid">
flowchart TB
    A[2023.06 OpenAI Function Calling] --> B[2023 LangChain Agents · AutoGPT]
    B --> C[2024.01 AutoGen · CrewAI · MetaGPT]
    C --> D[2024.03 Devin / SWE-Agent]
    D --> E[2024.06 Cline · OpenDevin]
    E --> F[2024.10 🏆 Claude Computer Use]
    F --> G[2024.11 🏆 MCP 协议发布]
    G --> H[2024.12 OpenAI Operator + Realtime]
    H --> I[2025.01 DeepSeek-R1 推理 Agent]
    I --> J[2025.02 OpenAI Deep Research]
    J --> K[2025.04 Google A2A 协议]
    K --> L[2025 Q2 🏆 Claude Skills 公开<br/>渐进披露能力包]
    L --> M[2025.08 🏆 Hermes 4 + Hermes Agent 发布<br/>14B/70B/405B 混合推理]
    M --> N[2025 H2 🏆 OpenClaw 出圈<br/>持续在线 + 多通道 + Markdown 配置]
    N --> O[2025-26 Skills + MCP + 自演化 → Agent 互联 A2A]
        </div>
      </div>

      <div class="section">
        <h2>2. Claude Skills 详解 — 渐进披露的"能力包"</h2>
        ${MCH.code(skill_md, "yaml")}
        ${MCH.info(`
          <b>为什么 Skills 比传统 RAG / 工具调用高一个层级</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>分层加载</b>：50 token 元数据 → 按需展开 → 真要执行才跑脚本，节约 90% 上下文</li>
            <li><b>三位一体</b>：能力 + 文档 + 工具脚本 一起打包，不是只送文档片段（RAG）也不是只送 schema（FC）</li>
            <li><b>可分发</b>：一个 zip / Git 仓库即可分享给团队或社区，Anthropic 已开放 Skills 注册中心</li>
            <li><b>可组合</b>：多个 Skill 同时挂载，Claude 自行路由 + 互相调用</li>
            <li><b>对模型可移植</b>：理论上不依赖 Claude 专有特性 → OpenClaw / Hermes / Codex 已全部跟进采用</li>
          </ul>
          <b>Claude Skills 三大典型用法</b>：(a) 个人级 — "我的写作风格"、"我的简历模板"；(b) 团队级 — "公司 RFP 流程"、"内部 API 调用规范"；(c) 行业级 — "PDF 发票解析"、"SQL 优化建议"、"K8s 排障"。
        `, "tip")}
      </div>

      <div class="section" style="background:linear-gradient(135deg,#fef9c3 0%,#fde68a 100%);border:1px solid #fbbf24;">
        <h2 style="color:#92400e;">3. OpenClaw 详解 — 开源持续在线 Agent 框架（用户重点关注 🆕）</h2>
        <p class="text-sm text-slate-700">
          OpenClaw 是 2025 出圈的开源 Agent 框架，建立在 <b>pi-agent</b> 之上，
          被 Hermes Agent 官方文档点名归类为 "与 Claude Code、Codex、Hermes 同类的自治编程 / 任务执行 Agent"。
          它的差异化在于：<b>不是临时聊天机器人，而是 7×24 持续在线的数字员工</b>，
          通过 Gateway 同时挂在多个消息平台 / Web / 邮件 / Cron 上，由 Goal → Plan → Act → Observe → Evaluate → Repeat 闭环驱动。
        </p>
        ${MCH.code(openclaw_code, "python")}
        <div class="grid-3 mt-3">
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>🏗 架构核心 — Gateway 中枢</h3>
            <p style="font-size:13px;line-height:1.6">
              单一 Gateway 进程统一管理：<b>会话状态</b>、<b>跨通道路由</b>（Telegram / WA / Discord / Slack / iMessage / Signal / Web）、
              <b>工具执行</b>、<b>权限策略</b>、<b>cron / webhook 调度</b>。设计目标是把 Agent 当成 Linux 服务部署 — install / start / stop / status 全套。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>📝 Markdown 三件套零代码定制</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>SOUL.md</b>（人格 / 价值观 / 拒答边界）、<b>AGENTS.md</b>（工具白名单 / 高危动作策略 / 产出格式）、<b>USER.md</b>（用户画像 / 长期目标）。
              非工程同学用 Markdown 就能改 Agent 行为，避免代码改造。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>🔧 内置工具 + 零代码 Skill 引擎</h3>
            <p style="font-size:13px;line-height:1.6">
              内置 exec / read-write-edit / web_fetch / browser 等基础工具；
              "Drop-in" 技能：把脚本扔进 <code>~/.openclaw/skills/&lt;name&gt;/</code> 即被加载，
              格式与 Claude Skills 互兼容（SKILL.md + scripts/ + reference/）。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>👥 子 Agent 派生（Sub-Agent Spawning）</h3>
            <p style="font-size:13px;line-height:1.6">
              父 Agent 可在隔离工作区里派生子 Agent，独立权限 + 独立 timeout，做完回调返回。
              典型场景：父 Agent 是产品经理，派生"工程师"子 Agent 写代码、"QA"子 Agent 跑测试 — 类似 MetaGPT 但跑在生产基础设施上。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>💾 持续会话 + 长期记忆</h3>
            <p style="font-size:13px;line-height:1.6">
              所有对话以 session 为单位持久化，可跨天 resume；
              短期记忆走对话历史，长期记忆走向量库（可插拔 Honcho / Mem0 等后端，与 Hermes Agent 共享生态）。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>🌐 多通道接入</h3>
            <p style="font-size:13px;line-height:1.6">
              Gateway 同时挂在 Telegram / WhatsApp / Discord / Slack / iMessage / Signal / web chat 上，
              所有通道共享同一个 Agent 大脑、同一份记忆 — 你在 Telegram 派的活儿，回到 PC 还能接着追问。
            </p>
          </div>
        </div>
        ${MCH.info(`
          <b>OpenClaw 与 Hermes 的关系</b>：两者技术理念高度同源（都强调持续在线 + 多通道 + Skills），社区有大量从 OpenClaw 迁移到 Hermes 的需求，
          Hermes Agent 直接内置 <code>hermes claw migrate</code> 一键迁移命令。本质上可看作"同一类持续在线 Agent 的两种实现"，
          OpenClaw 早期更轻量、Hermes 4 + Hermes Agent 更强（68 工具、profile、worktree、provider-agnostic 等更完备）。
        `, "tip")}
      </div>

      <div class="section">
        <h2>4. Hermes 4 + Hermes Agent 详解 — Nous Research 自演化旗舰</h2>
        ${MCH.code(hermes_code, "python")}
        <div class="grid-3 mt-3">
          <div class="card" style="border-left:3px solid #6366f1;">
            <h3>🧠 Hermes 4 模型族（2025.08）</h3>
            <p style="font-size:13px;line-height:1.6">
              三尺寸 <b>14B / 70B / 405B</b> 混合推理模型；通过 <code>&lt;think&gt;…&lt;/think&gt;</code> 显式切换思考模式，
              训练数据较 Hermes 3 <b>扩大 50×</b>，由 Atropos 合成框架生成（重点强化创意写作 + 复杂推理）。
              许可：开源（base 系），权重可自托管。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #6366f1;">
            <h3>🛠 Hermes Agent · 68 内置工具</h3>
            <p style="font-size:13px;line-height:1.6">
              一个 CLI / IDE / 消息平台共用的开源 Agent。68 个工具覆盖：fs / shell / git / browser (CDP) /
              web_fetch / search / image / audio / cron / scheduling / mcp_proxy / ……可按 toolset 分组开关。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #6366f1;">
            <h3>♻️ 自演化 Skills (Procedural Memory)</h3>
            <p style="font-size:13px;line-height:1.6">
              Agent 解决一次复杂问题或被纠正后，<b>主动把流程总结成 SKILL.md 存到 <code>~/.hermes/skills/</code></b>，
              下次自动加载 — 真实意义的"越用越聪明"。配套 <code>hermes skills publish / install / search</code> 注册中心。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #6366f1;">
            <h3>🔌 Provider-Agnostic + MCP</h3>
            <p style="font-size:13px;line-height:1.6">
              OpenRouter / Anthropic / OpenAI / DeepSeek / Nous / 本地 (Ollama / vLLM / llama.cpp) + 15+ 提供商，
              中途可换；MCP 既能当 Client（接社区 Server），也能 <code>hermes mcp serve</code> 反向暴露自己给别的 Agent。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #6366f1;">
            <h3>👤 Profiles + Worktree</h3>
            <p style="font-size:13px;line-height:1.6">
              多个独立 profile（个人 / 工作 / 客户）配置/会话/技能/记忆隔离；
              <code>--worktree</code> 模式开多个 git worktree 并行让多个 Agent 在不同分支同时干活，互不干扰。
            </p>
          </div>
          <div class="card" style="border-left:3px solid #6366f1;">
            <h3>🌍 多平台 Gateway</h3>
            <p style="font-size:13px;line-height:1.6">
              Telegram / Discord / Slack / WhatsApp / Signal / Matrix / Email + 10+ 通道全工具访问，
              不是只能聊天 — 在 Telegram 让它跑测试 / 改代码 / 部署线上都行（叠加 <code>/yolo</code> 关闭审批）。
            </p>
          </div>
        </div>
        ${MCH.info(`
          <b>Hermes Agent 的"杀器"在 Slash Commands</b> — 比之于 Claude Code、Codex CLI 多出一批 Agent 控制原语：
          <code>/goal</code> 跨轮长目标、<code>/steer</code> 工具调用之间注入指令、<code>/branch</code> 分叉并行、
          <code>/rollback N</code> 文件系统回滚 N 个 checkpoint、<code>/curator</code> 后台技能维护、
          <code>/kanban</code> 多 profile 协作板。这些是把"Agent 当生产力工具长期使用"沉淀出的工程原语。
        `, "tip")}
      </div>

      <div class="section">
        <h2>5. 三大代表方案核心对比（横评）</h2>
        <table class="table">
          <thead>
            <tr><th>维度</th><th>🏛 Claude Skills (Anthropic)</th><th>🌐 OpenClaw (pi-agent 系开源)</th><th>🤖 Hermes 4 + Hermes Agent (NousResearch)</th></tr>
          </thead>
          <tbody>
            <tr><td><b>定位</b></td><td>专有运行时的"能力包"</td><td>持续在线开源数字员工</td><td>开源自演化 Agent + 配套混合推理模型</td></tr>
            <tr><td><b>底层模型</b></td><td>Claude 4 / 4.5 (闭源)</td><td>任意 LLM（接 OpenAI / Claude / Qwen / DeepSeek / Hermes / 本地）</td><td>任意 LLM（OpenRouter / Anthropic / OpenAI / DeepSeek / Nous / 本地）+ 自家 Hermes 4</td></tr>
            <tr><td><b>核心创新</b></td><td>渐进披露 + SKILL.md 三级加载 + 文件系统级权限</td><td>SOUL/AGENTS/USER.md 三件套 + Gateway 多通道 + 子 Agent 派生</td><td>自演化 Skills（Agent 自己沉淀技能）+ 68 工具 + Profile 隔离 + Worktree 并行</td></tr>
            <tr><td><b>能力分发载体</b></td><td>SKILL.md + scripts/ + reference/</td><td>同 Skills 格式（兼容）· 落到 ~/.openclaw/skills/</td><td>同 Skills 格式 + Hub 商店（hermes skills install）</td></tr>
            <tr><td><b>工具协议</b></td><td>原生 Function Calling + MCP</td><td>原生工具 + MCP</td><td>原生 + MCP（双向：Client 接 + serve 反向暴露）</td></tr>
            <tr><td><b>多平台触达</b></td><td>API / Claude Code / 浏览器（Computer Use）</td><td>Telegram / WA / Discord / Slack / iMessage / Signal / web</td><td>Telegram / Discord / Slack / WA / Signal / Matrix / Email + 10+</td></tr>
            <tr><td><b>多 Agent / 子 Agent</b></td><td>Computer Use 内可派生 Tasks</td><td>父 → 子 Agent，隔离 workspace + 权限 + timeout</td><td>worktree 并行 + spawn task；可作为 MCP server 被别人调度</td></tr>
            <tr><td><b>记忆</b></td><td>Projects / Memory（专有）</td><td>session 持久化 + 长期向量库</td><td>session + Honcho / Mem0 多后端可插拔</td></tr>
            <tr><td><b>商业模式</b></td><td>Anthropic 订阅 + API</td><td>开源（社区 + 服务）</td><td>开源 + Nous API + 社区 Hub</td></tr>
            <tr><td><b>典型场景</b></td><td>编程助手、企业内部专家、ToB SaaS</td><td>客服中台、多通道运营、跨群的"数字员工"</td><td>本地强 Agent、研究自动化、Crypto / Web3 / 投研社区</td></tr>
            <tr><td><b>优势</b></td><td>体验最好，安全/对齐严谨，商业稳定</td><td>厂商中立，Markdown 即配置，多平台同时挂</td><td>开源 + 自演化 + 提供商无关，Slash 命令体系最全</td></tr>
            <tr><td><b>劣势</b></td><td>绑定 Claude，闭源，价格高</td><td>生态早期，Skill 商店不如 Hermes Hub 大</td><td>能力上限受 base 模型影响，运维门槛中等</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. Computer Use / GUI Agent 路线对比</h2>
        <table class="table">
          <thead><tr><th>方案</th><th>厂商</th><th>核心机制</th><th>成熟度</th></tr></thead>
          <tbody>
            <tr><td><b>Claude Computer Use</b></td><td>Anthropic</td><td>VLM 看截图 + 输出鼠标/键盘动作</td><td>🏆 商用</td></tr>
            <tr><td><b>OpenAI Operator</b></td><td>OpenAI</td><td>云端浏览器 Agent + GPT-4o</td><td>商用 (Pro)</td></tr>
            <tr><td><b>Project Mariner</b></td><td>Google</td><td>Chrome 扩展 + Gemini 2</td><td>研究预览</td></tr>
            <tr><td><b>Browser Use</b></td><td>开源</td><td>Playwright + LLM 驱动</td><td>开源主流</td></tr>
            <tr><td><b>OS-Atlas / CogAgent</b></td><td>开源</td><td>专门 GUI VLM + 操作 grounding</td><td>研究 → 工程</td></tr>
            <tr><td><b>UI-TARS / UI-TARS-1.5</b></td><td>字节</td><td>原生 GUI 操作大模型</td><td>2025 开源</td></tr>
            <tr><td><b>Hermes Agent + browser CDP</b></td><td>NousResearch</td><td>桌面 / 浏览器 + 任意 LLM</td><td>开源生产可用</td></tr>
            <tr><td><b>OpenClaw + browser tool</b></td><td>开源</td><td>持续在线 + 多通道触发屏幕操作</td><td>开源</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 开源 Agent 全景（2025-2026）</h2>
        <div class="grid-2">
          <div class="card">
            <h3>💻 编程 / 工程类（CLI 与 IDE）</h3>
            <ul style="font-size:13px;line-height:1.65;padding-left:18px;list-style:disc inside;">
              <li><b>Claude Code</b> (Anthropic) — 官方 CLI 编程 Agent，Skills 一等公民</li>
              <li><b>OpenAI Codex CLI / Operator</b> — 浏览器 + 终端 Agent</li>
              <li><b>Cursor</b> — 闭源 IDE，Composer / Agent 模式</li>
              <li><b>Cline</b> (formerly Claude Dev) — 开源 VS Code · MCP 全集成</li>
              <li><b>Aider</b> (Paul Gauthier) — 开源 CLI 配对程序员、Git 友好</li>
              <li><b>Continue (continue.dev)</b> — 开源 IDE 插件</li>
              <li><b>Bolt.new / StackBlitz · Replit Agent</b> — 浏览器内全栈生成</li>
              <li><b>Devin (Cognition)</b> — 闭源软件工程师 Agent</li>
              <li><b>OpenHands / OpenDevin</b> — 开源 Devin 复刻 · MIT</li>
              <li><b>SWE-Agent (Princeton)</b> — SWE-bench SOTA 早期</li>
              <li><b>OpenCode</b> — 开源 Claude Code 替身</li>
              <li><b>Hermes Agent</b> — 同时是编程 + 任务 + 知识 + 多通道</li>
              <li><b>OpenClaw</b> — 持续在线 Agent，含编程能力</li>
            </ul>
          </div>
          <div class="card">
            <h3>🔍 研究 / 知识 / 通用执行</h3>
            <ul style="font-size:13px;line-height:1.65;padding-left:18px;list-style:disc inside;">
              <li><b>Perplexity Pro / Deep Research</b> — 网页检索 + 长报告</li>
              <li><b>OpenAI Deep Research</b> · <b>Gemini Deep Research</b></li>
              <li><b>STORM (Stanford)</b> — 开源研究写作</li>
              <li><b>GPT-Researcher / Sci-Agent</b> — 开源研究自动化</li>
              <li><b>AutoGPT</b> (deprecated) → <b>MetaGPT / AutoGen / CrewAI</b></li>
              <li><b>LangGraph (LangChain)</b> — 状态机 Agent · 工业级</li>
              <li><b>OpenAI Swarm / Agents SDK</b></li>
              <li><b>Smol Agents (HF)</b> — 极简 Code Agent</li>
              <li><b>Browser Use</b> — Playwright + LLM</li>
              <li><b>AgentOps · Langfuse · LangSmith</b> — 监控评测</li>
            </ul>
          </div>
          <div class="card">
            <h3>📡 协议 / 标准 / 基础设施</h3>
            <ul style="font-size:13px;line-height:1.65;padding-left:18px;list-style:disc inside;">
              <li><b>MCP (Model Context Protocol, Anthropic 2024.11)</b> — 🏆 工具 / 数据源接口标准（USB-C of AI），1000+ Server</li>
              <li><b>A2A (Agent-to-Agent, Google 2025)</b> — Agent 间互联</li>
              <li><b>OpenAPI / JSON Schema for Tool Calling</b></li>
              <li><b>Skills 格式</b>（事实标准）— Claude / OpenClaw / Hermes 共用</li>
              <li><b>ACP (Agent Client Protocol)</b> — IDE 集成层</li>
            </ul>
          </div>
          <div class="card">
            <h3>🦾 模型层 · 开源 Agent 大脑</h3>
            <ul style="font-size:13px;line-height:1.65;padding-left:18px;list-style:disc inside;">
              <li><b>Hermes 4 (14B/70B/405B)</b> — 🏆 开源混合推理</li>
              <li><b>Qwen3 / QwQ-32B</b> — 中文推理 + Tool Use</li>
              <li><b>DeepSeek-V3 / R1</b> — 开源极致性价比 + 推理</li>
              <li><b>Llama 3.3 / 4</b> — Meta 开源大底座</li>
              <li><b>Mistral / Mixtral / Pixtral</b></li>
              <li><b>OLMo / Tülu / Molmo (AI2)</b> — 完全开放</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>8. 核心技术方案总结</h2>
        <div class="grid-3">
          <div class="card">
            <h3>📡 MCP (Model Context Protocol)</h3>
            <p style="font-size:13px;line-height:1.6">Anthropic 2024.11 发布的工具调用标准协议。Server 暴露 tools/resources/prompts；Client（Claude/OpenAI/Cursor/Cline/OpenClaw/Hermes）统一接入。<b>已成事实标准</b>，社区 1000+ MCP Server。</p>
          </div>
          <div class="card">
            <h3>🧱 渐进披露 (Progressive Disclosure)</h3>
            <p style="font-size:13px;line-height:1.6">Skills 核心思想：分级加载、按需读取。从 50-token 元数据 → 全文 → 子文件。和 RAG 的本质区别：能力包还含可执行脚本与示例。Claude / OpenClaw / Hermes 共用。</p>
          </div>
          <div class="card">
            <h3>♻️ 自演化 Skills (Procedural Memory)</h3>
            <p style="font-size:13px;line-height:1.6">Hermes 首创：Agent 解决问题后自动总结流程为 SKILL.md。下次直接复用，越用越聪明。这是把"短期 trace 记忆"升维到"过程性知识"的关键一跳。</p>
          </div>
          <div class="card">
            <h3>🔄 长任务循环 + Checkpoint</h3>
            <p style="font-size:13px;line-height:1.6">真实任务往往 50-500 步：截图 → 思考 → 行动 → 观察 → 反思。需要 KV-Cache 复用、错误恢复、用户中断点、文件系统 checkpoint（Hermes <code>/rollback N</code> 模式）、沙箱权限分级。</p>
          </div>
          <div class="card">
            <h3>📋 计划 + 执行解耦</h3>
            <p style="font-size:13px;line-height:1.6">由强推理模型（o1/R1/Hermes 4 405B）做计划，由便宜模型（Sonnet 4 / Haiku / Qwen3）执行。OpenAI o1 + Codex、Cursor Tab + Composer、Hermes 4 + Sonnet 4 都是这个套路。</p>
          </div>
          <div class="card">
            <h3>📝 Markdown 即配置</h3>
            <p style="font-size:13px;line-height:1.6">OpenClaw SOUL/AGENTS/USER.md、Cursor .cursorrules、Claude CLAUDE.md、Hermes Personalities — 业界共识：Agent 行为不该写在 Python 里，应该可由产品经理 / 用户直接编辑 Markdown。</p>
          </div>
          <div class="card">
            <h3>👤 Profile + Worktree 隔离</h3>
            <p style="font-size:13px;line-height:1.6">Hermes 把 git worktree 概念带入 Agent — 不同分支 / 不同客户 / 不同安全级别的 Agent 同时跑互不影响。这是从"实验玩具"走向"生产生产力工具"的标志。</p>
          </div>
          <div class="card">
            <h3>🛡 沙箱 + Human-in-loop</h3>
            <p style="font-size:13px;line-height:1.6">敏感动作（删除文件、付款、发邮件、推 main、部署 prod）必须人工确认。Anthropic API trust-tier、Cursor "Yolo Mode"、OpenClaw <code>require_human_approval_on</code>、Hermes <code>--yolo</code> 都是产品化方案。</p>
          </div>
          <div class="card">
            <h3>🌐 多通道 Gateway</h3>
            <p style="font-size:13px;line-height:1.6">同一个 Agent 大脑同时挂在 Telegram / Slack / Discord / WhatsApp / Web / 邮件 / cron 上 — 异步任务在哪都能下达，状态在中枢统一。OpenClaw / Hermes 是这条路线代表。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>9. 应用现状（2025-2026）</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>典型工作流</th><th>代表产品 / 框架</th></tr></thead>
          <tbody>
            <tr><td>💻 软件工程</td><td>需求 → 多文件改 → 测试 → PR</td><td>Cursor · Claude Code · Cline · Devin · OpenHands · Hermes Agent · OpenClaw</td></tr>
            <tr><td>🔍 知识研究</td><td>检索数十网页 → 长报告</td><td>Perplexity · OpenAI/Gemini Deep Research · STORM</td></tr>
            <tr><td>🖥 桌面自动化</td><td>填表 / 录入 / 跨应用搬运</td><td>Claude Computer Use · UiPath × LLM · UI-TARS · OpenClaw + browser</td></tr>
            <tr><td>🌐 浏览器 RPA</td><td>订机票 / 比价 / 抓取</td><td>Operator · Browser Use · MultiOn · Convergence · Hermes /browser</td></tr>
            <tr><td>📊 数据分析</td><td>读 CSV / SQL → 出图 + 结论</td><td>ChatGPT ADA · Code Interpreter · Cursor SQL Agent · Hermes</td></tr>
            <tr><td>📞 客服 + 工单 + 多通道运营</td><td>对话 + 业务系统 + 升级</td><td>🏆 OpenClaw / Hermes Gateway · Salesforce Agentforce · 阿里千问客服</td></tr>
            <tr><td>📅 个人助手</td><td>日历 / 邮件 / 通讯录</td><td>Apple Intelligence · Google Astra · Notion AI · Hermes profile=personal</td></tr>
            <tr><td>🤝 多 Agent 团队</td><td>PM / Dev / QA 协作</td><td>MetaGPT · CrewAI · Swarm · OpenClaw 子 Agent · Hermes worktree</td></tr>
            <tr><td>📈 投研 / 量化 / Web3</td><td>多源信号 + 自动决策</td><td>Hermes (Nous 社区强势) · OpenClaw 持续监控 · Bittensor 子网</td></tr>
            <tr><td>🦾 机器人 / Embodied</td><td>VLM → VLA → 物理动作</td><td>Figure 02 · Helix · OpenVLA · π0</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>10. 未来发展方向 + 关键难点</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>📈 长任务可靠性 (50% → 90%)</h3>
            <p style="font-size:13px;line-height:1.6">100+ 步任务成功率是行业大瓶颈。组合拳：(1) 推理模型 + Reflexion；(2) 状态可回溯（Hermes <code>/rollback</code>）；(3) 中间检查点 + 人审；(4) Sub-Agent 分层；(5) 自演化 Skills 沉淀失败模式。SWE-bench 50% → 80% 是关键里程碑。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>🛡 安全 / 沙箱 / Prompt Injection</h3>
            <p style="font-size:13px;line-height:1.6">Prompt Injection、跨网站数据泄漏、Agent 误删除/误付款已是现实风险。OWASP LLM Top 10 + Constitutional AI + 权限白名单 + tool tier + checkpoint 是必备工程基础设施。<b>OpenClaw 的 require_human_approval_on / Hermes 的 trust-tier</b> 都在产品化这层。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>♻️ 自演化 / 终身学习</h3>
            <p style="font-size:13px;line-height:1.6">Agent 工作产生新数据 → 沉淀新 Skills → 再训练 base model。Voyager (Minecraft)、AutoMix、<b>Hermes 自演化 Skills</b> 是早期实验。但目前仍偏研究 — 怎么避免"沉淀垃圾技能"是开放问题。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>🌐 Agent 互联 (A2A)</h3>
            <p style="font-size:13px;line-height:1.6">Google A2A、Anthropic 多 Agent SDK、Hermes <code>mcp serve</code> 反向暴露。下一代 SaaS = Agent 网络，需要：身份 / 计费 / 信任协议 / 跨厂商互操作。当前还很早期。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>💰 成本可控</h3>
            <p style="font-size:13px;line-height:1.6">单一长任务可能消耗 10-100 万 token。Caching、推测解码、Mixture-of-Tokenizer、动态思考预算（Hermes <code>/reasoning</code> level）、Sub-Agent 用便宜模型是关键。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>📦 Skills 生态商店</h3>
            <p style="font-size:13px;line-height:1.6">类比 App Store。需要：发现机制（Hermes Skills Hub / Anthropic Skills 注册中心）、版本管理、安全审核、计费分成。短期看 Hermes Hub + Anthropic Skills 双生态领先；OpenClaw 兼容两边。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>👤 拟人化 / 长期记忆</h3>
            <p style="font-size:13px;line-height:1.6">能记住"我是谁、我喜欢什么、上次聊到哪"。Claude Projects/Memory · ChatGPT Memory · Mem.ai · Hermes Honcho/Mem0 是雏形。隐私同时是大坑（GDPR / 中国个人信息保护法）。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>📏 统一评测 + 生产监控</h3>
            <p style="font-size:13px;line-height:1.6">SWE-bench / OSWorld / WebArena / TAU-Bench / GAIA 是研究基准；生产侧 AgentOps · Langfuse · LangSmith 监控线上 trace。但缺真实"日 PV 千万级"生产 benchmark。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>🌍 多通道 → 真"数字员工"</h3>
            <p style="font-size:13px;line-height:1.6">从聊天框走到 Telegram / Slack / 邮件 / cron — Agent 能否真正"7×24 执勤"是它从工具升格为雇员的关键。OpenClaw / Hermes Gateway 是当下最完备的实验。</p>
          </div>
          <div class="card" style="border-left:3px solid #7c3aed;">
            <h3>🇨🇳 中文 / 行业本地化</h3>
            <p style="font-size:13px;line-height:1.6">现有开源 Agent 框架默认英文 + 国外接入。中文场景下 OpenClaw / Hermes 需配 Qwen3 / DeepSeek-V3 + 国内 IM (微信 / 飞书 / 钉钉) 网关 + 国产 MCP Server。这是巨大空白市场。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>11. 优缺点总结 + 选型建议矩阵</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3>✅ 总体优势</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>"会聊"→"会做"，业务真闭环</li>
              <li>开源生态从模型到协议 (MCP / A2A) 齐备</li>
              <li>Skills 让能力可分发可组合（三家共识）</li>
              <li>推理模型大幅提升复杂任务成功率</li>
              <li>商业模式清晰（订阅 + API + Per-Agent SaaS）</li>
              <li>多通道 Gateway 让 Agent "下沉到所有入口"</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #ef4444;">
            <h3>❌ 主要难点</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>长任务可靠性仍不足（50% 关）</li>
              <li>Prompt Injection / 越权是结构性风险</li>
              <li>成本不可预测（单次 10-100 万 token）</li>
              <li>评测缺真实生产 benchmark</li>
              <li>跨厂商 / 跨 Agent 互操作仍早期</li>
              <li>中文 + 行业本地化生态稀缺</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>◎ 选型建议</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>商用顶配编程：Claude Code + Cursor + Skills</li>
              <li>开源私有化 + 多通道：<b>OpenClaw</b> 或 <b>Hermes Agent</b> + DeepSeek-V3 / Qwen3</li>
              <li>开源单兵编程：Cline + OpenHands + Aider</li>
              <li>研究 / 报告：Perplexity Pro · Deep Research</li>
              <li>桌面自动化：Claude Computer Use · Hermes /browser</li>
              <li>多 Agent 协作：LangGraph · CrewAI · OpenClaw 子 Agent</li>
              <li>本地 / 离线 / Crypto / Web3：Hermes 4 + Hermes Agent</li>
              <li>客服 + 多群运营："数字员工"角色 → OpenClaw + 国产 IM 网关</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>12. 资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://docs.claude.com/en/docs/agents-and-tools/agent-skills/overview" target="_blank">Anthropic Agent Skills 官方文档</a></li>
          <li><a href="https://modelcontextprotocol.io/" target="_blank">MCP 官方</a> · <a href="https://github.com/modelcontextprotocol/servers" target="_blank">MCP Servers</a></li>
          <li><a href="https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo" target="_blank">Claude Computer Use Demo</a></li>
          <li><a href="https://docs.openclaw.ai/" target="_blank">OpenClaw 官方文档</a> · <a href="https://claw-stack.com/en/openclaw/" target="_blank">Claw-Stack 研究平台</a></li>
          <li><a href="https://hermes4.nousresearch.com/" target="_blank">Hermes 4 (NousResearch)</a> · <a href="https://hermes-agent.nousresearch.com/" target="_blank">Hermes Agent 官方</a></li>
          <li><a href="https://github.com/All-Hands-AI/OpenHands" target="_blank">OpenHands (开源 Devin)</a> · <a href="https://github.com/cline/cline" target="_blank">Cline</a> · <a href="https://github.com/Aider-AI/aider" target="_blank">Aider</a></li>
          <li><a href="https://www.swebench.com/" target="_blank">SWE-bench</a> · <a href="https://os-world.github.io/" target="_blank">OSWorld</a> · <a href="https://gaia-benchmark-leaderboard.hf.space/" target="_blank">GAIA</a></li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
