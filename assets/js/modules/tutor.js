/* 模块：AI 算法学习助理（多轮对话 + 历史持久化） */
MCH.register("tutor", {
  render() {
    return `
      ${MCH.hero({
        icon: "👨‍🏫",
        name: "AI 算法学习助理",
        en: "Multi-turn Tutor · Persistent Chat History",
        tags: ["多轮对话", "历史持久化", "示例问题", "算法专家"],
        meta: ["◈ 基于你配置的 LLM（混元/GPT/Claude/DeepSeek ...）", "⚡ 像私人导师一样，随时答疑"],
      })}

      <div class="section" style="padding:0;overflow:hidden;height:calc(100vh - 220px);min-height:600px">
        <div style="display:grid;grid-template-columns:260px 1fr;height:100%;overflow:hidden">
          <!-- ━━━━━ 左侧历史列表 ━━━━━ -->
          <aside id="tutor-sidebar" style="border-right:1px solid var(--section-border);background:var(--bg-secondary);display:flex;flex-direction:column;min-height:0;overflow:hidden">
            <div style="padding:12px;border-bottom:1px solid var(--section-border);flex-shrink:0">
              <button id="tutor-new" style="width:100%;padding:8px 12px;border-radius:8px;background:var(--accent);color:#fff;font-size:13px;font-weight:600;border:none;cursor:pointer">➕ 新对话</button>
            </div>
            <div style="padding:8px 12px;font-size:11px;color:var(--text-muted);font-weight:700;flex-shrink:0">历史对话</div>
            <div id="tutor-history" style="flex:1 1 0;overflow-y:auto;padding:0 8px;min-height:0"></div>
            <div style="padding:10px;border-top:1px solid var(--section-border);font-size:11px;color:var(--text-muted);flex-shrink:0">
              <div>💾 自动保存到 localStorage</div>
              <div id="tutor-server-status"></div>
            </div>
          </aside>

          <!-- ━━━━━ 右侧对话区 ━━━━━ -->
          <main style="display:flex;flex-direction:column;background:var(--section-bg);min-height:0;overflow:hidden">
            <!-- 顶栏（固定不滚） -->
            <div style="padding:12px 18px;border-bottom:1px solid var(--section-border);display:flex;justify-content:space-between;align-items:center;gap:10px;flex-shrink:0">
              <div style="min-width:0;flex:1">
                <div style="font-weight:700;font-size:15px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" id="tutor-title">AI 算法学习助理</div>
                <div style="font-size:11px;color:var(--text-muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap" id="tutor-subtitle">选择一个示例问题开始，或直接提问</div>
              </div>
              <div style="display:flex;gap:6px;flex-shrink:0">
                <select id="tutor-llm-select" style="padding:5px 8px;font-size:12px;border-radius:6px;border:1px solid var(--section-border);background:var(--bg-secondary);color:var(--text-primary)"></select>
                <button id="tutor-export" title="导出当前对话为 Markdown" style="padding:5px 10px;font-size:12px;border-radius:6px;border:1px solid var(--section-border);background:var(--bg-secondary);color:var(--text-primary);cursor:pointer">💾 MD</button>
                <button id="tutor-export-all" title="导出所有对话为 JSON（供备份/迁移）" style="padding:5px 10px;font-size:12px;border-radius:6px;border:1px solid var(--section-border);background:var(--bg-secondary);color:var(--text-primary);cursor:pointer">📦 全部</button>
                <button id="tutor-import-all" title="从 JSON 文件导入对话" style="padding:5px 10px;font-size:12px;border-radius:6px;border:1px solid var(--section-border);background:var(--bg-secondary);color:var(--text-primary);cursor:pointer">📥 导入</button>
                <button id="tutor-clear" title="清空当前对话" style="padding:5px 10px;font-size:12px;border-radius:6px;border:1px solid var(--section-border);background:var(--bg-secondary);color:var(--text-primary);cursor:pointer">🗑</button>
              </div>
            </div>

            <!-- 对话流（可滚动） -->
            <div id="tutor-chat" style="flex:1 1 0;overflow-y:auto;overflow-x:hidden;padding:18px;display:flex;flex-direction:column;gap:12px;min-height:0">
              <!-- 欢迎屏（有示例问题） -->
              <div id="tutor-welcome" class="card" style="max-width:100%;text-align:center;padding:24px">
                <div style="font-size:32px">👨‍🏫</div>
                <h3 style="margin-top:10px">你好！我是 AI 算法学习助理</h3>
                <p style="color:var(--text-secondary);margin-bottom:12px">可以问我任何算法相关问题。选一个示例开始 👇</p>
                <!-- 分类筛选 tabs -->
                <div id="tutor-example-tabs" style="display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin:12px 0"></div>
                <div id="tutor-examples" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:12px;text-align:left"></div>
              </div>
            </div>

            <!-- 输入框（固定底部不滚） -->
            <div style="padding:14px 18px;border-top:1px solid var(--section-border);background:var(--bg-secondary);flex-shrink:0">
              <div style="position:relative;display:flex;gap:8px;align-items:flex-end">
                <textarea id="tutor-input" placeholder="输入你的问题（Shift+Enter 换行，Enter 发送）..." rows="2"
                          style="flex:1;padding:10px 14px;border-radius:10px;border:1px solid var(--section-border);background:var(--section-bg);color:var(--text-primary);font-size:14px;resize:none;font-family:inherit;line-height:1.5;max-height:180px;overflow-y:auto"></textarea>
                <button id="tutor-send" style="padding:10px 18px;border-radius:10px;background:var(--accent);color:#fff;font-size:14px;font-weight:600;border:none;cursor:pointer;white-space:nowrap">📨 发送</button>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:6px;font-size:11px;color:var(--text-muted)">
                <div id="tutor-model-info">—</div>
                <div id="tutor-token-info">—</div>
              </div>
            </div>
          </main>
        </div>
      </div>
    `;
  },

  mount() {
    if (window.AIAL_TUTOR) { AIAL_TUTOR.mount(); return; }

    /* ════════════════════════════════════════════════
     *   AIAL_TUTOR — 核心控制器（单例）
     * ════════════════════════════════════════════════ */
    const TUTOR_KEY = "aial-tutor-sessions";
    const TUTOR_ACTIVE = "aial-tutor-active";

    // —— 内置 25 个示例问题（6 大类）——
    const EXAMPLES = [
      // 入门
      { cat: "入门", icon: "🌱", q: "我是零基础小白，想转 AI 算法，我该从哪里开始？" },
      { cat: "入门", icon: "🤔", q: "机器学习、深度学习、大模型的区别是什么？" },
      { cat: "入门", icon: "📐", q: "学 AI 需要什么数学基础？有没有速成路径？" },
      { cat: "入门", icon: "💻", q: "不会编程能学 AI 吗？用 Python 够吗？" },
      // 基础 ML
      { cat: "基础", icon: "🌳", q: "决策树、随机森林、XGBoost 的核心区别和选型建议？" },
      { cat: "基础", icon: "🔍", q: "什么时候该用 SVM？什么时候该用神经网络？" },
      { cat: "基础", icon: "⚖", q: "过拟合和欠拟合怎么判断？有哪些解决方法？" },
      { cat: "基础", icon: "📊", q: "特征工程有哪些常用技巧？类别特征怎么处理？" },
      { cat: "基础", icon: "🎯", q: "样本不均衡问题怎么解决？下采样/过采样/权重对比？" },
      // 深度学习
      { cat: "深度", icon: "🧠", q: "Transformer 里的 Attention 是怎么工作的？自注意力 vs 交叉注意力？" },
      { cat: "深度", icon: "🎨", q: "CNN 为什么能处理图像？卷积核和池化分别干嘛？" },
      { cat: "深度", icon: "📈", q: "我想做时序预测，LSTM、Transformer、Prophet 选哪个？" },
      { cat: "深度", icon: "🔬", q: "BatchNorm 和 LayerNorm 的本质区别？为什么 LLM 都用后者？" },
      // 大模型
      { cat: "大模型", icon: "🤖", q: "我想给公司搭建一个内部 AI 助手，用 RAG 还是微调？" },
      { cat: "大模型", icon: "🦾", q: "LoRA / QLoRA / DoRA 分别适合什么场景？显存怎么算？" },
      { cat: "大模型", icon: "💭", q: "什么是 CoT、ReAct、Agent？它们之间是什么关系？" },
      { cat: "大模型", icon: "🔧", q: "Function Calling 怎么用？和 MCP 协议什么区别？" },
      { cat: "大模型", icon: "📚", q: "RAG 如何优化？Rerank / HyDE / GraphRAG 什么时候用？" },
      // 工程落地
      { cat: "工程", icon: "🚢", q: "训练好的模型怎么部署到生产？vLLM 和 TGI 怎么选？" },
      { cat: "工程", icon: "💰", q: "如何估算训练一个 7B 模型需要多少 GPU 和多少预算？" },
      { cat: "工程", icon: "📊", q: "推荐系统的召回、粗排、精排、重排各自要解决什么问题？" },
      { cat: "工程", icon: "🔒", q: "AI 应用上线前必做的安全/合规检查清单？" },
      // 职业发展
      { cat: "职业", icon: "🎯", q: "算法工程师面试八股文重点是什么？如何高效准备？" },
      { cat: "职业", icon: "📝", q: "如何写出让 HR 秒约面试的算法岗简历？" },
      { cat: "职业", icon: "🚀", q: "工作 3 年了，P6→P7 在算法方向该怎么规划突破？" },
    ];

    const T = {
      // ============ 状态 ============
      activeId: null,
      sessions: {},

      // ============ 存储（localStorage + 服务端双通道）============
      loadSessions() {
        try { this.sessions = JSON.parse(localStorage.getItem(TUTOR_KEY) || "{}"); }
        catch { this.sessions = {}; }
        this.activeId = localStorage.getItem(TUTOR_ACTIVE) || null;
      },
      persistLocal() {
        localStorage.setItem(TUTOR_KEY, JSON.stringify(this.sessions));
        if (this.activeId) localStorage.setItem(TUTOR_ACTIVE, this.activeId);
      },
      // 保存到服务端（统一 /api/config/tutor_sessions）
      async persistServer() {
        if (!this._hasServer()) return;
        try {
          await fetch("/api/config/tutor_sessions", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sessions: this.sessions, active: this.activeId }),
          });
        } catch {}
      },
      async loadFromServer() {
        if (!this._hasServer()) return false;
        try {
          const r = await fetch("/api/config/tutor_sessions");
          if (!r.ok) return false;
          const { exists, data } = await r.json();
          if (!exists || !data || !data.sessions) return false;
          // 合并：服务端 + 本地，以 updatedAt 较新的为准
          const merged = { ...data.sessions };
          Object.entries(this.sessions).forEach(([id, s]) => {
            const srv = merged[id];
            if (!srv || new Date(s.updatedAt||0) > new Date(srv.updatedAt||0)) merged[id] = s;
          });
          this.sessions = merged;
          this.activeId = this.activeId || data.active;
          this.persistLocal();
          return true;
        } catch { return false; }
      },
      _hasServer() { return window.AIAL_LLM && AIAL_LLM._unifiedServer === "yes"; },

      persist() { this.persistLocal(); this.persistServer(); },

      // ============ 会话管理 ============
      newSession() {
        const id = "s_" + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
        const s = {
          id, title: "新对话", messages: [],
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          model: this._currentModel(),
        };
        this.sessions[id] = s;
        this.activeId = id;
        this.persist();
        return s;
      },
      switchSession(id) {
        if (!this.sessions[id]) return;
        this.activeId = id;
        localStorage.setItem(TUTOR_ACTIVE, id);
        this._renderHistory();
        this._renderChat();
      },
      deleteSession(id) {
        if (!confirm("确定删除这个对话吗？")) return;
        delete this.sessions[id];
        if (this.activeId === id) this.activeId = null;
        this.persist();
        this._renderHistory();
        this._renderChat();
      },
      clearActive() {
        if (!this.activeId || !confirm("清空当前对话的所有消息？")) return;
        this.sessions[this.activeId].messages = [];
        this.sessions[this.activeId].updatedAt = new Date().toISOString();
        this.persist();
        this._renderChat();
      },
      _activeSession() {
        if (!this.activeId || !this.sessions[this.activeId]) return null;
        return this.sessions[this.activeId];
      },
      _currentModel() {
        const cfg = window.AIAL_LLM && AIAL_LLM.getActive ? AIAL_LLM.getActive() : null;
        if (!cfg) return "未配置";
        return `${cfg.provider}/${cfg.model || "default"}`;
      },

      // ============ 渲染 ============
      mount() {
        this.loadSessions();
        // 启动时尝试从服务端恢复
        this.loadFromServer().then(() => { this._renderHistory(); this._renderChat(); });

        // 示例问题（含分类 tabs）
        this._exampleFilter = "全部";
        this._renderExamples();

        // 按钮
        document.getElementById("tutor-new").addEventListener("click", () => {
          this.newSession(); this._renderHistory(); this._renderChat();
        });
        document.getElementById("tutor-send").addEventListener("click", () => this.send());
        document.getElementById("tutor-export").addEventListener("click", () => this.exportActive());
        const btnAll = document.getElementById("tutor-export-all");
        if (btnAll) btnAll.addEventListener("click", () => this.exportAll());
        const btnImp = document.getElementById("tutor-import-all");
        if (btnImp) btnImp.addEventListener("click", () => this.importAll());
        document.getElementById("tutor-clear").addEventListener("click", () => this.clearActive());
        const ta = document.getElementById("tutor-input");
        ta.addEventListener("keydown", e => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            this.send();
          }
        });
        // 自适应高度
        ta.addEventListener("input", () => {
          ta.style.height = "auto";
          ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
        });

        // LLM 选择器
        this._initLlmSelect();

        // 服务端状态提示
        const ss = document.getElementById("tutor-server-status");
        if (ss) ss.textContent = this._hasServer() ? "🟢 已连接统一服务器" : "⚪ 仅本地存储";

        this._renderHistory();
        this._renderChat();
      },

      _catColor(cat) {
        return { "入门":"#10b981", "基础":"#4f46e5", "深度":"#8b5cf6", "大模型":"#f59e0b", "工程":"#ec4899", "职业":"#14b8a6" }[cat] || "#64748b";
      },

      _renderExamples() {
        const tabsEl = document.getElementById("tutor-example-tabs");
        const exEl = document.getElementById("tutor-examples");
        if (!tabsEl || !exEl) return;

        const cats = ["全部", ...new Set(EXAMPLES.map(e => e.cat))];
        const active = this._exampleFilter || "全部";

        // 分类 tabs
        tabsEl.innerHTML = cats.map(c => {
          const color = c === "全部" ? "var(--accent)" : this._catColor(c);
          const isActive = c === active;
          return `<button class="tutor-cat-tab" data-cat="${c}"
                          style="padding:5px 12px;border-radius:14px;font-size:12px;font-weight:600;cursor:pointer;border:1px solid ${isActive?color:'var(--section-border)'};background:${isActive?color:'var(--bg-secondary)'};color:${isActive?'#fff':'var(--text-primary)'};transition:all .15s">
                    ${c === "全部" ? "🌍 "+c : c}
                  </button>`;
        }).join("");

        // 过滤后的示例
        const filtered = EXAMPLES
          .map((e, i) => ({ ...e, _idx: i }))
          .filter(e => active === "全部" || e.cat === active);

        exEl.innerHTML = filtered.map(e => `
          <div class="tutor-example-card" data-idx="${e._idx}" style="cursor:pointer;padding:10px 12px;border:1px solid var(--section-border);border-radius:8px;background:var(--bg-secondary);transition:all .15s">
            <div style="font-size:11px;color:${this._catColor(e.cat)};font-weight:700;margin-bottom:3px">${e.icon} ${e.cat}</div>
            <div style="font-size:13px;color:var(--text-primary);line-height:1.4">${e.q}</div>
          </div>
        `).join("");

        // 绑定 tab 切换
        tabsEl.querySelectorAll(".tutor-cat-tab").forEach(btn => {
          btn.addEventListener("click", () => {
            this._exampleFilter = btn.dataset.cat;
            this._renderExamples();
          });
        });

        // 绑定示例点击
        exEl.querySelectorAll(".tutor-example-card").forEach(el => {
          el.addEventListener("mouseenter", () => { el.style.borderColor = "var(--accent)"; el.style.transform = "translateY(-1px)"; });
          el.addEventListener("mouseleave", () => { el.style.borderColor = "var(--section-border)"; el.style.transform = "translateY(0)"; });
          el.addEventListener("click", () => {
            const idx = parseInt(el.dataset.idx);
            const q = EXAMPLES[idx].q;
            document.getElementById("tutor-input").value = q;
            this.send();
          });
        });
      },

      _initLlmSelect() {
        const sel = document.getElementById("tutor-llm-select");
        if (!sel || !window.AIAL_LLM) return;
        const list = AIAL_LLM.list ? AIAL_LLM.list() : [];
        const active = AIAL_LLM.getActive ? AIAL_LLM.getActive() : null;
        if (list.length === 0) {
          sel.innerHTML = `<option value="">⚠ 未配置 LLM</option>`;
          document.getElementById("tutor-model-info").textContent = "请先在右上角 ⚙️ 配置 LLM";
        } else {
          sel.innerHTML = list.map(c => `<option value="${c.id}" ${active && c.id===active.id ? "selected":""}>${c.name || c.provider}</option>`).join("");
          sel.addEventListener("change", () => { AIAL_LLM.setActive(sel.value); this._updateModelInfo(); });
          this._updateModelInfo();
        }
      },
      _updateModelInfo() {
        const cfg = AIAL_LLM.getActive();
        document.getElementById("tutor-model-info").textContent = cfg ? `🤖 ${cfg.provider} / ${cfg.model || "default"}` : "—";
      },

      _renderHistory() {
        const el = document.getElementById("tutor-history");
        if (!el) return;
        const list = Object.values(this.sessions).sort((a,b) => new Date(b.updatedAt) - new Date(a.updatedAt));
        if (list.length === 0) { el.innerHTML = `<div style="padding:16px;text-align:center;color:var(--text-muted);font-size:12px">还没有对话历史</div>`; return; }
        el.innerHTML = list.map(s => `
          <div class="tutor-hist-item" data-id="${s.id}" style="padding:10px 10px;border-radius:8px;cursor:pointer;margin:2px 0;position:relative;${s.id===this.activeId ? 'background:var(--accent);color:#fff':''}">
            <div style="font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding-right:20px">${this._escape(s.title)}</div>
            <div style="font-size:11px;opacity:.75;margin-top:2px">${new Date(s.updatedAt).toLocaleString("zh-CN",{month:"2-digit",day:"2-digit",hour:"2-digit",minute:"2-digit"})} · ${s.messages.length} 条</div>
            <span class="tutor-hist-del" data-id="${s.id}" style="position:absolute;right:6px;top:50%;transform:translateY(-50%);cursor:pointer;opacity:.5;font-size:14px">×</span>
          </div>
        `).join("");
        el.querySelectorAll(".tutor-hist-item").forEach(node => {
          node.addEventListener("click", e => {
            if (e.target.classList.contains("tutor-hist-del")) return;
            this.switchSession(node.dataset.id);
          });
        });
        el.querySelectorAll(".tutor-hist-del").forEach(node => {
          node.addEventListener("click", e => {
            e.stopPropagation();
            this.deleteSession(node.dataset.id);
          });
        });
      },

      _renderChat() {
        const chat = document.getElementById("tutor-chat");
        const welcome = document.getElementById("tutor-welcome");
        const titleEl = document.getElementById("tutor-title");
        const subEl = document.getElementById("tutor-subtitle");
        if (!chat) return;

        const s = this._activeSession();
        if (!s || s.messages.length === 0) {
          // 显示欢迎屏
          chat.innerHTML = "";
          chat.appendChild(welcome || this._mkWelcome());
          if (titleEl) titleEl.textContent = s ? s.title : "AI 算法学习助理";
          if (subEl) subEl.textContent = "选择一个示例问题开始，或直接提问";
          return;
        }

        if (titleEl) titleEl.textContent = s.title;
        if (subEl) subEl.textContent = `${s.messages.length} 条消息 · ${new Date(s.updatedAt).toLocaleString("zh-CN")}`;

        chat.innerHTML = s.messages.map(m => this._renderMessage(m)).join("");
        chat.scrollTop = chat.scrollHeight;

        // 代码高亮
        if (window.hljs) chat.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));
      },

      _mkWelcome() {
        const div = document.createElement("div");
        div.id = "tutor-welcome";
        div.className = "card";
        div.innerHTML = `<div style="text-align:center;padding:24px"><div style="font-size:32px">👨‍🏫</div><h3>新对话已创建</h3><p style="color:var(--text-secondary)">输入你的问题开始</p></div>`;
        return div;
      },

      _renderMessage(m) {
        const isUser = m.role === "user";
        const bg = isUser ? "var(--accent)" : "var(--bg-secondary)";
        const fg = isUser ? "#fff" : "var(--text-primary)";
        const align = isUser ? "flex-end" : "flex-start";
        const icon = isUser ? "🧑" : "👨‍🏫";
        const content = this._renderMarkdown(m.content || "");
        const time = m.ts ? new Date(m.ts).toLocaleTimeString("zh-CN",{hour:"2-digit",minute:"2-digit"}) : "";
        return `
          <div style="display:flex;flex-direction:column;align-items:${align};gap:4px">
            <div style="display:flex;align-items:flex-start;gap:8px;max-width:85%;flex-direction:${isUser?'row-reverse':'row'}">
              <div style="width:30px;height:30px;border-radius:50%;background:var(--bg-tertiary);display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0">${icon}</div>
              <div style="padding:10px 14px;border-radius:12px;background:${bg};color:${fg};font-size:14px;line-height:1.6;word-wrap:break-word;overflow-wrap:break-word">${content}</div>
            </div>
            <div style="font-size:10px;color:var(--text-muted);padding:0 42px">${time}${m.model?" · "+m.model:""}</div>
          </div>
        `;
      },

      _renderMarkdown(text) {
        // 轻量级 MD → HTML：代码块 + 加粗 + 斜体 + 列表 + 段落
        let html = this._escape(text);
        // 代码块 ```xxx```
        html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_, lang, code) =>
          `<pre style="background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:6px;overflow-x:auto;margin:6px 0"><code class="language-${lang}">${code}</code></pre>`);
        // 行内代码
        html = html.replace(/`([^`\n]+)`/g, '<code style="background:var(--bg-tertiary);padding:1px 6px;border-radius:3px;font-size:12.5px">$1</code>');
        // 加粗 & 斜体
        html = html.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
        html = html.replace(/(?<!\*)\*([^\*\n]+)\*(?!\*)/g, "<i>$1</i>");
        // 标题
        html = html.replace(/^### (.+)$/gm, '<h4 style="margin:10px 0 4px;font-weight:700">$1</h4>');
        html = html.replace(/^## (.+)$/gm, '<h3 style="margin:12px 0 6px;font-weight:700">$1</h3>');
        html = html.replace(/^# (.+)$/gm, '<h2 style="margin:14px 0 8px;font-weight:800">$1</h2>');
        // 列表
        html = html.replace(/^(\d+)\. (.+)$/gm, '<div style="margin:3px 0"><b>$1.</b> $2</div>');
        html = html.replace(/^[-*] (.+)$/gm, '<div style="margin:3px 0;padding-left:12px">• $1</div>');
        // 换行
        html = html.replace(/\n\n/g, "<br><br>").replace(/(?<!>)\n/g, "<br>");
        return html;
      },

      _escape(s) {
        return String(s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));
      },

      // ============ 发送消息 ============
      async send() {
        const ta = document.getElementById("tutor-input");
        const text = (ta.value || "").trim();
        if (!text) return;

        // 确保有 active session
        if (!this._activeSession()) this.newSession();
        const s = this._activeSession();

        // 添加用户消息
        const userMsg = { role: "user", content: text, ts: new Date().toISOString() };
        s.messages.push(userMsg);
        // 首次消息作为标题
        if (s.messages.length === 1) s.title = text.slice(0, 40);
        s.updatedAt = new Date().toISOString();
        this.persist();

        ta.value = "";
        ta.style.height = "auto";
        this._renderHistory();
        this._renderChat();

        // LLM 调用
        const cfg = AIAL_LLM && AIAL_LLM.getActive ? AIAL_LLM.getActive() : null;
        if (!cfg) {
          s.messages.push({ role:"assistant", content:"⚠️ 未配置 LLM。请点击右上角 ⚙️ 添加 API 配置后重试。", ts: new Date().toISOString() });
          this.persist();
          this._renderChat();
          return;
        }

        // 构造上下文（带 system prompt + 最近 20 轮）
        const system = {
          role: "system",
          content: `你是 AI 算法学习助理，专精机器学习、深度学习、大模型、工程落地。
特点：
- 回答条理清晰，多用小标题/列表/代码示例
- 涉及数学公式时用 LaTeX（$...$ 行内，$$...$$ 块）
- 推荐相关学习资源（论文/书/开源项目）
- 若用户问题模糊，先问清具体场景再回答
- 如果涉及本平台模块，用 [模块名](#/模块id) 格式链接，例如 [Transformer](#/nn_transformer)

当前可用的平台模块 id 前缀：
- ml_* : 机器学习（ml_decision_tree, ml_xgboost, ml_kmeans 等）
- nn_* : 神经网络（nn_mlp, nn_cnn, nn_transformer 等）
- graph_* : 图算法
- llm_* : 大模型（llm_foundation, llm_cot, llm_rag, llm_react, llm_agent, llm_tool_use 等）
- ts_* : 时序
- coding_* : 基础编程算法
- case_* : 工业案例
- frontier_* : 前沿算法`,
        };
        const recent = s.messages.slice(-20);
        const messages = [system, ...recent.map(m => ({ role: m.role, content: m.content }))];

        // 占位助手消息
        const asstMsg = { role:"assistant", content:"正在思考...", ts: new Date().toISOString(), model: this._currentModel() };
        s.messages.push(asstMsg);
        this._renderChat();

        // 调用 LLM（流式）
        try {
          let full = "";
          const onChunk = (delta, acc) => {
            full = acc;
            asstMsg.content = full;
            this._renderChat();
          };
          const result = await AIAL_LLM.chat(cfg, messages, { temperature: 0.4, max_tokens: 4096, stream: true, onChunk });
          asstMsg.content = result || full || "(空响应)";
        } catch (e) {
          asstMsg.content = "❌ 调用失败：" + (e.message || String(e));
        }
        s.updatedAt = new Date().toISOString();
        this.persist();
        this._renderChat();
        this._renderHistory();
      },

      // ============ 导出 / 导入 ============
      exportActive() {
        const s = this._activeSession();
        if (!s) { alert("当前没有对话"); return; }
        const md = this._sessionToMarkdown(s);
        const blob = new Blob([md], { type: "text/markdown" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = `tutor_${s.id}_${(s.title||"chat").slice(0,30)}.md`;
        a.click();
        URL.revokeObjectURL(a.href);
      },

      // 导出全部对话为 JSON（备份/迁移用）
      exportAll() {
        const payload = {
          exported_at: new Date().toISOString(),
          version: "v6.4",
          count: Object.keys(this.sessions).length,
          active: this.activeId,
          sessions: this.sessions,
        };
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        const ts = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
        a.download = `tutor_all_${ts}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
      },

      // 从 JSON 文件导入对话（合并到现有数据）
      importAll() {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".json";
        input.onchange = async e => {
          const file = e.target.files[0];
          if (!file) return;
          try {
            const text = await file.text();
            const data = JSON.parse(text);
            if (!data.sessions) throw new Error("JSON 格式错误：缺少 sessions 字段");

            const existing = Object.keys(this.sessions).length;
            const incoming = Object.keys(data.sessions).length;
            if (!confirm(`即将导入 ${incoming} 个对话（当前已有 ${existing} 个）。\n重复 id 将覆盖。是否继续？`)) return;

            Object.assign(this.sessions, data.sessions);
            // 可选：迁移 active 指针（仅当当前无 active 时）
            if (!this.activeId && data.active) this.activeId = data.active;

            this.persist();
            this._renderHistory();
            this._renderChat();
            alert(`✅ 已导入 ${incoming} 个对话`);
          } catch (err) {
            alert("❌ 导入失败：" + (err.message || String(err)));
          }
        };
        input.click();
      },
      _sessionToMarkdown(s) {
        const lines = [`# ${s.title}`, ``, `> 创建：${s.createdAt} · 最近：${s.updatedAt} · 模型：${s.model||"-"}`, ``];
        s.messages.forEach((m, i) => {
          lines.push(`## ${m.role === "user" ? "🧑 我" : "👨‍🏫 助理"}  \`${m.ts||""}\``);
          lines.push("");
          lines.push(m.content);
          lines.push("");
        });
        return lines.join("\n");
      },
    };

    window.AIAL_TUTOR = T;
    T.mount();
  },
});
