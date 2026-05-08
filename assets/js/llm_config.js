/* ============================================================================
 * llm_config.js — 大模型 API 配置管理
 * 支持：OpenAI / Claude / DeepSeek / Qwen / Gemini / Azure / Ollama / 自定义
 * 存储：localStorage (主) + 导入/导出 JSON 文件 (备份)
 * ========================================================================== */
(function () {
  const STORAGE_KEY = "aial-llm-configs";
  const ACTIVE_KEY = "aial-llm-active";

  // 腾讯混元内置演示 Key（参考 ai_task_manager 项目）
  // 零配置即用 - 不填 Key 选择混元时自动使用此 Key
  const HUNYUAN_DEMO_KEY = "***REMOVED***";

  // 内置 Provider 预设
  const PROVIDERS = {
    hunyuan: {
      name: "🌟 腾讯混元 (内置演示 Key)",
      endpoint: "https://api.hunyuan.cloud.tencent.com/v1",
      models: ["hunyuan-turbos-latest", "hunyuan-turbo", "hunyuan-pro", "hunyuan-standard", "hunyuan-standard-256K", "hunyuan-lite"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
      // 混元专属额外参数：启用内容增强（联网/工具增强）
      extra_body: { enable_enhancement: true },
      demo_key: HUNYUAN_DEMO_KEY,
      docs: "https://cloud.tencent.com/document/product/1729",
      console: "https://console.cloud.tencent.com/hunyuan/api-key",
    },
    openai: {
      name: "OpenAI",
      endpoint: "https://api.openai.com/v1",
      models: ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "o1-preview", "o1-mini"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
    anthropic: {
      name: "Anthropic Claude",
      endpoint: "https://api.anthropic.com/v1",
      models: ["claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
      format: "anthropic",
      header: (key) => ({ "x-api-key": key, "anthropic-version": "2023-06-01" }),
    },
    deepseek: {
      name: "DeepSeek",
      endpoint: "https://api.deepseek.com/v1",
      models: ["deepseek-chat", "deepseek-reasoner"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
    qwen: {
      name: "阿里 Qwen (DashScope)",
      endpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1",
      models: ["qwen-max", "qwen-plus", "qwen-turbo", "qwen2.5-72b-instruct", "qwen2.5-coder-32b-instruct"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
    zhipu: {
      name: "智谱 GLM",
      endpoint: "https://open.bigmodel.cn/api/paas/v4",
      models: ["glm-4-plus", "glm-4-air", "glm-4-flash"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
    moonshot: {
      name: "Kimi (Moonshot)",
      endpoint: "https://api.moonshot.cn/v1",
      models: ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
    groq: {
      name: "Groq Cloud (免费层)",
      endpoint: "https://api.groq.com/openai/v1",
      models: ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
      docs: "https://console.groq.com",
    },
    siliconflow: {
      name: "硅基流动 SiliconFlow",
      endpoint: "https://api.siliconflow.cn/v1",
      models: ["Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3", "meta-llama/Meta-Llama-3.1-405B-Instruct"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
      docs: "https://siliconflow.cn",
    },
    baichuan: {
      name: "百川智能 Baichuan",
      endpoint: "https://api.baichuan-ai.com/v1",
      models: ["Baichuan4-Turbo", "Baichuan4-Air", "Baichuan3-Turbo"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
    minimax: {
      name: "MiniMax",
      endpoint: "https://api.minimax.chat/v1",
      models: ["abab6.5s-chat", "abab6.5-chat", "abab5.5-chat"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
    doubao: {
      name: "字节豆包 (火山引擎)",
      endpoint: "https://ark.cn-beijing.volces.com/api/v3",
      models: ["doubao-pro-4k", "doubao-pro-32k", "doubao-pro-128k", "doubao-lite-4k"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
    gemini: {
      name: "Google Gemini",
      endpoint: "https://generativelanguage.googleapis.com/v1beta",
      models: ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash-exp"],
      format: "gemini",
      header: () => ({}),
    },
    ollama: {
      name: "Ollama 本地",
      endpoint: "http://localhost:11434/v1",
      models: ["llama3.1", "qwen2.5", "deepseek-r1", "mistral"],
      format: "openai",
      header: () => ({}),
    },
    azure: {
      name: "Azure OpenAI",
      endpoint: "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT",
      models: ["gpt-4o", "gpt-4"],
      format: "azure",
      header: (key) => ({ "api-key": key }),
    },
    custom: {
      name: "自定义 (OpenAI 兼容)",
      endpoint: "https://your-api.com/v1",
      models: ["custom-model"],
      format: "openai",
      header: (key) => ({ Authorization: "Bearer " + key }),
    },
  };

  const LLM = {
    PROVIDERS,
    list() { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); } catch { return []; } },
    save(configs) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(configs));
      // 🔥 双通道持久化：若 unified server 可用，同步写入 data/configs/llm.json
      this._syncToServer();
    },
    getActive() {
      const id = localStorage.getItem(ACTIVE_KEY);
      const list = this.list();
      return id ? list.find(c => c.id === id) : (list[0] || null);
    },
    setActive(id) {
      localStorage.setItem(ACTIVE_KEY, id);
      this._syncToServer();
    },

    addOrUpdate(cfg) {
      const list = this.list();
      const idx = list.findIndex(c => c.id === cfg.id);
      if (idx >= 0) list[idx] = cfg;
      else list.push(cfg);
      this.save(list);
    },

    remove(id) {
      this.save(this.list().filter(c => c.id !== id));
      if (localStorage.getItem(ACTIVE_KEY) === id) localStorage.removeItem(ACTIVE_KEY);
    },

    /* ─── 服务端双通道持久化 ──────────────────── */
    _syncToServer() {
      if (this._unifiedServer !== "yes") return;
      const payload = {
        exported_at: new Date().toISOString(),
        configs: this.list(),
        active: localStorage.getItem(ACTIVE_KEY) || "",
      };
      fetch("/api/config/llm", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).catch(() => {});
    },

    /** 启动时从服务端拉取配置，合并到 localStorage（localStorage 缺失时） */
    async loadFromServer() {
      if (!await this.probeUnifiedServer()) return false;
      try {
        const r = await fetch("/api/config/llm");
        if (!r.ok) return false;
        const { exists, data } = await r.json();
        if (!exists || !data) return false;

        // 如果 localStorage 已有数据，以 localStorage 为准（避免覆盖用户最新改动）
        const localConfigs = this.list();
        if (localConfigs.length === 0 && Array.isArray(data.configs) && data.configs.length > 0) {
          localStorage.setItem(STORAGE_KEY, JSON.stringify(data.configs));
          if (data.active) localStorage.setItem(ACTIVE_KEY, data.active);
          console.info("[AIAL] ✅ 从服务端恢复 LLM 配置：" + data.configs.length + " 条");
          return true;
        }
      } catch {}
      return false;
    },

    exportAll() {
      return JSON.stringify({
        exported_at: new Date().toISOString(),
        configs: this.list().map(c => ({ ...c, apiKey: c.apiKey ? "***REDACTED***" : "" })),  // 导出时脱敏
        active: localStorage.getItem(ACTIVE_KEY),
        _note: "API Keys 出于安全考虑已脱敏，请在目标环境重新填写",
      }, null, 2);
    },

    exportAllWithKeys() {
      return JSON.stringify({
        exported_at: new Date().toISOString(),
        configs: this.list(),
        active: localStorage.getItem(ACTIVE_KEY),
      }, null, 2);
    },

    importAll(json) {
      const data = typeof json === "string" ? JSON.parse(json) : json;
      if (!data.configs || !Array.isArray(data.configs)) throw new Error("Invalid config format");
      this.save(data.configs.filter(c => c.apiKey !== "***REDACTED***"));
      if (data.active) localStorage.setItem(ACTIVE_KEY, data.active);
    },

    // 根据 base_url 自动识别服务商特殊参数（参考 ai_task_manager 的 _make_openai_compat）
    _autoExtra(endpoint) {
      const low = (endpoint || "").toLowerCase();
      if (low.includes("hunyuan")) return { enable_enhancement: true };
      return null;
    },

    // 解析有效的 API Key（支持混元的演示 Key 兜底）
    _resolveKey(cfg) {
      if (cfg.apiKey && cfg.apiKey.trim()) return cfg.apiKey.trim();
      // 未配置时，若有 demo_key 则用演示 Key
      const provider = PROVIDERS[cfg.provider];
      if (provider && provider.demo_key && cfg.useDemoKey !== false) {
        return provider.demo_key;
      }
      return "";
    },

    // 状态：是否自动检测到 unified server
    _unifiedServer: null,  // 'yes' | 'no' | null (未探测)

    // 探测当前是否在 AI_Algo_Lab 统一服务器上运行（同源代理可用）
    async probeUnifiedServer() {
      if (this._unifiedServer !== null) return this._unifiedServer === "yes";
      // 非 http(s) 协议（file://）直接判定为否
      if (!location.protocol.startsWith("http")) {
        this._unifiedServer = "no";
        return false;
      }
      try {
        const r = await fetch("/api/health", { method: "GET" });
        if (r.ok) {
          const data = await r.json();
          if (data.ok && data.proxy) {
            this._unifiedServer = "yes";
            console.info("[AIAL] 🎉 检测到统一服务器，已启用同源代理");
            return true;
          }
        }
      } catch {}
      this._unifiedServer = "no";
      return false;
    },

    // Provider 是否支持浏览器直连（无需代理）
    supportsDirectBrowser(provider) {
      // Ollama 本地默认支持 CORS（设置 OLLAMA_ORIGINS=* 后）
      // Gemini 使用 URL 参数 key，无 Authorization 头，CORS 友好
      // Anthropic 官方支持 anthropic-dangerous-direct-browser-access 头
      return ["ollama", "gemini", "anthropic"].includes(provider);
    },

    // 通过代理 URL 改写目标 endpoint
    // 优先级: 1) 同源 unified server /api/llm/ 2) 用户配置的 proxyUrl 3) 全局 localStorage 代理 4) 直连
    _applyProxy(endpoint, cfg) {
      // 1) 同源 unified server：自动使用 /api/llm/<encoded>
      if (this._unifiedServer === "yes") {
        return location.origin + "/api/llm/" + encodeURIComponent(endpoint);
      }
      // 2) 用户配置的代理
      const proxy = (cfg.proxyUrl || localStorage.getItem("aial-cors-proxy") || "").trim();
      if (proxy) {
        const p = proxy.replace(/\/$/, "");
        // 支持两种代理风格：?url=<target> 或路径拼接
        if (p.includes("?url=")) return p + encodeURIComponent(endpoint);
        return p + "/" + endpoint;
      }
      // 3) 直连（仅对支持 CORS 的 provider 有效）
      return endpoint;
    },

    // 核心调用方法（支持 OpenAI/Anthropic/Gemini 三种格式）
    async chat(cfg, messages, { temperature = 0.2, max_tokens = 4096, stream = false, onChunk } = {}) {
      if (!cfg) throw new Error("未选择 LLM 配置");
      // 首次调用时探测统一服务器（同源代理）
      await this.probeUnifiedServer();
      const provider = PROVIDERS[cfg.provider] || PROVIDERS.custom;
      const endpoint = this._applyProxy(cfg.endpoint || provider.endpoint, cfg);
      const model = cfg.model || provider.models[0];
      const apiKey = this._resolveKey(cfg);

      // 合并 extra_body：provider 预设 + 自动识别 + 用户自定义
      const extraBody = {
        ...(provider.extra_body || {}),
        ...(this._autoExtra(cfg.endpoint || provider.endpoint) || {}),
        ...(cfg.extra_body || {}),
      };

      try {
        if (provider.format === "openai") {
          return await this._callOpenAI(endpoint, model, apiKey, messages, { temperature, max_tokens, stream, onChunk, extraBody });
        } else if (provider.format === "anthropic") {
          return await this._callAnthropic(endpoint, model, apiKey, messages, { temperature, max_tokens });
        } else if (provider.format === "gemini") {
          return await this._callGemini(endpoint, model, apiKey, messages, { temperature });
        } else if (provider.format === "azure") {
          return await this._callAzure(endpoint, apiKey, messages, { temperature, max_tokens, stream, onChunk });
        }
      } catch (e) {
        // 识别 CORS / 网络错误并给出更详细的提示
        throw this._wrapError(e, cfg);
      }
      throw new Error("Unsupported provider format: " + provider.format);
    },

    // 把模糊的 "Failed to fetch" 转成带诊断的可操作错误
    _wrapError(err, cfg) {
      const msg = String(err.message || err);
      const isFetchError = /Failed to fetch|NetworkError|ERR_NETWORK/i.test(msg);
      if (!isFetchError) return err;
      const proxyInfo = (cfg.proxyUrl || localStorage.getItem("aial-cors-proxy") || "").trim();
      const supportsDirect = this.supportsDirectBrowser(cfg.provider);
      const isFile = location.protocol === "file:";

      const lines = [
        "🚫 浏览器 CORS 跨域限制 — 请求被浏览器拦截",
        "",
        "4 种解决方案（按推荐度排序）：",
        "",
        "⭐ 方案 1：一键启动（强烈推荐，零配置）",
        "   用 Python 启动统一服务器（HTML + LLM 代理同源）：",
        "     cd <项目根目录>\\AI_Algo_Lab",
        "     python start.py",
        "   或双击 AI_Algo_Lab\\启动平台.bat",
        "   → 服务器会自动打开浏览器，CORS 问题彻底消失",
        "",
        "⚡ 方案 2：切换到支持浏览器直连的 Provider（无需任何服务器）",
        "   以下 Provider 原生支持浏览器调用：",
        "   • Ollama 本地（http://localhost:11434/v1，需设 OLLAMA_ORIGINS=*）",
        "   • Google Gemini（用 URL key 认证）",
        "   • Anthropic Claude（已自动加 anthropic-dangerous-direct-browser-access 头）",
        "",
        "🛡 方案 3：独立 CORS 代理（已有方案）",
        "   python AI_Algo_Lab/data/proxy/cors_proxy.py",
        "   配置面板 → 代理 URL 填 http://localhost:8787",
        "",
        "🔧 方案 4：Chrome 禁 CORS 启动（仅调试用）",
        "   双击 AI_Algo_Lab/data/launch_chrome_no_cors.bat",
        "",
      ];
      if (isFile) lines.push("⚠ 检测到 file:// 协议，某些 API 会更严格拒绝。建议用方案 1 的 http:// 打开。", "");
      if (supportsDirect) lines.push("💡 当前 Provider 本应支持直连，失败可能是 API Key 无效或网络问题。", "");
      if (proxyInfo) lines.push(`当前已配置代理：${proxyInfo}，但仍失败 → 请检查代理进程是否存活。`);
      return new Error(lines.join("\n"));
    },

    async _callOpenAI(endpoint, model, apiKey, messages, opts) {
      const headers = { "Content-Type": "application/json" };
      if (apiKey) headers.Authorization = "Bearer " + apiKey;
      // 合并 extra_body 到 payload 顶层（与 OpenAI SDK extra_body 行为一致）
      const body = {
        model, messages,
        temperature: opts.temperature,
        max_tokens: opts.max_tokens,
        stream: opts.stream,
        ...(opts.extraBody || {}),
      };
      const res = await fetch(endpoint.replace(/\/$/, "") + "/chat/completions", {
        method: "POST", headers, body: JSON.stringify(body),
      });
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`${res.status}: ${text.slice(0, 500)}`);
      }
      if (opts.stream && opts.onChunk) {
        return this._consumeSSE(res, opts.onChunk);
      }
      const data = await res.json();
      return data.choices?.[0]?.message?.content || "";
    },

    async _callAnthropic(endpoint, model, apiKey, messages, opts) {
      const systemMsg = messages.find(m => m.role === "system");
      const chatMsgs = messages.filter(m => m.role !== "system");
      const body = {
        model, max_tokens: opts.max_tokens, temperature: opts.temperature,
        messages: chatMsgs,
        ...(systemMsg ? { system: systemMsg.content } : {}),
      };
      const res = await fetch(endpoint.replace(/\/$/, "") + "/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json", "x-api-key": apiKey,
          "anthropic-version": "2023-06-01",
          "anthropic-dangerous-direct-browser-access": "true",
        },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 500)}`);
      const data = await res.json();
      return data.content?.[0]?.text || "";
    },

    async _callGemini(endpoint, model, apiKey, messages, opts) {
      const url = `${endpoint.replace(/\/$/, "")}/models/${model}:generateContent?key=${apiKey}`;
      const systemMsg = messages.find(m => m.role === "system");
      const contents = messages.filter(m => m.role !== "system").map(m => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
      }));
      const body = {
        contents,
        ...(systemMsg ? { systemInstruction: { parts: [{ text: systemMsg.content }] } } : {}),
        generationConfig: { temperature: opts.temperature },
      };
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 500)}`);
      const data = await res.json();
      return data.candidates?.[0]?.content?.parts?.[0]?.text || "";
    },

    async _callAzure(endpoint, apiKey, messages, opts) {
      // Azure endpoint 已含 deployment
      const url = endpoint.replace(/\/$/, "") + "/chat/completions?api-version=2024-02-15-preview";
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json", "api-key": apiKey },
        body: JSON.stringify({ messages, temperature: opts.temperature, max_tokens: opts.max_tokens }),
      });
      if (!res.ok) throw new Error(`${res.status}: ${(await res.text()).slice(0, 500)}`);
      const data = await res.json();
      return data.choices?.[0]?.message?.content || "";
    },

    async _consumeSSE(res, onChunk) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let full = "", buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") return full;
          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta?.content || "";
            if (delta) { full += delta; onChunk(delta, full); }
          } catch {}
        }
      }
      return full;
    },

    async testConnection(cfg) {
      try {
        const resp = await this.chat(cfg, [
          { role: "user", content: "Reply OK in 2 chars." },
        ], { temperature: 0, max_tokens: 10 });
        return { ok: true, response: resp };
      } catch (e) {
        return { ok: false, error: e.message };
      }
    },
  };

  window.AIAL_LLM = LLM;
})();
