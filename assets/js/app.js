/* ============================================================================
 * app.js — 路由 & 启动
 * ========================================================================== */

/* ─── Theme System（主题切换）────────────────────────────── */
const THEME_KEY = "mch-theme";
const THEME_MAP = {
  day:   { icon: "☀️", label: "日间", labelKey: "theme.day", mermaid: "neutral" },
  eye:   { icon: "🌿", label: "护眼", labelKey: "theme.eye", mermaid: "neutral" },
  night: { icon: "🌙", label: "夜视", labelKey: "theme.night", mermaid: "dark"    },
};

function setTheme(theme) {
  if (!THEME_MAP[theme]) theme = "day";
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem(THEME_KEY, theme);

  // 更新按钮图标和文字（i18n 感知）
  const iconEl = document.getElementById("theme-icon");
  const nameEl = document.getElementById("theme-name");
  if (iconEl) iconEl.textContent = THEME_MAP[theme].icon;
  if (nameEl) {
    const labelKey = THEME_MAP[theme].labelKey;
    nameEl.setAttribute("data-i18n", labelKey);
    nameEl.textContent = (window.MCH && MCH.t) ? MCH.t(labelKey) : THEME_MAP[theme].label;
  }

  // 更新下拉菜单激活态
  document.querySelectorAll(".theme-opt").forEach(btn => {
    const isActive = btn.dataset.theme === theme;
    btn.classList.toggle("bg-indigo-50", isActive);
    btn.classList.toggle("text-indigo-700", isActive);
    btn.classList.toggle("font-bold", isActive);
  });

  // 更新 header 内联渐变（始终深色）
  const header = document.querySelector("header");
  if (header) {
    header.style.background = "linear-gradient(to right, #0f172a, #1e293b, #0f172a)";
    header.style.color = "#ffffff";
  }

  // 更新 mermaid 主题
  if (window.mermaid) {
    mermaid.initialize({ theme: THEME_MAP[theme].mermaid });
    // 重新渲染页面中的 mermaid 图表
    document.querySelectorAll(".mermaid").forEach(el => {
      const id = el.getAttribute("data-processed");
      if (id) {
        const src = el.getAttribute("_src") || el.textContent;
        el.removeAttribute("data-processed");
        el.textContent = src;
        mermaid.run({ nodes: [el] });
      }
    });
  }

  // 更新 ECharts 图表背景（深色主题下图表容器背景跟随）
  const chartBg = theme === "night" ? "#0f172a" : "#ffffff";
  MCH.charts.forEach(c => {
    try {
      c.setOption({ backgroundColor: chartBg });
      c.resize();
    } catch (e) {}
  });
}

function initTheme() {
  // 读取上次保存的主题（默认 day）
  const saved = localStorage.getItem(THEME_KEY) || "day";
  setTheme(saved);

  // 绑定切换按钮点击 → 显示/隐藏下拉
  const themeBtn = document.getElementById("theme-btn");
  const dropdown = document.getElementById("theme-dropdown");
  if (themeBtn && dropdown) {
    themeBtn.addEventListener("click", e => {
      e.stopPropagation();
      dropdown.classList.toggle("hidden");
    });
    // 点击页面其他地方关闭
    document.addEventListener("click", () => dropdown.classList.add("hidden"));
  }

  // 绑定每个主题选项
  document.querySelectorAll(".theme-opt").forEach(btn => {
    btn.addEventListener("click", () => {
      setTheme(btn.dataset.theme);
      dropdown.classList.add("hidden");
    });
  });
}

/* ─── UI Preferences（字号 / 界面偏好）───────────────────── */
const UI_PREFS_KEY = "mch-ui-prefs";
const UI_PREFS_DEFAULT = {
  appFontSize: 14,         // 全局基础字号 (12-18)
  sidebarFontSize: 14,     // 侧栏 nav-item 字号 (11-18)
  sidebarCatFontSize: 12,  // 侧栏一级分类字号 (10-16)
  sidebarSubcatFontSize: 11.5,  // 侧栏二级分类字号
};

function loadUiPrefs() {
  try {
    const raw = localStorage.getItem(UI_PREFS_KEY);
    if (!raw) return { ...UI_PREFS_DEFAULT };
    return { ...UI_PREFS_DEFAULT, ...JSON.parse(raw) };
  } catch (e) { return { ...UI_PREFS_DEFAULT }; }
}

function saveUiPrefs(prefs) {
  localStorage.setItem(UI_PREFS_KEY, JSON.stringify(prefs));
}

function applyUiPrefs(prefs) {
  const root = document.documentElement;
  root.style.setProperty("--app-font-size", prefs.appFontSize + "px");
  root.style.setProperty("--sidebar-font-size", prefs.sidebarFontSize + "px");
  root.style.setProperty("--sidebar-cat-font-size", prefs.sidebarCatFontSize + "px");
  root.style.setProperty("--sidebar-subcat-font-size", prefs.sidebarSubcatFontSize + "px");
}

function initUiPrefs() {
  applyUiPrefs(loadUiPrefs());
}

/* ─── Settings Panel（设置面板弹窗）────────────────────── */
function openSettingsPanel() {
  let panel = document.getElementById("settings-panel");
  if (panel) { panel.classList.remove("hidden"); return; }

  const prefs = loadUiPrefs();
  const currentTheme = localStorage.getItem(THEME_KEY) || "day";
  const t = (k) => (window.MCH && MCH.t) ? MCH.t(k) : k;
  const currentLang = (window.MCH && MCH.i18n) ? MCH.i18n.lang : "zh";

  panel = document.createElement("div");
  panel.id = "settings-panel";
  panel.className = "fixed inset-0 z-[100] flex items-center justify-center";
  panel.innerHTML = `
    <div class="settings-backdrop absolute inset-0 bg-black/50"></div>
    <div class="settings-dialog relative w-[560px] max-w-[92vw] max-h-[88vh] overflow-y-auto rounded-2xl shadow-2xl"
         style="background:var(--section-bg);border:1px solid var(--section-border);color:var(--text-primary)">
      <div class="px-6 py-4 border-b flex items-center justify-between" style="border-color:var(--section-border)">
        <div class="flex items-center gap-2">
          <span class="text-xl">⚙️</span>
          <h2 class="text-lg font-bold" data-i18n="settings.title">${t("settings.title")}</h2>
        </div>
        <button class="settings-close text-2xl leading-none opacity-60 hover:opacity-100" style="color:var(--text-primary)">×</button>
      </div>
      <div class="p-6 space-y-6">
        <!-- 语言 / Language -->
        <section>
          <h3 class="text-sm font-bold mb-3" style="color:var(--text-primary)" data-i18n="settings.lang_title">${t("settings.lang_title")}</h3>
          <div class="grid grid-cols-2 gap-3">
            <button class="settings-lang-opt rounded-lg p-3 border-2 transition-all"
                    data-lang="zh"
                    style="${currentLang==='zh'?'border-color:var(--accent);background:var(--bg-secondary)':'border-color:var(--card-border);background:var(--card-bg)'};color:var(--text-primary)">
              <div class="text-2xl mb-1">🇨🇳</div>
              <div class="text-sm font-semibold" data-i18n="settings.lang_zh">${t("settings.lang_zh")}</div>
            </button>
            <button class="settings-lang-opt rounded-lg p-3 border-2 transition-all"
                    data-lang="en"
                    style="${currentLang==='en'?'border-color:var(--accent);background:var(--bg-secondary)':'border-color:var(--card-border);background:var(--card-bg)'};color:var(--text-primary)">
              <div class="text-2xl mb-1">🇺🇸</div>
              <div class="text-sm font-semibold" data-i18n="settings.lang_en">${t("settings.lang_en")}</div>
            </button>
          </div>
        </section>

        <!-- 主题颜色 -->
        <section>
          <h3 class="text-sm font-bold mb-3" style="color:var(--text-primary)" data-i18n="settings.theme_title">${t("settings.theme_title")}</h3>
          <div class="grid grid-cols-3 gap-3">
            ${["day","eye","night"].map(th => `
              <button class="settings-theme-opt rounded-lg p-3 border-2 transition-all ${currentTheme===th?'ring-2':''}"
                      data-theme="${th}"
                      style="${currentTheme===th?'border-color:var(--accent);background:var(--bg-secondary)':'border-color:var(--card-border);background:var(--card-bg)'}">
                <div class="text-2xl mb-1">${THEME_MAP[th].icon}</div>
                <div class="text-sm font-semibold" style="color:var(--text-primary)" data-i18n="theme.${th}">${t("theme."+th)}</div>
                <div class="text-[10px] mt-0.5" style="color:var(--text-muted)" data-i18n="theme.${th}_desc">${t("theme."+th+"_desc")}</div>
              </button>`).join("")}
          </div>
        </section>

        <!-- 字体大小 -->
        <section>
          <h3 class="text-sm font-bold mb-3" style="color:var(--text-primary)" data-i18n="settings.font_title">${t("settings.font_title")}</h3>
          <div class="space-y-4">
            <div>
              <div class="flex justify-between mb-1.5">
                <label class="text-xs font-semibold" style="color:var(--text-secondary)" data-i18n="settings.font_sidebar">${t("settings.font_sidebar")}</label>
                <span class="text-xs font-mono" style="color:var(--accent)"><span id="pref-sb-size-val">${prefs.sidebarFontSize}</span>px</span>
              </div>
              <input type="range" id="pref-sidebar-size" min="11" max="18" step="0.5" value="${prefs.sidebarFontSize}" class="w-full">
              <div class="flex justify-between text-[10px] mt-0.5" style="color:var(--text-muted)">
                <span data-i18n="settings.font_small">${t("settings.font_small")}</span>
                <span data-i18n="settings.font_medium">${t("settings.font_medium")}</span>
                <span data-i18n="settings.font_large">${t("settings.font_large")}</span>
              </div>
            </div>

            <div>
              <div class="flex justify-between mb-1.5">
                <label class="text-xs font-semibold" style="color:var(--text-secondary)" data-i18n="settings.font_cat">${t("settings.font_cat")}</label>
                <span class="text-xs font-mono" style="color:var(--accent)"><span id="pref-cat-size-val">${prefs.sidebarCatFontSize}</span>px</span>
              </div>
              <input type="range" id="pref-cat-size" min="10" max="16" step="0.5" value="${prefs.sidebarCatFontSize}" class="w-full">
            </div>

            <div>
              <div class="flex justify-between mb-1.5">
                <label class="text-xs font-semibold" style="color:var(--text-secondary)" data-i18n="settings.font_app">${t("settings.font_app")}</label>
                <span class="text-xs font-mono" style="color:var(--accent)"><span id="pref-app-size-val">${prefs.appFontSize}</span>px</span>
              </div>
              <input type="range" id="pref-app-size" min="12" max="18" step="0.5" value="${prefs.appFontSize}" class="w-full">
            </div>
          </div>
        </section>

        <!-- 预设字号 -->
        <section>
          <h3 class="text-sm font-bold mb-3" style="color:var(--text-primary)" data-i18n="settings.preset_title">${t("settings.preset_title")}</h3>
          <div class="grid grid-cols-3 gap-2">
            <button class="settings-preset rounded-md py-2 text-xs font-semibold border transition-all"
                    style="border-color:var(--card-border);background:var(--card-bg);color:var(--text-primary)"
                    data-preset="compact" data-i18n="settings.preset_compact">${t("settings.preset_compact")}</button>
            <button class="settings-preset rounded-md py-2 text-xs font-semibold border transition-all"
                    style="border-color:var(--card-border);background:var(--card-bg);color:var(--text-primary)"
                    data-preset="default" data-i18n="settings.preset_default">${t("settings.preset_default")}</button>
            <button class="settings-preset rounded-md py-2 text-xs font-semibold border transition-all"
                    style="border-color:var(--card-border);background:var(--card-bg);color:var(--text-primary)"
                    data-preset="large" data-i18n="settings.preset_large">${t("settings.preset_large")}</button>
          </div>
        </section>

        <!-- 底部按钮 -->
        <div class="flex justify-between pt-2 border-t" style="border-color:var(--section-border)">
          <button class="settings-reset px-4 py-1.5 text-xs font-semibold rounded-md border transition-colors"
                  style="border-color:var(--card-border);background:var(--card-bg);color:var(--text-secondary)" data-i18n="settings.reset">${t("settings.reset")}</button>
          <div class="flex gap-2">
            <button class="settings-close px-4 py-1.5 text-xs font-semibold rounded-md border"
                    style="border-color:var(--card-border);background:var(--card-bg);color:var(--text-primary)" data-i18n="settings.close">${t("settings.close")}</button>
          </div>
        </div>
      </div>
    </div>
  `;
  document.body.appendChild(panel);

  // 语言选项
  panel.querySelectorAll(".settings-lang-opt").forEach(btn => {
    btn.addEventListener("click", () => {
      if (!window.MCH || !MCH.i18n) return;
      MCH.i18n.setLang(btn.dataset.lang);
      // 刷新激活态
      panel.querySelectorAll(".settings-lang-opt").forEach(b => {
        const active = b.dataset.lang === btn.dataset.lang;
        b.style.borderColor = active ? "var(--accent)" : "var(--card-border)";
        b.style.background = active ? "var(--bg-secondary)" : "var(--card-bg)";
      });
    });
  });

  // 关闭按钮
  panel.querySelectorAll(".settings-close, .settings-backdrop").forEach(el => {
    el.addEventListener("click", () => panel.classList.add("hidden"));
  });

  // 主题选项
  panel.querySelectorAll(".settings-theme-opt").forEach(btn => {
    btn.addEventListener("click", () => {
      setTheme(btn.dataset.theme);
      // 刷新激活态
      panel.querySelectorAll(".settings-theme-opt").forEach(b => {
        const active = b.dataset.theme === btn.dataset.theme;
        b.style.borderColor = active ? "var(--accent)" : "var(--card-border)";
        b.style.background = active ? "var(--bg-secondary)" : "var(--card-bg)";
      });
    });
  });

  // 字号 slider 实时更新
  const bindSlider = (id, valId, key) => {
    const slider = document.getElementById(id);
    const valEl = document.getElementById(valId);
    if (!slider) return;
    slider.addEventListener("input", () => {
      const v = parseFloat(slider.value);
      valEl.textContent = v;
      const p = loadUiPrefs();
      p[key] = v;
      saveUiPrefs(p);
      applyUiPrefs(p);
    });
  };
  bindSlider("pref-sidebar-size", "pref-sb-size-val", "sidebarFontSize");
  bindSlider("pref-cat-size", "pref-cat-size-val", "sidebarCatFontSize");
  bindSlider("pref-app-size", "pref-app-size-val", "appFontSize");

  // 预设
  const PRESETS = {
    compact: { appFontSize: 13, sidebarFontSize: 12, sidebarCatFontSize: 11, sidebarSubcatFontSize: 10.5 },
    default: { ...UI_PREFS_DEFAULT },
    large:   { appFontSize: 16, sidebarFontSize: 16, sidebarCatFontSize: 14, sidebarSubcatFontSize: 13 },
  };
  panel.querySelectorAll(".settings-preset").forEach(btn => {
    btn.addEventListener("click", () => {
      const p = PRESETS[btn.dataset.preset];
      saveUiPrefs(p);
      applyUiPrefs(p);
      // 刷新 slider 显示
      document.getElementById("pref-sidebar-size").value = p.sidebarFontSize;
      document.getElementById("pref-sb-size-val").textContent = p.sidebarFontSize;
      document.getElementById("pref-cat-size").value = p.sidebarCatFontSize;
      document.getElementById("pref-cat-size-val").textContent = p.sidebarCatFontSize;
      document.getElementById("pref-app-size").value = p.appFontSize;
      document.getElementById("pref-app-size-val").textContent = p.appFontSize;
    });
  });

  // 重置
  panel.querySelector(".settings-reset").addEventListener("click", () => {
    const p = { ...UI_PREFS_DEFAULT };
    saveUiPrefs(p);
    applyUiPrefs(p);
    document.getElementById("pref-sidebar-size").value = p.sidebarFontSize;
    document.getElementById("pref-sb-size-val").textContent = p.sidebarFontSize;
    document.getElementById("pref-cat-size").value = p.sidebarCatFontSize;
    document.getElementById("pref-cat-size-val").textContent = p.sidebarCatFontSize;
    document.getElementById("pref-app-size").value = p.appFontSize;
    document.getElementById("pref-app-size-val").textContent = p.appFontSize;
  });
}

/* ─── Mermaid ─────────────────────────────────────────── */
mermaid.initialize({
  startOnLoad: false,
  theme: "neutral",
  securityLevel: "loose",
  flowchart: { useMaxWidth: true, htmlLabels: true, curve: "basis" },
});

function render(route) {
  MCH.disposeCharts();
  const content = document.getElementById("app-content");
  const mod = MCH.modules[route] || MCH.modules["overview"];
  try {
    const html = mod.render();
    content.innerHTML = html;
    // i18n: replace any data-i18n* placeholders embedded in module output
    if (window.MCH && MCH.i18n && MCH.i18n.applyDOM) {
      MCH.i18n.applyDOM(content);
    }
    MCH.rerender(content);
    if (typeof mod.mount === "function") {
      setTimeout(() => mod.mount(content), 30);
    }
  } catch (e) {
    console.error(e);
    const msg = (window.MCH && MCH.t) ? MCH.t("common.render_error") : "模块渲染出错：";
    content.innerHTML = `<div class="section text-red-600">${msg}${e.message}</div>`;
  }
  // 更新导航激活状态
  document.querySelectorAll(".nav-item").forEach((el) => {
    el.classList.toggle("active", el.dataset.route === route);
  });
  window.scrollTo(0, 0);
}

// 暴露给 i18n 在切换语言后重渲染当前路由
window.MCH_render = render;

function onHashChange() {
  const h = window.location.hash.replace(/^#\//, "") || "overview";
  render(h);
  // 路由变化时自动展开当前类别
  autoExpandActive();
}

window.addEventListener("hashchange", onHashChange);
window.addEventListener("DOMContentLoaded", () => {
  if (!window.location.hash) window.location.hash = "#/overview";
  else onHashChange();
  // 主题初始化（必须最先执行，确保页面加载时就是正确主题）
  initTheme();
  // 界面偏好（字号等）初始化
  initUiPrefs();
  // 🔥 从服务端恢复 LLM 配置和历史记录（unified server 可用时）
  (async () => {
    if (window.AIAL_LLM && AIAL_LLM.loadFromServer) {
      await AIAL_LLM.loadFromServer();
    }
    if (window.AIAL_ASSISTANT && AIAL_ASSISTANT.loadHistoryFromServer) {
      await AIAL_ASSISTANT.loadHistoryFromServer();
    }
  })();
  // 绑定头部搜索按钮到全局搜索
  const btn = document.getElementById("header-search-btn");
  if (btn) btn.addEventListener("click", () => { if (window.MCH && MCH.openSearch) MCH.openSearch(); });
  // 绑定 AI 助手按钮
  const asstBtn = document.getElementById("header-assistant-btn");
  if (asstBtn) asstBtn.addEventListener("click", () => { if (window.AIAL_ASSISTANT) AIAL_ASSISTANT.openAssistant(); });
  // 界面设置按钮（⚙️） → 打开设置面板
  const settingsBtn = document.getElementById("header-settings-btn");
  if (settingsBtn) settingsBtn.addEventListener("click", openSettingsPanel);
  // LLM 配置按钮 → 保留单独入口
  const cfgBtn = document.getElementById("header-config-btn");
  if (cfgBtn) cfgBtn.addEventListener("click", () => { if (window.AIAL_ASSISTANT) AIAL_ASSISTANT.openConfig(); });
  // 语言切换按钮（中 ↔ EN）
  const langBtn = document.getElementById("header-lang-btn");
  if (langBtn) langBtn.addEventListener("click", () => {
    if (!window.MCH || !MCH.i18n) return;
    const next = MCH.i18n.lang === "zh" ? "en" : "zh";
    MCH.i18n.setLang(next);
  });
  // 侧栏折叠逻辑（持久化到 localStorage）
  initSidebarCollapse();
  // 全部折叠/展开按钮
  bindSidebarToggleAll();
  // 移动端汉堡菜单
  initMobileMenu();
});

/* ─── 移动端汉堡菜单抽屉 ─── */
function initMobileMenu() {
  const btn = document.getElementById("mobile-menu-btn");
  const sidebar = document.getElementById("app-sidebar");
  const overlay = document.getElementById("mobile-overlay");
  if (!btn || !sidebar || !overlay) return;

  const open = () => {
    sidebar.classList.add("open");
    overlay.classList.add("open");
    document.body.style.overflow = "hidden";
  };
  const close = () => {
    sidebar.classList.remove("open");
    overlay.classList.remove("open");
    document.body.style.overflow = "";
  };
  const toggle = () => {
    sidebar.classList.contains("open") ? close() : open();
  };

  btn.addEventListener("click", e => { e.stopPropagation(); toggle(); });
  overlay.addEventListener("click", close);

  // 路由变化时自动关闭侧栏（手机上）
  window.addEventListener("hashchange", () => {
    if (window.innerWidth < 768) close();
  });

  // 点击侧栏里的 nav-item 后自动关闭
  sidebar.addEventListener("click", e => {
    const navItem = e.target.closest(".nav-item");
    if (navItem && window.innerWidth < 768) {
      setTimeout(close, 150);  // 等一下让路由先生效
    }
  });

  // ESC 关闭
  document.addEventListener("keydown", e => {
    if (e.key === "Escape" && sidebar.classList.contains("open")) close();
  });
}

/* ============== 侧栏折叠核心 ============== */
const COLLAPSE_KEY = "mch-sidebar-collapsed";        // 大类 collapsed key 集合
const SUBCAT_COLLAPSE_KEY = "mch-subcat-collapsed";  // 子类折叠 key 集合

function getCollapsedSet(key) {
  try { return new Set(JSON.parse(localStorage.getItem(key) || "[]")); }
  catch (e) { return new Set(); }
}
function saveCollapsedSet(key, set) {
  localStorage.setItem(key, JSON.stringify([...set]));
}

function initSidebarCollapse() {
  const collapsed = getCollapsedSet(COLLAPSE_KEY);
  const subCollapsed = getCollapsedSet(SUBCAT_COLLAPSE_KEY);
  const activeCat = getActiveCategory();
  const activeSubcat = getActiveSubcat();

  // —— 一级大类折叠 ——
  document.querySelectorAll(".cat-section").forEach(sec => {
    const key = sec.dataset.cat;
    const shouldCollapse = collapsed.has(key);
    if (shouldCollapse && key !== activeCat) sec.classList.add("collapsed");
    const header = sec.querySelector(":scope > .cat-header");
    if (header && !header.dataset.bound) {
      header.dataset.bound = "1";
      header.addEventListener("click", (e) => {
        e.stopPropagation();
        sec.classList.toggle("collapsed");
        const s = getCollapsedSet(COLLAPSE_KEY);
        if (sec.classList.contains("collapsed")) s.add(key); else s.delete(key);
        saveCollapsedSet(COLLAPSE_KEY, s);
      });
    }
  });

  // —— 二级子类折叠 ——
  document.querySelectorAll(".subcat-section").forEach(sec => {
    const key = sec.dataset.subcat;
    // 二级子类默认折叠（除非当前路由落在其中）
    const firstVisit = !localStorage.getItem(SUBCAT_COLLAPSE_KEY);
    const shouldCollapse = firstVisit ? true : subCollapsed.has(key);
    if (shouldCollapse && key !== activeSubcat) sec.classList.add("collapsed");
    const header = sec.querySelector(".subcat-header");
    if (header && !header.dataset.bound) {
      header.dataset.bound = "1";
      header.addEventListener("click", (e) => {
        e.stopPropagation();
        sec.classList.toggle("collapsed");
        const s = getCollapsedSet(SUBCAT_COLLAPSE_KEY);
        if (sec.classList.contains("collapsed")) s.add(key); else s.delete(key);
        saveCollapsedSet(SUBCAT_COLLAPSE_KEY, s);
      });
    }
  });
  // 首次访问的标记：防止"默认全折叠"在刷新后丢失
  if (!localStorage.getItem(SUBCAT_COLLAPSE_KEY)) {
    saveCollapsedSet(SUBCAT_COLLAPSE_KEY, new Set([...document.querySelectorAll(".subcat-section")].map(s => s.dataset.subcat).filter(Boolean)));
  }
}

function bindSidebarToggleAll() {
  const collapseBtn = document.getElementById("sidebar-collapse-all");
  const expandBtn = document.getElementById("sidebar-expand-all");
  if (collapseBtn) collapseBtn.addEventListener("click", () => {
    // 折叠所有大类（含子类）
    const cats = new Set();
    document.querySelectorAll(".cat-section").forEach(sec => {
      sec.classList.add("collapsed");
      if (sec.dataset.cat) cats.add(sec.dataset.cat);
    });
    const subs = new Set();
    document.querySelectorAll(".subcat-section").forEach(sec => {
      sec.classList.add("collapsed");
      if (sec.dataset.subcat) subs.add(sec.dataset.subcat);
    });
    saveCollapsedSet(COLLAPSE_KEY, cats);
    saveCollapsedSet(SUBCAT_COLLAPSE_KEY, subs);
  });
  if (expandBtn) expandBtn.addEventListener("click", () => {
    document.querySelectorAll(".cat-section").forEach(sec => sec.classList.remove("collapsed"));
    document.querySelectorAll(".subcat-section").forEach(sec => sec.classList.remove("collapsed"));
    saveCollapsedSet(COLLAPSE_KEY, new Set());
    saveCollapsedSet(SUBCAT_COLLAPSE_KEY, new Set());
  });
}

// 路由切换后自动展开对应类别（并不修改其他类别的折叠状态）
function autoExpandActive() {
  const activeCat = getActiveCategory();
  const activeSubcat = getActiveSubcat();
  if (activeCat) {
    const sec = document.querySelector(`.cat-section[data-cat="${activeCat}"]`);
    if (sec && sec.classList.contains("collapsed")) {
      sec.classList.remove("collapsed");
      const s = getCollapsedSet(COLLAPSE_KEY); s.delete(activeCat); saveCollapsedSet(COLLAPSE_KEY, s);
    }
  }
  if (activeSubcat) {
    const sec = document.querySelector(`.subcat-section[data-subcat="${activeSubcat}"]`);
    if (sec && sec.classList.contains("collapsed")) {
      sec.classList.remove("collapsed");
      const s = getCollapsedSet(SUBCAT_COLLAPSE_KEY); s.delete(activeSubcat); saveCollapsedSet(SUBCAT_COLLAPSE_KEY, s);
    }
  }
}

function getActiveCategory() {
  const route = window.location.hash.replace(/^#\//, "") || "overview";
  // 简易映射：route 前缀 → cat key
  if (route.startsWith("ml_decision") || route.startsWith("ml_random") || route.startsWith("ml_xgboost") || route.startsWith("ml_lightgbm") || route.startsWith("ml_logistic") || route.startsWith("ml_svm") || route.startsWith("ml_naive") || route === "ensemble_methods") return "ml_sup";
  if (route.startsWith("ml_kmeans") || route.startsWith("ml_dbscan")) return "ml_unsup";
  if (route.startsWith("ml_pca") || route.startsWith("ml_tsne") || route.startsWith("ml_isolation") || route === "ad_deep" || route === "ad_llm") return "ml_dr";
  if (route.startsWith("ts_")) return "time_series";
  if (route.startsWith("nn_") || route === "seq_encoder" || route === "image_encoder" || route === "fusion" || route === "mmoe" || route === "tab_dl") return "nn";
  if (route.startsWith("graph_")) return "graph";
  if (route.startsWith("loss") || route === "losses") return "loss";
  if (route.startsWith("llm_") || route === "text_encoder") return "llm";
  if (route.startsWith("frontier_")) return "frontier";
  if (route === "case_tts" || route === "case_video_gen") return "case";
  if (route.startsWith("coding_")) return "coding";
  if (route === "learning_paths" || route === "learning_stories" || route === "tutor") return "learning";
  // 工业案例（含商户识别子模块：它们现已并入 case 分类下）
  if (route.startsWith("case_") || route === "case_overview") return "case";
  if (["aux_encoder", "heads", "samplers", "training", "distillation"].includes(route)) return "case";
  return null;
}

function getActiveSubcat() {
  const route = window.location.hash.replace(/^#\//, "") || "overview";
  if (["aux_encoder", "heads", "samplers", "training", "distillation"].includes(route)) return "case_merchant_detail";
  return null;
}
