/* ============================================================================
 * common.js — 通用工具库（数学渲染 / 代码高亮 / 图表创建 / HTML 辅助）
 * ========================================================================== */
window.MCH = window.MCH || { modules: {}, charts: [] };

// 模块注册
MCH.register = function (name, mod) { MCH.modules[name] = mod; };

// Tagged template：把 HTML 模板字符串去除前导缩进
MCH.html = function (strings, ...values) {
  const raw = String.raw({ raw: strings }, ...values);
  return raw;
};

// 渲染某个 DOM 里的 LaTeX / 代码块
MCH.rerender = function (root) {
  if (!root) root = document;
  // 数学公式
  if (window.renderMathInElement) {
    renderMathInElement(root, {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
        { left: "\\(", right: "\\)", display: false },
        { left: "\\[", right: "\\]", display: true },
      ],
      throwOnError: false,
    });
  }
  // 代码高亮
  if (window.hljs) {
    root.querySelectorAll("pre code").forEach((el) => {
      if (!el.dataset.highlighted) {
        hljs.highlightElement(el);
        el.dataset.highlighted = "1";
      }
    });
  }
  // mermaid
  if (window.mermaid) {
    try { mermaid.run({ nodes: root.querySelectorAll(".mermaid") }); } catch (e) {}
  }
};

// Dispose 已创建的 echarts 实例，防止路由切换内存泄漏
MCH.disposeCharts = function () {
  if (!MCH.charts) return;
  MCH.charts.forEach((c) => { try { c.dispose(); } catch (e) {} });
  MCH.charts = [];
};

MCH.echart = function (el, option) {
  const c = echarts.init(el, null, { renderer: "canvas" });
  c.setOption(option);
  MCH.charts.push(c);
  window.addEventListener("resize", () => c.resize());
  return c;
};

// 预定义配色
MCH.palette = {
  indigo: "#4f46e5",
  purple: "#7c3aed",
  violet: "#8b5cf6",
  emerald: "#10b981",
  amber: "#f59e0b",
  rose: "#f43f5e",
  sky: "#0ea5e9",
  slate: "#64748b",
  red: "#ef4444",
  orange: "#f97316",
};
MCH.colorSeq = [
  "#4f46e5", "#7c3aed", "#0ea5e9", "#10b981", "#f59e0b",
  "#f43f5e", "#06b6d4", "#8b5cf6", "#ec4899", "#14b8a6",
];

// 通用 hero
MCH.hero = function ({ icon, name, en, tags, meta }) {
  const tagHtml = (tags || []).map((t) => `<span class="tag tag-slate">${t}</span>`).join(" ");
  const metaHtml = (meta || []).map((m) => `<span>${m}</span>`).join("");
  return `
    <div class="module-hero">
      <div class="flex items-center gap-3 mb-2">
        <div class="w-11 h-11 rounded-xl bg-white/10 flex items-center justify-center text-2xl">${icon || "◎"}</div>
        <div>
          <h1>${name}</h1>
          <div class="text-sm text-slate-300 mt-1">${en || ""}</div>
        </div>
      </div>
      <div class="flex items-center gap-2 mt-2">${tagHtml}</div>
      <div class="module-meta">${metaHtml}</div>
    </div>
  `;
};

// code block
MCH.code = function (source, lang = "python") {
  const esc = source.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return `<pre><code class="language-${lang}">${esc}</code></pre>`;
};

// 信息框
MCH.info = function (content, kind = "info") {
  const map = {
    info: { bg: "#eff6ff", border: "#3b82f6", fg: "#1e40af" },
    tip: { bg: "#f0fdf4", border: "#22c55e", fg: "#15803d" },
    warn: { bg: "#fef3c7", border: "#f59e0b", fg: "#b45309" },
    biz: { bg: "#faf5ff", border: "#a855f7", fg: "#7e22ce" },
  };
  const c = map[kind] || map.info;
  return `<div style="background:${c.bg};border-left:4px solid ${c.border};color:${c.fg};padding:12px 16px;border-radius:0 8px 8px 0;font-size:13.5px;line-height:1.6;margin:12px 0;">${content}</div>`;
};

// 生成 slider
MCH.slider = function ({ id, label, min, max, step, value, format }) {
  format = format || ((v) => v);
  return `
    <div style="margin-bottom:10px;">
      <div class="ctrl-label">
        <span>${label}</span>
        <span class="ctrl-val" id="${id}-val">${format(value)}</span>
      </div>
      <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}" />
    </div>
  `;
};

MCH.bindSlider = function (id, fn, format) {
  format = format || ((v) => v);
  const el = document.getElementById(id);
  const valEl = document.getElementById(id + "-val");
  if (!el) return;
  const update = () => {
    const v = parseFloat(el.value);
    if (valEl) valEl.textContent = format(v);
    fn(v);
  };
  el.addEventListener("input", update);
  update();
};

// 小工具：线性空间 & 基础数学
MCH.linspace = function (a, b, n) {
  const out = new Array(n);
  if (n === 1) { out[0] = a; return out; }
  for (let i = 0; i < n; i++) out[i] = a + ((b - a) * i) / (n - 1);
  return out;
};

MCH.randn = function (n, seed) {
  seed = seed || 1;
  let s = seed;
  const rand = () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
  const out = new Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = Math.max(1e-9, rand());
    const v = rand();
    const mag = Math.sqrt(-2 * Math.log(u));
    out[i] = mag * Math.cos(2 * Math.PI * v);
    if (i + 1 < n) out[i + 1] = mag * Math.sin(2 * Math.PI * v);
  }
  return out;
};

// 种子随机
MCH.seedRng = function (seed) {
  let s = seed || 1;
  return () => { s = (s * 9301 + 49297) % 233280; return s / 233280; };
};

// 标准 section 包装
MCH.section = function (title, content) {
  return `<div class="section"><h2>${title}</h2>${content}</div>`;
};

// 3 列优缺点卡片
MCH.prosCons = function (pros, cons, scenarios) {
  const li = (arr) => arr.map(x => `<li style="margin-bottom:4px;">${x}</li>`).join("");
  return `
    <div class="grid-3">
      <div class="card" style="border-left:3px solid #10b981;">
        <div class="font-semibold text-emerald-700 text-sm mb-2">✓ 优点</div>
        <ul class="text-xs text-slate-600" style="list-style:disc inside;">${li(pros)}</ul>
      </div>
      <div class="card" style="border-left:3px solid #ef4444;">
        <div class="font-semibold text-red-700 text-sm mb-2">✗ 缺点</div>
        <ul class="text-xs text-slate-600" style="list-style:disc inside;">${li(cons)}</ul>
      </div>
      <div class="card" style="border-left:3px solid #4f46e5;">
        <div class="font-semibold text-indigo-700 text-sm mb-2">◎ 最佳场景</div>
        <ul class="text-xs text-slate-600" style="list-style:disc inside;">${li(scenarios)}</ul>
      </div>
    </div>`;
};

// 版本演进表（registry 里 versions / highlights / papers）
MCH.versionSection = function (algoId) {
  const a = MCH.getById ? MCH.getById(algoId) : null;
  if (!a) return "";
  let html = "";
  if (a.highlights) {
    html += `<div class="section" style="background:linear-gradient(135deg,#fef3c7 0%,#fde68a 100%);border:1px solid #fcd34d;">
      <h2 style="color:#92400e;border:none;padding:0;margin:0 0 6px 0;">${a.highlights.startsWith("🆕") ? "" : "🆕 "}${a.highlights}</h2>
    </div>`;
  }
  if (a.versions && a.versions.length) {
    html += `<div class="section"><h2>📦 版本演进 / 实现变体</h2>
      <table class="table"><thead><tr><th style="width:220px;">版本 / 变体</th><th>亮点</th></tr></thead><tbody>`;
    a.versions.forEach(v => {
      html += `<tr><td><b>${v.name}</b></td><td class="text-xs text-slate-600">${v.note || "-"}</td></tr>`;
    });
    html += `</tbody></table>`;
    if (a.papers) html += `<div class="text-xs text-slate-500 mt-3">📄 参考文献 &amp; 链接：${a.papers}</div>`;
    html += `</div>`;
  } else if (a.papers) {
    html += `<div class="section"><h2>📄 参考文献</h2><div class="text-sm text-slate-700">${a.papers}</div></div>`;
  }
  return html;
};

// 超参表
MCH.hyperTable = function (rows) {
  let h = `<table class="table"><thead><tr><th>超参</th><th>推荐范围</th><th>作用</th><th>调优建议</th></tr></thead><tbody>`;
  rows.forEach(r => {
    h += `<tr><td><code>${r[0]}</code></td><td>${r[1]}</td><td>${r[2]}</td><td>${r[3]}</td></tr>`;
  });
  return h + `</tbody></table>`;
};
