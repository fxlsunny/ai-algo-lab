/* ============================================================================
 * search.js — 全局搜索功能
 * 索引算法名、类别、tags、优缺点、场景、超参、论文等 registry 内容
 * 快捷键：Ctrl+K / Cmd+K 打开, Esc 关闭, ↑↓ 切换, Enter 跳转
 * ========================================================================== */
(function () {
  // 构建搜索 Index
  function buildIndex() {
    return MCH.registry.map(a => {
      const parts = [];
      parts.push({ field: "name", text: a.name, weight: 12 });
      parts.push({ field: "en", text: a.en || "", weight: 10 });
      parts.push({ field: "category", text: a.cat_name || "", weight: 4 });
      (a.tags || []).forEach(t => parts.push({ field: "tag", text: t, weight: 7 }));
      (a.pros || []).forEach(p => parts.push({ field: "pros", text: p, weight: 4 }));
      (a.cons || []).forEach(p => parts.push({ field: "cons", text: p, weight: 4 }));
      (a.best_for || []).forEach(p => parts.push({ field: "scenario", text: p, weight: 5 }));
      (a.not_for || []).forEach(p => parts.push({ field: "not_for", text: p, weight: 3 }));
      (Array.isArray(a.key_hyperparams) ? a.key_hyperparams : [a.key_hyperparams || ""]).forEach(h => parts.push({ field: "hyperparam", text: String(h), weight: 5 }));
      parts.push({ field: "papers", text: a.papers || "", weight: 3 });
      parts.push({ field: "complexity", text: a.complexity || "", weight: 3 });
      return {
        algo: a,
        corpus: parts.map(p => p.text.toLowerCase()).join(" | "),
        parts,
      };
    });
  }

  function highlight(text, query) {
    if (!query) return text;
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    let out = text;
    // 按长度降序，避免子串冲突
    terms.sort((a, b) => b.length - a.length);
    terms.forEach(t => {
      const re = new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
      out = out.replace(re, '<mark style="background:#fef08a;color:#854d0e;padding:0 2px;border-radius:2px;">$1</mark>');
    });
    return out;
  }

  function scoreItem(item, terms) {
    if (terms.length === 0) return { score: 0, hits: [] };
    let total = 0;
    const hits = [];
    for (const part of item.parts) {
      const lc = part.text.toLowerCase();
      let partScore = 0;
      let allMatched = true;
      for (const t of terms) {
        if (lc.includes(t)) {
          partScore += part.weight;
          // 精确匹配 / 开头匹配加成
          if (lc === t) partScore += part.weight * 2;
          else if (lc.startsWith(t)) partScore += part.weight * 0.5;
        } else allMatched = false;
      }
      if (allMatched && partScore > 0) {
        total += partScore;
        hits.push(part);
      }
    }
    // AND-of-terms 至少所有词在 corpus 里出现
    const corpusHasAll = terms.every(t => item.corpus.includes(t));
    if (!corpusHasAll) total = 0;
    return { score: total, hits };
  }

  function search(query) {
    const index = buildIndex();
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
    if (terms.length === 0) return [];
    const scored = index.map(item => ({ ...scoreItem(item, terms), item }))
      .filter(r => r.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 20);
    return scored;
  }

  // UI 渲染
  function renderResults(results, query) {
    const t = (k, vars) => (window.MCH && MCH.t) ? MCH.t(k, vars) : k;
    const tname = (a) => (window.MCH && MCH.tname) ? MCH.tname(a) : a.name;
    const tcat = (zh) => (window.MCH && MCH.tcat) ? MCH.tcat(zh) : zh;
    const lang = (window.MCH && MCH.i18n) ? MCH.i18n.lang : "zh";

    if (results.length === 0) {
      const tryItems = lang === "en"
        ? '<code>XGBoost</code> · <code>long-tail</code> · <code>attention</code> · <code>community</code> · <code>quantization</code>'
        : '<code>XGBoost</code> · <code>长尾</code> · <code>注意力</code> · <code>社区挖掘</code> · <code>量化</code>';
      return `<div style="padding:32px;text-align:center;color:#64748b;">
        <div style="font-size:32px;opacity:0.3;">🔍</div>
        <div class="text-sm">${t("search.no_match")}</div>
        <div class="text-xs mt-2 text-slate-400">${t("search.try_label")}${tryItems}</div>
      </div>`;
    }
    return results.map((r, i) => {
      const a = r.item.algo;
      const color = MCH.catColors[a.category] || "#4f46e5";
      // 取第一个非 name/en 的命中作为预览
      const preview = r.hits.find(h => !["name", "en"].includes(h.field));
      const previewText = preview ? preview.text : (a.best_for && a.best_for[0]) || "";
      const fieldLabel = preview ? t("search.field." + preview.field) : "";
      const fieldTag = preview && fieldLabel && fieldLabel !== ("search.field." + preview.field)
        ? `<span class="tag tag-slate" style="font-size:10px;margin-right:6px;">${fieldLabel}</span>` : "";
      const primary = lang === "en" ? (a.en || a.name) : a.name;
      const secondary = lang === "en" ? (a.name || "") : (a.en || "");
      return `
        <div class="search-result" data-idx="${i}" data-route="${a.route}" style="padding:12px 14px;cursor:pointer;border-left:3px solid transparent;">
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <div>
              <span style="font-size:18px;margin-right:8px;">${a.icon}</span>
              <b style="color:${color};">${highlight(primary, query)}</b>
              <span class="text-[11px] text-slate-500" style="margin-left:8px;">${highlight(secondary, query)}</span>
            </div>
            <span class="tag" style="background:${color}14;color:${color};font-size:10px;">${tcat(a.cat_name)}</span>
          </div>
          ${previewText ? `<div class="text-[12px] text-slate-600 mt-1" style="padding-left:34px;line-height:1.45;">${fieldTag}${highlight(previewText, query)}</div>` : ""}
        </div>
      `;
    }).join("");
  }

  // 创建搜索 DOM
  function injectUI() {
    if (document.getElementById("global-search-overlay")) return;
    const t = (k, vars) => (window.MCH && MCH.t) ? MCH.t(k, vars) : k;
    const overlay = document.createElement("div");
    overlay.id = "global-search-overlay";
    overlay.style.cssText = "position:fixed;inset:0;background:rgba(15,23,42,0.55);z-index:100;display:none;backdrop-filter:blur(4px);align-items:flex-start;justify-content:center;padding-top:12vh;";
    overlay.innerHTML = `
      <div style="width:min(640px,90vw);background:white;border-radius:12px;box-shadow:0 20px 60px -15px rgba(15,23,42,0.5);overflow:hidden;">
        <div style="padding:12px 16px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;gap:10px;">
          <span style="font-size:18px;">🔍</span>
          <input id="global-search-input" type="text" data-i18n-placeholder="search.placeholder" placeholder="${t("search.placeholder")}" style="flex:1;border:none;outline:none;font-size:15px;background:transparent;" />
          <span class="text-[11px] text-slate-400" data-i18n="search.esc_hint" style="padding:2px 8px;background:#f1f5f9;border-radius:4px;">${t("search.esc_hint")}</span>
        </div>
        <div id="global-search-results" style="max-height:60vh;overflow-y:auto;"></div>
        <div style="padding:8px 16px;border-top:1px solid #e2e8f0;font-size:11px;color:#94a3b8;display:flex;justify-content:space-between;">
          <span data-i18n="search.kbd_hint">${t("search.kbd_hint")}</span>
          <span data-i18n="search.kbd_hint2">${t("search.kbd_hint2")}</span>
        </div>
      </div>
      <style>
        .search-result.active { background: #eef2ff; border-left-color: #4f46e5 !important; }
        .search-result:hover { background: #f8fafc; }
      </style>
    `;
    document.body.appendChild(overlay);

    const input = overlay.querySelector("#global-search-input");
    const resultsEl = overlay.querySelector("#global-search-results");
    let currentResults = [];
    let activeIdx = 0;

    const setActive = (idx) => {
      activeIdx = Math.max(0, Math.min(currentResults.length - 1, idx));
      overlay.querySelectorAll(".search-result").forEach((el, i) => {
        el.classList.toggle("active", i === activeIdx);
        if (i === activeIdx) el.scrollIntoView({ block: "nearest" });
      });
    };

    const doSearch = () => {
      const lang = (window.MCH && MCH.i18n) ? MCH.i18n.lang : "zh";
      const q = input.value.trim();
      currentResults = search(q);
      const tryItems = lang === "en"
        ? '<code>Transformer</code> · <code>long-tail</code> · <code>MoE</code> · <code>BFS DFS</code> · <code>risk</code> · <code>RoPE</code>'
        : '<code>Transformer</code> · <code>长尾</code> · <code>MoE</code> · <code>BFS DFS</code> · <code>风控</code> · <code>RoPE</code>';
      resultsEl.innerHTML = q ? renderResults(currentResults, q) : `
        <div style="padding:24px;text-align:center;color:#94a3b8;">
          <div class="text-sm mb-2">${t("search.idle_tip", { N: MCH.registry.length })}</div>
          <div class="text-xs">${t("search.idle_try")}${tryItems}</div>
        </div>
      `;
      activeIdx = 0; setActive(0);
      overlay.querySelectorAll(".search-result").forEach((el, i) => {
        el.addEventListener("click", () => navigate(currentResults[i]));
        el.addEventListener("mouseenter", () => setActive(i));
      });
    };

    const navigate = (r) => {
      if (!r) return;
      window.location.hash = "#/" + r.item.algo.route;
      close();
    };

    const open = () => {
      overlay.style.display = "flex";
      setTimeout(() => input.focus(), 30);
      doSearch();
    };
    const close = () => {
      overlay.style.display = "none";
      input.value = "";
    };
    MCH.openSearch = open;

    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    input.addEventListener("input", doSearch);
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        if (overlay.style.display === "none") open(); else close();
      } else if (overlay.style.display !== "none") {
        if (e.key === "Escape") close();
        else if (e.key === "ArrowDown") { e.preventDefault(); setActive(activeIdx + 1); }
        else if (e.key === "ArrowUp") { e.preventDefault(); setActive(activeIdx - 1); }
        else if (e.key === "Enter") { e.preventDefault(); navigate(currentResults[activeIdx]); }
      }
    });
  }

  // 启动
  window.addEventListener("DOMContentLoaded", () => {
    setTimeout(injectUI, 10);
  });
})();
