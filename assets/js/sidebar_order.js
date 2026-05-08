/* ============================================================================
 * sidebar_order.js — 侧栏导航手动排序（拖拽编辑 + 持久化）
 *
 * 行为：
 *   - 点击侧栏顶部 "✏️ 排序" 进入编辑模式
 *   - 编辑模式下：cat-section / subcat-section / nav-item 都成为 draggable
 *     可在同一父容器内任意拖动重排
 *   - 拖动结束自动序列化保存到 localStorage["mch-sidebar-order"]
 *   - 页面加载时（DOMContentLoaded 早期）自动恢复保存的顺序
 *   - "↩ 重置" 清空 localStorage 并刷新
 *   - 编辑模式下，点击 nav-item 不会触发跳路由（避免误触）
 *
 * 实现要点：
 *   - 容器分三类：sidebar 顶部 .p-4 / 每个 .cat-items / 每个 .subcat-items
 *   - 每个 sortable 子元素有唯一 key：cat:<X> / subcat:<X> / route:<X>
 *   - 同容器内有些子元素（折叠按钮组、提示框）非 sortable，重排时它们保持原位
 * ========================================================================== */
(function () {
  const STORAGE_KEY = "mch-sidebar-order";
  let editing = false;
  let dragSrc = null;

  /* ────────── localStorage 读写 ────────── */
  function loadOrder() {
    try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "{}"); }
    catch (e) { return {}; }
  }
  function saveOrderObj(ord) {
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(ord)); } catch (e) {}
  }
  function clearOrder() {
    try { localStorage.removeItem(STORAGE_KEY); } catch (e) {}
  }

  /* ────────── 元素 → 唯一 key ────────── */
  function keyOf(el) {
    if (!el || !el.classList) return null;
    if (el.classList.contains("cat-section") && el.dataset.cat) return "cat:" + el.dataset.cat;
    if (el.classList.contains("subcat-section") && el.dataset.subcat) return "subcat:" + el.dataset.subcat;
    if (el.classList.contains("nav-item") && el.dataset.route) return "route:" + el.dataset.route;
    return null;
  }

  /* ────────── 容器（决定排序作用域）────────── */
  function containerKey(container) {
    const sidebar = document.getElementById("app-sidebar");
    if (sidebar && container === sidebar.querySelector(":scope > .p-4")) return "_top";
    const p = container.parentElement;
    if (!p) return null;
    if (p.classList.contains("cat-section") && p.dataset.cat) return "cat:" + p.dataset.cat;
    if (p.classList.contains("subcat-section") && p.dataset.subcat) return "subcat:" + p.dataset.subcat;
    return null;
  }

  function getContainers() {
    const list = [];
    const sidebar = document.getElementById("app-sidebar");
    if (!sidebar) return list;
    const top = sidebar.querySelector(":scope > .p-4");
    if (top) list.push(top);
    sidebar.querySelectorAll(".cat-items").forEach(c => list.push(c));
    sidebar.querySelectorAll(".subcat-items").forEach(c => list.push(c));
    return list;
  }

  function getSortables(container) {
    return Array.from(container.children).filter(c => keyOf(c) != null);
  }

  /* ────────── 应用保存的顺序 ────────── */
  function applyOrder() {
    const ord = loadOrder();
    if (!ord || Object.keys(ord).length === 0) return;
    getContainers().forEach(container => {
      const ck = containerKey(container);
      if (!ck) return;
      const saved = ord[ck];
      if (!Array.isArray(saved) || saved.length === 0) return;
      const all = Array.from(container.children);
      const map = new Map();
      all.forEach(c => {
        const k = keyOf(c);
        if (k) map.set(k, c);
      });
      const used = new Set();
      const newSortable = [];
      saved.forEach(k => {
        if (map.has(k) && !used.has(k)) {
          newSortable.push(map.get(k));
          used.add(k);
        }
      });
      // 没在保存列表里的 sortable 元素（比如新增的模块）按原顺序追加到末尾
      all.forEach(c => {
        const k = keyOf(c);
        if (k && !used.has(k)) {
          newSortable.push(c);
          used.add(k);
        }
      });
      // 重新构造完整子元素列表：sortable 替换为新顺序，非 sortable 保持原位
      let ni = 0;
      const newAll = all.map(c => (keyOf(c) ? newSortable[ni++] : c));
      // appendChild 自动 detach + 插到末尾，按 newAll 顺序逐个 append 即可
      newAll.forEach(c => container.appendChild(c));
    });
  }

  /* ────────── 把当前 DOM 顺序写回 localStorage ────────── */
  function recordOrder() {
    const ord = {};
    getContainers().forEach(container => {
      const ck = containerKey(container);
      if (!ck) return;
      const list = [];
      Array.from(container.children).forEach(c => {
        const k = keyOf(c);
        if (k) list.push(k);
      });
      if (list.length > 0) ord[ck] = list;
    });
    saveOrderObj(ord);
  }

  /* ────────── 拖拽事件 ────────── */
  function onDragStart(e) {
    if (!editing) return;
    const t = e.currentTarget;
    if (!keyOf(t)) return;
    dragSrc = t;
    e.dataTransfer.effectAllowed = "move";
    try { e.dataTransfer.setData("text/plain", keyOf(t)); } catch (err) {}
    t.classList.add("drag-source");
    e.stopPropagation();
  }

  function onDragEnd() {
    if (dragSrc) dragSrc.classList.remove("drag-source");
    document.querySelectorAll(".drag-over-top, .drag-over-bottom").forEach(el => {
      el.classList.remove("drag-over-top", "drag-over-bottom");
    });
    dragSrc = null;
  }

  function onDragOver(e) {
    if (!editing || !dragSrc) return;
    const t = e.currentTarget;
    if (!t || t === dragSrc || t.parentElement !== dragSrc.parentElement) return;
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = "move";
    const rect = t.getBoundingClientRect();
    const mid = rect.top + rect.height / 2;
    const before = e.clientY < mid;
    t.classList.toggle("drag-over-top", before);
    t.classList.toggle("drag-over-bottom", !before);
  }

  function onDragLeave(e) {
    e.currentTarget.classList.remove("drag-over-top", "drag-over-bottom");
  }

  function onDrop(e) {
    if (!editing || !dragSrc) return;
    const t = e.currentTarget;
    if (!t || t === dragSrc || t.parentElement !== dragSrc.parentElement) return;
    e.preventDefault();
    e.stopPropagation();
    const rect = t.getBoundingClientRect();
    const mid = rect.top + rect.height / 2;
    const before = e.clientY < mid;
    if (before) t.parentElement.insertBefore(dragSrc, t);
    else t.parentElement.insertBefore(dragSrc, t.nextSibling);
    t.classList.remove("drag-over-top", "drag-over-bottom");
    recordOrder();
  }

  /* ────────── 编辑模式下点击 nav-item 不跳路由 ────────── */
  function navItemClickGuard(e) {
    if (editing) {
      e.preventDefault();
      e.stopPropagation();
    }
  }

  /* ────────── 折叠头点击守卫（编辑模式下不收起）────────── */
  function catHeaderClickGuard(e) {
    if (editing) e.stopPropagation();
  }

  /* ────────── 绑定/解绑拖拽事件 ────────── */
  function bindDragHandlers() {
    document.querySelectorAll(
      "#app-sidebar .cat-section, #app-sidebar .subcat-section, #app-sidebar a.nav-item"
    ).forEach(el => {
      if (el.dataset._dragBound) return;
      if (!keyOf(el)) return;
      // 默认不可拖（防止 <a> 浏览器原生 URL 拖拽干扰正常点击导航）
      el.setAttribute("draggable", "false");
      el.addEventListener("dragstart", onDragStart);
      el.addEventListener("dragend", onDragEnd);
      el.addEventListener("dragover", onDragOver);
      el.addEventListener("dragleave", onDragLeave);
      el.addEventListener("drop", onDrop);
      el.dataset._dragBound = "1";
    });
    document.querySelectorAll("#app-sidebar a.nav-item").forEach(a => {
      if (a.dataset._clickGuardBound) return;
      // 用 capture 阶段，确保比 hashchange 跳转早一步
      a.addEventListener("click", navItemClickGuard, true);
      a.dataset._clickGuardBound = "1";
    });
    document.querySelectorAll("#app-sidebar .cat-header, #app-sidebar .subcat-header").forEach(h => {
      if (h.dataset._headerGuardBound) return;
      h.addEventListener("click", catHeaderClickGuard, true);
      h.dataset._headerGuardBound = "1";
    });
  }

  /* ────────── 切换编辑模式 ────────── */
  function setEditing(on) {
    editing = !!on;
    const sidebar = document.getElementById("app-sidebar");
    if (!sidebar) return;
    sidebar.classList.toggle("sidebar-edit-mode", editing);

    document.querySelectorAll(
      "#app-sidebar .cat-section, #app-sidebar .subcat-section, #app-sidebar a.nav-item"
    ).forEach(el => {
      if (keyOf(el)) el.setAttribute("draggable", editing ? "true" : "false");
    });

    refreshButtonLabels();
  }

  function refreshButtonLabels() {
    const t = (k, fb) => (window.MCH && MCH.t) ? MCH.t(k) : fb;
    const editBtn     = document.getElementById("sidebar-edit-toggle");
    const resetBtn    = document.getElementById("sidebar-edit-reset");
    const collapseBtn = document.getElementById("sidebar-collapse-all");
    const expandBtn   = document.getElementById("sidebar-expand-all");
    if (editBtn) {
      editBtn.textContent = editing
        ? t("sidebar.edit_done",    "✓ 完成")
        : t("sidebar.edit_reorder", "✏️ 排序");
      editBtn.title = editing
        ? t("sidebar.edit_done_title",    "退出排序模式")
        : t("sidebar.edit_title",         "拖拽编辑模块顺序");
    }
    if (resetBtn) {
      resetBtn.textContent = t("sidebar.edit_reset",       "↩ 重置");
      resetBtn.title       = t("sidebar.edit_reset_title", "重置为默认顺序");
      resetBtn.style.display = editing ? "" : "none";
    }
    if (collapseBtn) collapseBtn.style.display = editing ? "none" : "";
    if (expandBtn)   expandBtn.style.display   = editing ? "none" : "";
  }

  /* ────────── 在折叠按钮组旁注入 编辑/重置 按钮 ────────── */
  function attachEditButtons() {
    const expandBtn = document.getElementById("sidebar-expand-all");
    if (!expandBtn) return;
    const row = expandBtn.parentElement;
    if (!row || document.getElementById("sidebar-edit-toggle")) return;

    const editBtn = document.createElement("button");
    editBtn.id = "sidebar-edit-toggle";
    editBtn.type = "button";
    editBtn.className =
      "text-[10px] px-2 py-0.5 rounded bg-amber-50 text-amber-700 hover:bg-amber-100";
    editBtn.textContent = "✏️ 排序";
    editBtn.title = "拖拽编辑模块顺序";
    editBtn.addEventListener("click", () => setEditing(!editing));

    const resetBtn = document.createElement("button");
    resetBtn.id = "sidebar-edit-reset";
    resetBtn.type = "button";
    resetBtn.className =
      "text-[10px] px-2 py-0.5 rounded bg-rose-50 text-rose-700 hover:bg-rose-100";
    resetBtn.textContent = "↩ 重置";
    resetBtn.title = "重置为默认顺序";
    resetBtn.style.display = "none";
    resetBtn.addEventListener("click", () => {
      const msg = (window.MCH && MCH.t)
        ? MCH.t("sidebar.edit_reset_confirm")
        : "重置侧栏顺序到默认？";
      if (confirm(msg)) {
        clearOrder();
        location.reload();
      }
    });

    row.insertBefore(editBtn, expandBtn);
    row.insertBefore(resetBtn, expandBtn);
  }

  /* ────────── 注入样式（编辑模式视觉反馈）────────── */
  function injectStyle() {
    if (document.getElementById("sidebar-order-style")) return;
    const s = document.createElement("style");
    s.id = "sidebar-order-style";
    s.textContent = `
      #app-sidebar.sidebar-edit-mode a.nav-item,
      #app-sidebar.sidebar-edit-mode .cat-section,
      #app-sidebar.sidebar-edit-mode .subcat-section {
        cursor: grab;
        position: relative;
      }
      #app-sidebar.sidebar-edit-mode a.nav-item[draggable="true"]::before,
      #app-sidebar.sidebar-edit-mode .cat-header::before,
      #app-sidebar.sidebar-edit-mode .subcat-header::before {
        content: "☰";
        display: inline-block;
        margin-right: 4px;
        color: var(--text-muted, #94a3b8);
        font-size: 11px;
        cursor: grab;
        user-select: none;
        opacity: 0.65;
      }
      #app-sidebar.sidebar-edit-mode a.nav-item[draggable="true"]:hover {
        outline: 1px dashed var(--accent, #4f46e5);
        outline-offset: -1px;
        background: var(--sidebar-hover, #f1f5f9);
      }
      #app-sidebar.sidebar-edit-mode .cat-section[draggable="true"] > .cat-header:hover,
      #app-sidebar.sidebar-edit-mode .subcat-section[draggable="true"] > .subcat-header:hover {
        outline: 1px dashed var(--accent, #4f46e5);
        outline-offset: -1px;
      }
      #app-sidebar .drag-source {
        opacity: 0.35;
      }
      #app-sidebar .drag-over-top {
        box-shadow: inset 0 2px 0 0 var(--accent, #4f46e5);
      }
      #app-sidebar .drag-over-bottom {
        box-shadow: inset 0 -2px 0 0 var(--accent, #4f46e5);
      }
      #app-sidebar.sidebar-edit-mode #sidebar-edit-toggle {
        background: #fde68a;
        color: #92400e;
        font-weight: 600;
      }
      #app-sidebar.sidebar-edit-mode .cat-section,
      #app-sidebar.sidebar-edit-mode .subcat-section {
        outline: 1px dashed transparent;
        outline-offset: -1px;
      }
      #app-sidebar.sidebar-edit-mode .cat-section[draggable="true"]:hover {
        outline-color: var(--accent, #4f46e5);
      }
    `;
    document.head.appendChild(s);
  }

  /* ────────── 初始化 ────────── */
  function init() {
    const sidebar = document.getElementById("app-sidebar");
    if (!sidebar) return;
    injectStyle();
    // 1) 先按保存的顺序重排（必须在 app.js 的 initSidebarCollapse 之前完成 DOM）
    applyOrder();
    // 2) 注入编辑按钮 + 绑定拖拽事件
    attachEditButtons();
    bindDragHandlers();
    // 3) 语言切换时刷新按钮文本
    document.addEventListener("mch:lang-changed", refreshButtonLabels);
    // 4) 初始按当前语言渲染按钮 label
    refreshButtonLabels();
  }

  // 暴露 API（方便调试 / 与 app.js 协作）
  window.MCH_SidebarOrder = {
    init,
    applyOrder,
    setEditing,
    clearOrder,
    isEditing: () => editing,
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init, { once: true });
  } else {
    init();
  }
})();
