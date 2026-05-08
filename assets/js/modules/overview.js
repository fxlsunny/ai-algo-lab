/* 模块：平台总览 */
MCH.register("overview", {
  render() {
    const t = (k) => (window.MCH && MCH.t) ? MCH.t(k) : k;
    const tname = (a) => (window.MCH && MCH.tname) ? MCH.tname(a) : a.name;
    const tcat = (zh) => (window.MCH && MCH.tcat) ? MCH.tcat(zh) : zh;
    const tdim = (d) => (window.MCH && MCH.tdim) ? MCH.tdim(d) : d.name;
    const lang = (window.MCH && MCH.i18n) ? MCH.i18n.lang : "zh";

    const byCategory = {};
    MCH.registry.forEach(a => {
      if (!byCategory[a.category]) byCategory[a.category] = { name: a.cat_name, items: [] };
      byCategory[a.category].items.push(a);
    });

    const renderCategory = (catId, cat) => {
      const color = MCH.catColors[catId] || "#4f46e5";
      const items = cat.items.map(a => `
        <a href="#/${a.route}" class="card block" style="text-decoration:none;color:inherit;border-top:3px solid ${color};">
          <div class="flex items-center gap-2 mb-1">
            <div style="width:28px;height:28px;border-radius:6px;background:${color}1a;color:${color};display:inline-flex;align-items:center;justify-content:center;font-weight:700;">${a.icon}</div>
            <div class="text-[13px] font-bold text-slate-800">${tname(a)}</div>
          </div>
          <div class="text-[11px] text-slate-500">${lang === "en" ? (a.name || "") : (a.en || "")}</div>
          <div class="flex flex-wrap gap-1 mt-2">
            ${a.tags.slice(0, 3).map(tag => `<span class="tag" style="background:${color}14;color:${color};">${tag}</span>`).join("")}
          </div>
          <div class="text-[10px] text-slate-400 mt-2">${t("common.enter_module")}</div>
        </a>`).join("");
      return `
        <div class="section">
          <h2 style="border-bottom-color:${color}33;">
            <span style="background:${color};color:#fff;padding:4px 12px;border-radius:6px;font-size:15px;display:inline-block;">${tcat(cat.name)}</span>
          </h2>
          <div class="grid-3">${items}</div>
        </div>`;
    };

    return `
      <div class="module-hero" style="background:linear-gradient(135deg,#1e1b4b 0%,#581c87 100%);">
        <div class="flex items-center gap-3 mb-2">
          <div class="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center text-2xl">🧪</div>
          <div>
            <h1>${t("overview.hero_title")}</h1>
            <div class="text-sm text-slate-300 mt-1">
              ${t("overview.hero_sub")}
            </div>
          </div>
        </div>
        <div class="flex gap-6 mt-4 text-sm text-slate-300 flex-wrap">
          <span>📚 <b class="text-white">${MCH.registry.length}</b> ${t("overview.stat.algos")}</span>
          <span>⚡ <b class="text-white">${Object.keys(byCategory).length}</b> ${t("overview.stat.cats")}</span>
          <span>📊 <b class="text-white">${MCH.dimensions.length}</b> ${t("overview.stat.dims")}</span>
          <span>${t("overview.stat.zero_build")}</span>
        </div>
      </div>

      <div class="section" style="background:linear-gradient(135deg,#eef2ff 0%,#fae8ff 100%);border:1px solid #c7d2fe;">
        <div class="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h2 style="color:#4338ca;border:none;padding:0;margin:0;">${t("overview.compare.title")}</h2>
            <p class="text-sm text-slate-600 mt-2 max-w-3xl">
              ${t("overview.compare.body")}
            </p>
          </div>
          <a href="#/compare" class="px-5 py-3 rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 text-white text-sm font-semibold shadow-lg hover:opacity-90 whitespace-nowrap">${t("overview.compare.btn")}</a>
        </div>
      </div>

      <div class="section">
        <h2>${t("overview.arch.title")}</h2>
        <div class="grid-2">
          <div>
            <h3 class="text-sm font-semibold text-slate-700">${t("overview.arch.taxonomy")}</h3>
            <div id="chart-taxonomy" style="height:300px;"></div>
          </div>
          <div>
            <h3 class="text-sm font-semibold text-slate-700">${t("overview.arch.cat_radar")}</h3>
            <div id="chart-cat-radar" style="height:340px;"></div>
          </div>
        </div>
      </div>

      ${Object.entries(byCategory).map(([catId, cat]) => renderCategory(catId, cat)).join("")}

      <div class="section">
        <h2>${t("overview.path.title")}</h2>
        <div class="grid-3">
          <div class="card" style="border-left:4px solid #10b981;">
            <h3 class="font-bold text-emerald-700">${t("overview.path.beginner")}</h3>
            <ol class="text-xs text-slate-600 mt-2" style="padding-left:20px;list-style:decimal;">
              ${(t("overview.path.beginner_steps") || []).map(s => `<li>${s}</li>`).join("")}
            </ol>
          </div>
          <div class="card" style="border-left:4px solid #f59e0b;">
            <h3 class="font-bold text-amber-700">${t("overview.path.business")}</h3>
            <ol class="text-xs text-slate-600 mt-2" style="padding-left:20px;list-style:decimal;">
              ${(t("overview.path.business_steps") || []).map(s => `<li>${s}</li>`).join("")}
            </ol>
          </div>
          <div class="card" style="border-left:4px solid #ec4899;">
            <h3 class="font-bold text-pink-700">${t("overview.path.llm")}</h3>
            <ol class="text-xs text-slate-600 mt-2" style="padding-left:20px;list-style:decimal;">
              ${(t("overview.path.llm_steps") || []).map(s => `<li>${s}</li>`).join("")}
            </ol>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>${t("overview.table.title")}</h2>
        <div style="max-height:500px;overflow-y:auto;">
          <table class="table">
            <thead style="position:sticky;top:0;">
              <tr>
                <th>${t("overview.col.algo")}</th>
                <th>${t("overview.col.cat")}</th>
                <th>${t("overview.col.acc")}</th>
                <th>${t("overview.col.intp")}</th>
                <th>${t("overview.col.train")}</th>
                <th>${t("overview.col.infer")}</th>
                <th>${t("overview.col.scenario")}</th>
              </tr>
            </thead>
            <tbody>
              ${MCH.registry.map(a => `
                <tr>
                  <td><a href="#/${a.route}" class="text-indigo-600 hover:underline"><b>${a.icon} ${tname(a)}</b></a><br/><span class="text-[10px] text-slate-500">${lang === "en" ? (a.name || "") : (a.en || "")}</span></td>
                  <td><span class="tag" style="background:${MCH.catColors[a.category]}14;color:${MCH.catColors[a.category]};">${tcat(a.cat_name)}</span></td>
                  <td>${this._bar(a.scores.accuracy)}</td>
                  <td>${this._bar(a.scores.interpretability)}</td>
                  <td>${this._bar(a.scores.training_speed)}</td>
                  <td>${this._bar(a.scores.inference_speed)}</td>
                  <td class="text-xs text-slate-500">${(a.best_for || [])[0] || "-"}</td>
                </tr>`).join("")}
            </tbody>
          </table>
        </div>
      </div>
    `;
  },

  _bar(val) {
    const pct = val * 10;
    const color = val >= 8 ? "#10b981" : val >= 6 ? "#f59e0b" : "#ef4444";
    return `<div style="display:flex;align-items:center;gap:6px;"><div style="width:60px;height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;"><div style="height:100%;background:${color};width:${pct}%;"></div></div><span class="text-xs font-mono">${val}</span></div>`;
  },

  mount() {
    const tcat = (zh) => (window.MCH && MCH.tcat) ? MCH.tcat(zh) : zh;
    const tdim = (d) => (window.MCH && MCH.tdim) ? MCH.tdim(d) : d.name;
    const tArchX = (window.MCH && MCH.t) ? MCH.t("overview.arch.x_count") : "算法数";

    // Taxonomy (bar)
    const tax = {};
    MCH.registry.forEach(a => { tax[a.cat_name] = (tax[a.cat_name] || 0) + 1; });
    const taxNamesZh = Object.keys(tax);
    const taxNamesDisplay = taxNamesZh.map(n => tcat(n));
    MCH.echart(document.getElementById("chart-taxonomy"), {
      grid: { left: 130, right: 30, top: 20, bottom: 30 },
      tooltip: {},
      xAxis: { type: "value", name: tArchX },
      yAxis: { type: "category", data: taxNamesDisplay },
      series: [{
        type: "bar", barWidth: 22,
        data: taxNamesZh.map((n, i) => ({
          value: tax[n],
          itemStyle: { color: Object.values(MCH.catColors)[i] || "#4f46e5" },
        })),
        label: { show: true, position: "right", color: "#475569" },
      }],
    });

    // Category avg radar
    const byCat = {};
    MCH.registry.forEach(a => {
      if (!byCat[a.cat_name]) byCat[a.cat_name] = [];
      byCat[a.cat_name].push(a);
    });
    const avgs = Object.entries(byCat).map(([name, arr]) => {
      const avg = {};
      MCH.dimensions.forEach(d => { avg[d.key] = arr.reduce((s, a) => s + a.scores[d.key], 0) / arr.length; });
      return { name, avg };
    });
    MCH.echart(document.getElementById("chart-cat-radar"), {
      tooltip: {},
      legend: { bottom: 0, textStyle: { fontSize: 11 } },
      radar: {
        indicator: MCH.dimensions.map(d => ({ name: tdim(d), max: 10 })),
        splitNumber: 5,
        radius: "55%",
        axisName: { fontSize: 11, color: "#475569" },
      },
      series: [{
        type: "radar", areaStyle: { opacity: 0.12 },
        data: avgs.map((a, i) => ({
          name: tcat(a.name),
          value: MCH.dimensions.map(d => +a.avg[d.key].toFixed(1)),
          lineStyle: { width: 2 },
          itemStyle: { color: Object.values(MCH.catColors)[i] || "#4f46e5" },
        })),
      }],
    });
  },
});
