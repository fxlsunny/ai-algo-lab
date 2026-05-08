/* 模块：算法对比分析 */
MCH.register("compare", {
  // 内部状态
  state: { selected: new Set(["ml_xgboost", "ml_lightgbm", "ml_random_forest"]) },

  render() {
    const t = (k, vars) => (window.MCH && MCH.t) ? MCH.t(k, vars) : k;
    const tname = (a) => (window.MCH && MCH.tname) ? MCH.tname(a) : a.name;
    const tcat = (zh) => (window.MCH && MCH.tcat) ? MCH.tcat(zh) : zh;

    const byCat = {};
    MCH.registry.forEach(a => {
      if (!byCat[a.cat_name]) byCat[a.cat_name] = [];
      byCat[a.cat_name].push(a);
    });

    const catHtml = Object.entries(byCat).map(([catName, arr]) => {
      const color = MCH.catColors[arr[0].category] || "#4f46e5";
      return `
        <div class="mb-3">
          <div class="text-[11px] font-bold uppercase mb-1" style="color:${color};">${tcat(catName)}</div>
          <div class="flex flex-wrap gap-1.5">
            ${arr.map(a => `
              <label class="algo-chip" data-id="${a.id}" data-color="${color}">
                <input type="checkbox" class="algo-cb" data-id="${a.id}" ${this.state.selected.has(a.id) ? "checked" : ""} style="display:none;" />
                <span class="chip-inner">${a.icon} ${tname(a)}</span>
              </label>
            `).join("")}
          </div>
        </div>`;
    }).join("");

    return `
      <div class="module-hero" style="background:linear-gradient(135deg,#312e81 0%,#6b21a8 100%);">
        <div class="flex items-center gap-3 mb-2">
          <div class="w-12 h-12 rounded-xl bg-white/10 flex items-center justify-center text-2xl">⚖️</div>
          <div>
            <h1>${t("compare.hero_title")}</h1>
            <div class="text-sm text-slate-300 mt-1">
              ${t("compare.hero_sub")}
            </div>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>${t("compare.step1.title")}</h2>
        <div class="ctrl-panel">${catHtml}</div>
        <div class="mt-3 border-t pt-3">
          <div class="text-xs text-slate-500 font-semibold mb-2">${t("compare.preset.tip")}</div>
          <div class="text-[11px] text-slate-500 font-semibold mb-1">${t("compare.preset.same")}</div>
          <div class="flex flex-wrap gap-2 mb-3">
            <button class="preset-btn" data-preset="trees">🌲 树模型家族（4）</button>
            <button class="preset-btn" data-preset="nn">🧠 神经网络基础（5）</button>
            <button class="preset-btn" data-preset="seq">⏳ 序列模型全家族（4）</button>
            <button class="preset-btn" data-preset="graph_emb">📐 图嵌入方法（4）</button>
            <button class="preset-btn" data-preset="community">👥 社区挖掘算法（2）</button>
            <button class="preset-btn" data-preset="gnn">🕸️ GNN 家族（3）</button>
            <button class="preset-btn" data-preset="loss">📉 损失函数（3）</button>
            <button class="preset-btn" data-preset="llm">🤖 LLM 专题（4）</button>
          </div>
          <div class="text-[11px] text-slate-500 font-semibold mb-1">${t("compare.preset.same_more")}</div>
          <div class="flex flex-wrap gap-2 mb-3">
            <button class="preset-btn" data-preset="supervised_ml">📚 监督 ML 全家族（7）</button>
            <button class="preset-btn" data-preset="clustering">🎯 聚类算法对比（2）</button>
            <button class="preset-btn" data-preset="dim_reduce">📉 降维方法对比（2）</button>
            <button class="preset-btn" data-preset="anomaly">🚨 异常检测方案</button>
          </div>
          <div class="text-[11px] text-slate-500 font-semibold mb-1">${t("compare.preset.cross")}</div>
          <div class="flex flex-wrap gap-2 mb-3">
            <button class="preset-btn preset-cross" data-preset="tabular_vs">🎯 表格场景：树模型 vs DNN vs LR</button>
            <button class="preset-btn preset-cross" data-preset="classic_vs_llm">⚔️ 经典 ML vs 大模型</button>
            <button class="preset-btn preset-cross" data-preset="seq_ancient_modern">⏰ 时序方法进化：RNN→Transformer→GPT</button>
            <button class="preset-btn preset-cross" data-preset="graph_all">🌐 图算法全谱系（7）</button>
            <button class="preset-btn preset-cross" data-preset="risk_lineup">🛡️ 风控任务候选</button>
            <button class="preset-btn preset-cross" data-preset="interpretable">🔍 可解释性 TOP</button>
            <button class="preset-btn preset-cross" data-preset="small_data">📊 小样本友好算法</button>
            <button class="preset-btn preset-cross" data-preset="billion_scale">🏭 十亿级规模可用</button>
            <button class="preset-btn preset-cross" data-preset="imbalance">⚖️ 长尾/不均衡方案</button>
            <button class="preset-btn preset-cross" data-preset="multitask">🎭 多任务学习方案</button>
            <button class="preset-btn preset-cross" data-preset="text_nlp">📝 文本/NLP 算法栈</button>
            <button class="preset-btn preset-cross" data-preset="cluster_vs_community">🔗 聚类 vs 社区挖掘</button>
            <button class="preset-btn preset-cross" data-preset="case_all">🏭 所有工业案例对比（5）</button>
            <button class="preset-btn preset-cross" data-preset="case_recsys_stack">🛒 推荐系统技术栈</button>
            <button class="preset-btn preset-cross" data-preset="case_fraud_stack">💳 欺诈识别技术栈</button>
            <button class="preset-btn preset-cross" data-preset="ts_all">📈 时序预测全谱系</button>
            <button class="preset-btn preset-cross" data-preset="anomaly_all">⚠ 异常检测演进（IF→Deep→LLM）</button>
            <button class="preset-btn preset-cross" data-preset="tabular_all">🧮 表格数据方案大比拼</button>
            <button class="preset-btn preset-cross" data-preset="frontier_all">🚀 前沿算法巡礼</button>
            <button class="preset-btn preset-cross" data-preset="long_seq">🧭 长序列建模方案 (Transformer/Mamba/TS)</button>
            <button class="preset-btn preset-cross" data-preset="coding_all">🧮 基础编程算法全谱系（8）🆕</button>
            <button class="preset-btn preset-cross" data-preset="coding_graph">🕸️ 图遍历 vs 图算法（编程 vs AI）🆕</button>
            <button class="preset-btn preset-cross" data-preset="coding_vs_ai_tree">🌳 数据结构树 vs ML 决策树 🆕</button>
          </div>
        </div>

        <div class="flex items-center justify-between mt-2">
          <div id="sel-count" class="text-xs text-slate-500"></div>
          <div class="flex gap-2">
            <button id="select-all-cat" class="text-xs px-3 py-1 bg-indigo-50 text-indigo-600 rounded hover:bg-indigo-100">${t("compare.btn.select_all")}</button>
            <button id="clear-sel" class="text-xs px-3 py-1 bg-red-50 text-red-600 rounded hover:bg-red-100">${t("compare.btn.clear_sel")}</button>
          </div>
        </div>
      </div>

      <div id="compare-result"></div>

      <style>
        .algo-chip { display: inline-block; cursor: pointer; }
        .algo-chip .chip-inner {
          display: inline-block; padding: 4px 10px;
          background: #f1f5f9; color: #475569;
          border-radius: 999px; font-size: 12px;
          border: 1px solid transparent;
          transition: all .15s;
        }
        .algo-chip:hover .chip-inner { background: #e2e8f0; }
        .algo-chip input:checked + .chip-inner {
          background: var(--chip-color, #4f46e5); color: white;
          border-color: var(--chip-color, #4f46e5);
          box-shadow: 0 2px 6px -1px rgba(79,70,229,.35);
        }
        .preset-btn {
          font-size: 11.5px;
          padding: 4px 10px;
          background: #f1f5f9;
          color: #475569;
          border-radius: 6px;
          border: 1px solid transparent;
          transition: all .15s;
        }
        .preset-btn:hover { background: #e0e7ff; color: #4338ca; border-color: #c7d2fe; }
        .preset-cross {
          background: linear-gradient(90deg, #eef2ff 0%, #fae8ff 100%);
          color: #5b21b6;
          border-color: #ddd6fe;
        }
        .preset-cross:hover {
          background: linear-gradient(90deg, #e0e7ff 0%, #f3e8ff 100%);
          border-color: #a78bfa;
        }
      </style>
    `;
  },

  mount(root) {
    // Chip color setup
    document.querySelectorAll(".algo-chip").forEach(c => {
      c.style.setProperty("--chip-color", c.dataset.color);
      c.addEventListener("click", (e) => {
        const id = c.dataset.id;
        if (this.state.selected.has(id)) this.state.selected.delete(id);
        else this.state.selected.add(id);
        // 强制切换 checkbox
        const cb = c.querySelector(".algo-cb");
        cb.checked = this.state.selected.has(id);
        this._updateResult();
      });
    });

    // Presets
    const setSel = (ids) => {
      this.state.selected = new Set(ids);
      document.querySelectorAll(".algo-cb").forEach(cb => {
        cb.checked = this.state.selected.has(cb.dataset.id);
      });
      this._updateResult();
    };

    const PRESETS = {
      // 同类别
      trees: ["ml_decision_tree", "ml_random_forest", "ml_xgboost", "ml_lightgbm"],
      nn: ["nn_mlp", "nn_cnn", "nn_rnn", "nn_attention", "nn_gpt"],
      seq: ["nn_rnn", "nn_attention", "nn_gpt", "seq_encoder"],
      graph_emb: ["graph_node2vec", "graph_gcn", "graph_sage", "graph_gat"],
      community: ["graph_louvain", "graph_label_prop"],
      gnn: ["graph_gcn", "graph_sage", "graph_gat"],
      loss: ["loss_basics", "losses", "loss_multitask"],
      llm: ["nn_attention", "nn_gpt", "llm_params", "llm_finetune", "llm_quantization"],
      // 跨类别选型场景
      tabular_vs: ["ml_logistic", "ml_xgboost", "ml_lightgbm", "nn_mlp"],
      classic_vs_llm: ["ml_xgboost", "nn_mlp", "nn_attention", "nn_gpt"],
      seq_ancient_modern: ["nn_rnn", "nn_attention", "nn_gpt"],
      graph_all: ["graph_pagerank", "graph_louvain", "graph_label_prop", "graph_node2vec", "graph_gcn", "graph_sage", "graph_gat"],
      risk_lineup: ["ml_logistic", "ml_xgboost", "ml_lightgbm", "graph_sage", "case_overview"],
      interpretable: ["ml_decision_tree", "ml_logistic", "graph_pagerank", "graph_louvain"],
      small_data: ["ml_logistic", "ml_decision_tree", "ml_random_forest", "loss_basics"],
      billion_scale: ["ml_lightgbm", "graph_sage", "graph_label_prop", "llm_quantization"],
      imbalance: ["losses", "samplers", "ml_xgboost", "loss_multitask"],
      multitask: ["mmoe", "loss_multitask", "heads", "losses"],
      // 新增
      supervised_ml: ["ml_decision_tree", "ml_random_forest", "ml_xgboost", "ml_lightgbm", "ml_logistic", "ml_svm", "ml_naive_bayes"],
      clustering: ["ml_kmeans", "ml_dbscan"],
      dim_reduce: ["ml_pca", "ml_tsne_umap"],
      anomaly: ["ml_isolation_forest", "ml_dbscan", "losses"],
      text_nlp: ["ml_naive_bayes", "ml_svm", "nn_attention", "nn_gpt", "llm_foundation"],
      cluster_vs_community: ["ml_kmeans", "ml_dbscan", "graph_louvain", "graph_label_prop"],
      case_all: ["case_recommendation", "case_fraud", "case_ads_ctr", "case_credit_score", "case_merchant"],
      case_recsys_stack: ["case_recommendation", "nn_attention", "seq_encoder", "mmoe", "graph_sage"],
      case_fraud_stack: ["case_fraud", "ml_xgboost", "ml_isolation_forest", "graph_sage", "graph_louvain"],
      ts_all: ["ts_classical", "ts_deep", "ts_foundation"],
      anomaly_all: ["ml_isolation_forest", "ad_deep", "ad_llm"],
      tabular_all: ["ml_xgboost", "ml_lightgbm", "tab_dl", "ml_logistic"],
      frontier_all: ["frontier_mamba", "frontier_diffusion", "frontier_rag"],
      long_seq: ["nn_attention", "nn_transformer", "frontier_mamba", "ts_deep", "ts_foundation"],
      // 编程算法 (v6.0)
      coding_all: ["coding_sort", "coding_search", "coding_ds_stack_queue", "coding_ds_linked_tree", "coding_graph_traversal", "coding_graph_shortest", "coding_dp", "coding_string_math"],
      coding_graph: ["coding_graph_traversal", "coding_graph_shortest", "graph_pagerank", "graph_sage"],
      coding_vs_ai_tree: ["coding_ds_linked_tree", "ml_decision_tree", "ml_random_forest", "ml_xgboost"],
    };
    document.querySelectorAll(".preset-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const key = btn.dataset.preset;
        if (PRESETS[key]) setSel(PRESETS[key]);
      });
    });
    document.getElementById("clear-sel").onclick = () => setSel([]);
    document.getElementById("select-all-cat").onclick = () => {
      // 找当前选中里最多的类别，全选该类别
      const counts = {};
      this.state.selected.forEach(id => {
        const a = MCH.getById(id); if (a) counts[a.category] = (counts[a.category] || 0) + 1;
      });
      const topCat = Object.entries(counts).sort((a, b) => b[1] - a[1])[0];
      const targetCat = topCat ? topCat[0] : "classical_ml";
      setSel(MCH.registry.filter(a => a.category === targetCat).map(a => a.id));
    };

    this._updateResult();
  },

  _updateResult() {
    const t = (k) => (window.MCH && MCH.t) ? MCH.t(k) : k;
    const lang = (window.MCH && MCH.i18n) ? MCH.i18n.lang : "zh";
    const result = document.getElementById("compare-result");
    const selArr = [...this.state.selected].map(id => MCH.getById(id)).filter(Boolean);
    const countLabel = lang === "en"
      ? `<b style="color:#4f46e5;">${selArr.length}</b> algorithm${selArr.length === 1 ? "" : "s"} selected`
      : `已选 <b style="color:#4f46e5;">${selArr.length}</b> 个算法`;
    document.getElementById("sel-count").innerHTML = countLabel;

    if (selArr.length === 0) {
      const empty = lang === "en"
        ? "👆 Select at least one algorithm above to start comparing"
        : "👆 请在上方勾选至少一个算法开始对比";
      result.innerHTML = `<div class="section text-center text-slate-400 py-10">${empty}</div>`;
      MCH.disposeCharts();
      return;
    }
    MCH.disposeCharts();

    // Render all comparison panels
    result.innerHTML = `
      ${this._renderRadar(selArr)}
      ${this._renderScoreTable(selArr)}
      ${this._renderResources(selArr)}
      ${this._renderProsCons(selArr)}
      ${this._renderMetaTable(selArr)}
      ${this._renderScenarioMatrix(selArr)}
      ${this._renderRecommendation(selArr)}
    `;
    MCH.rerender(result);
    this._mountCharts(selArr);
    this._mountResources(selArr);
  },

  _renderResources(sel) {
    const scales = ["10w", "100w", "1000w", "1亿"];
    const scaleNames = { "10w": "10 万样本", "100w": "100 万样本", "1000w": "1000 万样本", "1亿": "1 亿样本" };

    // 参数量表
    const paramHtml = sel.map(a => {
      const r = MCH.getResource(a.id, "10w");
      const params = r ? r.paramsTypical : 0;
      return { algo: a, params, min: r?.paramsMin || 0, max: r?.paramsMax || 0 };
    });

    let resourceTable = `<div class="section"><h2>4️⃣ 💻 计算资源评估（样本规模 → 训练/推理需求）</h2>
      <p class="text-sm text-slate-600 mb-3">基于工业经验的粗略估算，实际取决于特征维度、模型复杂度、硬件代际等。</p>

      <h3>· 模型参数量与部署体积</h3>
      <div id="res-chart-params" style="height:${Math.max(260, sel.length * 40)}px;margin-bottom:20px;"></div>

      <h3>· 训练资源需求矩阵</h3>
      <div style="overflow-x:auto;">
        <table class="table">
          <thead>
            <tr>
              <th style="min-width:180px;">算法 \\ 样本量</th>`;
    scales.forEach(s => { resourceTable += `<th>${scaleNames[s]}</th>`; });
    resourceTable += `</tr>
          </thead>
          <tbody>`;
    sel.forEach(a => {
      resourceTable += `<tr><td><b>${a.icon} ${a.name}</b><div class="text-[10px] text-slate-500">${a.cat_name}</div></td>`;
      scales.forEach(s => {
        const r = MCH.getResource(a.id, s);
        if (!r || !r.train) { resourceTable += `<td class="text-xs text-slate-400">-</td>`; return; }
        const t = r.train;
        const hasGPU = (t.hw || "").includes("GPU") || (t.hw || "").includes("A100") || (t.hw || "").includes("H100");
        const notViable = (t.hw || "").includes("❌") || (t.hw || "").includes("不") || (t.hw || "").includes("N/A");
        const color = notViable ? "#ef4444" : (hasGPU ? "#4f46e5" : "#10b981");
        resourceTable += `<td style="padding:8px;">
          <div style="font-size:11px;line-height:1.45;">
            <div style="color:${color};font-weight:600;"><b>⏱</b> ${t.time || "-"}</div>
            <div style="color:#475569;"><b>💾</b> ${t.mem || "-"}</div>
            <div style="color:${color};"><b>🖥️</b> ${t.hw || "-"}</div>
          </div>
        </td>`;
      });
      resourceTable += `</tr>`;
    });
    resourceTable += `</tbody></table></div>

      <h3 style="margin-top:18px;">· 推理能力估算</h3>
      <table class="table">
        <thead><tr><th>算法</th><th>典型参数量</th><th>模型体积 (FP32)</th><th>单核 QPS</th><th>相对延迟</th></tr></thead>
        <tbody>`;
    sel.forEach(a => {
      const r = MCH.getResource(a.id, "10w");
      if (!r) { return; }
      const params = r.paramsTypical;
      const size = r.infer?.size_mb || (params * 4 / 1024 / 1024);
      const qps = r.infer?.qps_per_core || "-";
      const latency = typeof qps === "number" ? (1000 / qps).toFixed(2) + " ms" : "-";
      resourceTable += `<tr>
          <td><b>${a.icon} ${a.name}</b></td>
          <td>${params ? (params >= 1e9 ? (params / 1e9).toFixed(1) + "B" : params >= 1e6 ? (params / 1e6).toFixed(1) + "M" : (params / 1e3).toFixed(1) + "K") : "-"}</td>
          <td>${typeof size === "number" ? (size < 1 ? size.toFixed(2) + " MB" : size < 1000 ? size.toFixed(0) + " MB" : (size / 1024).toFixed(1) + " GB") : size}</td>
          <td>${typeof qps === "number" ? qps.toLocaleString() : qps}</td>
          <td>${latency}</td>
        </tr>`;
    });
    resourceTable += `</tbody></table>

      <h3 style="margin-top:18px;">· 1 亿样本规模下的成本预估</h3>
      <div id="res-chart-cost" style="height:300px;"></div>
    </div>`;
    return resourceTable;
  },

  _mountResources(sel) {
    // 参数量对比
    const pEl = document.getElementById("res-chart-params");
    if (pEl) {
      MCH.echart(pEl, {
        tooltip: { trigger: "axis", valueFormatter: v => (v >= 1e9 ? (v / 1e9).toFixed(2) + "B" : (v / 1e6).toFixed(1) + "M") },
        legend: { top: 0 },
        grid: { left: 140, right: 40, top: 40, bottom: 30 },
        xAxis: { type: "log", name: "参数量", logBase: 10 },
        yAxis: { type: "category", data: sel.map(a => `${a.icon} ${a.name}`) },
        series: [{
          type: "bar", barWidth: 16,
          data: sel.map(a => {
            const r = MCH.getResource(a.id, "10w");
            const v = r ? Math.max(10, r.paramsTypical) : 10;
            return { value: v, itemStyle: { color: MCH.catColors[a.category] } };
          }),
          label: {
            show: true, position: "right", color: "#475569",
            formatter: (p) => { const v = p.value; return v >= 1e9 ? (v / 1e9).toFixed(2) + " B" : v >= 1e6 ? (v / 1e6).toFixed(1) + " M" : v >= 1e3 ? (v / 1e3).toFixed(1) + " K" : v; },
          },
        }],
      });
    }

    // 1 亿样本下的训练成本（估算 GPU 小时）
    const costEl = document.getElementById("res-chart-cost");
    if (costEl) {
      const costEstimate = sel.map(a => {
        const r = MCH.getResource(a.id, "1亿");
        if (!r || !r.train) return { name: a.name, cost: 0, hw: "-" };
        const t = r.train.time || "";
        // 粗估 GPU 小时
        let hours = 0;
        if (t.includes("月")) hours = 720;
        else if (t.includes("周") || t.includes("5 天") || t.includes("数天")) hours = 168;
        else if (t.includes("天")) hours = 48;
        else if (t.includes("半天") || t.includes("10 h") || t.includes("10-30 h")) hours = 12;
        else if (t.includes("h") || t.includes("小时")) hours = 4;
        else if (t.includes("min") || t.includes("分钟")) hours = 0.5;
        // GPU 数：按硬件估
        let gpus = 1;
        const hw = r.train.hw || "";
        if (hw.includes("1000") || hw.includes("千")) gpus = 1000;
        else if (hw.includes("256") || hw.includes("集群")) gpus = 256;
        else if (hw.includes("16")) gpus = 16;
        else if (hw.includes("8")) gpus = 8;
        else if (hw.includes("4")) gpus = 4;
        else if (hw.includes("2-4")) gpus = 3;
        else if (hw.includes("多机")) gpus = 32;
        else if (hw.includes("多 GPU") || hw.includes("多 A100")) gpus = 4;
        else if (hw.includes("GPU") || hw.includes("A100") || hw.includes("H100")) gpus = 1;
        const gpuHours = hours * gpus;
        return {
          name: `${a.icon} ${a.name}`,
          cost: gpuHours,
          hw: hw,
          time: t,
          cny: Math.round(gpuHours * 15),   // 约 15 元/GPU·h 估算
        };
      });
      MCH.echart(costEl, {
        tooltip: {
          trigger: "axis",
          formatter: (p) => {
            const item = costEstimate[p[0].dataIndex];
            return `${item.name}<br/>GPU·h: <b>${item.cost.toLocaleString()}</b><br/>硬件: ${item.hw}<br/>时长: ${item.time}<br/>估算成本 ≈ ¥${item.cny.toLocaleString()}`;
          },
        },
        grid: { left: 140, right: 80, top: 40, bottom: 40 },
        xAxis: { type: "log", name: "GPU·小时 (1亿样本训练)", logBase: 10 },
        yAxis: { type: "category", data: costEstimate.map(c => c.name) },
        series: [{
          type: "bar", barWidth: 16,
          data: costEstimate.map((c, i) => ({ value: Math.max(0.1, c.cost), itemStyle: { color: MCH.catColors[sel[i].category] } })),
          label: { show: true, position: "right", color: "#475569", formatter: (p) => { const c = costEstimate[p.dataIndex]; return c.cost > 0 ? `${c.cost.toLocaleString()} h · ¥${c.cny > 1000 ? (c.cny / 10000).toFixed(1) + "万" : c.cny}` : "-"; } },
        }],
      });
    }
  },

  _renderRadar(sel) {
    return `
      <div class="section">
        <h2>2️⃣ 多维度雷达对比</h2>
        <p class="text-xs text-slate-500 mb-3">10 维能力打分（1-10，越高越好）。面积越大表示综合能力越强；但"最适合"要结合具体场景看各维度。</p>
        <div id="radar-chart" style="height:480px;"></div>
      </div>`;
  },

  _renderScoreTable(sel) {
    let html = `<div class="section"><h2>3️⃣ 维度明细打分（每项 1-10 分，越高越好）</h2><table class="table"><thead><tr><th style="min-width:140px;">维度</th>`;
    sel.forEach(a => { html += `<th><div class="flex items-center gap-1"><span style="color:${MCH.catColors[a.category]};">${a.icon}</span> ${a.name}</div></th>`; });
    html += `<th style="width:60px;">最高</th></tr></thead><tbody>`;
    MCH.dimensions.forEach(d => {
      const vals = sel.map(a => a.scores[d.key]);
      const max = Math.max(...vals);
      html += `<tr><td><b>${d.name}</b><div class="text-[10px] text-slate-400">${d.en}</div></td>`;
      sel.forEach((a, i) => {
        const v = vals[i];
        const isMax = v === max;
        html += `<td>
          <div style="display:flex;align-items:center;gap:6px;">
            <div style="flex:1;height:6px;background:#e2e8f0;border-radius:3px;overflow:hidden;">
              <div style="height:100%;background:${MCH.catColors[a.category]};width:${v * 10}%;"></div>
            </div>
            <span style="font-family:monospace;font-size:12px;font-weight:${isMax ? 700 : 400};color:${isMax ? MCH.catColors[a.category] : "#64748b"};">${v}${isMax ? " 🏆" : ""}</span>
          </div>
        </td>`;
      });
      const maxAlgo = sel[vals.indexOf(max)];
      html += `<td class="text-xs" style="color:${MCH.catColors[maxAlgo.category]};font-weight:600;">${maxAlgo.icon}</td>`;
      html += `</tr>`;
    });
    html += `</tbody></table></div>`;
    return html;
  },

  _renderProsCons(sel) {
    let html = `<div class="section"><h2>5️⃣ 优缺点并排分析</h2>`;
    html += `<div style="display:grid;grid-template-columns:repeat(${sel.length}, 1fr);gap:14px;">`;
    sel.forEach(a => {
      const color = MCH.catColors[a.category];
      html += `
        <div class="card" style="border-top:3px solid ${color};">
          <div class="flex items-center gap-2 mb-2">
            <span style="font-size:18px;">${a.icon}</span>
            <div>
              <div class="font-bold text-slate-800 text-sm">${a.name}</div>
              <div class="text-[10px] text-slate-500">${a.en}</div>
            </div>
          </div>
          <div class="mt-3">
            <div class="text-[11px] font-semibold text-emerald-700 mb-1">✓ 优点</div>
            <ul class="text-[12px] text-slate-600" style="padding-left:16px;list-style:disc;">
              ${a.pros.map(p => `<li>${p}</li>`).join("")}
            </ul>
          </div>
          <div class="mt-3">
            <div class="text-[11px] font-semibold text-red-700 mb-1">✗ 缺点 / 限制</div>
            <ul class="text-[12px] text-slate-600" style="padding-left:16px;list-style:disc;">
              ${a.cons.map(p => `<li>${p}</li>`).join("")}
            </ul>
          </div>
        </div>`;
    });
    html += `</div></div>`;
    return html;
  },

  _renderMetaTable(sel) {
    let html = `<div class="section"><h2>6️⃣ 关键参数与复杂度</h2><table class="table"><thead><tr><th>算法</th><th>复杂度</th><th>关键超参</th><th>参考论文</th></tr></thead><tbody>`;
    sel.forEach(a => {
      html += `<tr>
        <td style="vertical-align:top;"><b>${a.icon} ${a.name}</b><div class="text-[10px] text-slate-500">${a.cat_name}</div></td>
        <td style="vertical-align:top;"><code class="text-xs">${a.complexity || "-"}</code></td>
        <td style="vertical-align:top;"><div class="flex flex-wrap gap-1">${(Array.isArray(a.key_hyperparams) ? a.key_hyperparams : [a.key_hyperparams]).map(h => `<code class="tag tag-slate" style="font-family:monospace;">${h}</code>`).join("")}</div></td>
        <td style="vertical-align:top;" class="text-xs text-slate-600">${a.papers || "-"}</td>
      </tr>`;
    });
    html += `</tbody></table></div>`;
    return html;
  },

  _renderScenarioMatrix(sel) {
    // 场景 vs 算法适配矩阵
    const scenarios = [
      { name: "小样本 (<1k)", key: (a) => (a.id === "ml_logistic" ? 8 : a.scores.low_feature_eng < 6 ? 3 : a.id.startsWith("nn_") ? 2 : 6) },
      { name: "大样本 (>1亿)", key: (a) => a.scores.scalability },
      { name: "可解释强要求", key: (a) => a.scores.interpretability },
      { name: "极致精度", key: (a) => a.scores.accuracy },
      { name: "实时低延迟 (<10ms)", key: (a) => a.scores.inference_speed },
      { name: "内存受限（端侧）", key: (a) => a.scores.memory_efficient },
      { name: "表格结构化数据", key: (a) => (a.category === "classical_ml" ? 9 : a.category === "neural_net" && a.id === "nn_mlp" ? 6 : 4) },
      { name: "图像 / CV", key: (a) => (a.id === "nn_cnn" ? 10 : a.id === "nn_attention" ? 9 : a.id === "image_encoder" ? 8 : 2) },
      { name: "文本 / NLP", key: (a) => (a.id === "nn_attention" ? 10 : a.id === "text_encoder" ? 9 : a.id.startsWith("llm_") ? 10 : 3) },
      { name: "长尾不均衡", key: (a) => (a.id === "losses" ? 10 : a.id === "samplers" ? 9 : a.id === "ml_xgboost" || a.id === "ml_lightgbm" ? 6 : 4) },
      { name: "多任务学习", key: (a) => (a.id === "mmoe" ? 10 : a.id === "heads" ? 8 : a.scores.scalability > 7 ? 5 : 3) },
      { name: "需要增量/在线", key: (a) => (a.id === "ml_logistic" ? 9 : a.id === "ml_lightgbm" ? 6 : 3) },
    ];
    let html = `<div class="section"><h2>7️⃣ 场景适配矩阵</h2>
      <p class="text-xs text-slate-500 mb-3">对每个典型业务场景，为所选算法启发式打分（基于上方能力维度）。深色单元格表示更合适。</p>
      <div style="overflow-x:auto;"><table class="table"><thead><tr><th style="min-width:160px;">场景 / 算法</th>`;
    sel.forEach(a => { html += `<th>${a.icon} ${a.name}</th>`; });
    html += `</tr></thead><tbody>`;
    scenarios.forEach(s => {
      html += `<tr><td><b>${s.name}</b></td>`;
      sel.forEach(a => {
        const v = s.key(a);
        const alpha = Math.min(1, v / 10);
        const bg = a.category ? MCH.catColors[a.category] : "#4f46e5";
        html += `<td style="background:${bg}${Math.round(alpha * 40).toString(16).padStart(2, "0")};text-align:center;font-weight:${v >= 8 ? 700 : 400};color:${v >= 8 ? bg : "#475569"};font-family:monospace;">${v} / 10</td>`;
      });
      html += `</tr>`;
    });
    html += `</tbody></table></div></div>`;
    return html;
  },

  _renderRecommendation(sel) {
    // 计算综合分
    const dimWeights = { interpretability: 1, accuracy: 1.2, training_speed: 0.8, inference_speed: 1, memory_efficient: 0.8, scalability: 0.9, nonlinear: 0.9, low_feature_eng: 0.8, tuning_ease: 0.9, robustness: 1 };
    const ranked = sel.map(a => {
      let score = 0, total = 0;
      for (const k in dimWeights) { score += a.scores[k] * dimWeights[k]; total += dimWeights[k]; }
      return { a, score: score / total };
    }).sort((x, y) => y.score - x.score);

    // 找出每个算法"最突出"维度和"最弱"维度
    const analysis = sel.map(a => {
      const entries = Object.entries(a.scores).map(([k, v]) => {
        const dim = MCH.dimensions.find(d => d.key === k);
        return { k, v, name: dim ? dim.name : k };
      });
      const top = [...entries].sort((x, y) => y.v - x.v).slice(0, 2);
      const bot = [...entries].sort((x, y) => x.v - y.v).slice(0, 2);
      return { a, top, bot };
    });

    return `
      <div class="section">
        <h2>8️⃣ 综合推荐与差异化分析</h2>

        <div class="grid-2">
          <div>
            <h3>· 综合得分排名（加权平均）</h3>
            <div id="rank-chart" style="height:${Math.max(200, ranked.length * 50)}px;"></div>
          </div>
          <div>
            <h3>· 差异化亮点</h3>
            ${analysis.map(({ a, top, bot }) => `
              <div class="card mb-2" style="padding:10px 14px;">
                <div class="font-semibold text-sm" style="color:${MCH.catColors[a.category]};">${a.icon} ${a.name}</div>
                <div class="text-xs text-slate-600 mt-1">
                  <span class="text-emerald-600">💪 最强：</span>${top.map(t => `<b>${t.name}</b>(${t.v})`).join(" / ")}
                  &nbsp;·&nbsp;
                  <span class="text-red-600">⚠️ 较弱：</span>${bot.map(t => `<b>${t.name}</b>(${t.v})`).join(" / ")}
                </div>
              </div>`).join("")}
          </div>
        </div>

        <h3 style="margin-top:20px;">· 选型建议（基于最佳场景交集）</h3>
        <div class="grid-3 mt-3">
          ${sel.map(a => `
            <div class="card" style="border-left:3px solid ${MCH.catColors[a.category]};">
              <div class="font-semibold text-sm" style="color:${MCH.catColors[a.category]};">${a.icon} ${a.name}</div>
              <div class="text-[11px] text-emerald-700 mt-2 font-semibold">🎯 选它，如果你的场景是：</div>
              <ul class="text-xs text-slate-600" style="padding-left:16px;list-style:disc;">
                ${(a.best_for || []).map(s => `<li>${s}</li>`).join("")}
              </ul>
              <div class="text-[11px] text-red-700 mt-2 font-semibold">🚫 别选它，如果：</div>
              <ul class="text-xs text-slate-600" style="padding-left:16px;list-style:disc;">
                ${(a.not_for || []).map(s => `<li>${s}</li>`).join("")}
              </ul>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  },

  _mountCharts(sel) {
    // Radar
    MCH.echart(document.getElementById("radar-chart"), {
      tooltip: { trigger: "item" },
      legend: { bottom: 0, textStyle: { fontSize: 12 } },
      radar: {
        indicator: MCH.dimensions.map(d => ({ name: d.name, max: 10 })),
        radius: "62%",
        splitNumber: 5,
        axisName: { color: "#475569", fontSize: 12 },
        splitArea: { areaStyle: { color: ["rgba(241,245,249,0.3)", "rgba(241,245,249,0.6)"] } },
      },
      series: [{
        type: "radar",
        emphasis: { areaStyle: { opacity: 0.45 } },
        data: sel.map(a => ({
          name: `${a.icon} ${a.name}`,
          value: MCH.dimensions.map(d => a.scores[d.key]),
          lineStyle: { width: 2 },
          areaStyle: { opacity: 0.18 },
          itemStyle: { color: MCH.catColors[a.category] },
        })),
      }],
    });

    // Rank bar
    const dimWeights = { interpretability: 1, accuracy: 1.2, training_speed: 0.8, inference_speed: 1, memory_efficient: 0.8, scalability: 0.9, nonlinear: 0.9, low_feature_eng: 0.8, tuning_ease: 0.9, robustness: 1 };
    const ranked = sel.map(a => {
      let score = 0, total = 0;
      for (const k in dimWeights) { score += a.scores[k] * dimWeights[k]; total += dimWeights[k]; }
      return { a, score: +(score / total).toFixed(2) };
    }).sort((x, y) => x.score - y.score);
    const rankEl = document.getElementById("rank-chart");
    if (rankEl) {
      MCH.echart(rankEl, {
        tooltip: {},
        grid: { left: 140, right: 40, top: 20, bottom: 30 },
        xAxis: { type: "value", max: 10, name: "综合分" },
        yAxis: { type: "category", data: ranked.map(r => `${r.a.icon} ${r.a.name}`) },
        series: [{
          type: "bar", barWidth: 20,
          data: ranked.map(r => ({ value: r.score, itemStyle: { color: MCH.catColors[r.a.category] } })),
          label: { show: true, position: "right", color: "#475569", formatter: "{c}" },
        }],
      });
    }
  },
});
