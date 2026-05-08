/* ============================================================================
 * assistant.js — AI 算法选型助手
 *
 * 功能：
 * 1. 接收用户输入（场景/数据/规模/目标/样例）
 * 2. 调用 LLM 或规则引擎给出算法选型方案
 * 3. 多种算法类型：传统/深度/大模型/多模态/长时序
 * 4. 导出方案到本地（File System Access API + 下载）
 * 5. 历史记录保存到 data/history/
 * ========================================================================== */
(function () {
  const HIST_KEY = "aial-assistant-history";
  const DATA_HANDLE_KEY = "aial-data-handle";

  const Assistant = {
    state: {
      dirHandle: null,  // File System Access API 目录句柄
    },

    // ---------------- 历史记录 ----------------
    listHistory() {
      try { return JSON.parse(localStorage.getItem(HIST_KEY) || "[]"); } catch { return []; }
    },
    saveHistory(record) {
      const hist = this.listHistory();
      hist.unshift(record);
      localStorage.setItem(HIST_KEY, JSON.stringify(hist.slice(0, 100)));  // 最多存 100 条
      // 🔥 同时写服务端（如果 unified server 可用）
      this._saveHistoryToServer(record);
    },
    clearHistory() {
      localStorage.removeItem(HIST_KEY);
      // 服务端也清空
      this._deleteHistoryOnServer();
    },

    // 判定当前是否在 unified server 下（复用 LLM 探测结果）
    _hasServer() {
      return window.AIAL_LLM && AIAL_LLM._unifiedServer === "yes";
    },

    _saveHistoryToServer(record) {
      if (!this._hasServer()) return;
      fetch("/api/history/" + (record.id || "unknown"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(record),
      }).catch(() => {});
    },

    _deleteHistoryOnServer() {
      if (!this._hasServer()) return;
      fetch("/api/history", { method: "DELETE" }).catch(() => {});
    },

    /** 启动时从服务端恢复历史（localStorage 为空或少于服务端时合并） */
    async loadHistoryFromServer() {
      if (!window.AIAL_LLM) return false;
      if (!await AIAL_LLM.probeUnifiedServer()) return false;
      try {
        const r = await fetch("/api/history?full=1");
        if (!r.ok) return false;
        const { items } = await r.json();
        if (!Array.isArray(items) || items.length === 0) return false;

        // 合并策略：以 id 为 key，服务端+本地去重
        const localList = this.listHistory();
        const map = new Map();
        // 先放服务端（旧优先），再用本地覆盖（本地更新）
        items.forEach(it => { if (it && it.id) map.set(it.id, it); });
        localList.forEach(it => { if (it && it.id) map.set(it.id, it); });
        const merged = [...map.values()].sort((a, b) => {
          const ta = new Date(a.ts || a.timestamp || 0).getTime();
          const tb = new Date(b.ts || b.timestamp || 0).getTime();
          return tb - ta;
        });
        localStorage.setItem(HIST_KEY, JSON.stringify(merged.slice(0, 100)));
        console.info("[AIAL] ✅ 从服务端恢复 " + merged.length + " 条历史记录");

        // 若 UI 已打开，刷新列表
        if (this._refreshHistoryList) setTimeout(() => this._refreshHistoryList(), 100);
        return true;
      } catch { return false; }
    },

    // ---------------- 导出到本地 data 目录 ----------------
    async saveFileToData(subDir, filename, content) {
      // 方案 A：File System Access API（Chrome/Edge）
      if (this.state.dirHandle) {
        try {
          const sub = await this.state.dirHandle.getDirectoryHandle(subDir, { create: true });
          const handle = await sub.getFileHandle(filename, { create: true });
          const writable = await handle.createWritable();
          await writable.write(content);
          await writable.close();
          return { method: "fs-api", path: `data/${subDir}/${filename}` };
        } catch (e) {
          console.warn("FS API 失败，降级到下载：", e);
        }
      }
      // 方案 B：浏览器下载
      const blob = new Blob([content], { type: filename.endsWith(".json") ? "application/json" : "text/markdown" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
      return { method: "download", path: `Downloads/${filename}` };
    },

    async pickDataDirectory() {
      if (!window.showDirectoryPicker) {
        alert("当前浏览器不支持 File System Access API，将使用下载方式保存。建议用 Chrome/Edge。");
        return null;
      }
      try {
        const handle = await window.showDirectoryPicker({ mode: "readwrite", startIn: "documents" });
        this.state.dirHandle = handle;
        return handle;
      } catch { return null; }
    },

    // ---------------- 核心推理：构造 prompt ----------------
    buildPrompt(input) {
      // 注入 registry 关键元数据作为知识库
      const kb = (MCH.registry || []).map(a => ({
        id: a.id, name: a.name, en: a.en, category: a.cat_name,
        tags: a.tags, pros: a.pros?.slice(0, 3), cons: a.cons?.slice(0, 3),
        best_for: a.best_for, not_for: a.not_for,
        complexity: a.complexity,
      }));

      const system = `你是资深 AI 算法架构师，精通机器学习、深度学习、大模型、图算法、时序预测、异常检测、多模态等领域。
你的任务是基于用户需求，从下面的算法库（共 ${kb.length} 个算法）给出算法选型方案。

【算法库 JSON】（每个算法有 id/name/category/tags/pros/cons/best_for/not_for）：
${JSON.stringify(kb)}

【输出要求】
严格输出 JSON，结构如下：
{
  "summary": "一句话总结推荐方案",
  "difficulty": "low|medium|high",
  "estimated_effort": "工程人日估算",
  "recommendations": [
    {
      "tier": "primary|secondary|alternative",
      "algorithm_id": "对应算法库 id",
      "algorithm_name": "算法名",
      "reason": "选择理由（结合用户需求）",
      "fit_score": 0-100
    }
  ],
  "pipeline": [
    { "step": 1, "name": "步骤名", "detail": "具体做什么" }
  ],
  "data_preparation": ["数据准备要点"],
  "evaluation_metrics": ["推荐的评估指标"],
  "deployment_notes": ["部署注意事项"],
  "risks": ["潜在风险与对策"],
  "alternatives_by_category": {
    "traditional_ml": ["备选 id1", "id2"],
    "deep_learning": ["id1"],
    "llm": ["id1"],
    "multi_modal": ["id1"],
    "time_series": ["id1"]
  },
  "reading_list": ["推荐阅读的算法详情页链接 (#/ID)"]
}
必须是合法 JSON，不要包含 markdown 标记，不要包含任何解释性文字。`;

      const user = `## 场景
${input.scenario || "(未填)"}

## 数据内容
${input.data_content || "(未填)"}

## 数据规模
${input.data_scale || "(未填)"}

## 业务目标
${input.goal || "(未填)"}

## 样例数据 / Schema
\`\`\`
${input.sample_data || "(未提供)"}
\`\`\`

## 偏好算法类型（用户勾选）
${(input.categories || []).join(", ") || "(无偏好, 请综合推荐)"}

## 约束（如延迟/解释性/预算）
${input.constraints || "(无特别约束)"}

请严格按照上述 JSON schema 输出算法选型方案。`;

      return [{ role: "system", content: system }, { role: "user", content: user }];
    },

    // ---------------- 规则引擎 Fallback ----------------
    ruleBasedRecommend(input) {
      const kb = MCH.registry || [];
      const text = [input.scenario, input.data_content, input.goal, input.constraints].join(" ").toLowerCase();
      const hasCat = (input.categories || []).length > 0;

      // 关键词 → 候选算法
      const scores = {};
      const score = (id, delta, reason) => {
        if (!scores[id]) scores[id] = { score: 0, reasons: [] };
        scores[id].score += delta;
        if (reason) scores[id].reasons.push(reason);
      };

      // 数据规模映射
      const scale = input.data_scale || "";
      const scaleMatch = {
        small: /\b(10k|1w|小|数千|<\s*1\s*万|1,?000)/i.test(scale),
        medium: /\b(10w|100w|中|百万|数万|千级)/i.test(scale),
        large: /\b(1000w|千万|1亿|亿|十亿|大规模)/i.test(scale),
      };

      // 数据类型关键词
      const kw = {
        text: /文本|nlp|语言|评论|邮件|日志|对话|chat/i.test(text),
        image: /图像|图片|视觉|照片|ocr|cv|cnn/i.test(text),
        time_series: /时序|时间序列|预测|趋势|arima|季节|sales|traffic/i.test(text),
        graph: /图|关系|团伙|社区|社交|node|edge|gnn/i.test(text),
        tabular: /表格|结构化|特征|csv|excel|行列/i.test(text),
        recommend: /推荐|召回|排序|ctr|个性化/i.test(text),
        fraud: /欺诈|反欺诈|风控|黑产|盗刷|套现/i.test(text),
        anomaly: /异常|outlier|故障|入侵|反常/i.test(text),
        multimodal: /多模态|multi-?modal|图文|图\s*文/i.test(text),
        cluster: /聚类|分群|clustering|客群|cohort/i.test(text),
        llm_needed: /问答|生成|总结|翻译|agent|rag|知识库/i.test(text),
        explainable: /可解释|可审计|合规|监管|评分卡/i.test(text),
        realtime: /实时|低延迟|&lt;\s*\d+\s*ms|<\s*\d+\s*ms|online|streaming/i.test(text),
        // —— 工程/编程算法关键词（v6.0 新增）——
        coding_sort: /排序|sort|topk|top-?k|排行榜|归并|快排|堆排/i.test(text),
        coding_search: /查找|搜索|二分|binary\s*search|哈希|hash|索引|lookup|去重/i.test(text),
        coding_stack: /栈|队列|stack|queue|滑动窗口|sliding\s*window|lru|单调/i.test(text),
        coding_tree: /链表|二叉树|trie|前缀树|敏感词|自动补全|字典树|红黑树|avl/i.test(text),
        coding_graph: /bfs|dfs|拓扑|连通分量|岛屿|迷宫|最短路|dijkstra|a\s*\*/i.test(text),
        coding_dp: /动态规划|dp|背包|最长.{0,4}子序列|编辑距离|子数组|lcs|lis/i.test(text),
        coding_str: /字符串匹配|kmp|manacher|回文|快速幂|gcd|欧几里得/i.test(text),
        engineering: /工程|系统|后端|服务|pipeline|etl|kafka|sdk|微服务|中间件|面试|刷题|leetcode|算法题/i.test(text),
      };

      // 按关键词加分
      if (kw.text) { score("ml_naive_bayes", 3, "文本分类 NB 快"); score("nn_attention", 4, "文本 NLP"); score("llm_foundation", 5, "现代 NLP 用 LLM"); score("text_encoder", 3, "LoRA 微调"); }
      if (kw.image) { score("nn_cnn", 5, "图像 CNN 基础"); score("nn_transformer", 4, "ViT"); score("frontier_diffusion", 3, "生成"); score("image_encoder", 3); }
      if (kw.time_series) { score("ts_classical", 4, "小样本时序"); score("ts_deep", 5, "深度时序"); score("ts_foundation", 5, "零样本基座"); if (kw.anomaly) score("ad_deep", 3, "时序异常"); }
      if (kw.graph) { score("graph_sage", 5, "大规模图"); score("graph_gat", 4, "注意力"); score("graph_louvain", 3, "社区挖掘"); score("graph_node2vec", 3, "无监督嵌入"); }
      if (kw.tabular) { score("ml_xgboost", 5, "表格 XGB"); score("ml_lightgbm", 5, "大数据 LGB"); score("tab_dl", 3, "深度表格"); score("ml_logistic", 2, "baseline"); }
      if (kw.recommend) { score("case_recommendation", 6, "推荐案例"); score("mmoe", 4, "多任务"); score("nn_attention", 3, "DIN/SASRec"); score("seq_encoder", 3, "行为序列"); }
      if (kw.fraud) { score("case_fraud", 6, "欺诈案例"); score("ml_xgboost", 4); score("ml_isolation_forest", 4, "异常"); score("graph_sage", 4, "团伙识别"); score("losses", 3, "长尾"); }
      if (kw.anomaly) { score("ml_isolation_forest", 5); score("ad_deep", 4); score("ad_llm", 3, "零样本"); }
      if (kw.multimodal) { score("fusion", 5, "多模态融合"); score("case_merchant", 4, "工业案例"); score("frontier_diffusion", 3); }
      if (kw.cluster) { score("ml_kmeans", 5); score("ml_dbscan", 4, "非球形"); score("ml_pca", 3, "降维"); score("ml_tsne_umap", 2, "可视化"); }
      if (kw.llm_needed) { score("llm_foundation", 5); score("llm_finetune", 4); score("frontier_rag", 5, "RAG"); score("llm_params", 3); }
      if (kw.explainable) { score("ml_logistic", 5, "评分卡"); score("ml_decision_tree", 4); score("case_credit_score", 4); }
      if (kw.realtime) { score("ml_logistic", 3); score("ml_lightgbm", 3); score("llm_quantization", 3, "量化加速"); }

      // —— 编程算法直接命中（v6.0）——
      if (kw.coding_sort) { score("coding_sort", 5, "排序/TopK 基本功"); }
      if (kw.coding_search) { score("coding_search", 5, "二分/哈希 O(log n)/O(1) 查找"); }
      if (kw.coding_stack) { score("coding_ds_stack_queue", 5, "栈/队列/单调栈 滑窗"); }
      if (kw.coding_tree) { score("coding_ds_linked_tree", 5, "Trie/堆/BST 前缀查询 & TopK"); }
      if (kw.coding_graph) { score("coding_graph_traversal", 5, "BFS/DFS/拓扑"); if (/最短路|dijkstra|a\s*\*/i.test(text)) score("coding_graph_shortest", 5, "Dijkstra/A*"); }
      if (kw.coding_dp) { score("coding_dp", 5, "动态规划"); }
      if (kw.coding_str) { score("coding_string_math", 5, "字符串/数学"); }

      // —— 业务场景 → 编程算法组合（v6.0）——
      // 这些是 AI 方案落地时"一定会用到"的工程基石组件，给适度加分做出现在推荐列表底部
      if (kw.recommend) { score("coding_ds_linked_tree", 2, "Top-K 推荐召回用堆"); score("coding_sort", 1, "粗排/精排"); }
      if (kw.fraud) { score("coding_graph_traversal", 2, "团伙关系 BFS/DFS"); score("coding_graph_shortest", 1, "欺诈路径分析"); }
      if (kw.text) { score("coding_string_math", 2, "敏感词 Trie / KMP 匹配"); score("coding_ds_linked_tree", 1, "Trie"); }
      if (kw.realtime) { score("coding_ds_stack_queue", 2, "单调队列滑窗"); }
      if (kw.llm_needed) { score("coding_dp", 1, "Beam Search"); score("coding_ds_linked_tree", 1, "Top-K 采样用堆"); }
      if (kw.graph) { score("coding_graph_traversal", 2, "图遍历基石"); score("coding_graph_shortest", 1); }
      if (kw.engineering) { score("coding_sort", 1, "工程基础"); score("coding_search", 1); score("coding_ds_stack_queue", 1); }

      // 数据规模加分
      if (scaleMatch.small) {
        score("ml_logistic", 3, "小数据稳定"); score("ml_decision_tree", 3); score("tab_dl", -2, "小数据 DL 易过拟合");
      }
      if (scaleMatch.medium) {
        score("ml_xgboost", 3); score("ml_lightgbm", 3); score("nn_mlp", 2);
      }
      if (scaleMatch.large) {
        score("ml_lightgbm", 5, "亿级工业首选"); score("graph_sage", 4, "大图"); score("llm_quantization", 3); score("frontier_mamba", 2, "长序列");
      }

      // 用户偏好类型加分
      if (hasCat) {
        const cats = new Set(input.categories.map(c => c.toLowerCase()));
        kb.forEach(a => {
          const catKey = a.category === "classical_ml" ? "traditional" :
                        a.category === "neural_net" ? "deep" :
                        a.category === "llm" ? "llm" :
                        a.category === "time_series" ? "time_series" :
                        a.category === "graph" ? "graph" :
                        a.category === "frontier" ? "frontier" : "other";
          const hasMultimodal = a.tags?.some(t => /多模态|multi-?modal/i.test(t));
          const hasLongSeq = a.tags?.some(t => /长序列|long|100k/i.test(t));
          if (cats.has(catKey)) score(a.id, 2);
          if (cats.has("multimodal") && hasMultimodal) score(a.id, 3);
          if (cats.has("long_sequence") && hasLongSeq) score(a.id, 3);
        });
      }

      // 排序取 Top
      const ranked = Object.entries(scores)
        .map(([id, { score, reasons }]) => ({ id, score, reasons, algo: kb.find(a => a.id === id) }))
        .filter(x => x.algo && x.score > 0)
        .sort((a, b) => b.score - a.score);

      if (ranked.length === 0) {
        // 兜底
        ranked.push(
          { id: "ml_lightgbm", score: 5, reasons: ["通用工业级 baseline"], algo: MCH.getById("ml_lightgbm") },
          { id: "ml_xgboost", score: 4, reasons: ["精度 SOTA"], algo: MCH.getById("ml_xgboost") },
        );
      }

      const top = ranked.slice(0, 3);
      const secondary = ranked.slice(3, 6);
      const alt = ranked.slice(6, 10);

      // 分类备选
      const byCat = { traditional_ml: [], deep_learning: [], llm: [], multi_modal: [], time_series: [], coding: [] };
      ranked.forEach(r => {
        const cat = r.algo?.category;
        if (cat === "classical_ml") byCat.traditional_ml.push(r.id);
        else if (cat === "neural_net" || cat === "graph") byCat.deep_learning.push(r.id);
        else if (cat === "llm") byCat.llm.push(r.id);
        else if (cat === "time_series") byCat.time_series.push(r.id);
        else if (cat === "coding") byCat.coding.push(r.id);
        if (r.algo?.tags?.some(t => /多模态|multi/i.test(t))) byCat.multi_modal.push(r.id);
      });

      // 分离 AI 算法和工程组件，让推荐列表更清晰：AI 主方案在前，编程算法组件放 alternative 末尾
      const codingRanked = ranked.filter(r => r.algo?.category === "coding");
      const nonCodingRanked = ranked.filter(r => r.algo?.category !== "coding");
      const top2 = nonCodingRanked.slice(0, 3);
      const secondary2 = nonCodingRanked.slice(3, 6);
      const alt2 = [...nonCodingRanked.slice(6, 9), ...codingRanked.slice(0, 3)];

      return {
        summary: `基于规则引擎匹配，推荐 ${top2[0]?.algo?.name || top[0]?.algo?.name || "LightGBM"} 为主方案，配合 ${(secondary2[0]?.algo?.name || secondary[0]?.algo?.name || "XGBoost")} 做备选。` + (kw.llm_needed ? " 场景涉及 LLM 能力，建议配合 RAG 方案。" : "") + (codingRanked.length ? ` 工程落地推荐配套使用 ${codingRanked.slice(0, 2).map(r => r.algo.name).join(" / ")} 等编程算法组件。` : ""),
        difficulty: scaleMatch.large ? "high" : scaleMatch.medium ? "medium" : "low",
        estimated_effort: scaleMatch.large ? "2-4 人月" : scaleMatch.medium ? "2-4 人周" : "1-2 人周",
        recommendations: [
          ...top2.map(r => ({ tier: "primary", algorithm_id: r.id, algorithm_name: r.algo.name, reason: r.reasons.join("；"), fit_score: Math.min(95, 60 + r.score * 6) })),
          ...secondary2.map(r => ({ tier: "secondary", algorithm_id: r.id, algorithm_name: r.algo.name, reason: r.reasons.join("；"), fit_score: Math.min(80, 40 + r.score * 5) })),
          ...alt2.map(r => ({ tier: "alternative", algorithm_id: r.id, algorithm_name: r.algo.name, reason: (r.algo.category === "coding" ? "[工程组件] " : "") + r.reasons.join("；"), fit_score: Math.min(65, 30 + r.score * 4) })),
        ],
        pipeline: [
          { step: 1, name: "数据收集与清洗", detail: "构建训练/验证/测试集，处理缺失、异常、编码" },
          { step: 2, name: "baseline 建模", detail: `先用 ${top2[0]?.algo?.name || top[0]?.algo?.name || "LightGBM"} 跑通基线` },
          { step: 3, name: "特征工程/调参", detail: "WOE/交叉特征、GridSearch、Optuna" },
          { step: 4, name: "高级模型", detail: `尝试 ${secondary2[0]?.algo?.name || secondary[0]?.algo?.name || "XGBoost"} 或深度方案` },
          { step: 5, name: "评估与上线", detail: "离线指标 + A/B 测试 + 灰度发布" },
          ...(codingRanked.length ? [{ step: 6, name: "工程侧实现", detail: `使用 ${codingRanked.slice(0, 3).map(r => r.algo.name).join(" / ")} 等基础算法组件支撑上线` }] : []),
        ],
        data_preparation: [
          "样本去重与时间分割（防穿越）",
          "特征编码（WOE/one-hot/target encoding）",
          kw.text ? "文本清洗 + Tokenizer" : "数值特征标准化",
          scaleMatch.large ? "Spark/Flink 分布式特征构造" : "pandas 内存处理",
        ],
        evaluation_metrics: [
          kw.fraud || text.includes("分类") ? "AUC / KS / Precision@TopK / F1" : "Accuracy / MAE / RMSE",
          kw.recommend ? "NDCG@10, HitRate@K, AUC" : "",
          kw.anomaly ? "Recall, Precision, F1（异常类）" : "",
        ].filter(Boolean),
        deployment_notes: [
          kw.realtime ? "实时推理 &lt; 100ms: ONNX + Triton / vLLM / LightGBM C++" : "离线批量推理 Spark/Ray",
          kw.explainable ? "SHAP 或评分卡: 监管友好" : "",
          scaleMatch.large ? "分布式训练 + 模型版本管理 MLflow" : "",
        ].filter(Boolean),
        risks: [
          kw.fraud ? "坏人会对抗绕过 → 模型需月级迭代" : "",
          kw.explainable ? "监管审核 → 必须可解释 + 公平性审计" : "",
          scaleMatch.large ? "数据飘移 → PSI/CSI 监控" : "",
          "标签噪声 → 需人工抽检 + 主动学习",
        ].filter(Boolean),
        alternatives_by_category: byCat,
        reading_list: top.slice(0, 3).map(r => `#/${r.id}`),
        _engine: "rule-based",
      };
    },

    // ---------------- 主推荐调用 ----------------
    async recommend(input, { useLLM = true } = {}) {
      const cfg = window.AIAL_LLM?.getActive();
      // 能用 LLM 的条件：配置存在 + (有 API Key OR 有 demo_key)
      const hasUsableKey = cfg && (
        (cfg.apiKey && cfg.apiKey.trim()) ||
        (window.AIAL_LLM.PROVIDERS[cfg.provider]?.demo_key && cfg.useDemoKey !== false)
      );
      if (useLLM && hasUsableKey) {
        try {
          const messages = this.buildPrompt(input);
          const raw = await window.AIAL_LLM.chat(cfg, messages, { temperature: 0.2, max_tokens: 3000 });
          const clean = raw.replace(/```(?:json)?/g, "").replace(/```$/g, "").trim();
          const result = JSON.parse(clean);
          result._engine = "llm-" + (cfg.provider || "custom") + "/" + (cfg.model || "?");
          return result;
        } catch (e) {
          console.warn("LLM 调用失败，降级规则引擎：", e);
          const fallback = this.ruleBasedRecommend(input);
          fallback._engine = "rule-based (LLM fallback: " + e.message.slice(0, 100) + ")";
          return fallback;
        }
      }
      return this.ruleBasedRecommend(input);
    },

    // ---------------- UI 注入 ----------------
    injectUI() {
      const overlay = document.createElement("div");
      overlay.id = "aial-assistant-overlay";
      overlay.style.cssText = "position:fixed;inset:0;background:rgba(15,23,42,0.55);z-index:110;display:none;backdrop-filter:blur(4px);align-items:center;justify-content:center;";
      overlay.innerHTML = this._renderHTML();
      document.body.appendChild(overlay);

      const cfgOverlay = document.createElement("div");
      cfgOverlay.id = "aial-config-overlay";
      cfgOverlay.style.cssText = "position:fixed;inset:0;background:rgba(15,23,42,0.55);z-index:110;display:none;backdrop-filter:blur(4px);align-items:center;justify-content:center;";
      cfgOverlay.innerHTML = this._renderConfigHTML();
      document.body.appendChild(cfgOverlay);

      this._bindEvents();

      window.AIAL_ASSISTANT = this;
    },

    openAssistant() {
      document.getElementById("aial-assistant-overlay").style.display = "flex";
      this._refreshHistoryList();
    },
    closeAssistant() { document.getElementById("aial-assistant-overlay").style.display = "none"; },

    openConfig() {
      document.getElementById("aial-config-overlay").style.display = "flex";
      this._refreshConfigList();
    },
    closeConfig() { document.getElementById("aial-config-overlay").style.display = "none"; },

    _renderHTML() {
      return `
        <div style="width:min(1100px, 94vw);max-height:92vh;background:white;border-radius:14px;box-shadow:0 30px 80px -20px rgba(0,0,0,0.5);display:flex;flex-direction:column;overflow:hidden;">
          <div style="padding:16px 24px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#ec4899 100%);color:white;">
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="font-size:28px;">🧙‍♂️</div>
              <div>
                <div style="font-size:18px;font-weight:800;">AI 算法选型助手</div>
                <div style="font-size:12px;opacity:0.85;">描述你的场景，我来帮你推荐最合适的算法方案</div>
              </div>
            </div>
            <div class="flex gap-2 items-center">
              <span id="aial-engine-badge" style="font-size:11px;padding:3px 10px;background:rgba(255,255,255,0.2);border-radius:999px;"></span>
              <button id="aial-close" style="color:white;font-size:24px;background:transparent;border:0;cursor:pointer;padding:0 8px;">×</button>
            </div>
          </div>
          <div style="display:grid;grid-template-columns:400px 1fr;flex:1;overflow:hidden;">
            <div style="padding:20px;border-right:1px solid #e2e8f0;overflow-y:auto;">
              <h3 style="font-size:15px;font-weight:700;margin-bottom:12px;">📋 输入需求</h3>
              <div style="margin-bottom:10px;">
                <label style="font-size:12px;color:#475569;font-weight:600;display:block;margin-bottom:4px;">🎯 业务场景</label>
                <textarea id="a-scenario" rows="2" placeholder="例：商户风控黑白名单判定；电商搜索精排；用户流失预测；金融反欺诈..." style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;resize:vertical;"></textarea>
              </div>
              <div style="margin-bottom:10px;">
                <label style="font-size:12px;color:#475569;font-weight:600;display:block;margin-bottom:4px;">📊 数据内容</label>
                <textarea id="a-data-content" rows="2" placeholder="例：用户近 30 天交易流水 + 画像（性别/年龄/职业）+ 设备指纹；图像 + 文本；日志序列..." style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;resize:vertical;"></textarea>
              </div>
              <div style="margin-bottom:10px;">
                <label style="font-size:12px;color:#475569;font-weight:600;display:block;margin-bottom:4px;">📏 数据规模</label>
                <input id="a-data-scale" placeholder="例：1000万条样本，每样本 120 维特征；图像 50万张；日志 500GB/天" style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;" />
              </div>
              <div style="margin-bottom:10px;">
                <label style="font-size:12px;color:#475569;font-weight:600;display:block;margin-bottom:4px;">🎖 业务目标</label>
                <textarea id="a-goal" rows="2" placeholder="例：AUC 提升 3% 以上；召回@Top1% 达 80%；推理 P99 < 50ms..." style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;resize:vertical;"></textarea>
              </div>
              <div style="margin-bottom:10px;">
                <label style="font-size:12px;color:#475569;font-weight:600;display:block;margin-bottom:4px;">🧪 样例数据 / Schema（可选）</label>
                <textarea id="a-sample" rows="3" placeholder="user_id, age, gmv_30d, device_os, label&#10;u1, 28, 1250.5, ios, 0&#10;u2, 42, 250.1, android, 1&#10;..." style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;font-family:monospace;resize:vertical;"></textarea>
              </div>
              <div style="margin-bottom:10px;">
                <label style="font-size:12px;color:#475569;font-weight:600;display:block;margin-bottom:4px;">🎨 偏好算法类型（多选）</label>
                <div id="a-cats" style="display:flex;flex-wrap:wrap;gap:6px;">
                  ${["traditional", "deep", "llm", "multimodal", "time_series", "graph", "frontier"].map(c => `<label style="font-size:12px;padding:3px 8px;background:#f1f5f9;border-radius:12px;cursor:pointer;"><input type="checkbox" value="${c}" class="a-cat-cb" style="margin-right:4px;" />${{traditional: "传统 ML", deep: "深度学习", llm: "大模型 LLM", multimodal: "多模态", time_series: "时序", graph: "图算法", frontier: "前沿"}[c]}</label>`).join("")}
                </div>
              </div>
              <div style="margin-bottom:14px;">
                <label style="font-size:12px;color:#475569;font-weight:600;display:block;margin-bottom:4px;">⚖️ 约束（延迟/成本/可解释性等）</label>
                <textarea id="a-constraints" rows="2" placeholder="例：在线 P99 &lt; 100ms；监管合规必须可解释；单台 A100 GPU..." style="width:100%;padding:8px;border:1px solid #e2e8f0;border-radius:6px;font-size:13px;resize:vertical;"></textarea>
              </div>
              <div style="display:flex;gap:8px;flex-wrap:wrap;">
                <button id="a-run" style="flex:1;padding:10px;background:linear-gradient(90deg,#4f46e5,#7c3aed);color:white;border:0;border-radius:8px;font-weight:700;cursor:pointer;">🪄 分析并推荐</button>
                <button id="a-example" style="padding:10px 14px;background:#f1f5f9;border:0;border-radius:8px;cursor:pointer;font-size:12px;">📝 加载示例</button>
              </div>
              <div style="margin-top:14px;border-top:1px solid #e2e8f0;padding-top:12px;">
                <h3 style="font-size:13px;font-weight:700;margin-bottom:6px;">📜 历史记录</h3>
                <div id="a-history" style="max-height:200px;overflow-y:auto;font-size:12px;"></div>
              </div>
            </div>
            <div id="a-result" style="padding:20px;overflow-y:auto;background:#f8fafc;"></div>
          </div>
        </div>
      `;
    },

    _renderConfigHTML() {
      return `
        <div style="width:min(900px, 94vw);max-height:90vh;background:white;border-radius:14px;box-shadow:0 30px 80px -20px rgba(0,0,0,0.5);display:flex;flex-direction:column;overflow:hidden;">
          <div style="padding:16px 24px;border-bottom:1px solid #e2e8f0;display:flex;align-items:center;justify-content:space-between;background:linear-gradient(135deg,#1e293b,#334155);color:white;">
            <div style="display:flex;align-items:center;gap:12px;">
              <div style="font-size:22px;">⚙️</div>
              <div style="font-weight:800;">LLM API 配置管理</div>
            </div>
            <button id="cfg-close" style="color:white;font-size:24px;background:transparent;border:0;cursor:pointer;">×</button>
          </div>
          <div style="padding:20px;overflow-y:auto;flex:1;">
            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">
              <button id="cfg-quick-hunyuan" style="padding:7px 14px;background:linear-gradient(135deg,#10b981,#059669);color:white;border:0;border-radius:6px;font-size:13px;cursor:pointer;font-weight:700;">🎁 一键添加混元（内置演示 Key）</button>
              <button id="cfg-add" style="padding:7px 14px;background:#4f46e5;color:white;border:0;border-radius:6px;font-size:13px;cursor:pointer;">+ 新增配置</button>
              <button id="cfg-import" style="padding:7px 14px;background:#f1f5f9;border:0;border-radius:6px;font-size:13px;cursor:pointer;">📥 导入 JSON</button>
              <button id="cfg-export" style="padding:7px 14px;background:#f1f5f9;border:0;border-radius:6px;font-size:13px;cursor:pointer;">📤 导出（脱敏）</button>
              <button id="cfg-export-full" style="padding:7px 14px;background:#fef3c7;color:#78350f;border:0;border-radius:6px;font-size:13px;cursor:pointer;">📤 导出（含 Key）⚠</button>
              <button id="cfg-pick-dir" style="padding:7px 14px;background:#dbeafe;color:#1e40af;border:0;border-radius:6px;font-size:13px;cursor:pointer;">📁 绑定 data 目录</button>
              <button id="cfg-help-cors" style="padding:7px 14px;background:#fce7f3;color:#be185d;border:0;border-radius:6px;font-size:13px;cursor:pointer;">🛡 CORS 跨域帮助</button>
            </div>
            <div id="cfg-dir-status" style="font-size:11px;color:#64748b;margin-bottom:10px;"></div>

            <div style="padding:12px;background:linear-gradient(135deg,#fef3c7,#fde68a);border:1px solid #f59e0b;border-radius:8px;margin-bottom:12px;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="font-size:18px;">🛡</span>
                <b style="font-size:13px;color:#78350f;">全局 CORS 代理（解决 "Failed to fetch"）</b>
              </div>
              <div style="display:flex;gap:8px;align-items:center;">
                <input id="cfg-proxy" type="text" placeholder="例：http://localhost:8787" style="flex:1;padding:6px;border:1px solid #e2e8f0;border-radius:4px;font-size:12px;font-family:monospace;" />
                <button id="cfg-proxy-save" style="padding:6px 12px;background:#f59e0b;color:white;border:0;border-radius:4px;font-size:12px;cursor:pointer;">保存</button>
                <button id="cfg-proxy-clear" style="padding:6px 12px;background:#fef3c7;color:#78350f;border:0;border-radius:4px;font-size:12px;cursor:pointer;">清除</button>
              </div>
              <div id="cfg-proxy-status" style="font-size:11px;color:#78350f;margin-top:6px;"></div>
              <div style="font-size:11px;color:#92400e;margin-top:4px;">
                💡 点击右边 "🛡 CORS 跨域帮助" 按钮查看启动代理的 3 种方式。本地代理推荐 <code>python AI_Algo_Lab/data/proxy/cors_proxy.py</code>
              </div>
            </div>

            <div id="cfg-list"></div>
          </div>
        </div>
      `;
    },

    _bindEvents() {
      // 关闭
      document.getElementById("aial-close").onclick = () => this.closeAssistant();
      document.getElementById("cfg-close").onclick = () => this.closeConfig();
      document.getElementById("aial-assistant-overlay").addEventListener("click", (e) => {
        if (e.target.id === "aial-assistant-overlay") this.closeAssistant();
      });
      document.getElementById("aial-config-overlay").addEventListener("click", (e) => {
        if (e.target.id === "aial-config-overlay") this.closeConfig();
      });

      // 引擎 badge
      this._updateEngineBadge();

      // 运行推荐
      document.getElementById("a-run").onclick = () => this._run();
      document.getElementById("a-example").onclick = () => this._loadExample();

      // 配置管理
      document.getElementById("cfg-add").onclick = () => this._editConfig(null);
      document.getElementById("cfg-quick-hunyuan").onclick = () => this._quickAddHunyuan();
      document.getElementById("cfg-import").onclick = () => this._importConfig();
      document.getElementById("cfg-export").onclick = () => this._exportConfig(false);
      document.getElementById("cfg-export-full").onclick = () => this._exportConfig(true);
      document.getElementById("cfg-pick-dir").onclick = async () => {
        const h = await this.pickDataDirectory();
        const s = document.getElementById("cfg-dir-status");
        if (h) { s.innerHTML = `✓ 已绑定到: <b>${h.name}</b> （将自动写入 data/configs 和 data/history）`; s.style.color = "#10b981"; }
        else { s.textContent = "未绑定，文件将通过浏览器下载方式保存"; }
      };

      // 代理配置
      const proxyInput = document.getElementById("cfg-proxy");
      const proxyStatus = document.getElementById("cfg-proxy-status");
      const refreshProxy = async () => {
        const p = localStorage.getItem("aial-cors-proxy") || "";
        proxyInput.value = p;
        // 探测统一服务器
        const hasUnified = await window.AIAL_LLM.probeUnifiedServer();
        if (hasUnified) {
          proxyStatus.innerHTML = `🎉 <b style="color:#10b981;">检测到统一服务器（同源代理已自动启用）</b>，无需额外配置代理！<br/>当前 URL: <code>${location.origin}/api/llm/*</code>`;
          proxyInput.disabled = true;
          proxyInput.placeholder = "（统一服务器已启用，无需填写）";
        } else {
          proxyInput.disabled = false;
          proxyStatus.innerHTML = p
            ? `✓ 代理已启用：<code>${p}</code>`
            : "⚠ 未检测到统一服务器。建议：<b>双击 AI_Algo_Lab/启动平台.bat</b>（一键解决 CORS）或配置独立代理";
        }
      };
      refreshProxy();
      document.getElementById("cfg-proxy-save").onclick = () => {
        const v = proxyInput.value.trim().replace(/\/$/, "");
        if (v) localStorage.setItem("aial-cors-proxy", v);
        else localStorage.removeItem("aial-cors-proxy");
        refreshProxy();
      };
      document.getElementById("cfg-proxy-clear").onclick = () => {
        localStorage.removeItem("aial-cors-proxy");
        refreshProxy();
      };
      document.getElementById("cfg-help-cors").onclick = () => this._showCorsHelp();
    },

    _showCorsHelp() {
      const html = `
        <div id="cors-help-modal" style="position:fixed;inset:0;background:rgba(0,0,0,0.6);z-index:130;display:flex;align-items:center;justify-content:center;">
          <div style="background:white;padding:24px;border-radius:12px;width:min(780px,94vw);max-height:92vh;overflow-y:auto;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
              <h3 style="font-weight:700;font-size:17px;">🛡 解决 "Failed to fetch" 跨域问题</h3>
              <button onclick="document.getElementById('cors-help-modal').remove()" style="font-size:22px;background:0;border:0;cursor:pointer;">×</button>
            </div>
            <div style="background:#fef2f2;border-left:4px solid #ef4444;padding:10px 14px;border-radius:6px;margin-bottom:14px;font-size:13px;color:#991b1b;">
              <b>根因</b>：浏览器同源策略禁止 JS 直接请求跨域 LLM API。<b>4 种方案</b>任选其一：
            </div>

            <h4 style="font-weight:700;font-size:15px;margin:10px 0 6px;color:#059669;">⭐ 方案 1：一键启动（零配置，强烈推荐）</h4>
            <div style="background:linear-gradient(135deg,#f0fdf4,#dcfce7);border:2px solid #10b981;padding:12px;border-radius:8px;font-size:12px;">
              <div style="font-weight:700;color:#065f46;margin-bottom:6px;">用 Python 启动 HTML + LLM 代理二合一的统一服务器：</div>
              <pre style="background:#fff;padding:8px;border-radius:4px;margin:4px 0;font-size:11px;overflow-x:auto;line-height:1.5;">cd &lt;项目根目录&gt;\\AI_Algo_Lab
python start.py</pre>
              <div style="margin-top:6px;"><b>或双击</b>：<code style="background:#fff;padding:2px 4px;">AI_Algo_Lab\\启动平台.bat</code></div>
              <div style="margin-top:8px;color:#047857;">
                ✨ <b>优点</b>：
                <ul style="padding-left:20px;margin:4px 0;">
                  <li>浏览器和 API 代理在<b>同源</b>，天然无 CORS 问题</li>
                  <li><b>自动检测</b>并启用 — 无需在配置面板填任何代理 URL</li>
                  <li>自动打开浏览器 — 一步到位</li>
                  <li>零外部依赖（仅 Python 3.6+ 标准库）</li>
                </ul>
              </div>
            </div>

            <h4 style="font-weight:700;font-size:15px;margin:14px 0 6px;color:#2563eb;">⚡ 方案 2：使用原生支持浏览器直连的 Provider（无需任何服务器）</h4>
            <div style="background:#eff6ff;border:1px solid #93c5fd;padding:12px;border-radius:6px;font-size:12px;">
              <div style="color:#1e40af;margin-bottom:6px;"><b>以下 3 个 Provider 原生支持浏览器调用，直接用即可：</b></div>
              <table style="width:100%;font-size:12px;">
                <tr><td style="padding:4px 8px;background:#fff;border-radius:4px;"><b>🏠 Ollama 本地</b></td><td style="padding:4px 8px;">endpoint: <code>http://localhost:11434/v1</code><br/>前置条件：安装 Ollama 并设置环境变量 <code>OLLAMA_ORIGINS=*</code></td></tr>
                <tr><td style="padding:4px 8px;background:#fff;border-radius:4px;"><b>🌐 Google Gemini</b></td><td style="padding:4px 8px;">使用 URL 参数 key 认证（本平台已适配），直接可用</td></tr>
                <tr><td style="padding:4px 8px;background:#fff;border-radius:4px;"><b>🤖 Anthropic Claude</b></td><td style="padding:4px 8px;">本平台已自动添加 <code>anthropic-dangerous-direct-browser-access: true</code> 头，直接可用</td></tr>
              </table>
            </div>

            <h4 style="font-weight:700;font-size:15px;margin:14px 0 6px;color:#7c3aed;">🛡 方案 3：独立 CORS 代理</h4>
            <div style="background:#f5f3ff;border:1px solid #c4b5fd;padding:12px;border-radius:6px;font-size:12px;">
              <pre style="background:#fff;padding:8px;border-radius:4px;margin:4px 0;font-size:11px;overflow-x:auto;">python AI_Algo_Lab\\data\\proxy\\cors_proxy.py</pre>
              然后在上方"全局 CORS 代理"输入框填：<code style="background:#fff;padding:2px 4px;">http://localhost:8787</code>
            </div>

            <h4 style="font-weight:700;font-size:15px;margin:14px 0 6px;color:#ea580c;">🔧 方案 4：Chrome 禁用 CORS 启动（仅调试）</h4>
            <div style="background:#fff7ed;border:1px solid #fdba74;padding:10px;border-radius:6px;font-size:12px;">
              <b>Windows</b>：双击 <code>AI_Algo_Lab/data/launch_chrome_no_cors.bat</code><br/>
              <b>macOS/Linux</b>：
              <pre style="background:#fff;padding:6px;border-radius:4px;margin:4px 0;font-size:11px;">open -a "Google Chrome" --args --user-data-dir=/tmp/chrome_dev --disable-web-security</pre>
              ⚠ 降低浏览器安全，仅本地调试用。
            </div>

            <div style="margin-top:14px;padding:10px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;font-size:12px;color:#475569;">
              💡 <b>对比参考</b>：<code>ai_task_manager</code> 是 Python 桌面应用用 urllib 直连，不受同源策略约束。<br/>
              本平台是纯浏览器 Web 应用，方案 1 实质上是把它变成"迷你桌面应用"（Python 托管页面 + 代理），达到等价效果。
            </div>

            <div style="margin-top:14px;text-align:right;">
              <button onclick="document.getElementById('cors-help-modal').remove()" style="padding:8px 16px;background:#4f46e5;color:white;border:0;border-radius:6px;cursor:pointer;">知道了</button>
            </div>
          </div>
        </div>
      `;
      document.body.insertAdjacentHTML("beforeend", html);
    },

    async _updateEngineBadge() {
      const cfg = window.AIAL_LLM?.getActive();
      const badge = document.getElementById("aial-engine-badge");
      if (!badge) return;
      // 异步探测统一服务器
      const hasUnified = await window.AIAL_LLM.probeUnifiedServer();
      const serverTag = hasUnified ? "🎉 统一服务器" : "";
      if (!cfg) {
        badge.textContent = `🎯 规则引擎（未配置 LLM）${serverTag ? " · " + serverTag : ""}`;
        return;
      }
      const providers = window.AIAL_LLM.PROVIDERS;
      const p = providers[cfg.provider];
      const hasKey = cfg.apiKey && cfg.apiKey.trim();
      const hasDemo = p && p.demo_key && cfg.useDemoKey !== false;
      let main;
      if (hasKey) main = `⚡ ${cfg.name || cfg.provider} / ${cfg.model || "?"}`;
      else if (hasDemo) main = `🎁 ${cfg.name || cfg.provider} (演示 Key) / ${cfg.model || "?"}`;
      else main = "🎯 规则引擎（未配置 Key）";
      badge.textContent = serverTag ? main + " · " + serverTag : main;
    },

    _loadExample() {
      document.getElementById("a-scenario").value = "电商平台用户下单转化率 (CVR) 预测，用于广告系统出价";
      document.getElementById("a-data-content").value = "用户近 30 天点击/加购/下单行为序列、用户画像（年龄/性别/地域）、商品特征（类目/价格/销量）、上下文（时段/设备/页面位置）";
      document.getElementById("a-data-scale").value = "日均曝光 2 亿次，每样本约 150 个特征（含 30 个类别特征，高基数 10w+）";
      document.getElementById("a-goal").value = "CVR AUC 提升 3%+，在线 P99 &lt; 30ms，支持小时级模型更新";
      document.getElementById("a-sample").value = "user_id, item_id, click_seq, purchase_seq, age_bin, gender, price_bin, category_l2, hour, position, label_cvr\nu1, i101, [i99,i98], [], 3, M, 2, 8015, 14, 2, 0\nu2, i205, [i201], [i12], 5, F, 4, 8023, 21, 1, 1";
      document.getElementById("a-constraints").value = "在线 P99 &lt; 30ms；小时级增量训练；模型体积 &lt; 500MB；有 GPU 资源";
      document.querySelectorAll(".a-cat-cb").forEach(cb => cb.checked = ["deep", "traditional"].includes(cb.value));
    },

    async _run() {
      const input = this._collectInput();
      const btn = document.getElementById("a-run");
      const result = document.getElementById("a-result");
      btn.disabled = true; btn.textContent = "🔄 分析中...";
      result.innerHTML = `<div style="text-align:center;padding:60px;color:#64748b;"><div style="font-size:40px;">⏳</div><div style="margin-top:10px;">正在调用算法推理...</div></div>`;
      try {
        const reco = await this.recommend(input);
        const record = {
          id: "reco_" + Date.now(),
          timestamp: new Date().toISOString(),
          input, output: reco,
        };
        this.saveHistory(record);
        this._renderResult(reco, record);
        this._refreshHistoryList();
        // 自动保存到 data/history/
        const filename = `${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
        this.saveFileToData("history", filename, JSON.stringify(record, null, 2)).catch(console.warn);
      } catch (e) {
        result.innerHTML = `<div style="color:#ef4444;padding:30px;">❌ 错误：${e.message}</div>`;
      } finally {
        btn.disabled = false; btn.textContent = "🪄 分析并推荐";
      }
    },

    _collectInput() {
      const cats = [...document.querySelectorAll(".a-cat-cb:checked")].map(x => x.value);
      return {
        scenario: document.getElementById("a-scenario").value,
        data_content: document.getElementById("a-data-content").value,
        data_scale: document.getElementById("a-data-scale").value,
        goal: document.getElementById("a-goal").value,
        sample_data: document.getElementById("a-sample").value,
        categories: cats,
        constraints: document.getElementById("a-constraints").value,
      };
    },

    _renderResult(reco, record) {
      const el = document.getElementById("a-result");
      const tierColor = { primary: "#10b981", secondary: "#f59e0b", alternative: "#94a3b8" };
      const tierLabel = { primary: "🎯 主推方案", secondary: "🔸 备选方案", alternative: "⚪ 替代方案" };

      const recos = (reco.recommendations || []).map(r => {
        const algo = MCH.getById(r.algorithm_id);
        const color = tierColor[r.tier] || "#4f46e5";
        return `
          <div style="padding:12px;background:white;border-left:4px solid ${color};border-radius:6px;margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <div>
                <span style="font-size:11px;color:${color};font-weight:700;">${tierLabel[r.tier] || r.tier}</span>
                <a href="#/${r.algorithm_id}" onclick="AIAL_ASSISTANT.closeAssistant()" style="display:block;font-size:15px;font-weight:700;color:#1e293b;text-decoration:none;margin-top:2px;">${algo?.icon || ""} ${r.algorithm_name}</a>
                <span style="font-size:11px;color:#64748b;">${algo?.cat_name || ""} · ${algo?.en || ""}</span>
              </div>
              <div style="text-align:right;">
                <div style="font-size:22px;font-weight:800;color:${color};">${r.fit_score || "-"}</div>
                <div style="font-size:10px;color:#64748b;">契合度</div>
              </div>
            </div>
            <div style="font-size:12px;color:#475569;margin-top:6px;line-height:1.5;">${r.reason || ""}</div>
          </div>
        `;
      }).join("");

      el.innerHTML = `
        <div style="display:flex;gap:8px;margin-bottom:14px;justify-content:space-between;align-items:center;flex-wrap:wrap;">
          <div>
            <div style="font-size:11px;color:#64748b;">${reco._engine || "-"} · ${record.timestamp.slice(0, 19).replace("T", " ")}</div>
            <div style="font-size:15px;font-weight:700;color:#1e293b;margin-top:2px;">${reco.summary || ""}</div>
          </div>
          <div style="display:flex;gap:6px;">
            <button onclick="AIAL_ASSISTANT._exportReco('${record.id}', 'json')" style="padding:6px 12px;background:#4f46e5;color:white;border:0;border-radius:6px;font-size:12px;cursor:pointer;">📄 导出 JSON</button>
            <button onclick="AIAL_ASSISTANT._exportReco('${record.id}', 'md')" style="padding:6px 12px;background:#10b981;color:white;border:0;border-radius:6px;font-size:12px;cursor:pointer;">📝 导出 Markdown</button>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:auto auto auto;gap:8px;margin-bottom:14px;">
          <div style="background:white;padding:10px;border-radius:8px;"><div style="font-size:10px;color:#64748b;">难度</div><div style="font-size:14px;font-weight:700;">${reco.difficulty || "-"}</div></div>
          <div style="background:white;padding:10px;border-radius:8px;"><div style="font-size:10px;color:#64748b;">工作量</div><div style="font-size:14px;font-weight:700;">${reco.estimated_effort || "-"}</div></div>
          <div style="background:white;padding:10px;border-radius:8px;"><div style="font-size:10px;color:#64748b;">推荐数</div><div style="font-size:14px;font-weight:700;">${(reco.recommendations || []).length}</div></div>
        </div>

        <h3 style="font-size:14px;font-weight:700;margin:16px 0 8px;">🎯 算法推荐</h3>
        ${recos}

        <h3 style="font-size:14px;font-weight:700;margin:16px 0 8px;">🔨 实施流程</h3>
        <ol style="padding-left:20px;font-size:13px;color:#475569;">
          ${(reco.pipeline || []).map(p => `<li style="margin-bottom:6px;"><b>${p.name || ""}</b>：${p.detail || ""}</li>`).join("")}
        </ol>

        ${reco.data_preparation?.length ? `
          <h3 style="font-size:14px;font-weight:700;margin:16px 0 8px;">📦 数据准备</h3>
          <ul style="padding-left:20px;font-size:13px;color:#475569;">${reco.data_preparation.map(x => `<li>${x}</li>`).join("")}</ul>
        ` : ""}

        ${reco.evaluation_metrics?.length ? `
          <h3 style="font-size:14px;font-weight:700;margin:16px 0 8px;">📏 评估指标</h3>
          <div style="display:flex;gap:6px;flex-wrap:wrap;">${reco.evaluation_metrics.map(x => `<span style="padding:3px 8px;background:#e0e7ff;color:#3730a3;border-radius:4px;font-size:12px;">${x}</span>`).join("")}</div>
        ` : ""}

        ${reco.deployment_notes?.length ? `
          <h3 style="font-size:14px;font-weight:700;margin:16px 0 8px;">🚀 部署要点</h3>
          <ul style="padding-left:20px;font-size:13px;color:#475569;">${reco.deployment_notes.map(x => `<li>${x}</li>`).join("")}</ul>
        ` : ""}

        ${reco.risks?.length ? `
          <h3 style="font-size:14px;font-weight:700;margin:16px 0 8px;">⚠️ 潜在风险</h3>
          <ul style="padding-left:20px;font-size:13px;color:#475569;">${reco.risks.map(x => `<li>${x}</li>`).join("")}</ul>
        ` : ""}

        <h3 style="font-size:14px;font-weight:700;margin:16px 0 8px;">📚 详情查看</h3>
        <div style="display:flex;gap:6px;flex-wrap:wrap;">${(reco.reading_list || []).map(p => `<a href="${p}" onclick="AIAL_ASSISTANT.closeAssistant()" style="padding:4px 10px;background:#4f46e5;color:white;border-radius:4px;font-size:12px;text-decoration:none;">${p}</a>`).join("")}</div>
      `;
    },

    _refreshHistoryList() {
      const list = this.listHistory();
      const el = document.getElementById("a-history");
      if (!el) return;
      if (list.length === 0) {
        el.innerHTML = `<div style="color:#94a3b8;font-size:11px;">暂无历史记录</div>`;
        return;
      }
      el.innerHTML = list.slice(0, 10).map(r => `
        <div style="padding:6px 8px;border-bottom:1px dashed #e2e8f0;display:flex;justify-content:space-between;gap:6px;align-items:center;">
          <div style="flex:1;min-width:0;cursor:pointer;" onclick="AIAL_ASSISTANT._viewHistory('${r.id}')">
            <div style="font-size:11px;color:#64748b;">${new Date(r.timestamp).toLocaleString()}</div>
            <div style="font-size:11px;color:#1e293b;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${r.input.scenario || "(无场景)"}</div>
          </div>
          <button onclick="AIAL_ASSISTANT._exportReco('${r.id}', 'md')" style="padding:2px 6px;background:#10b981;color:white;border:0;border-radius:4px;font-size:10px;cursor:pointer;">导出</button>
        </div>
      `).join("") + (list.length > 10 ? `<div style="font-size:10px;color:#94a3b8;padding:4px;">共 ${list.length} 条，显示最新 10 条</div>` : "");
    },

    _viewHistory(id) {
      const r = this.listHistory().find(x => x.id === id);
      if (!r) return;
      ["scenario", "data-content", "data-scale", "goal", "sample", "constraints"].forEach((f, i) => {
        const key = ["scenario", "data_content", "data_scale", "goal", "sample_data", "constraints"][i];
        const el = document.getElementById("a-" + f);
        if (el) el.value = r.input[key] || "";
      });
      document.querySelectorAll(".a-cat-cb").forEach(cb => {
        cb.checked = (r.input.categories || []).includes(cb.value);
      });
      this._renderResult(r.output, r);
    },

    async _exportReco(id, format) {
      const r = this.listHistory().find(x => x.id === id);
      if (!r) return;
      const date = r.timestamp.replace(/[:.]/g, "-").slice(0, 19);
      let content, filename;
      if (format === "json") {
        content = JSON.stringify(r, null, 2);
        filename = `reco-${date}.json`;
      } else {
        content = this._toMarkdown(r);
        filename = `reco-${date}.md`;
      }
      const result = await this.saveFileToData("history", filename, content);
      alert(`✓ 已保存到: ${result.path}`);
    },

    _toMarkdown(r) {
      const reco = r.output;
      const recos = (reco.recommendations || []).map(x => `
### ${{primary: "🎯", secondary: "🔸", alternative: "⚪"}[x.tier] || ""} ${x.algorithm_name} (契合度 ${x.fit_score || "-"})
- **类型**：${x.tier}
- **理由**：${x.reason}
- **详情**：[#/${x.algorithm_id}](#/${x.algorithm_id})
`).join("\n");

      return `# AI 算法选型方案

**生成时间**：${r.timestamp}
**引擎**：${reco._engine || "-"}

## 📋 输入需求

- **场景**：${r.input.scenario || "-"}
- **数据内容**：${r.input.data_content || "-"}
- **数据规模**：${r.input.data_scale || "-"}
- **业务目标**：${r.input.goal || "-"}
- **偏好类型**：${(r.input.categories || []).join(", ") || "-"}
- **约束**：${r.input.constraints || "-"}

### 样例数据
\`\`\`
${r.input.sample_data || "(无)"}
\`\`\`

## 📊 推荐总览

**一句话总结**：${reco.summary || "-"}
**难度**：${reco.difficulty} · **工作量**：${reco.estimated_effort}

## 🎯 算法推荐
${recos}

## 🔨 实施流程

${(reco.pipeline || []).map(p => `${p.step}. **${p.name}**：${p.detail}`).join("\n")}

## 📦 数据准备
${(reco.data_preparation || []).map(x => `- ${x}`).join("\n")}

## 📏 评估指标
${(reco.evaluation_metrics || []).map(x => `- ${x}`).join("\n")}

## 🚀 部署要点
${(reco.deployment_notes || []).map(x => `- ${x}`).join("\n")}

## ⚠️ 潜在风险
${(reco.risks || []).map(x => `- ${x}`).join("\n")}

## 📚 分类备选

${Object.entries(reco.alternatives_by_category || {}).map(([k, v]) => `- **${k}**：${v.join(", ") || "-"}`).join("\n")}
`;
    },

    // ---------------- 配置 UI ----------------
    _refreshConfigList() {
      const list = window.AIAL_LLM.list();
      const active = window.AIAL_LLM.getActive();
      const el = document.getElementById("cfg-list");
      if (!el) return;
      if (list.length === 0) {
        el.innerHTML = `<div style="padding:40px;text-align:center;color:#94a3b8;">
          <div style="font-size:40px;">⚙️</div>
          <div style="margin-top:10px;">暂无 LLM 配置。点击 "新增配置" 开始。</div>
          <div style="margin-top:4px;font-size:12px;">未配置时会使用内置<b>规则引擎</b>，也能给出合理推荐。</div>
        </div>`;
        return;
      }
      el.innerHTML = list.map(c => {
        const isActive = active && active.id === c.id;
        const provider = window.AIAL_LLM.PROVIDERS[c.provider] || {};
        const usingDemo = !c.apiKey && provider.demo_key && c.useDemoKey !== false;
        const keyDisplay = c.apiKey ? '●●●●' + c.apiKey.slice(-4) : (usingDemo ? '<span style="color:#10b981;">🎁 演示 Key</span>' : '<span style="color:#ef4444;">(无)</span>');
        const extras = [];
        if (c.extra_body?.enable_enhancement) extras.push('🌐 enhancement');
        if (c.extra_body && Object.keys(c.extra_body).length > (c.extra_body.enable_enhancement !== undefined ? 1 : 0)) extras.push('⚙ 自定义参数');
        return `
          <div style="background:white;border:${isActive ? '2px solid #4f46e5' : '1px solid #e2e8f0'};border-radius:8px;padding:12px;margin-bottom:8px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:10px;">
              <div style="flex:1;min-width:0;">
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
                  ${isActive ? '<span style="font-size:10px;padding:2px 6px;background:#4f46e5;color:white;border-radius:3px;">● ACTIVE</span>' : ''}
                  <b style="font-size:14px;">${c.name || c.provider}</b>
                  <span style="font-size:11px;color:#64748b;">${c.provider}</span>
                  ${extras.map(e => `<span style="font-size:10px;padding:2px 6px;background:#fef3c7;color:#78350f;border-radius:3px;">${e}</span>`).join('')}
                </div>
                <div style="font-size:11px;color:#64748b;margin-top:4px;font-family:monospace;">${c.endpoint || '-'}</div>
                <div style="font-size:11px;color:#475569;margin-top:2px;">Model: <b>${c.model || '-'}</b> · Key: ${keyDisplay}</div>
              </div>
              <div style="display:flex;gap:4px;">
                ${isActive ? '' : `<button onclick="AIAL_LLM.setActive('${c.id}'); AIAL_ASSISTANT._refreshConfigList(); AIAL_ASSISTANT._updateEngineBadge()" style="padding:4px 8px;background:#10b981;color:white;border:0;border-radius:4px;font-size:11px;cursor:pointer;">设默认</button>`}
                <button onclick="AIAL_ASSISTANT._testConfig('${c.id}')" style="padding:4px 8px;background:#eef2ff;color:#4f46e5;border:0;border-radius:4px;font-size:11px;cursor:pointer;">测连接</button>
                <button onclick="AIAL_ASSISTANT._editConfig('${c.id}')" style="padding:4px 8px;background:#f1f5f9;border:0;border-radius:4px;font-size:11px;cursor:pointer;">编辑</button>
                <button onclick="AIAL_ASSISTANT._removeConfig('${c.id}')" style="padding:4px 8px;background:#fee2e2;color:#dc2626;border:0;border-radius:4px;font-size:11px;cursor:pointer;">删除</button>
              </div>
            </div>
          </div>
        `;
      }).join("");
    },

    _editConfig(id) {
      const list = window.AIAL_LLM.list();
      const cfg = id ? list.find(c => c.id === id) : {
        id: "cfg_" + Date.now(),
        provider: "hunyuan",   // 默认混元（有内置演示 Key，零配置即用）
        name: "",
        endpoint: "",
        model: "",
        apiKey: "",
        useDemoKey: true,
        extra_body: null,
      };
      const provider = window.AIAL_LLM.PROVIDERS[cfg.provider] || window.AIAL_LLM.PROVIDERS.hunyuan;
      if (!cfg.endpoint) cfg.endpoint = provider.endpoint;
      if (!cfg.model) cfg.model = provider.models[0];

      const modalHTML = `
        <div id="cfg-edit-modal" style="position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:120;display:flex;align-items:center;justify-content:center;">
          <div style="background:white;padding:24px;border-radius:12px;width:min(600px, 92vw);max-height:92vh;overflow-y:auto;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
              <h3 style="font-weight:700;font-size:16px;">${id ? '编辑' : '新增'} LLM 配置</h3>
              <button onclick="document.getElementById('cfg-edit-modal').remove()" style="font-size:22px;background:0;border:0;cursor:pointer;">×</button>
            </div>
            <label style="font-size:12px;font-weight:600;">配置名（自定义）</label>
            <input id="ce-name" value="${cfg.name || ''}" placeholder="例：我的混元 / 公司 DeepSeek" style="width:100%;padding:7px;border:1px solid #e2e8f0;border-radius:6px;margin-bottom:10px;" />

            <label style="font-size:12px;font-weight:600;">Provider</label>
            <select id="ce-provider" style="width:100%;padding:7px;border:1px solid #e2e8f0;border-radius:6px;margin-bottom:10px;">
              ${Object.entries(window.AIAL_LLM.PROVIDERS).map(([k, p]) => `<option value="${k}" ${cfg.provider === k ? 'selected' : ''}>${p.name}</option>`).join("")}
            </select>

            <div id="ce-provider-info" style="margin-bottom:10px;"></div>

            <label style="font-size:12px;font-weight:600;">Endpoint</label>
            <input id="ce-endpoint" value="${cfg.endpoint || ''}" style="width:100%;padding:7px;border:1px solid #e2e8f0;border-radius:6px;margin-bottom:10px;font-family:monospace;font-size:12px;" />

            <label style="font-size:12px;font-weight:600;">Model</label>
            <input id="ce-model" list="ce-model-list" value="${cfg.model || ''}" style="width:100%;padding:7px;border:1px solid #e2e8f0;border-radius:6px;margin-bottom:10px;" />
            <datalist id="ce-model-list">${provider.models.map(m => `<option value="${m}">`).join("")}</datalist>

            <label style="font-size:12px;font-weight:600;">API Key <span id="ce-key-label-hint" style="font-weight:400;color:#64748b;"></span></label>
            <input id="ce-key" type="password" value="${cfg.apiKey || ''}" placeholder="sk-..." style="width:100%;padding:7px;border:1px solid #e2e8f0;border-radius:6px;margin-bottom:4px;font-family:monospace;" />
            <div id="ce-demo-box" style="display:none;margin-bottom:10px;padding:10px;background:linear-gradient(135deg,#ecfdf5,#d1fae5);border:1px solid #6ee7b7;border-radius:6px;font-size:12px;">
              <label style="display:flex;align-items:center;gap:8px;cursor:pointer;">
                <input type="checkbox" id="ce-use-demo" ${cfg.useDemoKey !== false ? 'checked' : ''} />
                <span><b style="color:#065f46;">🎁 使用内置演示 Key（零配置即用）</b></span>
              </label>
              <div style="margin-top:6px;color:#047857;font-size:11px;">
                本项目预置了腾讯混元演示 Key，勾选后无需自备 Key 即可直接使用。<br/>
                ⚠️ 演示 Key 可能有速率限制，正式使用建议申请自己的 Key。
              </div>
            </div>

            <div id="ce-hunyuan-opts" style="display:none;margin-bottom:10px;padding:10px;background:#fff7ed;border:1px solid #fdba74;border-radius:6px;">
              <label style="display:flex;align-items:center;gap:8px;cursor:pointer;font-size:12px;">
                <input type="checkbox" id="ce-enhancement" checked />
                <span><b style="color:#9a3412;">🌐 启用 enable_enhancement（混元内容增强）</b></span>
              </label>
              <div style="margin-top:4px;color:#7c2d12;font-size:11px;">开启后混元会联网检索并调用工具增强（官方特性）。</div>
            </div>

            <div style="display:flex;gap:8px;justify-content:space-between;align-items:center;">
              <div>
                <a id="ce-docs" target="_blank" style="font-size:11px;color:#4f46e5;text-decoration:underline;display:none;">📄 官方文档</a>
                <a id="ce-console" target="_blank" style="font-size:11px;color:#4f46e5;text-decoration:underline;margin-left:8px;display:none;">🔗 控制台申请 Key</a>
              </div>
              <div style="display:flex;gap:8px;">
                <button onclick="document.getElementById('cfg-edit-modal').remove()" style="padding:7px 14px;background:#f1f5f9;border:0;border-radius:6px;cursor:pointer;">取消</button>
                <button id="ce-save" style="padding:7px 14px;background:#4f46e5;color:white;border:0;border-radius:6px;cursor:pointer;">保存</button>
              </div>
            </div>
            <div style="margin-top:10px;padding:10px;background:#f1f5f9;border-radius:6px;font-size:11px;color:#64748b;">
              ⚠️ API Key 保存在浏览器 localStorage，请勿在公共电脑使用。可通过 "📤 导出（含 Key）" 备份。
            </div>
          </div>
        </div>
      `;
      document.body.insertAdjacentHTML("beforeend", modalHTML);

      const updateProviderUI = (providerKey) => {
        const p = window.AIAL_LLM.PROVIDERS[providerKey];
        if (!p) return;
        // 自动填充
        document.getElementById("ce-endpoint").value = p.endpoint;
        document.getElementById("ce-model").value = p.models[0];
        const dl = document.getElementById("ce-model-list");
        dl.innerHTML = p.models.map(m => `<option value="${m}">`).join("");
        // 演示 Key 面板
        const demoBox = document.getElementById("ce-demo-box");
        const keyHint = document.getElementById("ce-key-label-hint");
        if (p.demo_key) {
          demoBox.style.display = "block";
          keyHint.textContent = " （可留空，将使用演示 Key）";
        } else {
          demoBox.style.display = "none";
          keyHint.textContent = "";
        }
        // 混元专属选项
        document.getElementById("ce-hunyuan-opts").style.display = providerKey === "hunyuan" ? "block" : "none";
        // 文档链接
        const docs = document.getElementById("ce-docs");
        const con = document.getElementById("ce-console");
        if (p.docs) { docs.href = p.docs; docs.style.display = "inline"; } else { docs.style.display = "none"; }
        if (p.console) { con.href = p.console; con.style.display = "inline"; } else { con.style.display = "none"; }
      };
      updateProviderUI(cfg.provider);

      document.getElementById("ce-provider").onchange = (e) => updateProviderUI(e.target.value);
      document.getElementById("ce-save").onclick = () => {
        const providerKey = document.getElementById("ce-provider").value;
        const p = window.AIAL_LLM.PROVIDERS[providerKey];
        const save = {
          id: cfg.id,
          name: document.getElementById("ce-name").value || p?.name || providerKey,
          provider: providerKey,
          endpoint: document.getElementById("ce-endpoint").value.trim(),
          model: document.getElementById("ce-model").value.trim(),
          apiKey: document.getElementById("ce-key").value.trim(),
          useDemoKey: p?.demo_key ? document.getElementById("ce-use-demo").checked : false,
        };
        // 混元专属：enable_enhancement
        if (providerKey === "hunyuan") {
          const enabled = document.getElementById("ce-enhancement").checked;
          save.extra_body = { enable_enhancement: enabled };
        }
        if (!save.endpoint || !save.model) { alert("endpoint / model 不能为空"); return; }
        // 校验：无 Key 且不用演示 Key 时警告
        if (!save.apiKey && !save.useDemoKey) {
          if (!confirm("未填 API Key 且未启用演示 Key，该配置将无法调用 LLM（仅能走规则引擎）。确认保存？")) return;
        }
        window.AIAL_LLM.addOrUpdate(save);
        if (!window.AIAL_LLM.getActive()) window.AIAL_LLM.setActive(save.id);
        document.getElementById("cfg-edit-modal").remove();
        this._refreshConfigList();
        this._updateEngineBadge();
      };
    },

    _removeConfig(id) {
      if (!confirm("确认删除该配置？")) return;
      window.AIAL_LLM.remove(id);
      this._refreshConfigList();
      this._updateEngineBadge();
    },

    _quickAddHunyuan() {
      const p = window.AIAL_LLM.PROVIDERS.hunyuan;
      const id = "cfg_hunyuan_" + Date.now();
      const cfg = {
        id,
        name: "腾讯混元（演示 Key）",
        provider: "hunyuan",
        endpoint: p.endpoint,
        model: p.models[0],
        apiKey: "",
        useDemoKey: true,
        extra_body: { enable_enhancement: true },
      };
      window.AIAL_LLM.addOrUpdate(cfg);
      window.AIAL_LLM.setActive(id);
      this._refreshConfigList();
      this._updateEngineBadge();
      alert("✓ 已添加腾讯混元配置并设为默认！\n可直接使用（内置演示 Key），建议后续替换为自己的 Key。");
    },

    async _testConfig(id) {
      const cfg = window.AIAL_LLM.list().find(c => c.id === id);
      if (!cfg) return;
      // 显示等待弹窗
      this._showModal(`<div style="padding:20px;text-align:center;"><div style="font-size:32px;">⏳</div><div style="margin-top:10px;">正在测试连接...</div><div style="font-size:11px;color:#64748b;margin-top:6px;">${cfg.name || cfg.provider}</div></div>`, "测试连接");
      const r = await window.AIAL_LLM.testConnection(cfg);
      if (r.ok) {
        this._showModal(`
          <div style="padding:16px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
              <span style="font-size:24px;">✅</span>
              <b style="font-size:16px;color:#10b981;">连接成功！</b>
            </div>
            <div style="font-size:13px;color:#475569;margin-bottom:8px;">模型回复：</div>
            <div style="padding:10px;background:#f0fdf4;border-left:3px solid #10b981;border-radius:4px;font-family:monospace;font-size:13px;">${r.response}</div>
          </div>
        `, "连接成功");
      } else {
        const isCors = /CORS|Failed to fetch|浏览器/i.test(r.error);
        this._showModal(`
          <div style="padding:16px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
              <span style="font-size:24px;">❌</span>
              <b style="font-size:16px;color:#ef4444;">连接失败</b>
            </div>
            <pre style="white-space:pre-wrap;font-size:12px;color:#475569;background:#fef2f2;padding:10px;border-radius:4px;border-left:3px solid #ef4444;max-height:400px;overflow-y:auto;font-family:-apple-system,Segoe UI,sans-serif;">${r.error}</pre>
            ${isCors ? `<div style="margin-top:10px;text-align:right;"><button onclick="document.getElementById('aial-modal').remove();AIAL_ASSISTANT._showCorsHelp()" style="padding:6px 14px;background:#ec4899;color:white;border:0;border-radius:6px;cursor:pointer;font-size:12px;">🛡 查看 CORS 解决方案</button></div>` : ""}
          </div>
        `, "连接失败");
      }
    },

    _showModal(html, title) {
      const existing = document.getElementById("aial-modal");
      if (existing) existing.remove();
      const modal = document.createElement("div");
      modal.id = "aial-modal";
      modal.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,0.5);z-index:130;display:flex;align-items:center;justify-content:center;";
      modal.innerHTML = `
        <div style="background:white;border-radius:10px;width:min(560px,92vw);max-height:85vh;overflow-y:auto;">
          <div style="display:flex;justify-content:space-between;align-items:center;padding:10px 14px;border-bottom:1px solid #e2e8f0;">
            <b style="font-size:13px;">${title || "提示"}</b>
            <button onclick="document.getElementById('aial-modal').remove()" style="background:0;border:0;font-size:18px;cursor:pointer;">×</button>
          </div>
          ${html}
        </div>`;
      modal.addEventListener("click", (e) => { if (e.target === modal) modal.remove(); });
      document.body.appendChild(modal);
    },

    async _exportConfig(includeKeys) {
      const content = includeKeys ? window.AIAL_LLM.exportAllWithKeys() : window.AIAL_LLM.exportAll();
      const filename = `llm-configs-${new Date().toISOString().slice(0, 10)}${includeKeys ? "-with-keys" : ""}.json`;
      const result = await this.saveFileToData("configs", filename, content);
      alert(`✓ 已保存到: ${result.path}`);
    },

    _importConfig() {
      const input = document.createElement("input");
      input.type = "file"; input.accept = ".json";
      input.onchange = async (e) => {
        const file = e.target.files[0];
        if (!file) return;
        const text = await file.text();
        try {
          window.AIAL_LLM.importAll(text);
          alert(`✓ 导入成功（${file.name}）`);
          this._refreshConfigList();
          this._updateEngineBadge();
        } catch (err) { alert("导入失败：" + err.message); }
      };
      input.click();
    },
  };

  // 启动时注入 UI
  window.addEventListener("DOMContentLoaded", () => {
    setTimeout(() => Assistant.injectUI(), 20);
  });
})();
