/* ============================================================================
 * i18n.js — Lightweight i18n framework
 * - Zh (default) / En toggle, persisted in localStorage("mch-lang")
 * - MCH.t(key)   : returns localized string for a given dictionary key
 * - MCH.tn(route): returns localized sidebar nav label for a given route
 * - MCH.tcat(key): returns localized category name (sidebar group / overview)
 * - MCH.i18n.applyDOM(root) : walks DOM, replaces:
 *     [data-i18n="key"]            -> textContent = t(key)
 *     [data-i18n-html="key"]       -> innerHTML   = t(key)
 *     [data-i18n-title="key"]      -> title       = t(key)
 *     [data-i18n-aria="key"]       -> aria-label  = t(key)
 *     [data-i18n-placeholder="key"]-> placeholder = t(key)
 *     [data-route]                 -> patches the visible nav text portion
 * - MCH.i18n.setLang(lang) : persists, applies DOM, re-renders current route.
 * ========================================================================== */
window.MCH = window.MCH || { modules: {}, charts: [] };

(function () {
  const LANG_KEY = "mch-lang";
  const FALLBACK = "zh";

  /* ───────── Shell UI dictionary ───────── */
  const DICT = {
    /* Header */
    "header.title":         { zh: "AI 算法研发可视化平台",                en: "AI Algo Visualization Lab" },
    "header.subtitle":      { zh: "Basic ML · Neural Networks · Graph · Loss · LLM · 案例研究 & 对比",
                              en: "Basic ML · Neural Networks · Graph · Loss · LLM · Case Study & Comparison" },
    "header.search_label":  { zh: "搜索算法 / 模块",                      en: "Search algorithms / modules" },
    "header.compare":       { zh: "⚖️ 对比",                              en: "⚖️ Compare" },
    "header.assistant":     { zh: "AI 算法选型助手",                       en: "AI Algorithm Selector" },
    "header.tutor":         { zh: "AI 学习助理",                           en: "AI Learning Tutor" },
    "header.tutor_short":   { zh: "AI 学习助理",                           en: "AI Tutor" },
    "header.menu_btn":      { zh: "菜单",                                  en: "Menu" },
    "header.menu_aria":     { zh: "打开侧边菜单",                          en: "Open sidebar menu" },
    "header.theme_title":   { zh: "切换主题",                              en: "Switch theme" },
    "header.settings_title":{ zh: "界面设置（主题/字号/语言）",            en: "UI settings (theme / font / language)" },
    "header.config_title":  { zh: "配置 LLM API",                          en: "Configure LLM API" },
    "header.lang_title":    { zh: "切换语言（中文 / English）",            en: "Switch language (Chinese / English)" },

    /* Theme */
    "theme.day":            { zh: "日间",     en: "Day"   },
    "theme.eye":            { zh: "护眼",     en: "Eye"   },
    "theme.night":          { zh: "夜视",     en: "Night" },
    "theme.day_desc":       { zh: "亮色",     en: "Light" },
    "theme.eye_desc":       { zh: "暖色",     en: "Warm"  },
    "theme.night_desc":     { zh: "深色",     en: "Dark"  },

    /* Sidebar */
    "sidebar.platform_overview": { zh: "平台总览",        en: "Platform Overview" },
    "sidebar.compare_label":     { zh: "算法对比分析",    en: "Algorithm Comparison" },
    "sidebar.modules_nav":       { zh: "模块导航",        en: "Module Navigation" },
    "sidebar.collapse_all":      { zh: "▸ 全折叠",         en: "▸ Collapse" },
    "sidebar.expand_all":        { zh: "▾ 全展开",         en: "▾ Expand" },
    "sidebar.collapse_all_t":    { zh: "全部折叠",         en: "Collapse all" },
    "sidebar.expand_all_t":      { zh: "全部展开",         en: "Expand all" },
    "sidebar.edit_reorder":      { zh: "✏️ 排序",          en: "✏️ Reorder" },
    "sidebar.edit_done":         { zh: "✓ 完成",           en: "✓ Done" },
    "sidebar.edit_done_title":   { zh: "退出排序模式",      en: "Exit reorder mode" },
    "sidebar.edit_title":        { zh: "拖拽编辑模块顺序",  en: "Drag to reorder modules" },
    "sidebar.edit_reset":        { zh: "↩ 重置",           en: "↩ Reset" },
    "sidebar.edit_reset_title":  { zh: "重置为默认顺序",    en: "Reset to default order" },
    "sidebar.edit_reset_confirm":{ zh: "重置侧栏顺序到默认？保存的拖拽排序将被清除。",
                                   en: "Reset sidebar order to default? Your custom order will be cleared." },
    "sidebar.tip_title":         { zh: "💡 使用提示",      en: "💡 Tips" },
    "sidebar.tip_body":          {
      zh: '每个模块包含：原理 / 代码 / 交互图 / 场景。点击顶部 <b>⚖️ 算法对比</b> 可多选算法从 10+ 维度并排分析。',
      en: 'Each module contains: principles / code / interactive charts / scenarios. Click the <b>⚖️ Compare</b> at the top to compare algorithms across 10+ dimensions.'
    },

    /* Sidebar group titles (cat headers) */
    "cat.ml_sup":               { zh: "▎ 基础机器学习 · 监督",               en: "▎ Classical ML · Supervised" },
    "cat.ml_unsup":             { zh: "▎ 基础机器学习 · 无监督",             en: "▎ Classical ML · Unsupervised" },
    "cat.ml_dr":                { zh: "▎ 降维 & 异常检测",                  en: "▎ Dim. Reduction & Anomaly Detection" },
    "cat.time_series":          { zh: "▎ 时序模型",                          en: "▎ Time Series" },
    "cat.nn":                   { zh: "▎ 神经网络基础",                      en: "▎ Neural Networks" },
    "cat.frontier":             { zh: "▎ 前沿算法",                          en: "▎ Frontier Algorithms" },
    "cat.graph":                { zh: "▎ 图算法 & 图嵌入",                  en: "▎ Graph Algos & Embeddings" },
    "cat.loss":                 { zh: "▎ 损失函数",                          en: "▎ Loss Functions" },
    "cat.llm":                  { zh: "▎ 大模型 LLM",                        en: "▎ Large Language Models" },
    "cat.learning":             { zh: "▎ 学习 & 助理 🆕",                    en: "▎ Learning & Tutor 🆕" },
    "cat.case":                 { zh: "▎ 工业案例",                          en: "▎ Industrial Cases" },
    "cat.coding":               { zh: "▎ 基础编程算法 🆕",                   en: "▎ Classical Coding Algorithms 🆕" },
    "cat.case_merchant_detail": { zh: "└ 商户识别 · 子模块",                 en: "└ Merchant Recognition · Submodules" },

    /* Overview category card titles (registry cat_name → translated) */
    "catname.基础机器学习":     { zh: "基础机器学习",     en: "Classical ML" },
    "catname.神经网络基础":     { zh: "神经网络基础",     en: "Neural Networks" },
    "catname.图算法":           { zh: "图算法",           en: "Graph Algorithms" },
    "catname.损失函数":         { zh: "损失函数",         en: "Loss Functions" },
    "catname.大模型":           { zh: "大模型",           en: "Large Language Models" },
    "catname.工业案例":         { zh: "工业案例",         en: "Industrial Cases" },
    "catname.时序模型":         { zh: "时序模型",         en: "Time Series" },
    "catname.前沿算法":         { zh: "前沿算法",         en: "Frontier Algorithms" },
    "catname.基础编程算法":     { zh: "基础编程算法",     en: "Classical Coding Algorithms" },

    /* Settings panel */
    "settings.title":           { zh: "界面设置",           en: "UI Settings" },
    "settings.theme_title":     { zh: "🎨 主题配色",        en: "🎨 Theme" },
    "settings.lang_title":      { zh: "🌐 语言 / Language", en: "🌐 Language" },
    "settings.lang_zh":         { zh: "中文",               en: "中文 (Chinese)" },
    "settings.lang_en":         { zh: "English (英文)",     en: "English" },
    "settings.font_title":      { zh: "🔠 字体大小",        en: "🔠 Font Size" },
    "settings.font_sidebar":    { zh: "侧栏菜单字号",       en: "Sidebar font" },
    "settings.font_cat":        { zh: "侧栏分类标题字号",   en: "Sidebar category font" },
    "settings.font_app":        { zh: "全局基础字号",       en: "Base font" },
    "settings.font_small":      { zh: "小 11",              en: "Small 11" },
    "settings.font_medium":     { zh: "适中 14",            en: "Medium 14" },
    "settings.font_large":      { zh: "大 18",              en: "Large 18" },
    "settings.preset_title":    { zh: "⚡ 快速预设",        en: "⚡ Presets" },
    "settings.preset_compact":  { zh: "🐭 紧凑",             en: "🐭 Compact" },
    "settings.preset_default":  { zh: "📋 默认",             en: "📋 Default" },
    "settings.preset_large":    { zh: "🔍 大字",             en: "🔍 Large" },
    "settings.reset":           { zh: "↩️ 恢复默认",          en: "↩️ Reset" },
    "settings.close":           { zh: "关闭",                en: "Close" },

    /* Search */
    "search.placeholder": { zh: "搜索算法 / 标签 / 场景 / 超参...", en: "Search algorithms / tags / scenarios / hyperparams..." },
    "search.esc_hint":    { zh: "Esc 关闭",                          en: "Esc to close" },
    "search.no_match":    { zh: "未找到匹配算法",                    en: "No matching algorithms" },
    "search.try_label":   { zh: "试试：",                            en: "Try: " },
    "search.idle_tip":    { zh: "💡 搜索所有 {N} 个算法的元数据",     en: "💡 Search all {N} algorithm metadata" },
    "search.idle_try":    { zh: "试试：",                            en: "Try: " },
    "search.kbd_hint":    { zh: "支持多关键词（空格分隔，AND 匹配）", en: "Multi-keyword (space-separated AND match)" },
    "search.kbd_hint2":   { zh: "↑ ↓ 选择 · Enter 打开 · Ctrl/⌘+K 再次打开", en: "↑ ↓ select · Enter open · Ctrl/⌘+K toggle" },
    "search.field.pros":       { zh: "✓ 优点",     en: "✓ Pros" },
    "search.field.cons":       { zh: "✗ 缺点",     en: "✗ Cons" },
    "search.field.scenario":   { zh: "◎ 场景",     en: "◎ Scenario" },
    "search.field.tag":        { zh: "🏷️ 标签",    en: "🏷️ Tag" },
    "search.field.hyperparam": { zh: "⚙ 超参",     en: "⚙ Hyperparam" },
    "search.field.papers":     { zh: "📄 论文",    en: "📄 Paper" },
    "search.field.category":   { zh: "📂 分类",    en: "📂 Category" },
    "search.field.not_for":    { zh: "⊘ 不适合",   en: "⊘ Not for" },
    "search.field.complexity": { zh: "Ω 复杂度",   en: "Ω Complexity" },

    /* Module loading */
    "common.loading":           { zh: "加载中...",                       en: "Loading..." },
    "common.render_error":      { zh: "模块渲染出错：",                   en: "Module render error: " },
    "common.enter_module":      { zh: "→ 进入研究",                       en: "→ Open" },
    "common.pros":              { zh: "✓ 优点",                          en: "✓ Pros" },
    "common.cons":              { zh: "✗ 缺点",                          en: "✗ Cons" },
    "common.best_scenario":     { zh: "◎ 最佳场景",                      en: "◎ Best scenarios" },

    /* Overview module */
    "overview.hero_title":      { zh: "AI 算法研发可视化平台",
                                  en: "AI Algorithm Visualization Lab" },
    "overview.hero_sub":        {
      zh: "交互式研究 · 原理解析 · 代码注释 · 多维对比 —— 覆盖基础机器学习、神经网络、图算法、损失函数、大模型和工业案例",
      en: "Interactive research · Principle analysis · Annotated code · Multi-dim comparison — Covering Classical ML, Neural Networks, Graph algorithms, Loss functions, LLMs and Industrial cases"
    },
    "overview.stat.algos":      { zh: "种算法",         en: "Algorithms" },
    "overview.stat.cats":       { zh: "大类别",         en: "Categories" },
    "overview.stat.dims":       { zh: "维度对比",       en: "Dimensions" },
    "overview.stat.zero_build": { zh: "🎯 零构建 · 浏览器打开",   en: "🎯 Zero-build · Open in browser" },

    "overview.compare.title":   { zh: "⚖️ 核心功能：算法对比分析",  en: "⚖️ Key Feature: Algorithm Comparison" },
    "overview.compare.body": {
      zh: "支持 <b>勾选 2-5 个算法</b>，从 <b>可解释性、精度、训练速度、推理速度、内存、调参难度、鲁棒性</b> 等 <b>10 个维度</b>做雷达图 + 表格并排对比，辅以<b>优缺点、最佳场景、复杂度、关键超参、论文出处</b>全方位解析。",
      en: "<b>Select 2-5 algorithms</b> and compare them across <b>10 dimensions</b> (interpretability, accuracy, training/inference speed, memory, tuning ease, robustness, etc.) via radar chart + side-by-side table, with <b>pros/cons, best scenarios, complexity, key hyperparameters, paper references</b> for full analysis."
    },
    "overview.compare.btn":     { zh: "进入对比分析 →",            en: "Open Comparison →" },

    "overview.arch.title":      { zh: "🎨 平台架构与维度",         en: "🎨 Platform Architecture & Dimensions" },
    "overview.arch.taxonomy":   { zh: "· 算法分类覆盖",            en: "· Algorithm Taxonomy" },
    "overview.arch.cat_radar":  { zh: "· 各类别平均能力雷达",      en: "· Per-category Average Capability Radar" },
    "overview.arch.x_count":    { zh: "算法数",                    en: "# Algos" },

    "overview.path.title":      { zh: "🗺️ 使用路径建议",            en: "🗺️ Recommended Paths" },
    "overview.path.beginner":   { zh: "🎓 初学者路径",              en: "🎓 Beginner Path" },
    "overview.path.business":   { zh: "💼 业务工程师路径",          en: "💼 Business Engineer Path" },
    "overview.path.llm":        { zh: "🔬 LLM 工程师路径",          en: "🔬 LLM Engineer Path" },
    "overview.path.beginner_steps": {
      zh: ["基础损失函数（MSE/CE/BCE 直观理解）", "逻辑回归 → 决策树 → 随机森林", "MLP → CNN → Attention", "LightGBM 工程化实战"],
      en: ["Basic losses (MSE/CE/BCE intuition)", "Logistic Regression → Decision Tree → Random Forest", "MLP → CNN → Attention", "LightGBM hands-on engineering"]
    },
    "overview.path.business_steps": {
      zh: ["算法对比分析（选择合适算法）", "XGBoost / LightGBM 深入", "长尾损失函数（Focal/LDAM/Seesaw）", "工业案例：商户识别完整链路"],
      en: ["Algorithm comparison (pick the right one)", "XGBoost / LightGBM deep dive", "Long-tail losses (Focal/LDAM/Seesaw)", "Industrial case: merchant recognition pipeline"]
    },
    "overview.path.llm_steps": {
      zh: ["Attention / Transformer 原理", "LLM 采样参数调控", "LoRA / QLoRA 高效微调", "量化 & KV Cache 部署优化"],
      en: ["Attention / Transformer principles", "LLM sampling parameters", "LoRA / QLoRA efficient fine-tuning", "Quantization & KV Cache deployment"]
    },

    "overview.table.title":     { zh: "📋 全算法速览表",          en: "📋 Full Algorithm Table" },
    "overview.col.algo":        { zh: "算法",     en: "Algorithm"  },
    "overview.col.cat":         { zh: "类别",     en: "Category"   },
    "overview.col.acc":         { zh: "精度",     en: "Accuracy"   },
    "overview.col.intp":        { zh: "解释性",   en: "Interpret." },
    "overview.col.train":       { zh: "训练速度", en: "Train"      },
    "overview.col.infer":       { zh: "推理速度", en: "Infer"      },
    "overview.col.scenario":    { zh: "典型场景", en: "Typical scenario" },

    /* Compare module */
    "compare.hero_title":       { zh: "算法对比分析工作台",      en: "Algorithm Comparison Workbench" },
    "compare.hero_sub":         { zh: "多选算法 · 多维度雷达图 · 并排详细表 · 场景推荐矩阵",
                                  en: "Multi-select · Radar chart · Side-by-side table · Scenario matrix" },
    "compare.step1.title":      { zh: "1️⃣ 选择要对比的算法（建议 2-5 个）",
                                  en: "1️⃣ Select algorithms (2-5 recommended)" },
    "compare.preset.tip":       { zh: "🎯 预设对比组合（点击快速选择）", en: "🎯 Presets (click to apply)" },
    "compare.preset.same":      { zh: "同类别横向对比：",         en: "Same-category presets:" },
    "compare.preset.same_more": { zh: "同类别新增组合：",         en: "More same-category presets:" },
    "compare.preset.cross":     { zh: "跨类别对比（经典选型场景）：", en: "Cross-category presets (classic scenarios):" },
    "compare.btn.select_all":   { zh: "⊞ 全选当前类别",            en: "⊞ Select all in category" },
    "compare.btn.clear_sel":    { zh: "✗ 清空",                    en: "✗ Clear" },
    "compare.sel.tip":          { zh: "已选择 {N} 个算法",         en: "{N} algorithms selected" },
    "compare.sel.suggest":      { zh: "（建议 2-5 个）",            en: "(suggest 2-5)" },
  };

  /* ───────── Sidebar nav route → label ─────────
     Icon stays in HTML; we only swap the trailing text portion.
  */
  const NAV = {
    overview:                 { zh: "平台总览",                       en: "Platform Overview" },
    compare:                  { zh: "算法对比分析",                   en: "Algorithm Comparison" },

    /* Classical ML */
    ml_decision_tree:         { zh: "决策树",                          en: "Decision Tree" },
    ml_random_forest:         { zh: "随机森林",                        en: "Random Forest" },
    ml_xgboost:               { zh: "XGBoost",                         en: "XGBoost" },
    ml_lightgbm:              { zh: "LightGBM",                        en: "LightGBM" },
    ml_logistic:              { zh: "逻辑回归",                        en: "Logistic Regression" },
    ml_svm:                   { zh: "SVM 支持向量机",                  en: "SVM" },
    ml_naive_bayes:           { zh: "朴素贝叶斯",                      en: "Naive Bayes" },
    ensemble_methods:         { zh: "集成融合方法",                    en: "Ensemble Methods" },
    ml_kmeans:                { zh: "K-Means 聚类",                    en: "K-Means" },
    ml_dbscan:                { zh: "DBSCAN / HDBSCAN",                en: "DBSCAN / HDBSCAN" },
    ml_pca:                   { zh: "PCA 主成分分析",                  en: "PCA" },
    ml_tsne_umap:             { zh: "t-SNE / UMAP",                    en: "t-SNE / UMAP" },
    ml_isolation_forest:      { zh: "Isolation Forest",                en: "Isolation Forest" },
    ad_deep:                  { zh: "深度异常检测",                    en: "Deep Anomaly Detection" },
    ad_llm:                   { zh: "LLM 异常检测",                    en: "LLM Anomaly Detection" },

    /* Time Series */
    ts_classical:             { zh: "时序经典 (ARIMA/Prophet)",        en: "Classical TS (ARIMA/Prophet)" },
    ts_deep:                  { zh: "深度时序 (PatchTST/iTrans)",      en: "Deep TS (PatchTST/iTrans)" },
    ts_foundation:            { zh: "时序基座 (TimesFM/Chronos)",      en: "TS Foundation (TimesFM/Chronos)" },

    /* Neural Networks */
    nn_mlp:                   { zh: "MLP / DNN",                       en: "MLP / DNN" },
    nn_cnn:                   { zh: "CNN 卷积",                        en: "CNN" },
    nn_rnn:                   { zh: "RNN / LSTM / GRU",                en: "RNN / LSTM / GRU" },
    seq_encoder:              { zh: "序列 Encoder (BST+T2V)",          en: "Sequence Encoder (BST+T2V)" },
    nn_attention:             { zh: "Attention 机制",                  en: "Attention" },
    nn_transformer:           { zh: "Transformer 完整架构",            en: "Transformer (full)" },
    nn_gpt:                   { zh: "GPT / Decoder-only",              en: "GPT / Decoder-only" },
    image_encoder:            { zh: "图像 Attention Pool",             en: "Image Attention Pool" },
    fusion:                   { zh: "Perceiver-IO 融合",               en: "Perceiver-IO Fusion" },
    mmoe:                     { zh: "PLE-MMoE 多任务",                 en: "PLE-MMoE Multi-task" },
    tab_dl:                   { zh: "表格深度学习",                    en: "Tabular DL" },

    /* Frontier */
    frontier_mamba:           { zh: "Mamba / SSM",                     en: "Mamba / SSM" },
    frontier_diffusion:       { zh: "Diffusion 扩散模型",              en: "Diffusion Models" },
    frontier_rag:             { zh: "RAG 检索增强",                    en: "RAG" },
    frontier_multimodal:      { zh: "多模态大模型 🆕",                  en: "Multimodal Foundation Models 🆕" },
    frontier_image_gen:       { zh: "图像生成 🆕",                      en: "Image Generation 🆕" },
    frontier_video_gen:       { zh: "视频生成 🆕",                      en: "Video Generation 🆕" },
    frontier_asr:             { zh: "语音识别 🆕",                      en: "Speech Recognition (ASR) 🆕" },
    frontier_audio_gen:       { zh: "音频/歌曲生成 🆕",                 en: "Audio & Music Generation 🆕" },

    /* Graph */
    graph_pagerank:           { zh: "PageRank",                        en: "PageRank" },
    graph_louvain:            { zh: "Louvain 社区挖掘",                en: "Louvain Community" },
    graph_label_prop:         { zh: "标签传播 LPA",                    en: "Label Propagation (LPA)" },
    graph_node2vec:           { zh: "Node2Vec / DeepWalk",             en: "Node2Vec / DeepWalk" },
    graph_gcn:                { zh: "GCN 图卷积",                      en: "GCN" },
    graph_sage:               { zh: "GraphSAGE",                       en: "GraphSAGE" },
    graph_gat:                { zh: "GAT 图注意力",                    en: "GAT" },
    graph_encoder:            { zh: "HGT 异构图 Encoder",              en: "HGT Heterogeneous Graph Encoder" },

    /* Loss */
    loss_basics:              { zh: "基础损失函数",                    en: "Basic Losses" },
    losses:                   { zh: "长尾 / 不均衡",                   en: "Long-tail / Imbalance" },
    loss_multitask:           { zh: "多目标优化",                      en: "Multi-objective" },

    /* LLM */
    llm_foundation:           { zh: "基座模型原理",                    en: "Foundation Models" },
    text_encoder:             { zh: "文本 Encoder (MacBERT+LoRA)",     en: "Text Encoder (MacBERT+LoRA)" },
    llm_params:               { zh: "采样参数",                        en: "Sampling Parameters" },
    llm_finetune:             { zh: "高效微调",                        en: "Efficient Fine-tuning" },
    llm_quantization:         { zh: "量化 & KV Cache",                 en: "Quantization & KV Cache" },
    llm_serving:              { zh: "推理部署架构",                    en: "Serving Architecture" },
    llm_cot:                  { zh: "CoT 思维链 🆕",                   en: "Chain-of-Thought 🆕" },
    llm_rag:                  { zh: "RAG 检索增强 🆕",                 en: "RAG (Retrieval-Augmented) 🆕" },
    llm_react:                { zh: "ReAct 推理-行动 🆕",              en: "ReAct (Reason+Act) 🆕" },
    llm_tool_use:             { zh: "Tool Use / Function Calling 🆕",  en: "Tool Use / Function Calling 🆕" },
    llm_agent:                { zh: "LLM Agent 智能体 🆕",             en: "LLM Agent 🆕" },
    llm_models_zoo:           { zh: "主流大模型详介 🆕",                en: "Mainstream LLMs Zoo 🆕" },
    llm_agent_advanced:       { zh: "Agent 进阶 · Skills/OpenClaw/Hermes 🆕", en: "Agent Advances · Skills/OpenClaw/Hermes 🆕" },

    /* Learning */
    learning_paths:           { zh: "学习路径（四类人群）",            en: "Learning Paths (4 personas)" },
    learning_stories:         { zh: "真实转型案例 🆕",                 en: "Real-world Transition Stories 🆕" },
    tutor:                    { zh: "AI 学习助理（多轮对话）",         en: "AI Tutor (multi-turn chat)" },

    /* Cases */
    case_overview:            { zh: "案例总览",                         en: "Cases Overview" },
    case_recommendation:      { zh: "推荐算法",                         en: "Recommendation" },
    case_fraud:               { zh: "欺诈识别",                         en: "Fraud Detection" },
    case_ads_ctr:             { zh: "广告 CTR 预估",                    en: "Ads CTR Prediction" },
    case_credit_score:        { zh: "信用评分",                         en: "Credit Scoring" },
    case_tts:                 { zh: "语音合成 TTS 🆕",                  en: "Speech Synthesis (TTS) 🆕" },
    case_video_gen:           { zh: "视频生成 🆕",                       en: "Video Generation 🆕" },
    case_merchant:            { zh: "商户多模态识别",                   en: "Merchant Multi-modal Recognition" },
    aux_encoder:              { zh: "辅助 Encoder",                      en: "Auxiliary Encoder" },
    heads:                    { zh: "多任务 Heads",                      en: "Multi-task Heads" },
    samplers:                 { zh: "采样器",                            en: "Samplers" },
    training:                 { zh: "Decoupled cRT",                     en: "Decoupled cRT" },
    distillation:             { zh: "蒸馏 & 量化",                       en: "Distillation & Quantization" },

    /* Coding */
    coding_overview:          { zh: "总览（LeetCode / VisuAlgo）",        en: "Overview (LeetCode / VisuAlgo)" },
    coding_sort:              { zh: "排序算法全家族",                     en: "Sorting Algorithms" },
    coding_search:            { zh: "查找（二分/哈希）",                   en: "Search (Binary / Hash)" },
    coding_ds_stack_queue:    { zh: "栈/队列/单调栈",                     en: "Stack / Queue / Monotonic Stack" },
    coding_ds_linked_tree:    { zh: "链表/二叉树/堆",                     en: "Linked List / Binary Tree / Heap" },
    coding_graph_traversal:   { zh: "BFS/DFS/拓扑排序",                   en: "BFS / DFS / Topological Sort" },
    coding_graph_shortest:    { zh: "最短路径 (Dijkstra/A*)",             en: "Shortest Paths (Dijkstra/A*)" },
    coding_dp:                { zh: "动态规划 DP",                         en: "Dynamic Programming" },
    coding_string_math:       { zh: "字符串 & 数学",                       en: "Strings & Math" },
  };

  /* ───────── Public API ───────── */
  const i18n = {
    lang: FALLBACK,
    dict: DICT,
    nav: NAV,

    t(key, vars) {
      const entry = DICT[key];
      let s;
      if (entry) s = entry[this.lang] != null ? entry[this.lang] : (entry[FALLBACK] || key);
      else s = key;
      if (vars && typeof s === "string") {
        Object.keys(vars).forEach(k => {
          s = s.replace(new RegExp("\\{" + k + "\\}", "g"), vars[k]);
        });
      }
      return s;
    },

    /* sidebar nav label */
    navLabel(route) {
      const e = NAV[route];
      if (!e) return null;
      return e[this.lang] != null ? e[this.lang] : (e[FALLBACK] || route);
    },

    /* category name (registry cat_name → translated) */
    catName(zhCatName) {
      const k = "catname." + zhCatName;
      const e = DICT[k];
      if (!e) return zhCatName;
      return e[this.lang] != null ? e[this.lang] : zhCatName;
    },

    /* Algo display name (uses registry.en when in English mode) */
    algoName(a) {
      if (!a) return "";
      if (this.lang === "en") return a.en || a.name;
      return a.name || a.en;
    },

    /* Dimension translated name */
    dimName(d) {
      if (!d) return "";
      if (this.lang === "en") return d.en || d.name;
      return d.name || d.en;
    },

    /* Walk DOM and replace data-i18n* attributes */
    applyDOM(root) {
      root = root || document;
      // textContent
      root.querySelectorAll("[data-i18n]").forEach(el => {
        const k = el.getAttribute("data-i18n");
        if (k) el.textContent = this.t(k);
      });
      // innerHTML
      root.querySelectorAll("[data-i18n-html]").forEach(el => {
        const k = el.getAttribute("data-i18n-html");
        if (k) el.innerHTML = this.t(k);
      });
      // title attribute
      root.querySelectorAll("[data-i18n-title]").forEach(el => {
        const k = el.getAttribute("data-i18n-title");
        if (k) el.setAttribute("title", this.t(k));
      });
      // aria-label
      root.querySelectorAll("[data-i18n-aria]").forEach(el => {
        const k = el.getAttribute("data-i18n-aria");
        if (k) el.setAttribute("aria-label", this.t(k));
      });
      // placeholder
      root.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
        const k = el.getAttribute("data-i18n-placeholder");
        if (k) el.setAttribute("placeholder", this.t(k));
      });
      // sidebar nav-item: patch the visible text node (after the icon span)
      root.querySelectorAll(".nav-item[data-route]").forEach(el => {
        const route = el.getAttribute("data-route");
        const label = this.navLabel(route);
        if (label == null) return;
        // Find/replace the trailing text node
        let textNode = null;
        for (let n = el.lastChild; n; n = n.previousSibling) {
          if (n.nodeType === Node.TEXT_NODE && n.textContent.trim()) { textNode = n; break; }
        }
        if (textNode) textNode.textContent = " " + label;
        else el.appendChild(document.createTextNode(" " + label));
      });
      // Update <html lang>
      document.documentElement.setAttribute("lang", this.lang === "en" ? "en" : "zh-CN");
    },

    /* Set language: persist + apply DOM + re-render current module */
    setLang(lang) {
      if (lang !== "zh" && lang !== "en") lang = FALLBACK;
      this.lang = lang;
      try { localStorage.setItem(LANG_KEY, lang); } catch (e) {}
      // 1) header / sidebar / static index.html elements
      this.applyDOM(document);
      // 2) language indicator in header (if present)
      const ind = document.getElementById("lang-indicator");
      if (ind) ind.textContent = lang === "en" ? "EN" : "中";
      // 3) re-render currently active module so internal text updates too
      if (typeof window.MCH_render === "function") {
        const route = (window.location.hash.replace(/^#\//, "") || "overview");
        try { window.MCH_render(route); } catch (e) { /* noop */ }
      }
      // 4) notify listeners (e.g. settings panel highlight)
      document.dispatchEvent(new CustomEvent("mch:lang-changed", { detail: { lang } }));
    },

    /* Init: read storage, apply DOM */
    init() {
      let saved = FALLBACK;
      try { saved = localStorage.getItem(LANG_KEY) || FALLBACK; } catch (e) {}
      if (saved !== "zh" && saved !== "en") saved = FALLBACK;
      this.lang = saved;
      this.applyDOM(document);
      const ind = document.getElementById("lang-indicator");
      if (ind) ind.textContent = saved === "en" ? "EN" : "中";
    },
  };

  // Public exports
  MCH.i18n  = i18n;
  MCH.t     = i18n.t.bind(i18n);
  MCH.tn    = i18n.navLabel.bind(i18n);
  MCH.tcat  = i18n.catName.bind(i18n);
  MCH.tname = i18n.algoName.bind(i18n);
  MCH.tdim  = i18n.dimName.bind(i18n);

  // Auto-init on DOMContentLoaded (must run BEFORE app.js's DOM handler that
  // expects nav-items to have correct text for activation logic)
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => i18n.init());
  } else {
    i18n.init();
  }
})();
