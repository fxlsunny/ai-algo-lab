# AI 算法研发可视化平台 · AI Algo Lab

> 一站式的 AI 算法研究/学习/对比本地 Web 平台。覆盖**基础机器学习、神经网络、图算法、损失函数、大模型**五大技术领域，并配套**商户多模态多任务识别**完整工业案例。支持**多维度并排对比**。

> 🌐 **Live Demo**: [https://ai-algo-lab.vercel.app](https://ai-algo-lab.vercel.app)（Vercel 托管，纯前端，零后端）

---

## ✨ 核心能力

- **🧙‍♂️ AI 算法选型助手**（v5.0）：输入场景/数据/目标，自动推荐算法方案；支持 8+ 大模型 API（OpenAI/Claude/DeepSeek/Qwen/GLM/Kimi/Gemini/Ollama/自定义），无配置时走规则引擎 fallback；方案可导出 JSON/Markdown 到本地 `data/` 目录。**v6.0 新增**：规则引擎识别排序/查找/图遍历/最短路/DP/字符串等工程关键词，推荐方案会同时附带基础编程算法**工程组件**
- **🧩 65 种算法模块 + 5 大工业案例**，覆盖 **9 大类别**：基础 ML / 神经网络 / 图 / 损失 / LLM / 时序 / 前沿 / 案例 / **🆕 基础编程算法**
- **🧮 基础编程算法可视化（🆕 v6.0）**：对标 VisuAlgo +《算法导论》+ LeetCode，覆盖**排序/查找/栈队列/链表树堆/BFSDFS/最短路/动态规划/字符串数学** 8 大模块，每个模块含交互可视化
- **📦 每个算法都有版本演进节**：新旧实现变体对比 + 2024 最新亮点（XGBoost 2.0、LightGBM 4.0、GATv2、DeepSeek-V3 等）
- **🔗 完整参考文献 + 官方文档直达链接**（arXiv / GitHub / 官方 Docs 一键跳转）
- **⚖️ 多维度算法对比**：10 个维度雷达图 + 并排详细表 + 场景适配矩阵 + **18 个预设组合** + **💻 计算资源评估**
- **🎮 每模块 3-5 个交互可视化**：参数滑块实时调节，直观理解算法行为
- **📖 原理 / 代码 / 场景 三层解读**：LaTeX 公式 + Python 代码（带注释）+ 业务应用
- **🏭 工业案例**：日级 1 亿+ 商户的多模态识别完整链路
- **💻 零构建、零后端**：双击 HTML 即可在浏览器打开
- **🌍 移动端适配**：响应式布局，支持手机/平板访问（v6.8）

---

## 🚀 快速开始

### 方式一：直接打开（推荐）
下载 `index.html`，双击在浏览器中打开即可，无需任何服务器。

### 方式二：本地服务器（开发 / CORS 代理）

```bash
# Windows
python start.py --port 8899

# 或直接双击
启动平台.bat
```

如遇到跨域问题，运行 CORS 代理：

```bash
cd data/proxy
python cors_proxy.py
```

---

## 📁 项目结构

```
ai-algo-lab/
├── index.html              # 主入口（双击即用）
├── start.py                # 轻量 HTTP 服务器
├── 启动平台.bat            # Windows 一键启动
├── assets/
│   ├── css/
│   │   └── app.css         # 全局样式 + 响应式
│   └── js/
│       ├── app.js          # 主应用逻辑
│       ├── algorithms_registry.js  # 65+ 算法注册表
│       ├── assistant.js    # AI 选型助手
│       ├── common.js       # 通用工具
│       ├── i18n.js         # 国际化
│       ├── llm_config.js   # LLM 配置
│       ├── modules/        # 各模块可视化实现
│       ├── search.js       # 搜索
│       └── sidebar_order.js # 侧栏折叠状态
├── data/
│   └── configs/            # LLM 配置模板
├── skills/                 # 平台建设经验沉淀
└── _sync_github.py         # GitHub 同步脚本
```

---

## 📦 模块目录（部分）

| 类别 | 模块 |
|------|------|
| 🎓 基础机器学习 | 决策树、随机森林、XGBoost、LightGBM、逻辑回归、SVM、朴素贝叶斯、K-Means、DBSCAN、PCA、t-SNE/UMAP、Isolation Forest |
| 🧠 神经网络 | MLP/CNN/RNN/LSTM/GRU、Attention、Transformer、GPT/LLM |
| 🕸️ 图算法 | BFS/DFS、Dijkstra、A*、Bellman-Ford、Prim、Kruskal、Floyd-Warshall |
| 📉 损失函数 | Cross-Entropy、MSE/MAE、Focal Loss、Triplet Loss、GAN Loss |
| 🧮 基础算法 | 排序、查找、栈/队列、链表、树/堆、BFS/DFS、最短路、DP、字符串、数学 |
| 🏭 工业案例 | 商户多模态识别完整链路 |

---

## ⚙️ AI 助手配置

支持以下模型（可在界面设置中配置 API Key）：

| 模型 | 说明 |
|------|------|
| 🤖 Ollama | 本地部署（需 Ollama 运行） |
| 🐉 腾讯混元 | Hunyuan Turbo |
| ⚡ Groq | Groq API |
| 🔵 DeepSeek | DeepSeek V3 / Chat |
| 🌙 Moonshot | Kimi 系列 |
| 🤖 OpenAI | GPT-4 / GPT-3.5 |
| 💡 Qwen | 阿里通义千问 |
| 🧠 GLM | 智谱 GLM-4 |
| 🔮 Gemini | Google Gemini |

---

## 🛠️ 技术栈

- **纯前端**：HTML + CSS + JavaScript（ES6+），零框架依赖
- **可视化**：ECharts 5（图表）+ Canvas/SVG（算法动画）
- **公式渲染**：KaTeX（LaTeX 数学公式）
- **本地服务器**：Python 标准库 `http.server`（`start.py`）
- **AI 集成**：Fetch API + CORS 代理模式

---

## 📄 License

MIT © fxlsunny