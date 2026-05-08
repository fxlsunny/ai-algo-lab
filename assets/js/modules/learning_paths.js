/* 模块：AI 算法学习路径（四类人群） */
MCH.register("learning_paths", {
  render() {
    const paths = {
      beginner: {
        icon: "🌱",
        title: "小白入门路径",
        en: "For AI Beginners",
        duration: "3-6 个月",
        color: "#10b981",
        desc: "零基础 / 非计算机专业 / 跨行转 AI",
        prereq: ["高中数学（线性代数 + 概率基础）", "Python 会写简单脚本"],
        stages: [
          { w: "W1-4",  name: "Python + 数学补强", tasks: ["NumPy/Pandas/Matplotlib", "线性代数（向量/矩阵/特征值）", "概率统计基础"], modules: ["-"], proj: "Kaggle Titanic / 房价预测" },
          { w: "W5-8",  name: "机器学习基础", tasks: ["决策树/随机森林/XGBoost", "逻辑回归 & SVM", "K-Means/DBSCAN"], modules: ["ml_decision_tree","ml_random_forest","ml_xgboost","ml_logistic","ml_kmeans"], proj: "信用卡反欺诈" },
          { w: "W9-12", name: "神经网络入门", tasks: ["MLP 手写反向传播", "CNN 图像分类（MNIST）", "RNN/LSTM 情感分类"], modules: ["nn_mlp","nn_cnn","nn_rnn"], proj: "CIFAR-10 猫狗分类" },
          { w: "W13-16", name: "NLP + LLM 应用", tasks: ["Transformer 结构理解", "HuggingFace 调用", "RAG 问答机器人"], modules: ["nn_transformer","nn_attention","llm_rag"], proj: "搭建个人知识库问答" },
          { w: "W17-24", name: "实战项目 + 求职", tasks: ["2 个完整项目", "刷 LeetCode Top 100", "面试八股文"], modules: ["coding_overview","coding_sort","coding_ds_linked_tree","coding_dp"], proj: "作为 GitHub 作品集" },
        ],
        resources: ["李沐《动手学深度学习》", "吴恩达 Coursera ML 专项", "《统计学习方法》（李航）", "Kaggle Micro-courses"],
        output: "初级算法工程师（15-25K）",
      },
      engineer: {
        icon: "🔁",
        title: "工程师转型路径",
        en: "From SWE to ML Engineer",
        duration: "4-8 个月",
        color: "#4f46e5",
        desc: "已有 2-5 年工程经验（后端/前端/嵌入式）",
        prereq: ["熟练编码（Python/Java/C++）", "有系统设计经验", "理解分布式/并发"],
        stages: [
          { w: "M1",  name: "快速对齐数理", tasks: ["线性代数复习（重点：矩阵求导）", "概率（重点：最大似然/贝叶斯）", "优化（梯度下降变体）"], modules: ["-"], proj: "无需项目" },
          { w: "M2",  name: "机器学习全家桶", tasks: ["树模型家族", "集成方法", "PCA/异常检测"], modules: ["ml_decision_tree","ml_xgboost","ml_lightgbm","ensemble_methods","ml_pca","ml_isolation_forest"], proj: "用你现有业务数据建模" },
          { w: "M3",  name: "深度学习", tasks: ["CNN/RNN/Transformer 手撕", "分布式训练（DDP）", "模型加速（量化/蒸馏）"], modules: ["nn_cnn","nn_transformer","nn_attention","llm_quantization"], proj: "把一个业务场景迁移到 DL 方案" },
          { w: "M4",  name: "大模型 & LLM 应用", tasks: ["LoRA 微调", "RAG", "Function Calling", "Agent"], modules: ["llm_foundation","llm_finetune","llm_rag","llm_tool_use","llm_agent","llm_cot","llm_react"], proj: "为公司搭建内部 AI 助手" },
          { w: "M5-6", name: "工程化落地", tasks: ["MLOps（训练/推理/监控）", "性能优化", "A/B 实验"], modules: ["llm_serving","frontier_rag","case_ads_ctr","case_recommendation"], proj: "完整闭环项目" },
          { w: "M7-8", name: "深入行业场景", tasks: ["选定领域深挖：推荐/风控/广告/CV/NLP", "论文精读 + 复现"], modules: ["case_overview","case_merchant","case_fraud","case_credit_score"], proj: "垂直领域 end-to-end" },
        ],
        resources: ["《百面机器学习》", "Andrej Karpathy - Zero to Hero", "CS224n / CS231n", "Full Stack Deep Learning"],
        output: "中高级算法工程师（30-50K，部分大厂 P6/P7）",
      },
      student: {
        icon: "🎓",
        title: "学生求职路径",
        en: "Student → First Job",
        duration: "6-12 个月（校招季）",
        color: "#ec4899",
        desc: "本/硕在读，目标秋招算法岗",
        prereq: ["有 ML/DL 基础课程", "会一门主力语言（强烈建议 Python + C++）"],
        stages: [
          { w: "大三上 / 研一上",  name: "夯实基础 + 选定方向", tasks: ["CV/NLP/推荐/风控选一个主攻", "读 10 篇经典论文", "刷 LeetCode 300+"], modules: ["coding_overview","coding_sort","coding_dp","coding_ds_linked_tree"], proj: "可复现的 baseline 项目" },
          { w: "大三下 / 研一下", name: "打比赛 + 拿 Offer", tasks: ["Kaggle / 天池打到前 5%", "春招实习面试", "1-2 篇 Kaggle solution writeup"], modules: ["ml_xgboost","ml_lightgbm","ensemble_methods","nn_transformer","loss_basics","losses"], proj: "Kaggle Gold 奖牌 = 大厂直通卡" },
          { w: "大四上 / 研二上", name: "实习 + 论文", tasks: ["大厂实习（字节/腾讯/阿里）", "发一篇 workshop 论文或 arxiv"], modules: ["case_recommendation","case_ads_ctr","case_fraud","case_merchant","mmoe","fusion","frontier_rag"], proj: "实习做出线上收益指标" },
          { w: "大四下 / 研二下", name: "秋招决胜", tasks: ["八股文（Transformer 每个细节)", "项目深挖（能撑 30 分钟）", "Leetcode 随机可 AC hard"], modules: ["nn_transformer","nn_attention","llm_foundation","llm_finetune","llm_rag","llm_agent","coding_graph_traversal","coding_graph_shortest"], proj: "反复打磨已有项目" },
        ],
        resources: ["《深度学习》花书", "Papers With Code", "牛客 / CodeTop", "导师 / 学长 + 内推群"],
        output: "大厂校招算法岗（SP/SSP 40-80 万年包）",
      },
      senior: {
        icon: "🚀",
        title: "大模型高阶路径",
        en: "LLM Senior → Expert",
        duration: "6-12 个月（已有 ML 基础）",
        color: "#f59e0b",
        desc: "想从传统 ML 转入大模型核心赛道 / 冲击专家级",
        prereq: ["传统 ML/DL 熟练", "PyTorch 写过分布式训练", "至少读过 10 篇 Transformer/GPT 论文"],
        stages: [
          { w: "M1",  name: "精通 Transformer 内核", tasks: ["手撕 Multi-Head Attention", "RoPE / ALiBi / GQA", "KV Cache / Flash Attention", "MoE 结构"], modules: ["nn_transformer","nn_attention","nn_gpt","llm_foundation","llm_quantization"], proj: "从零用 PyTorch 实现 100M 参数 GPT" },
          { w: "M2",  name: "预训练 & Scaling", tasks: ["Tokenizer 训练（BPE/SentencePiece）", "数据清洗 pipeline", "Chinchilla 最优计算分配", "DeepSpeed ZeRO / FSDP"], modules: ["llm_foundation","llm_finetune"], proj: "预训练 1B 模型" },
          { w: "M3",  name: "对齐 & 微调", tasks: ["SFT / LoRA / QLoRA / DoRA", "DPO / KTO / GRPO", "Reward Model", "Constitutional AI"], modules: ["llm_finetune","loss_basics","losses","loss_multitask"], proj: "微调领域模型 + DPO 对齐" },
          { w: "M4",  name: "推理优化", tasks: ["vLLM / SGLang / TensorRT-LLM", "量化（GPTQ/AWQ/INT4）", "投机解码（Speculative Decoding）", "Continuous Batching"], modules: ["llm_serving","llm_quantization"], proj: "自建 7B 推理服务，&gt;1000 QPS" },
          { w: "M5",  name: "Agent & 应用", tasks: ["CoT / ToT / ReAct 精通", "RAG 进阶（GraphRAG/Self-RAG）", "Multi-Agent 架构", "MCP 工具协议"], modules: ["llm_cot","llm_react","llm_rag","llm_agent","llm_tool_use"], proj: "搭建垂直领域 AI Agent" },
          { w: "M6", name: "前沿追踪", tasks: ["Mamba/SSM", "Diffusion LLM", "o1/R1 级推理模型", "多模态 LLM"], modules: ["frontier_mamba","frontier_diffusion","frontier_rag"], proj: "复现一篇最新顶会工作" },
        ],
        resources: ["Andrej Karpathy - nanoGPT/llm.c", "《Build a Large Language Model (From Scratch)》Sebastian Raschka", "Hugging Face Cookbook", "arXiv daily"],
        output: "大模型核心算法专家（60-120 万年包 / 技术合伙人）",
      },
    };

    const pathCard = (key, p) => `
      <div class="card" style="border-top:4px solid ${p.color};cursor:pointer" onclick="document.getElementById('path-detail-${key}').scrollIntoView({behavior:'smooth'})">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
          <div style="width:48px;height:48px;border-radius:10px;background:${p.color}22;color:${p.color};display:flex;align-items:center;justify-content:center;font-size:24px">${p.icon}</div>
          <div>
            <h3 style="margin:0">${p.title}</h3>
            <div style="font-size:11px;color:var(--text-muted)">${p.en} · ${p.duration}</div>
          </div>
        </div>
        <p style="font-size:13px;color:var(--text-secondary);margin:6px 0">${p.desc}</p>
        <div style="font-size:11px;color:var(--text-muted);margin-top:6px">📌 目标：${p.output}</div>
      </div>
    `;

    const pathDetail = (key, p) => `
      <div class="section" id="path-detail-${key}" style="border-left:4px solid ${p.color}">
        <h2><span style="color:${p.color}">${p.icon}</span> ${p.title} — ${p.duration}</h2>
        <div style="display:grid;grid-template-columns:2fr 1fr;gap:16px;margin-bottom:16px">
          <div>
            <h3>🎯 适合人群</h3>
            <p style="color:var(--text-secondary)">${p.desc}</p>
            <h3>📋 前置要求</h3>
            <ul style="padding-left:20px;line-height:1.7">${p.prereq.map(x=>`<li>${x}</li>`).join("")}</ul>
          </div>
          <div>
            <h3>🏆 最终成果</h3>
            <div class="card" style="background:${p.color}15;border-color:${p.color}44">
              <div style="font-weight:700;color:${p.color}">${p.output}</div>
            </div>
            <h3 style="margin-top:14px">📚 推荐资源</h3>
            <ul style="padding-left:20px;line-height:1.7;font-size:13px">${p.resources.map(x=>`<li>${x}</li>`).join("")}</ul>
          </div>
        </div>

        <h3>🗓 阶段规划</h3>
        <table class="table">
          <thead><tr><th>时间</th><th>阶段</th><th>核心任务</th><th>对应平台模块</th><th>实战项目</th></tr></thead>
          <tbody>
            ${p.stages.map(s => `
              <tr>
                <td><span class="tag" style="background:${p.color}22;color:${p.color}">${s.w}</span></td>
                <td><b>${s.name}</b></td>
                <td><ul style="margin:0;padding-left:16px;font-size:12px">${s.tasks.map(t=>`<li>${t}</li>`).join("")}</ul></td>
                <td style="max-width:280px">${s.modules[0]==="-" ? '<span style="color:var(--text-muted);font-size:11px">先打基础</span>' : s.modules.map(m=>`<a href="#/${m}" class="tag" style="margin:1px 2px;font-size:10px">${m}</a>`).join("")}</td>
                <td style="font-size:12px;color:var(--text-secondary)">${s.proj}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </div>
    `;

    return `
      ${MCH.hero({
        icon: "🎯",
        name: "AI 算法学习路径",
        en: "Four Learning Paths for Different Backgrounds",
        tags: ["小白入门", "工程师转型", "学生求职", "大模型高阶"],
        meta: ["◈ 基于 57 个算法模块 · 工程师/学生/研究者闭环", "⚡ 从 0 到专家，按自己的基础选一条路"],
      })}

      <div class="section">
        <h2>📍 选择你的路径</h2>
        <p style="color:var(--text-secondary);margin-bottom:14px">点击卡片跳转到对应详细规划。每条路径都对接本平台的具体算法模块，形成可执行的学习闭环。</p>
        <div class="grid-4">
          ${Object.entries(paths).map(([k,p]) => pathCard(k, p)).join("")}
        </div>
      </div>

      ${Object.entries(paths).map(([k,p]) => pathDetail(k, p)).join("")}

      <div class="section">
        <h2>💡 选路径的 3 个原则</h2>
        <div class="grid-3">
          <div class="card">
            <h3>1. 别贪多</h3>
            <p style="line-height:1.7">一次只走一条路径。别今天 CV 明天 NLP 后天推荐，3 个月啥也不精通</p>
          </div>
          <div class="card">
            <h3>2. 项目驱动</h3>
            <p style="line-height:1.7">每阶段必须有可交付的项目（GitHub repo / Kaggle 分数 / 线上指标）</p>
          </div>
          <div class="card">
            <h3>3. 主动求反馈</h3>
            <p style="line-height:1.7">加入社群 / 找 mentor / 写博客。闭门造车 10 个月 ≈ 有反馈 3 个月</p>
          </div>
        </div>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
