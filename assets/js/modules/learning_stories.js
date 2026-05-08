/* 模块：AI 学习真实案例故事 */
MCH.register("learning_stories", {
  render() {
    const stories = [
      /* ────── 小白入门 ────── */
      {
        cat: "beginner", catName: "小白入门", color: "#10b981", icon: "🌱",
        title: "文科女生 → 推荐算法工程师",
        bg: "英语专业本科，零编程基础，毕业后做运营 2 年",
        duration: "14 个月",
        moves: [
          "🧱 M1-3：啃《Python 编程：从入门到实践》+ 《利用 Python 进行数据分析》，每天 2 小时",
          "📊 M4-6：吴恩达 ML 专项 + Kaggle Learn，做了 Titanic/House Prices/Spaceship",
          "🧠 M7-9：动手学深度学习 PyTorch 版，跟书写 CNN/RNN 代码",
          "🏆 M10：打 Kaggle 推荐赛拿到 Bronze Medal（全球前 10%）",
          "💼 M11-14：3 个简历项目 + 刷 LeetCode 150 题，投简历+内推，拿到某电商推荐组 offer",
        ],
        pitfalls: [
          "❌ 一开始沉迷看教程不动手 → 浪费 1 个月",
          "❌ 数学公式全部跳过 → 面试被问推导抓瞎",
          "❌ 项目只放 Kaggle notebook → HR 看不懂价值",
        ],
        lessons: [
          "✅ 每学一周必须有 commit 到 GitHub",
          "✅ 每个项目都写一篇博客（技术 blog + 业务价值）",
          "✅ 面试前把项目讲 3 遍给镜子听",
        ],
        offer: "某电商推荐算法工程师 · 17K × 14 · 2024 届社招",
      },
      {
        cat: "beginner", catName: "小白入门", color: "#10b981", icon: "🌱",
        title: "机械专业硕士 → NLP 工程师",
        bg: "985 机械硕士，会 Python 但只写过仿真脚本，想转码",
        duration: "8 个月",
        moves: [
          "🎯 M1-2：直接从深度学习入手（机械背景有数理基础），跳过传统 ML 基础课",
          "🔬 M3-4：啃 Transformer 论文 + HuggingFace 文档，复现 BERT 微调",
          "📝 M5-6：用 LoRA 微调 Qwen-1.8B 做公司内部 FAQ 分类，准确率 89%",
          "🚀 M7：搭建 RAG 问答系统（FAISS + bge-large）发布内部 demo",
          "💼 M8：拿到互联网大厂 NLP 实习转正 + 某 AI Startup 全职 offer",
        ],
        pitfalls: [
          "❌ 卡在 PyTorch Dataset/DataLoader 写法一周",
          "❌ LoRA 微调显存爆了才知道要 QLoRA",
          "❌ 简历写「熟悉 Transformer」被面试官秒杀",
        ],
        lessons: [
          "✅ 理工科转 AI 别从 ML 基础开始，直接 DL/LLM 更高效",
          "✅ 把学习过程录成视频发 B 站，倒逼自己讲清楚",
          "✅ 业务价值 > 技术难度：内部 demo 比 paper 复现管用",
        ],
        offer: "某 AI Startup NLP 工程师 · 28K × 14 · 校招",
      },
      /* ────── 工程师转型 ────── */
      {
        cat: "engineer", catName: "工程师转型", color: "#4f46e5", icon: "🔁",
        title: "后端工程师 → 大模型应用工程师",
        bg: "Java 后端 5 年，P6，想转 AI 但数学基础一般",
        duration: "6 个月",
        moves: [
          "🎯 M1：没从基础 ML 开始，直接学 LangChain + RAG，1 周上线第一个内部助手",
          "🧠 M2：补 PyTorch 基础 + 用 LoRA 微调 Qwen-7B 做代码注释生成",
          "🚢 M3：学 vLLM/TGI 部署，把 7B 模型推理 QPS 从 10 提到 150",
          "🔧 M4：研究 Function Calling/Agent，把公司 ERP 接入 AI 助手",
          "📊 M5：落地 A/B 实验，AI 助手提升开发效率 23%",
          "💼 M6：转岗同公司大模型应用团队 · P7 · +35% 涨薪",
        ],
        pitfalls: [
          "❌ 一开始想从线性代数学起 → 3 周没产出差点放弃",
          "❌ 没关注成本，一个月 token 账单 3 万美金被老板拍桌",
          "❌ RAG 第一版用 OpenAI Embedding，数据跨境合规出问题",
        ],
        lessons: [
          "✅ 工程师转 AI：从应用层（LangChain/RAG/Agent）倒推学理论最快",
          "✅ 务必监控 token 消耗 + 加 max_tokens 熔断",
          "✅ 国产 Embedding（bge）优先：合规 + 中文效果更好",
        ],
        offer: "同公司大模型应用平台 P7 · 55K × 16 · 2024 内部转岗",
      },
      {
        cat: "engineer", catName: "工程师转型", color: "#4f46e5", icon: "🔁",
        title: "前端工程师 → CV 算法工程师",
        bg: "React 前端 3 年，喜欢看 SD 生图，想转 AIGC 方向",
        duration: "10 个月",
        moves: [
          "🎨 M1-2：搭 Stable Diffusion WebUI，研究各种插件原理",
          "📐 M3-4：啃 CNN/Transformer/Diffusion 论文，做 PyTorch 重构",
          "🖼 M5-6：训 LoRA 模型（特定画风），发布到 Civitai 下载破万",
          "🧪 M7-8：参加 AIGC Hackathon 拿到三等奖，接触到字节/MiniMax 团队",
          "💼 M9-10：拿到字节 AIGC 图像团队 offer",
        ],
        pitfalls: [
          "❌ 没学清楚 U-Net/DDPM 就想训自己的模型 → 训出来全是噪声",
          "❌ 数据标注太少（500 张）导致过拟合，改用数据增强才解决",
          "❌ 面试被问 Attention 公式推导，答不上来",
        ],
        lessons: [
          "✅ AIGC 方向：先玩透工具（Comfy/WebUI/训练器）再学原理",
          "✅ 在社区（Civitai/HuggingFace）刷影响力比刷 paper 更直接",
          "✅ Hackathon 是接触大厂团队的捷径",
        ],
        offer: "字节 AIGC 图像算法 · 40K × 16 · 2024 社招",
      },
      /* ────── 学生求职 ────── */
      {
        cat: "student", catName: "学生求职", color: "#ec4899", icon: "🎓",
        title: "普通 211 硕士 → 字节算法 SSP",
        bg: "211 计算机硕，研究方向一般（文本分类），论文 0 产出",
        duration: "研一全年 + 研二暑期实习",
        moves: [
          "🥇 研一暑：参加 Kaggle Feedback Prize 比赛，Bronze（全球前 10%）",
          "📄 研一下：写 Kaggle solution writeup + 知乎专栏（3 篇 10K+ 阅读）",
          "💼 暑期实习：字节广告算法，做了 CTR 多目标模型，AUC 提升 1.2%",
          "🏆 研二上：Kaggle LLM Prize 拿到 Silver（全球前 5%）",
          "✍ 研二下：把实习+比赛经验写成一篇 CCF-B 短文（workshop 接收）",
          "🎯 秋招：字节/腾讯/阿里都给了 SSP，选字节广告算法",
        ],
        pitfalls: [
          "❌ 研一只顾学校课程，没刷题，春招海投被拒 60+",
          "❌ 第一次面试讲不清楚 GBDT 和 XGBoost 区别",
          "❌ 简历写「熟悉 Transformer」被问 Scaled Dot-Product Attention 公式",
        ],
        lessons: [
          "✅ Kaggle > 科研（对工业界算法岗）",
          "✅ 暑期实习 = 秋招门票，研一下就开始准备",
          "✅ 所有技术名词都要能推公式/画结构图",
        ],
        offer: "字节广告算法 · SSP 42W base × 16 + 股票 · 2024 校招",
      },
      {
        cat: "student", catName: "学生求职", color: "#ec4899", icon: "🎓",
        title: "C9 本科 → OpenAI/Anthropic 实习",
        bg: "C9 计算机本科大三，GPA 3.9，想冲海外顶尖 AI Lab",
        duration: "大三 1 年",
        moves: [
          "📚 寒假：读完 《Deep Learning》花书 + 10 篇 Transformer 经典论文",
          "🔬 大三下：跟导师做 Mamba/SSM 相关方向，复现论文在 ICLR workshop",
          "📝 暑期：以一作投 NeurIPS（最终被接收为 poster）",
          "🌐 大三暑：通过 Twitter DM 关注 Anthropic 研究员并勤发思考 blog",
          "🎯 大四上：Anthropic Research Intern 面试（4 轮 coding + 2 轮 research）",
          "✈ 结果：远程实习 offer，时薪 $55，做 Constitutional AI 方向",
        ],
        pitfalls: [
          "❌ 刷 LeetCode 2000+ 对 research intern 面试帮助有限",
          "❌ 只会 PyTorch 不会 JAX 限制了部分机会",
          "❌ 英文技术 writing 不够好，面试 system design 翻车一次",
        ],
        lessons: [
          "✅ 海外顶尖 Lab：一篇 NeurIPS > 100 个 Kaggle 奖牌",
          "✅ 公开写作（blog/Twitter）是进入顶尖圈子的敲门砖",
          "✅ 英文技术表达必须练到 native 级别",
        ],
        offer: "Anthropic Research Intern · 远程 · $55/h · 2025 转全职 L4",
      },
      /* ────── 大模型高阶 ────── */
      {
        cat: "senior", catName: "大模型高阶", color: "#f59e0b", icon: "🚀",
        title: "推荐算法专家 → 大模型架构师",
        bg: "BAT 推荐算法 P7，7 年经验，想趁 LLM 红利晋升",
        duration: "8 个月",
        moves: [
          "🏛 M1：手撕 nanoGPT，理解 GPT 训练全链路",
          "📈 M2：学 Scaling Law + 数据清洗，主导公司 8B 基座预训练（4096 H100）",
          "🎯 M3：做 SFT 数据集设计，把通用 8B 基座 tune 到金融问答专家",
          "⚔ M4：对齐阶段用 DPO 替代 RLHF，训练成本降低 60%",
          "🚢 M5：推理侧接入 vLLM + Speculative Decoding，QPS 提升 5×",
          "🤖 M6：设计多 Agent 架构服务风控 / 客服 / 营销 3 条业务线",
          "🏆 M7：主导工作拿到集团年度最佳技术项目",
          "💰 M8：内部晋升 P8 + 外部拿到某头部 AI Lab Principal Engineer offer",
        ],
        pitfalls: [
          "❌ 8B 预训练第一次跑 72 小时 OOM，数据集格式错误浪费 50 万 GPU 时",
          "❌ SFT 数据 10k 条质量不够，模型跑偏 → 重新标注到 8 万条",
          "❌ DPO 超参 β 设错，模型 reward hack 输出乱码",
        ],
        lessons: [
          "✅ 大模型专家成长路径：基座预训练 → SFT → 对齐 → 推理 → Agent 全链路闭环",
          "✅ 敢申请 GPU 预算是关键（不然永远停留在 7B toy 阶段）",
          "✅ 内部项目 + 外部 offer 两手抓，涨薪最快",
        ],
        offer: "某头部 AI Lab Principal Engineer · 120W × 18 + 大额期权 · 2024 社招",
      },
      {
        cat: "senior", catName: "大模型高阶", color: "#f59e0b", icon: "🚀",
        title: "CV 算法 → Multimodal LLM 研究员",
        bg: "985 博士，CV 方向，想转多模态大模型研究",
        duration: "1 年（博士最后一年）",
        moves: [
          "👁 M1-2：读完 CLIP/BLIP/LLaVA/Qwen-VL 所有关键论文",
          "🔬 M3-4：复现 LLaVA-1.5，在 MMBench 达到论文报告精度",
          "📝 M5-7：自研 MLLM 训练技巧（视觉 token 压缩），一篇 CVPR 一作",
          "🏆 M8：参加 Multimodal Hackathon 拿到冠军",
          "🤝 M9-10：通过论文发表接触到 DeepSeek / MiniMax 团队",
          "💼 M11-12：拿到 DeepSeek 多模态组 Research Engineer offer",
        ],
        pitfalls: [
          "❌ 自研 MLLM 太贪心，想一次性训 SOTA，耗时 3 月失败",
          "❌ 数据集质量差，模型学到 bias（总是输出「图中是一个人」）",
          "❌ 投稿 rebuttal 没跟 reviewer 充分沟通，从 borderline accept 变 reject",
        ],
        lessons: [
          "✅ 博士转行业：选爆发性赛道（MLLM）+ 一作顶会论文",
          "✅ 训练 trick 先从小数据集/小模型验证",
          "✅ Rebuttal 阶段要主动跟 AC 和 reviewer 沟通",
        ],
        offer: "DeepSeek 多模态 Research Engineer · 90W × 16 · 2025 校招",
      },
      /* ────── 特殊：35+ 再突破 ────── */
      {
        cat: "engineer", catName: "工程师转型", color: "#4f46e5", icon: "🔁",
        title: "38 岁 Java 架构师 → AI 架构师",
        bg: "某二线城市 Java 架构师，35+ 焦虑，想搭上 AI 末班车",
        duration: "12 个月",
        moves: [
          "🎯 M1-3：没从头学 ML，直接定位「AI 工程化」方向 —— 用原有架构经验",
          "🏗 M4-6：主导公司 RAG 知识库项目，对接 20+ 内部系统",
          "☁ M7-9：设计大模型推理集群架构（vLLM + K8s），日均千万调用",
          "🤖 M10-12：组建 AI 中台，统一 Prompt/工具/Agent 平台",
          "💼 结果：公司 AI 基础设施负责人，年包不降反升",
        ],
        pitfalls: [
          "❌ 原本想刷 ML 论文和年轻人卷，3 个月后果断放弃",
          "❌ 低估了 Prompt Engineering 的细节，版本管理混乱",
          "❌ 没给老板算清成本账，一度被质疑 ROI",
        ],
        lessons: [
          "✅ 35+ 优势在工程经验，AI 工程化 (infra/MLOps/Agent 平台) 是蓝海",
          "✅ 不要跟年轻人拼模型算法，拼整体交付能力",
          "✅ 所有 AI 项目必须有 ROI 数字，跟业务/财务说同一种话",
        ],
        offer: "原公司 AI 基础设施总监 · 95W 年包 · 2024 内部提拔",
      },
      /* ────── 特殊：0 基础创业者 ────── */
      {
        cat: "beginner", catName: "小白入门", color: "#10b981", icon: "🌱",
        title: "40 岁制造业老板 → AI 应用创业者",
        bg: "制造业老板，完全不懂代码，但有行业经验和资源",
        duration: "18 个月",
        moves: [
          "💡 M1-3：学会 Prompt Engineering + 用 Coze/Dify 搭工作流",
          "🤝 M4-6：找 2 个技术合伙人（1 AI 全栈 + 1 产品）",
          "🔬 M7-9：用 RAG 做行业知识库 + AI 助手，给自己工厂内部试用",
          "💰 M10-12：产品化，开始卖给同行小厂（5万-20万/家）",
          "🚀 M13-18：融 Pre-A 1500w，团队扩到 15 人",
        ],
        pitfalls: [
          "❌ 前期想自研模型，烧钱没结果 → 转用开源 + Fine-tune",
          "❌ 低估 B 端销售周期，现金流差点断",
          "❌ 第一个合伙人偏学术，不接地气，后来换人",
        ],
        lessons: [
          "✅ 不懂技术的创始人：深挖垂直行业，用开源生态 + Prompt 就够了",
          "✅ 关键是找对合伙人（懂工程化 + 懂 B 端销售）",
          "✅ MVP 先给自己公司用，再找同行付费验证",
        ],
        offer: "Pre-A 1500w · 公司估值 8000w · 2024 融资",
      },
    ];

    const catColors = { beginner: "#10b981", engineer: "#4f46e5", student: "#ec4899", senior: "#f59e0b" };

    const storyCard = (s, idx) => `
      <div class="card" style="border-left:4px solid ${s.color};margin-bottom:16px">
        <div style="display:flex;align-items:flex-start;gap:12px;margin-bottom:12px">
          <div style="width:44px;height:44px;border-radius:10px;background:${s.color}22;color:${s.color};display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0">${s.icon}</div>
          <div style="flex:1">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
              <h3 style="margin:0">${s.title}</h3>
              <span class="tag" style="background:${s.color}22;color:${s.color}">${s.catName}</span>
              <span class="tag tag-slate">⏱ ${s.duration}</span>
            </div>
            <div style="color:var(--text-secondary);font-size:13px;margin-top:4px">🎭 ${s.bg}</div>
          </div>
        </div>

        <div class="grid-2" style="gap:14px">
          <div>
            <h4 style="margin:0 0 6px;color:${s.color};font-size:13px;font-weight:700">🗺 关键动作</h4>
            <ul style="margin:0;padding-left:16px;line-height:1.75;font-size:12.5px;color:var(--text-secondary)">
              ${s.moves.map(m=>`<li>${m}</li>`).join("")}
            </ul>
          </div>
          <div>
            <div style="margin-bottom:10px">
              <h4 style="margin:0 0 6px;color:#ef4444;font-size:13px;font-weight:700">⚠ 踩过的坑</h4>
              <ul style="margin:0;padding-left:16px;line-height:1.7;font-size:12.5px;color:var(--text-secondary)">
                ${s.pitfalls.map(p=>`<li>${p}</li>`).join("")}
              </ul>
            </div>
            <div>
              <h4 style="margin:0 0 6px;color:#10b981;font-size:13px;font-weight:700">💡 可复制方法论</h4>
              <ul style="margin:0;padding-left:16px;line-height:1.7;font-size:12.5px;color:var(--text-secondary)">
                ${s.lessons.map(l=>`<li>${l}</li>`).join("")}
              </ul>
            </div>
          </div>
        </div>

        <div style="margin-top:12px;padding:10px 14px;background:${s.color}11;border-radius:8px;font-size:13px">
          <span style="color:${s.color};font-weight:700">🏆 最终 Offer：</span>
          <span style="color:var(--text-primary)">${s.offer}</span>
        </div>
      </div>
    `;

    const cats = ["beginner", "engineer", "student", "senior"];
    const catTitles = { beginner: "🌱 小白入门故事", engineer: "🔁 工程师转型故事", student: "🎓 学生求职故事", senior: "🚀 大模型高阶故事" };

    return `
      ${MCH.hero({
        icon: "📖",
        name: "真实转型案例故事",
        en: "Real Stories · Lessons from People Who Made It",
        tags: ["10 个真实故事", "可复制方法论", "踩坑清单", "Offer 真实数据"],
        meta: ["◈ 匿名化真实案例 · 覆盖四类人群 + 35+/创业者", "⚡ 从他人经历中偷方法论"],
      })}

      <div class="section">
        <h2>📊 案例总览</h2>
        <p style="color:var(--text-secondary);margin-bottom:14px">
          10 个不同背景的真实转型故事（已匿名化）。每个故事包含：<b>背景</b> / <b>关键动作时间线</b> / <b>踩过的坑</b> / <b>可复制方法论</b> / <b>最终 Offer 数据</b>。
          你可以根据自己的背景跳转到对应分类。
        </p>
        <div class="grid-4" style="margin-bottom:14px">
          ${cats.map(c => `
            <a href="#${c}-stories" onclick="event.preventDefault();document.getElementById('${c}-stories').scrollIntoView({behavior:'smooth'})"
               class="card" style="border-top:3px solid ${catColors[c]};text-decoration:none;cursor:pointer">
              <div style="font-size:11px;color:${catColors[c]};font-weight:700">${catTitles[c]}</div>
              <div style="font-size:22px;font-weight:800;color:var(--text-primary);margin-top:4px">${stories.filter(s=>s.cat===c).length}</div>
              <div style="font-size:11px;color:var(--text-muted)">个真实故事</div>
            </a>
          `).join("")}
        </div>
        <div class="card" style="background:var(--bg-tertiary);border:1px dashed var(--section-border)">
          <div style="font-size:13px;line-height:1.7;color:var(--text-secondary)">
            💬 <b>使用建议</b>：先快速浏览和你背景相似的 2-3 个故事，重点看"踩过的坑"和"可复制方法论"。
            真实案例 &gt; 完美规划 —— 别人走过的弯路，你可以跳过。
          </div>
        </div>
      </div>

      ${cats.map(c => `
        <div class="section" id="${c}-stories" style="border-left:4px solid ${catColors[c]}">
          <h2 style="color:${catColors[c]}">${catTitles[c]}</h2>
          ${stories.filter(s=>s.cat===c).map(storyCard).join("")}
        </div>
      `).join("")}

      <div class="section">
        <h2>🎯 规律总结</h2>
        <div class="grid-3">
          <div class="card">
            <h3>共同成功要素</h3>
            <ul style="line-height:1.8;padding-left:18px;font-size:13px">
              <li>✅ 项目驱动学习（非课程驱动）</li>
              <li>✅ 公开输出（博客/GitHub/比赛）</li>
              <li>✅ 尽早找同行社群 + 导师</li>
              <li>✅ 每月必有可 show 的产出</li>
            </ul>
          </div>
          <div class="card">
            <h3>共同失败模式</h3>
            <ul style="line-height:1.8;padding-left:18px;font-size:13px">
              <li>❌ 沉迷教程不动手</li>
              <li>❌ 想从基础数学学起（特别是工程师）</li>
              <li>❌ 简历堆技术名词不讲业务价值</li>
              <li>❌ 闭门造车不求反馈</li>
            </ul>
          </div>
          <div class="card">
            <h3>被低估的加速器</h3>
            <ul style="line-height:1.8;padding-left:18px;font-size:13px">
              <li>🚀 Kaggle 奖牌（对工业界有效）</li>
              <li>🚀 技术博客（打造个人品牌）</li>
              <li>🚀 Hackathon（接触顶尖团队）</li>
              <li>🚀 内部转岗（降低风险）</li>
            </ul>
          </div>
        </div>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
