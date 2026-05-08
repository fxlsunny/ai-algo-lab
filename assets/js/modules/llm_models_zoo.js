/* 模块：主流大模型详介（Models Zoo） */
MCH.register("llm_models_zoo", {
  render() {
    return `
      ${MCH.hero({
        icon: "🌐",
        name: "主流大模型 Models Zoo",
        en: "Mainstream LLMs · Qwen / DeepSeek / OpenAI / Claude / Llama / Gemini / Mistral",
        tags: ["Qwen3", "DeepSeek-V3/R1", "GPT-4/4o/o1", "Claude 3.5/4", "Llama 3", "Gemini 2.5"],
        meta: ["◈ 中美闭源 + 全球开源对比", "⚡ 选型 / 训练范式 / 商业模式 / 趋势"],
      })}

      <div class="section" style="background:linear-gradient(135deg,#dbeafe 0%,#e0e7ff 100%);border:1px solid #a5b4fc;">
        <h2 style="color:#3730a3;border:none;padding:0;margin:0 0 6px 0;">🆕 2024-2026 LLM 大局：闭源四强（OpenAI / Anthropic / Google / xAI）+ 中美双开源生态（Qwen / DeepSeek 国内、Llama / Mistral / Gemma 国外）+ 推理时代（o1 / R1 / Claude Thinking）+ Agent 化（Computer Use / Skills）</h2>
      </div>

      <div class="section">
        <h2>1. 总览：全球 LLM 阵营</h2>
        <div class="mermaid">
flowchart TB
    subgraph SC[闭源前沿]
      OA[🇺🇸 OpenAI<br/>GPT-3 → 4 → 4o → o1/o3 → GPT-5]
      AN[🇺🇸 Anthropic<br/>Claude 1 → 3 → 3.5 → 4 + Skills]
      GO[🇺🇸 Google DeepMind<br/>PaLM → Gemini 1 → 1.5 → 2.0/2.5]
      XA[🇺🇸 xAI<br/>Grok 1 → 2 → 3 → 4]
    end
    subgraph OS[开源 / 半开源]
      LL[🇺🇸 Meta Llama 1 → 3.x → 4]
      MS[🇫🇷 Mistral · 7B → Mixtral → Large/Pixtral]
      DS[🇨🇳 DeepSeek<br/>V1 → V2 → V3 → R1 → V3.2-Exp]
      QW[🇨🇳 Qwen<br/>1 → 2 → 2.5 → 3 + Coder/Math/VL/Omni]
      GLM[🇨🇳 智谱 GLM-4 → ChatGLM-3<br/>+ CogView/CogVideo]
      MN[🇨🇳 MiniMax<br/>abab → MiniMax-01 / Speech]
      KIMI[🇨🇳 月之暗面 Kimi · K2]
      SKY[🇨🇳 昆仑万维 Skywork]
      DOU[🇨🇳 字节豆包 Doubao]
      MX[🇨🇳 腾讯 Hunyuan]
      BD[🇨🇳 百度 文心 ERNIE]
      MIST[🇫🇷 Mistral]
      GEMMA[🇺🇸 Google Gemma]
      PHI[🇺🇸 Microsoft Phi]
    end
    SC --> OS
        </div>
      </div>

      <div class="section">
        <h2>2. 闭源四强深度对比</h2>
        <table class="table">
          <thead>
            <tr><th>模型族</th><th>厂商</th><th>关键代际</th><th>窗口</th><th>独门优势</th><th>典型短板</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>GPT 系</b></td><td>OpenAI</td>
              <td>GPT-3.5 (2022.11) · GPT-4 (2023.03) · GPT-4 Turbo · GPT-4o (2024.05 端到端 Omni) · o1-preview (2024.09) · o1 / o1-pro (2024.12) · o3 (2025) · GPT-5 (2025-26)</td>
              <td>128k-200k</td>
              <td>🏆 推理 + 编程 + 工具调用 + Realtime API · 用户基数大 · ChatGPT/GPTs/Operator 生态</td>
              <td>价格高 · 中文不如 Qwen/DeepSeek · 部分能力不公开权重</td>
            </tr>
            <tr>
              <td><b>Claude 系</b></td><td>Anthropic</td>
              <td>Claude 1 (2023.03) · 2 · 3 Haiku/Sonnet/Opus (2024.03) · 3.5 Sonnet (2024.06/10) · 3.7 / Sonnet 4 / Opus 4 (2025) · Skills (2025)</td>
              <td>200k-1M</td>
              <td>🏆 长文档 + 代码 + 安全（宪法 AI） · Computer Use · Artifacts · Projects · Skills 生态</td>
              <td>API 受限地区多 · 推理速度比 GPT 慢 · 工具调用 JSON 严格</td>
            </tr>
            <tr>
              <td><b>Gemini 系</b></td><td>Google</td>
              <td>PaLM 2 → Gemini 1.0 (2023.12) · 1.5 Pro (2024.02 1M ctx) · 2.0 Flash · 2.5 Pro (2025 推理) · Gemini Live</td>
              <td>1M-2M</td>
              <td>🏆 超长上下文 · 原生多模态（视频原生）· Workspace 集成 · TPU 算力优势</td>
              <td>早期翻车多 · 创意写作弱于 Claude · API 体验不稳</td>
            </tr>
            <tr>
              <td><b>Grok 系</b></td><td>xAI</td>
              <td>Grok-1 (开源 314B MoE) · Grok-2 · Grok-3 (Colossus 200k GPU) · Grok-4 (2025)</td>
              <td>128k+</td>
              <td>X 实时数据 · "Fun mode" 无审查 · 大算力快速迭代</td>
              <td>生态弱 · 准确度不稳 · 商业化局限于 X / Premium+</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 中国大模型阵营深度对比</h2>
        <table class="table">
          <thead>
            <tr><th>模型族</th><th>厂商</th><th>开闭源</th><th>关键代际</th><th>独门优势</th><th>典型场景</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>Qwen 通义千问</b></td><td>阿里</td><td>🏆 开源</td>
              <td>Qwen 1 (2023.08) · Qwen 1.5 / 2 / 2.5 (0.5B-72B/110B) · Qwen2-VL · Qwen2.5-Coder/Math · Qwen3 (2025 含 235B-A22B MoE) · Qwen3-VL · Qwen3-Omni · QwQ-32B (推理)</td>
              <td>🏆 全球开源 LLM 总下载量第一 · 模型矩阵最完整（基础/编程/数学/视觉/语音/Omni/Edge）· Apache 2.0 商用</td>
              <td>企业私有化部署、端侧、电商客服、千问助手</td>
            </tr>
            <tr>
              <td><b>DeepSeek</b></td><td>幻方量化系</td><td>🏆 开源 MIT</td>
              <td>DeepSeek-V1 (67B) · V2 (236B MoE) · V2.5 · V3 (671B-A37B 2024.12) · R1 (推理 2025.01) · V3.2-Exp (Sparse Attn 2025)</td>
              <td>🏆 训练效率 SOTA · MoE 极致优化 · MLA / DualPipe / FP8 训练 · R1 开源推理路线 · 极便宜 API</td>
              <td>研究复用、复杂推理、代码、工程性价比最高</td>
            </tr>
            <tr>
              <td><b>GLM / 智谱清言</b></td><td>智谱 AI</td><td>开源 + 商用</td>
              <td>GLM-130B → ChatGLM 2/3 → GLM-4 · GLM-4-Air/Plus · CogView/CogVideo 多模态</td>
              <td>清华学院派 · 代码能力强 · 生态完整（Qwen 之后第二大开源中文）</td>
              <td>政企客户、教育、双语任务</td>
            </tr>
            <tr>
              <td><b>Doubao 豆包 / Skylark</b></td><td>字节</td><td>闭源 SaaS</td>
              <td>云雀 → 豆包 1.5 Pro · Doubao Vision · Doubao-1.5-pro-32k</td>
              <td>🏆 价格屠夫 · 抖音/飞书生态 · 多模态投入大</td>
              <td>消费级 ToC 应用、SaaS API 调用量第一</td>
            </tr>
            <tr>
              <td><b>Hunyuan 混元</b></td><td>腾讯</td><td>开源 + 商用</td>
              <td>Hunyuan-Large (389B-A52B 开源) · Hunyuan3D · HunyuanVideo · HunyuanDiT</td>
              <td>开源 MoE · 多模态产品矩阵 · 微信 / 元宝 接入</td>
              <td>微信生态、视频生成、3D</td>
            </tr>
            <tr>
              <td><b>ERNIE 文心</b></td><td>百度</td><td>闭源 + 部分开源</td>
              <td>文心 1.0-4.5 · 文心轻舟 · ERNIE-Speed</td>
              <td>政企 ToB 大客户、搜索集成</td>
              <td>搜索增强、ToB 行业大模型</td>
            </tr>
            <tr>
              <td><b>Kimi (Moonshot)</b></td><td>月之暗面</td><td>闭源 SaaS · K2 开源</td>
              <td>Kimi Chat (2024 长上下文 200 万字) · Kimi K2 (2025 1T MoE 开源)</td>
              <td>🏆 长文档 ToC 现象级 · K2 开源后跻身全球前列</td>
              <td>论文 / 法律 / 学习场景</td>
            </tr>
            <tr>
              <td><b>MiniMax</b></td><td>稀宇</td><td>闭源 + MiniMax-01 开源</td>
              <td>abab 6/7 · MiniMax-01 (456B Lightning Attention 2025) · Speech</td>
              <td>Lightning Attention · 海螺 AI · 海外 Talkie ToC</td>
              <td>对话陪伴、长上下文、ToC 海外</td>
            </tr>
            <tr>
              <td><b>SenseChat / 商量</b></td><td>商汤</td><td>闭源</td>
              <td>商量 5 / 5.5 / 6 · SenseNova 系列</td>
              <td>多模态 + 视觉强 · 端云一体</td>
              <td>金融、政务、智能驾驶</td>
            </tr>
            <tr>
              <td><b>Yi 零一万物</b></td><td>李开复</td><td>开源</td>
              <td>Yi-34B / Yi-1.5 / Yi-Lightning</td>
              <td>开源中英平衡 · 推理速度优化</td>
              <td>双语、ToC API</td>
            </tr>
            <tr>
              <td><b>Baichuan 百川</b></td><td>百川智能</td><td>开源 + 闭源</td>
              <td>Baichuan 2 / 3 / 4</td>
              <td>金融 / 医疗垂直</td>
              <td>医疗 / 金融行业</td>
            </tr>
            <tr>
              <td><b>StepFun 阶跃星辰</b></td><td>阶跃</td><td>闭源 + 部分开源</td>
              <td>Step-1V/2 · Step-Video · Step-Audio</td>
              <td>多模态产品矩阵</td>
              <td>视频 + 语音原生</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>4. 全球开源阵营</h2>
        <table class="table">
          <thead><tr><th>模型族</th><th>厂商</th><th>规模</th><th>许可</th><th>核心亮点</th></tr></thead>
          <tbody>
            <tr><td><b>Llama 1 → 3.3 → 4</b></td><td>Meta</td><td>1B-405B / 4 系 17B-2T MoE</td><td>Llama Community License</td><td>🏆 开源王者 · 生态最大 · Llama 3.1 405B 接近 GPT-4 · Llama 4 走 MoE</td></tr>
            <tr><td><b>Mistral / Mixtral / Pixtral</b></td><td>Mistral AI</td><td>7B-141B MoE / 124B</td><td>Apache 2.0 / Mistral Research</td><td>欧洲玩家 · 7B 经典 · Mixtral SMoE 开创 · Pixtral 多模态</td></tr>
            <tr><td><b>Gemma 1 / 2 / 3</b></td><td>Google</td><td>2B-27B</td><td>Gemma Terms</td><td>Gemini 同源开源 · 端侧友好</td></tr>
            <tr><td><b>Phi 1 / 2 / 3 / 4</b></td><td>Microsoft</td><td>1.3B-14B</td><td>MIT</td><td>"Textbook is all you need" · 小而强 · Phi-4 14B 接近 Llama 70B</td></tr>
            <tr><td><b>Falcon</b></td><td>TII (UAE)</td><td>7B/40B/180B/Mamba</td><td>Apache 2.0</td><td>中东开源旗舰 · Falcon Mamba</td></tr>
            <tr><td><b>Command R / R+</b></td><td>Cohere</td><td>35B / 104B</td><td>CC-BY-NC</td><td>RAG 优化 · 多语种企业级</td></tr>
            <tr><td><b>OLMo / Tülu / Molmo</b></td><td>AI2</td><td>1B-13B</td><td>Apache 2.0</td><td>🏆 完全开放（数据 + 训练代码 + 中间 ckpt）</td></tr>
            <tr><td><b>Mamba / Jamba / Zamba</b></td><td>多家</td><td>7B-52B</td><td>Apache 2.0</td><td>SSM 替代 Transformer · 长上下文友好</td></tr>
            <tr><td><b>SmolLM / TinyLlama</b></td><td>HF / TinyLlama 项目</td><td>135M-1.7B</td><td>Apache 2.0</td><td>端侧 / 教育 / 实验</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 训练范式核心技术对比</h2>
        <table class="table">
          <thead><tr><th>维度</th><th>2022 范式 (GPT-3 时代)</th><th>2024-2025 范式</th><th>代表创新</th></tr></thead>
          <tbody>
            <tr><td>架构</td><td>Dense Transformer</td><td>MoE / SSM / Hybrid / Native Multimodal</td><td>Mixtral · Mamba · Jamba · Chameleon</td></tr>
            <tr><td>注意力</td><td>FlashAttention</td><td>MLA · GQA · MQA · Sliding Window · Lightning Attn</td><td>DeepSeek MLA · Llama 3 GQA · MiniMax-01</td></tr>
            <tr><td>训练数据</td><td>RefinedWeb / The Pile (~1.4T tokens)</td><td>15-30T tokens · 数据合成 · 多语种</td><td>Llama 3 · Qwen2.5</td></tr>
            <tr><td>训练精度</td><td>BF16</td><td>FP8 / 混合精度</td><td>DeepSeek-V3 FP8 训练</td></tr>
            <tr><td>对齐 / RLHF</td><td>InstructGPT 三阶段</td><td>DPO · IPO · KTO · SimPO · Online RLHF</td><td>Llama 3 / DeepSeek-R1 · Tulu 3</td></tr>
            <tr><td>推理</td><td>纯生成</td><td>Test-time scaling · CoT · Reasoning RL</td><td>o1 · DeepSeek-R1 · QwQ · Claude Thinking</td></tr>
            <tr><td>部署</td><td>vLLM / TGI</td><td>SGLang · vLLM v1 · TensorRT-LLM · 推测解码</td><td>EAGLE · Medusa · Lookahead</td></tr>
            <tr><td>评测</td><td>MMLU / GSM8K</td><td>LiveBench · GPQA · SWE-bench · ARC-AGI · 长上下文</td><td>SimpleBench · MT-Bench-Live</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 推理 (Reasoning) 模型革命</h2>
        ${MCH.info(`
          <b>2024.09 OpenAI o1 是分水岭</b>：把 RL + Long-CoT 在 test-time 大规模 scale，引爆"推理时代"。
          <br/>开源跟进路径：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>DeepSeek-R1 (2025.01)</b>：MIT 开源、数学/代码 SOTA · 蒸馏小模型(1.5B-70B) 全开源</li>
            <li><b>QwQ-32B / Qwen3-Thinking</b>：阿里推理系，性能媲美 o1-preview</li>
            <li><b>Claude 3.7 Sonnet / Opus 4 Thinking</b>：Anthropic 思考模式</li>
            <li><b>Gemini 2.5 Pro Thinking</b>：Google 思考模式</li>
            <li><b>Kimi K0/K1.5/K2</b>：月之暗面长 CoT 推理</li>
          </ul>
          核心三件套：(1) 自然语言长 CoT；(2) Process Reward Model；(3) 大规模 RL 训练。
        `, "tip")}
      </div>

      <div class="section">
        <h2>7. 选型决策树</h2>
        <div class="grid-3">
          <div class="card">
            <h3>🏢 企业 ToB / 私有部署</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>追求极致效果：DeepSeek-V3 / Qwen3-235B / Llama 4</li>
              <li>性价比：Qwen2.5-72B / Mistral Large / Hunyuan-Large</li>
              <li>边缘 / 端侧：Qwen3-4B / Phi-4 / Gemma 3</li>
              <li>金融 / 医疗：百川 / 商量 / Hunyuan</li>
            </ul>
          </div>
          <div class="card">
            <h3>📱 ToC SaaS API</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>对话最强：GPT-4o / Claude 3.5 Sonnet</li>
              <li>长文档：Gemini 2.5 Pro / Claude / Kimi</li>
              <li>价格优先：Doubao / DeepSeek / Gemini Flash</li>
              <li>无审查 / 开放：Grok / Mistral</li>
            </ul>
          </div>
          <div class="card">
            <h3>🧠 推理 / 复杂任务</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>顶配：o1 / o3-mini-high / Claude 4 Thinking</li>
              <li>开源：DeepSeek-R1 / QwQ-32B</li>
              <li>数学：DeepSeek-Math / Qwen2.5-Math</li>
              <li>代码：Claude Sonnet 4 / DeepSeek-Coder</li>
            </ul>
          </div>
          <div class="card">
            <h3>🎭 多模态</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>视觉：GPT-4o / Gemini 2.5 / Qwen2.5-VL</li>
              <li>视频：Gemini 2.5 / Qwen2.5-VL</li>
              <li>语音：GPT-4o Realtime / Qwen2.5-Omni</li>
              <li>图生：DALL-E 3 / Imagen 3 / FLUX.1</li>
            </ul>
          </div>
          <div class="card">
            <h3>🦾 Agent / Computer Use</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>桌面 GUI：Claude 3.5 + Computer Use</li>
              <li>编程：Claude Code · Cursor · GPT-Codex</li>
              <li>研究：Perplexity Pro · Deep Research (OpenAI/Gemini)</li>
              <li>开源：OpenHands · Cline · Aider</li>
            </ul>
          </div>
          <div class="card">
            <h3>💰 价格敏感</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>API：DeepSeek-V3 ¥0.5/M tok · Doubao</li>
              <li>免费层：Gemini Flash · Mistral Free</li>
              <li>自托管：Qwen3-Instruct / Llama 3.3</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>8. 商业模式格局</h2>
        <table class="table">
          <thead><tr><th>模式</th><th>主要玩家</th><th>典型客单价</th><th>商业护城河</th></tr></thead>
          <tbody>
            <tr><td>📱 ToC 订阅</td><td>ChatGPT Plus/Pro · Claude Pro · Gemini Advanced · Kimi · Doubao</td><td>$20-200/mo</td><td>用户体验 + 模型能力</td></tr>
            <tr><td>☁️ API 计量</td><td>所有大厂</td><td>$/M tokens</td><td>价格 + 延迟 + SLA</td></tr>
            <tr><td>🏢 私有化部署</td><td>百度 / 智谱 / 商汤 / 火山引擎 / Cohere</td><td>百万级 / 项目</td><td>合规 + 行业 know-how</td></tr>
            <tr><td>🎯 ToB 行业</td><td>金融 / 医疗 / 法律 / 政务定制</td><td>数百万 - 数千万</td><td>数据壁垒 + 客户关系</td></tr>
            <tr><td>📦 模型 + Agent SaaS</td><td>Cursor / Devin / Bolt / Replit Agent</td><td>$20-500/mo per seat</td><td>工作流 + 集成</td></tr>
            <tr><td>🔌 模型即基础设施</td><td>Together / Fireworks / Groq / Cerebras / 硅基流动</td><td>$/tok 折扣</td><td>推理性能 + 多模型聚合</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>9. 未来发展方向 & 难点</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🧠 推理深度 vs 成本</h3>
            <p style="font-size:13px;line-height:1.6">o1/R1 路线让"思考久 = 更准"，但 token 量爆炸。需要：(1) Process RM 优化；(2) 蒸馏长 CoT；(3) 自适应思考预算。OpenAI o3-mini-high 已开始动态分配。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📦 数据墙</h3>
            <p style="font-size:13px;line-height:1.6">高质量人类文本 ≈ 30T token 已被吃透。下一步：(1) 大规模合成数据（Phi 系列已验证）；(2) 多模态扩容；(3) 智能体交互产生自演化数据。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🦾 Agent 化</h3>
            <p style="font-size:13px;line-height:1.6">从"会聊"到"会做"。Computer Use / Browser Use / Skills / OpenHands。难点：长任务可靠性、错误恢复、安全沙箱。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📺 真长上下文</h3>
            <p style="font-size:13px;line-height:1.6">声称 1M-2M tokens 但实际"中段失忆"。需要：(1) 训练数据真长；(2) 检索增强一体；(3) 注意力稀疏化。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📱 端侧</h3>
            <p style="font-size:13px;line-height:1.6">Apple Intelligence 3B、Gemini Nano、Qwen3-4B。隐私 + 延迟 + 离线必备。难点：3-7B 范围内做到 70B 体验。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>⚖️ 安全与对齐</h3>
            <p style="font-size:13px;line-height:1.6">越强越难对齐。Constitutional AI、Sleeper Agents、Reward Hacking 等问题待解。开源后能力扩散加剧滥用风险。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🌐 全模态统一</h3>
            <p style="font-size:13px;line-height:1.6">从串接式 (LLaVA) → 原生融合 (GPT-4o, Qwen2.5-Omni)。任意模态输入 → 任意模态输出。算力 + 数据是壁垒。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>💎 推理硬件</h3>
            <p style="font-size:13px;line-height:1.6">NVIDIA H100/H200 → B200/GB200 → Rubin；Groq/Cerebras 大芯片；TPU v5p/v6e；国产 910B/910C；推理成本下降 10×/年。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>10. 客观评测 / 排行榜</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://lmarena.ai/" target="_blank">Chatbot Arena (LMSYS)</a> — 人类盲评 ELO，事实标准</li>
          <li><a href="https://livebench.ai/" target="_blank">LiveBench</a> — 防泄题 / 滚动更新</li>
          <li><a href="https://www.swebench.com/" target="_blank">SWE-bench / Verified</a> — 真实开源仓库 issue 修复</li>
          <li><a href="https://arcprize.org/leaderboard" target="_blank">ARC-AGI</a> — 抽象推理</li>
          <li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard" target="_blank">Open LLM Leaderboard</a></li>
          <li><a href="https://opencompass.org.cn/" target="_blank">OpenCompass</a> · <a href="https://www.superclueai.com/" target="_blank">SuperCLUE</a> — 中文榜</li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
