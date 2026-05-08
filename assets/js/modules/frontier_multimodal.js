/* 模块：多模态模型（前沿） */
MCH.register("frontier_multimodal", {
  render() {
    const code = `# ============================================================================
# 多模态大模型（VLM / MLLM）核心范式
# 经典三段式：Image Encoder → Modality Aligner → LLM Decoder
# 代表实现：LLaVA / Qwen-VL / InternVL / GPT-4V / Gemini / Claude 3
# ============================================================================

class VLM(nn.Module):
    """Vision-Language Model 通用骨架（LLaVA-style）"""
    def __init__(self, vit, projector, llm):
        super().__init__()
        self.vit       = vit          # 视觉编码器（CLIP-ViT-L/14、SigLIP、EVA-02）
        self.projector = projector    # 模态对齐（MLP / Q-Former / Resampler）
        self.llm       = llm          # 文本解码器（Qwen2、Llama-3、Vicuna）

    def forward(self, image, text_ids):
        # 1) 抽取视觉 token：(B, N_patches, D_vit)，N≈576 for 336²/14
        vfeat = self.vit(image, output_hidden_states=True).hidden_states[-2]
        # 2) 投影到 LLM 词向量空间：(B, N_v, D_llm)
        vtok = self.projector(vfeat)
        # 3) 文本 token：(B, T, D_llm)
        ttok = self.llm.embed_tokens(text_ids)
        # 4) 拼接：[<image>vtok</image>, ttok] → 走 LLM 自回归解码
        x = torch.cat([vtok, ttok], dim=1)
        return self.llm.forward_with_embeds(x)

# ─── 模态对齐三大流派 ────────────────────────────────────────────
# (a) Linear / MLP Projector（LLaVA-1.5）：最简单 2 层 MLP，参数 < 30M
# (b) Q-Former（BLIP-2）：可学习 query 抽取 32 个固定长度视觉 token
# (c) Cross-Attention Resampler（Flamingo）：每隔几层 LLM 插入交叉注意力

# ─── 训练范式：两阶段对齐 + 指令微调 ────────────────────────────
# Stage 1 · Pretrain Aligner：冻结 ViT/LLM，只训练 projector
#          数据：image-caption pair（CC3M / LAION-558K）
# Stage 2 · Visual Instruction Tuning：解冻 LLM（可选 LoRA），训练 projector + LLM
#          数据：图文混合多轮对话（LLaVA-Instruct-150K、ShareGPT4V、Cauldron）

# ─── 高分辨率 / 任意宽高比策略（2024 主流） ─────────────────────
# AnyRes（LLaVA-Next）   ：把图切成多 sub-image，每块 336²
# Pixel Shuffle / Compress：InternVL2 用 pixel-shuffle 把 token 压到 1/4
# Native Resolution      ：Qwen2-VL 引入 M-RoPE，支持任意分辨率 + 视频原生输入
# Visual Token Sparsing  ：FastV / VTW，推理时丢掉 50%+ 视觉 token
`;

    const sora_code = `# ============================================================================
# 全模态融合 · Native MoT（Mixture-of-Transformers，统一所有模态）
# Chameleon (Meta 2024) / Show-o (NUS 2024) / Janus (DeepSeek 2024)
# ============================================================================

# 把图像 / 音频 / 视频统一离散化（VQ-VAE token），与文本 token 共享一套 vocab
# 早期融合（Early Fusion）：所有模态 token 拼接，单 Transformer 解码

# 离散化：vision tokens (8192 codes) + text tokens (32k) → vocab=40k
# 训练目标：next-token prediction（图像/文本/语音都是 token 序列）

class UnifiedTransformer(nn.Module):
    def forward(self, mixed_tokens):
        # mixed_tokens 形如 [text..., <bos_image>, vis_tok..., <eos_image>, text...]
        return self.transformer(mixed_tokens)  # 只用一个 LM Loss
`;

    return `
      ${MCH.hero({
        icon: "🎭",
        name: "多模态大模型",
        en: "Multimodal Foundation Models · VLM / MLLM / Omni-Models",
        tags: ["VLM", "GPT-4V", "Qwen-VL", "Gemini", "InternVL", "AnyRes", "Native"],
        meta: ["◈ 视觉 + 语言 + 音频统一表征", "⚡ 2024 是 Omni-Model Year"],
      })}

      <div class="section" style="background:linear-gradient(135deg,#fef3c7 0%,#fde68a 100%);border:1px solid #fcd34d;">
        <h2 style="color:#92400e;border:none;padding:0;margin:0 0 6px 0;">🆕 关键里程碑：CLIP 2021 → BLIP-2 2023 → LLaVA 2023 → GPT-4V 2023 → Sora 2024 → Qwen2-VL 2024 → GPT-4o 2024 → Gemini 2.5 2025 → Qwen2.5-Omni / Janus / Chameleon 2024-25</h2>
      </div>

      <div class="section">
        <h2>1. 发展史：从对比学习到统一 Omni 架构</h2>
        <div class="mermaid">
flowchart TB
    A[CLIP 2021<br/>对比学习对齐<br/>4亿图文对] --> B[ALIGN/CoCa 2021-22<br/>更大规模]
    B --> C[Flamingo 2022<br/>DeepMind<br/>少样本 VLM]
    C --> D[BLIP-2 2023<br/>Q-Former 桥接]
    D --> E[LLaVA 2023<br/>开源 VLM 范式]
    E --> F[GPT-4V 2023.09<br/>商业化里程碑]
    F --> G[Gemini 1.0 2023.12<br/>原生多模态训练]
    G --> H[Claude 3 / Sonnet 2024<br/>视觉推理]
    G --> I[Qwen-VL → Qwen2-VL 2024<br/>开源 SOTA]
    I --> J[InternVL 2 2024<br/>76B 旗舰]
    F --> K[Sora 2024.02<br/>DiT 视频生成]
    H --> L[GPT-4o 2024.05<br/>Omni 端到端]
    L --> M[Gemini 2.0/2.5 2024-25<br/>多模态推理]
    L --> N[Qwen2.5-Omni / Janus 2024<br/>统一离散 token]
    M --> O[🆕 视频/语音/图统一<br/>2025-26 Omni Year]
    N --> O
        </div>
      </div>

      <div class="section">
        <h2>2. 三大主流架构对比</h2>
        <table class="table">
          <thead><tr><th>架构</th><th>原理</th><th>代表</th><th>优势</th><th>劣势</th></tr></thead>
          <tbody>
            <tr>
              <td><b>对比学习</b><br/>(Dual-Encoder)</td>
              <td>图/文分别编码，对比 loss 拉近正样本</td>
              <td>CLIP, ALIGN, SigLIP, EVA-CLIP</td>
              <td>检索快、零样本分类强</td>
              <td>不能生成、不会推理</td>
            </tr>
            <tr>
              <td><b>桥接式</b><br/>(Connector + LLM)</td>
              <td>视觉 Encoder + Projector + LLM 解码</td>
              <td>BLIP-2, LLaVA, MiniGPT-4, Qwen-VL, InternVL</td>
              <td>复用预训练 LLM、训练成本低</td>
              <td>视觉信息有损压缩</td>
            </tr>
            <tr>
              <td><b>原生融合</b><br/>(Native Multimodal)</td>
              <td>预训练就混合模态，单一 Transformer</td>
              <td>Gemini 1.0+, GPT-4o, Chameleon, Janus, Show-o, Qwen2.5-Omni</td>
              <td>跨模态理解上限高、可双向生成</td>
              <td>训练贵、对齐难、闭源占主导</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码：LLaVA-style VLM 骨架</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 全模态架构（Omni-Model）核心</h2>
        ${MCH.code(sora_code, "python")}
        ${MCH.info(`
          <b>2024 一个判断：</b>桥接式（LLaVA 范式）适合开源开发者快速跟进；但 Frontier 厂商（OpenAI / Google / Anthropic / Meta）正集体押注<b>原生融合</b>路线 —— 因为它在长视频理解、跨模态推理、多模态生成上限更高。开源社区也在快速跟进：DeepSeek <b>Janus-Pro</b>、Meta <b>Chameleon</b>、阿里 <b>Qwen2.5-Omni</b>、智谱 <b>CogVLM2-Video</b>。
        `, "tip")}
      </div>

      <div class="section">
        <h2>5. 代表模型横评（2024-2025）</h2>
        <table class="table">
          <thead><tr><th>模型</th><th>厂商</th><th>规模</th><th>开闭源</th><th>核心亮点</th></tr></thead>
          <tbody>
            <tr><td><b>GPT-4V / GPT-4o</b></td><td>OpenAI</td><td>未公开</td><td>闭源</td><td>商业 SOTA · GPT-4o 端到端语音/视觉/文本一次推理</td></tr>
            <tr><td><b>Gemini 1.5 / 2.0 / 2.5 Pro</b></td><td>Google</td><td>未公开</td><td>闭源</td><td>1M-2M tokens 长上下文、视频原生输入</td></tr>
            <tr><td><b>Claude 3 / 3.5 / 4 Sonnet</b></td><td>Anthropic</td><td>未公开</td><td>闭源</td><td>视觉推理 + 长文档 OCR + Computer Use</td></tr>
            <tr><td><b>Qwen-VL → Qwen2-VL → Qwen2.5-VL</b></td><td>阿里</td><td>2B/7B/72B</td><td>开源</td><td>🏆 开源 VLM SOTA · 任意分辨率 · 视频原生</td></tr>
            <tr><td><b>InternVL 1.5 / 2 / 2.5</b></td><td>OpenGVLab</td><td>1B-78B</td><td>开源</td><td>动态分辨率 · 接近 GPT-4V 商业级</td></tr>
            <tr><td><b>LLaVA / LLaVA-1.5 / NeXT</b></td><td>U.Wisconsin</td><td>7B/13B/34B</td><td>开源</td><td>开源 VLM 启蒙、生态最广</td></tr>
            <tr><td><b>MiniCPM-V 2.6</b></td><td>面壁智能</td><td>8B</td><td>开源</td><td>端侧推理 · 单图 1.8M 像素</td></tr>
            <tr><td><b>DeepSeek-VL2 / Janus-Pro</b></td><td>DeepSeek</td><td>4B/27B</td><td>开源</td><td>🆕 MoE-VLM · 双塔解耦理解/生成</td></tr>
            <tr><td><b>Chameleon</b></td><td>Meta</td><td>7B/34B</td><td>开源</td><td>原生离散 token · 早期融合</td></tr>
            <tr><td><b>Pixtral-12B / Large</b></td><td>Mistral</td><td>12B/124B</td><td>开源</td><td>欧洲玩家代表 · 任意分辨率</td></tr>
            <tr><td><b>Molmo</b></td><td>AI2</td><td>1B-72B</td><td>开源</td><td>纯开源数据训练 · 接近 GPT-4V</td></tr>
            <tr><td><b>NVLM-D-72B</b></td><td>NVIDIA</td><td>72B</td><td>开源</td><td>解码器架构对比研究</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 关键工程难点</h2>
        <div class="grid-3">
          <div class="card">
            <h3>🖼 高分辨率视觉 Token</h3>
            <p style="font-size:13px;line-height:1.6">单张 4K 图直接喂 ViT 会爆炸。AnyRes / 切块 / Pixel Shuffle / 原生分辨率 + 2D-RoPE 是当前主流方案。视频更难：1 分钟 30fps × 200 token/帧 = 360k token。</p>
          </div>
          <div class="card">
            <h3>🌀 模态间幻觉</h3>
            <p style="font-size:13px;line-height:1.6">VLM 经常"看图说瞎话"（Object Hallucination）。POPE / MMHal-Bench 评测、对比解码（VCD）、RLHF 视觉对齐是缓解方案。</p>
          </div>
          <div class="card">
            <h3>🔀 训练数据偏置</h3>
            <p style="font-size:13px;line-height:1.6">CC3M / LAION 等大规模 caption 噪声大，专家标注数据 (LLaVA-Instruct) 才能解锁推理能力，但贵。Synthesized GPT-4V caption 是当前主流"刷分捷径"。</p>
          </div>
          <div class="card">
            <h3>📺 视频长上下文</h3>
            <p style="font-size:13px;line-height:1.6">视频 = 海量帧 token。Token Pruning（FastV）、关键帧采样、视觉记忆压缩（VideoChat-Flash）、StreamingLLM 是关键。</p>
          </div>
          <div class="card">
            <h3>🗣 双向多模态生成</h3>
            <p style="font-size:13px;line-height:1.6">不仅看懂还要生成图。统一 token（Chameleon/Janus）vs 解耦 head（Show-o）vs 串接 Diffusion（GPT-4o 推测路线）三派并存。</p>
          </div>
          <div class="card">
            <h3>⚡ 端侧推理</h3>
            <p style="font-size:13px;line-height:1.6">MiniCPM-V / Phi-3-Vision / Qwen2-VL-2B 把 VLM 压到 2-8B，配合 INT4 / FP8 量化在手机上跑通，开启端侧多模态时代。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>7. 主流应用场景</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>典型需求</th><th>代表落地</th></tr></thead>
          <tbody>
            <tr><td>📋 文档智能 (DocVQA / OCR-Free)</td><td>表格、合同、票据理解</td><td>Qwen2-VL · InternVL · GPT-4o · 钉钉智能助理</td></tr>
            <tr><td>🛍 电商导购 / 视觉搜索</td><td>看图找同款、智能客服</td><td>淘宝问问、京东数科、小红书"识图"</td></tr>
            <tr><td>🏥 医学影像辅助诊断</td><td>X-Ray / CT / 病理切片</td><td>LLaVA-Med, RadFM, PathChat</td></tr>
            <tr><td>🚘 智能驾驶 World Model</td><td>多视角 + 时序场景理解</td><td>DriveVLM, GAIA-1, EmerNeRF</td></tr>
            <tr><td>🎬 视频内容理解</td><td>剪辑、总结、问答</td><td>Video-LLaVA, VideoChat, Gemini Long-Video</td></tr>
            <tr><td>📱 GUI Agent (屏幕理解)</td><td>识别界面元素 + 操作</td><td>Claude Computer Use · Anthropic 3.5 · CogAgent · OS-Atlas</td></tr>
            <tr><td>🎙 实时语音助手</td><td>边听边看边说</td><td>GPT-4o Voice · Qwen2.5-Omni · Moshi</td></tr>
            <tr><td>🎨 创意生成</td><td>图配文、视频脚本</td><td>Midjourney v6 · Sora · 可灵 · 即梦</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>8. 未来发展方向</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🌐 Any-to-Any Omni 模型</h3>
            <p style="font-size:13px;line-height:1.6">输入/输出都是任意模态混合：文本/图像/音频/视频/3D/动作。代表：GPT-4o、Gemini 2.5、AnyGPT、CoDi-2、Qwen2.5-Omni。<b>瓶颈</b>：跨模态对齐数据稀缺、计算预算极高。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📺 长视频 + 时序记忆</h3>
            <p style="font-size:13px;line-height:1.6">2 小时电影 → 完整理解 + 任意时间点提问。需要 Token 压缩 (1000×) + 时序 Memory 检索 + 流式推理。Gemini 2M / VideoChat-Flash / MA-LMM。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🦾 Embodied + Robotics</h3>
            <p style="font-size:13px;line-height:1.6">VLM → VLA (Vision-Language-Action)，输出机器人动作 token。RT-2 / OpenVLA / Octo / Helix。仿真 + 现实迁移是难点。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🧠 多模态推理 (o1-style)</h3>
            <p style="font-size:13px;line-height:1.6">把 OpenAI o1 的链式思考迁移到视觉：MM-o1 / Llava-CoT / Qwen2-VL-Reasoning。<b>开放问题</b>：视觉 reward model 怎么训。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📦 数据合成飞轮</h3>
            <p style="font-size:13px;line-height:1.6">用强 VLM 生成训练数据、过滤、再训新 VLM（self-improve）。Gemini Flash 的对话样例、ShareGPT4V、Cauldron 已是雏形。</p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📱 端侧 Omni</h3>
            <p style="font-size:13px;line-height:1.6">在手机/眼镜端跑视觉+语音+文本统一模型。Apple Intelligence、Google Astra、Qwen2-VL-2B-Edge。难点：电池、延迟、隐私。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>9. 优缺点 & 选型建议</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3>✅ 优势</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>统一感知：图/文/音/视频一套架构</li>
              <li>少样本视觉任务无需训练</li>
              <li>跨模态推理（看图回答 + 描述）</li>
              <li>降低 OCR/分割/检测专用模型成本</li>
              <li>端到端代替 ASR + NLU + TTS 流水线</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #ef4444;">
            <h3>❌ 局限</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>显存 + 推理慢于纯文本 LLM 数倍</li>
              <li>OCR / 细粒度定位仍弱于专用模型</li>
              <li>幻觉问题更严重（视觉不可证伪）</li>
              <li>训练数据 IP / 偏见 / 安全极复杂</li>
              <li>视频 token 爆炸尚未根本解决</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>◎ 推荐选型</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>📋 文档场景：Qwen2.5-VL-72B / InternVL2.5</li>
              <li>📱 端侧：MiniCPM-V 2.6 / Qwen2-VL-2B</li>
              <li>🎬 视频长上下文：Gemini 2.5 Pro / VideoLLaMA3</li>
              <li>🦾 GUI Agent：Claude 3.5 + Computer Use</li>
              <li>🎨 闭源极致：GPT-4o / Gemini 2.5 Pro</li>
              <li>🔬 学术魔改：LLaVA-NeXT / Cambrian</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>10. 资源推荐</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models" target="_blank">Awesome-MLLM</a> — 多模态大模型最全清单</li>
          <li><a href="https://github.com/QwenLM/Qwen2-VL" target="_blank">Qwen2-VL 官方仓库</a></li>
          <li><a href="https://github.com/OpenGVLab/InternVL" target="_blank">InternVL</a> · <a href="https://github.com/haotian-liu/LLaVA" target="_blank">LLaVA</a></li>
          <li><a href="https://huggingface.co/spaces/opencompass/open_vlm_leaderboard" target="_blank">OpenCompass VLM Leaderboard</a> — 客观评测</li>
          <li><a href="https://arxiv.org/abs/2306.13549" target="_blank">A Survey on MLLM</a> · <a href="https://arxiv.org/abs/2304.08485" target="_blank">LLaVA 论文</a></li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
