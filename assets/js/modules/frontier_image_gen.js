/* 模块：图像生成（前沿） */
MCH.register("frontier_image_gen", {
  render() {
    const code = `# ============================================================================
# 图像生成 · 主流路线技术对比
# (1) GAN  →  (2) VAE/VQ-VAE  →  (3) Autoregressive  →  (4) Diffusion / DiT  →  (5) Hybrid
# ============================================================================

# ─── (1) GAN：生成器 vs 判别器对抗 ─────────────────────────────────
# StyleGAN3 / BigGAN — 仍是最快的（单步生成），人脸领域 SOTA
# 训练困难（mode collapse）、可控性差，但小模型有效
class StyleGAN3:
    def G(self, z):
        # latent z -> 渐进式上采样 -> 1024×1024 image
        return img
    def D(self, x):
        return real_or_fake_score   # WGAN-GP / R1 正则化

# ─── (3) Autoregressive：把图当作 token 序列 ────────────────────
# DALL-E 1 (2021), Parti (2022), MUSE (2023), MaskGIT, VAR (2024 NeurIPS Best Paper)
class ARImageGen:
    def encode(self, img):
        # VQGAN / VQ-VAE：1024×1024 → 64×64 离散 codes (8192 codebook)
        return tokens
    def generate(self, prompt):
        # GPT 式 next-token：text tokens + <bos_image> + 4096 visual tokens
        return decode(self.transformer.generate(prompt))

# ─── (4) Diffusion / DiT：当前主流 ──────────────────────────────
# Stable Diffusion 1/2/XL/3, FLUX.1, SDXL Turbo, DiT, PixArt, Lumina
# 公式：x_t = √α_t · x_0 + √(1-α_t) · ε，训练去噪网络 ε_θ
# DiT (Peebles 2023) 用 Transformer 替代 UNet，是 Sora / SD3 / FLUX 共同选择

# ─── (5) Rectified Flow / Flow Matching（2024 新主流） ─────────
# SD3 / FLUX.1 / Lumina-Next 切换到 Rectified Flow：
#   - 在 x_0 (噪声) 与 x_1 (图) 之间学一条 直线 路径
#   - 训练：v_θ(x_t, t) ≈ x_1 - x_0
#   - 推理：欧拉法直线积分，10-20 步即可
# 比传统 DDPM 推理更稳、更快、更易训练

# ─── 文本条件 Encoder 演进 ─────────────────────────────────────
# CLIP-L (SD1.x) → CLIP-G + T5-XXL (SDXL/SD3/FLUX) → Multi-encoder Bag (FLUX.1)
# 趋势：T5 / Llama 提供长 prompt 语义，CLIP 提供视觉对齐
`;

    const ctrl_code = `# ============================================================================
# 控制 / 编辑核心方法（2023-2025 全谱）
# ============================================================================

# 🎯 ControlNet（Zhang ICCV 2023）：冻结 SD + 可训练分支，结构条件
#    线稿 / 深度 / 姿态 / Canny / 法线 / OpenPose 全套
class ControlNet:
    branch = copy(unet.encoder)         # 复制 Encoder
    zero_conv = nn.Conv2d(...)          # 零初始化卷积，平滑加入

# 🎯 IP-Adapter / InstantID：图像作为 prompt（人脸保持）
# 🎯 LoRA：少量参数微调风格 (8-32M trainable)
# 🎯 DreamBooth / Textual Inversion：定制概念
# 🎯 InstructPix2Pix / MagicBrush：自然语言编辑
# 🎯 Inpainting / Outpainting：局部修复 / 外扩
# 🎯 SDEdit / Img2Img：低 strength 强化已有图

# 2024 新趋势：Auto-prompting + Agent 化生成
#   FLUX + Prompt Refiner + Auto-LoRA-Selection
#   ComfyUI Node Graph 让生成 = 工作流编排
`;

    return `
      ${MCH.hero({
        icon: "🎨",
        name: "图像生成",
        en: "Image Generation · GAN / VAE / Autoregressive / Diffusion / Flow",
        tags: ["GAN", "Diffusion", "DiT", "Rectified Flow", "FLUX", "SD3", "ControlNet"],
        meta: ["◈ 文生图 · 图生图 · 编辑 · 个性化", "⚡ Stable Diffusion 生态 / FLUX.1 / Midjourney v6 / DALL-E 3"],
      })}

      <div class="section" style="background:linear-gradient(135deg,#fce7f3 0%,#fbcfe8 100%);border:1px solid #f9a8d4;">
        <h2 style="color:#9d174d;border:none;padding:0;margin:0 0 6px 0;">🆕 2024-2025 关键变化：UNet → DiT，DDPM → Rectified Flow，单图 → 多图编辑 / 动态生成；FLUX.1（开源 12B）追平 Midjourney v6；推理从 50 步压到 1-4 步。</h2>
      </div>

      <div class="section">
        <h2>1. 发展史：5 大技术路线接力</h2>
        <div class="mermaid">
flowchart TB
    GAN[GAN 2014<br/>Goodfellow] --> SG[StyleGAN 2019<br/>人脸 SOTA]
    SG --> SG3[StyleGAN3 2021<br/>抗混叠]

    VAE[VAE 2013] --> VQVAE[VQ-VAE 2017<br/>离散表征]
    VQVAE --> VQGAN[VQGAN 2021<br/>感知 GAN loss]

    VQGAN --> DALLE1[DALL-E 1 2021<br/>OpenAI]
    DALLE1 --> Parti[Parti 2022<br/>Google 20B]
    Parti --> MUSE[MUSE / MaskGIT 2023<br/>并行生成]
    MUSE --> VAR[🆕 VAR 2024<br/>NeurIPS Best Paper]

    DDPM[DDPM 2020] --> SD1[Stable Diffusion 1 2022<br/>Latent Diffusion]
    SD1 --> SDXL[SDXL 2023]
    SDXL --> SD3[SD3 2024<br/>MM-DiT + Rectified Flow]
    SD3 --> FLUX[🆕 FLUX.1 2024<br/>12B 开源 SOTA]

    SDXL --> ControlNet[ControlNet 2023]
    SDXL --> LCM[LCM 2023<br/>1-4 步采样]

    DDPM --> DiT[DiT 2023<br/>Peebles]
    DiT --> Sora[Sora 2024<br/>视频]
    DiT --> SD3
    DiT --> FLUX

    DALLE1 --> DALLE3[DALL-E 3 2023<br/>OpenAI]
    SDXL --> MJ[Midjourney v6 2024]
    MJ --> Imagen[Imagen 3 2024<br/>Google]

    FLUX --> Lumina[Lumina-Next 2024]
    FLUX --> Kolors[Kolors 2024<br/>快手开源]
    FLUX --> CogView[CogView4 2024<br/>智谱]
        </div>
      </div>

      <div class="section">
        <h2>2. 五大技术路线对比</h2>
        <table class="table">
          <thead><tr><th>路线</th><th>原理</th><th>代表</th><th>速度</th><th>质量</th><th>多样性</th><th>可控性</th></tr></thead>
          <tbody>
            <tr><td><b>GAN</b></td><td>对抗博弈，1 步生成</td><td>StyleGAN3, BigGAN</td><td>⭐⭐⭐⭐⭐</td><td>⭐⭐⭐⭐</td><td>⭐⭐</td><td>⭐⭐</td></tr>
            <tr><td><b>VAE/VQ-VAE</b></td><td>编码器+解码器，重构 + KL/VQ loss</td><td>VQ-VAE-2, NVAE</td><td>⭐⭐⭐⭐⭐</td><td>⭐⭐⭐</td><td>⭐⭐⭐</td><td>⭐⭐⭐</td></tr>
            <tr><td><b>Autoregressive</b></td><td>把图变 token 序列，next-token 预测</td><td>DALL-E 1, Parti, MUSE, VAR</td><td>⭐⭐</td><td>⭐⭐⭐⭐</td><td>⭐⭐⭐⭐</td><td>⭐⭐⭐⭐</td></tr>
            <tr><td><b>Diffusion / DiT</b></td><td>逐步去噪，UNet/DiT 预测噪声</td><td>SD1/2/XL, DALL-E 2/3, MJ, Imagen</td><td>⭐⭐⭐</td><td>⭐⭐⭐⭐⭐</td><td>⭐⭐⭐⭐⭐</td><td>⭐⭐⭐⭐⭐</td></tr>
            <tr><td><b>Rectified Flow</b></td><td>学习直线 ODE 路径，欧拉法积分</td><td>🆕 SD3, FLUX.1, Lumina</td><td>⭐⭐⭐⭐</td><td>⭐⭐⭐⭐⭐</td><td>⭐⭐⭐⭐⭐</td><td>⭐⭐⭐⭐⭐</td></tr>
            <tr><td><b>Hybrid</b></td><td>多路线混合（如 ARDiT、Show-o）</td><td>🆕 Show-o, Janus, VAR-Diff</td><td>⭐⭐⭐</td><td>⭐⭐⭐⭐</td><td>⭐⭐⭐⭐</td><td>⭐⭐⭐⭐</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码：路线对比</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 控制 / 编辑生态（一站式）</h2>
        ${MCH.code(ctrl_code, "python")}
      </div>

      <div class="section">
        <h2>5. 代表模型横评（2024-2025）</h2>
        <table class="table">
          <thead><tr><th>模型</th><th>厂商</th><th>规模</th><th>开闭源</th><th>核心亮点</th></tr></thead>
          <tbody>
            <tr><td><b>Stable Diffusion 1/2/XL</b></td><td>Stability AI</td><td>0.9B-2.6B</td><td>开源</td><td>开源生态奠基（A1111/ComfyUI/SDWebUI）</td></tr>
            <tr><td><b>SD3 / SD3.5</b></td><td>Stability AI</td><td>2B/8B</td><td>有限开源</td><td>MM-DiT + Rectified Flow + 三 encoder</td></tr>
            <tr><td><b>FLUX.1 [dev/schnell/pro]</b></td><td>Black Forest Labs</td><td>12B</td><td>开源 weights</td><td>🏆 2024 开源 SOTA · MJ 级别 · Schnell 4 步</td></tr>
            <tr><td><b>Midjourney v6 / v6.1</b></td><td>Midjourney</td><td>未公开</td><td>闭源</td><td>艺术性最强 · Discord/Web UI</td></tr>
            <tr><td><b>DALL-E 3</b></td><td>OpenAI</td><td>未公开</td><td>闭源</td><td>Prompt-Following 强 · ChatGPT 集成</td></tr>
            <tr><td><b>Imagen 3 / 4</b></td><td>Google</td><td>未公开</td><td>闭源</td><td>文字渲染、一致性</td></tr>
            <tr><td><b>Ideogram 2.0</b></td><td>Ideogram</td><td>未公开</td><td>闭源</td><td>🆕 文字渲染最强</td></tr>
            <tr><td><b>Kolors / Kling Image</b></td><td>快手</td><td>2.6B</td><td>开源</td><td>中文 prompt 友好</td></tr>
            <tr><td><b>CogView 3 / 4</b></td><td>智谱</td><td>3B-9B</td><td>开源</td><td>双语 + Diffusion + Refiner</td></tr>
            <tr><td><b>Lumina-Next</b></td><td>上海 AI Lab</td><td>2B</td><td>开源</td><td>统一 Flow 框架（图/视频/3D）</td></tr>
            <tr><td><b>HunyuanDiT / HunyuanImage</b></td><td>腾讯</td><td>1.5B</td><td>开源</td><td>双语 DiT</td></tr>
            <tr><td><b>VAR</b></td><td>字节</td><td>2B</td><td>开源</td><td>🆕 NeurIPS24 Best · AR 重夺王座</td></tr>
            <tr><td><b>Janus / Janus-Pro</b></td><td>DeepSeek</td><td>1.3B/7B</td><td>开源</td><td>🆕 解耦"理解"与"生成"双视觉编码器</td></tr>
            <tr><td><b>SDXL Turbo / Hyper-SD / DMD</b></td><td>多家</td><td>—</td><td>开源</td><td>蒸馏到 1-4 步实时生成</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 关键工程难点</h2>
        <div class="grid-3">
          <div class="card">
            <h3>📝 Prompt-Following</h3>
            <p style="font-size:13px;line-height:1.6">"两个红苹果在木桌左边" 模型经常忽视方位 / 数量。SD3 用三 encoder（CLIP-L + CLIP-G + T5-XXL）显著缓解，FLUX.1 沿用并加 Multi-encoder Bag。</p>
          </div>
          <div class="card">
            <h3>📐 文字渲染</h3>
            <p style="font-size:13px;line-height:1.6">"画一个写着 Hello 的招牌" 是经典翻车点。Ideogram 2 / DALL-E 3 / Imagen 3 已较好；开源模型仍弱。需要字符级 Token / 大 T5。</p>
          </div>
          <div class="card">
            <h3>👤 主体一致性</h3>
            <p style="font-size:13px;line-height:1.6">同一个角色在不同场景保持外貌：DreamBooth / IP-Adapter / InstantID / PuLID / Story-Maker。生产级方案少。</p>
          </div>
          <div class="card">
            <h3>🖐 手指 / 解剖</h3>
            <p style="font-size:13px;line-height:1.6">六指手是经典 meme。FLUX.1 / SD3.5 大幅缓解，但极端姿势仍翻车。OpenPose ControlNet 是稳定方案。</p>
          </div>
          <div class="card">
            <h3>⚡ 推理加速</h3>
            <p style="font-size:13px;line-height:1.6">DDIM(50) → DPM++(20) → LCM(4) → SDXL Turbo(1) → DMD2 一步。蒸馏 + Adversarial Distillation 是主线。</p>
          </div>
          <div class="card">
            <h3>⚖️ 数据 + 版权</h3>
            <p style="font-size:13px;line-height:1.6">LAION-5B 含未授权图，多家被诉讼。Adobe Firefly 走"全自有数据"路线。Nightshade / Glaze 帮艺术家"投毒反制"。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>7. 主流应用场景</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>典型需求</th><th>代表落地</th></tr></thead>
          <tbody>
            <tr><td>🎨 创意营销 / 广告素材</td><td>千人千面海报、A/B 测试</td><td>Adobe Firefly · Midjourney · 美图设计室</td></tr>
            <tr><td>🛍 电商商品图</td><td>白底图、模特换装、场景渲染</td><td>淘宝 LeoVerse · 京东京 ME · ZMO.ai</td></tr>
            <tr><td>🎮 游戏美术资产</td><td>原画概念图、贴图、角色立绘</td><td>米哈游、网易、Scenario.gg</td></tr>
            <tr><td>📺 短视频 / 表情包</td><td>动图、贴纸、AI 滤镜</td><td>抖音 / 快手特效、Snap AI Lens</td></tr>
            <tr><td>🏛 影视 / 动画</td><td>分镜、预演、动画补帧</td><td>RunwayML · 即梦 · 可灵</td></tr>
            <tr><td>👤 个性化（写真/换脸）</td><td>AI 写真馆、虚拟形象</td><td>妙鸭、Replika · Higgsfield</td></tr>
            <tr><td>🏗 建筑 / 工业设计</td><td>方案草图渲染、室内设计</td><td>LookX · Maket.ai · Stable Doodle</td></tr>
            <tr><td>👗 服装设计</td><td>概念图 → 样衣</td><td>Style3D · CALA</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>8. 未来发展方向</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #ec4899;">
            <h3>📦 Native Multi-Image / Story 生成</h3>
            <p style="font-size:13px;line-height:1.6">单次生成 4-12 张主体一致的图（漫画 / 故事板 / 教程）。Story Diffusion、Anole、SEED-Story、Pyramid Flow 是先驱。商业化潜力巨大。</p>
          </div>
          <div class="card" style="border-left:3px solid #ec4899;">
            <h3>✏️ 自然语言编辑</h3>
            <p style="font-size:13px;line-height:1.6">"把猫换成狗、保持其他不变" 一句话编辑。InstructPix2Pix → MagicBrush → UltraEdit → Emu Edit。下一代 Photoshop。</p>
          </div>
          <div class="card" style="border-left:3px solid #ec4899;">
            <h3>🤖 Agent 化生成</h3>
            <p style="font-size:13px;line-height:1.6">用户说"做一张电商主图"，Agent 自动选 LoRA、写 prompt、调 ControlNet、生成 + 评分。ComfyUI Manager + LLM 驱动 = 雏形。</p>
          </div>
          <div class="card" style="border-left:3px solid #ec4899;">
            <h3>🔄 统一图理解 + 生成</h3>
            <p style="font-size:13px;line-height:1.6">Janus-Pro / Show-o / Chameleon / VAR 探索"看图说话 + 看话画图"用同一模型。最终目标：一个 Omni 模型。</p>
          </div>
          <div class="card" style="border-left:3px solid #ec4899;">
            <h3>🧪 物理一致 / 3D 真实</h3>
            <p style="font-size:13px;line-height:1.6">2D 图 → 3D 一致：Stable Video 3D、Zero123++、TripoSR、Tripo3D。下一步：物理可信 + 模拟器闭环（Nvidia Cosmos 路线）。</p>
          </div>
          <div class="card" style="border-left:3px solid #ec4899;">
            <h3>🏃 实时 / 端侧</h3>
            <p style="font-size:13px;line-height:1.6">M2/M3 Mac 已能跑 SDXL Turbo 实时；下一步是手机原生 4 步生成。Apple Core ML、MediaPipe Diffusion、SD-Lite。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>9. 优缺点 & 选型</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3>✅ 优势</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>0 → 1 创意效率 100×</li>
              <li>开源生态完整（LoRA / ControlNet / IP-Adapter）</li>
              <li>商业化模式清晰（订阅 / 按图付费）</li>
              <li>可控性持续突破</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #ef4444;">
            <h3>❌ 局限</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>训练 / 推理算力门槛高</li>
              <li>版权 / 数据合规仍有风险</li>
              <li>Deepfake / 恶意内容滥用</li>
              <li>中文 prompt 支持仍弱（除阿里 / 智谱 / 快手系）</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>◎ 选型建议</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>开源生产：FLUX.1 / SD3.5 / Kolors</li>
              <li>极致艺术：Midjourney v6.1</li>
              <li>Prompt 严谨：DALL-E 3 / Ideogram</li>
              <li>实时 / 端侧：SDXL Turbo / LCM-LoRA</li>
              <li>个性化：DreamBooth + LoRA + IP-Adapter</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>10. 资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://huggingface.co/black-forest-labs/FLUX.1-dev" target="_blank">FLUX.1-dev (Black Forest Labs)</a></li>
          <li><a href="https://github.com/Stability-AI/sd3-ref" target="_blank">SD3 reference impl</a> · <a href="https://comfyanonymous.github.io/ComfyUI_examples/" target="_blank">ComfyUI Examples</a></li>
          <li><a href="https://huggingface.co/datasets/laion/laion2B-en" target="_blank">LAION-2B</a> · <a href="https://www.midjourney.com/explore" target="_blank">Midjourney Explore</a></li>
          <li><a href="https://arxiv.org/abs/2403.03206" target="_blank">Stable Diffusion 3 论文</a> · <a href="https://arxiv.org/abs/2404.02905" target="_blank">VAR 论文</a></li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
