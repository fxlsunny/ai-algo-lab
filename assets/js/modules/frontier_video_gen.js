/* 模块：视频生成（前沿） */
MCH.register("frontier_video_gen", {
  render() {
    const code = `# ============================================================================
# 视频生成 · DiT 主流路线（Sora / Veo / Kling / Open-Sora / CogVideoX）
# ============================================================================
# 1) 视频压缩到 时空潜空间：3D-VAE
# 2) DiT (Diffusion Transformer) 在潜空间做时空 patch 上的扩散去噪
# 3) 文本条件：T5-XXL / CLIP / Flan-T5 通过 cross-attn 或 in-context tokens

# ─── (1) 3D-VAE：时空压缩 ─────────────────────────────────────────
class CausalVideo3DVAE(nn.Module):
    """把 (T, H, W) 视频压缩到 (T', H/8, W/8) 潜变量。
    Causal：解码时仅依赖过去帧，支持任意长度。"""
    def encode(self, video_thw):
        return latent_t_hp_wp           # 通常 T:1/4, H/W:1/8 压缩
    def decode(self, latent):
        return video_recon

# ─── (2) DiT 时空块（patch + 时空注意力）─────────────────────────
class SpatioTemporalDiT(nn.Module):
    def __init__(self, depth=28, dim=1280, heads=16):
        self.blocks = nn.ModuleList([
            DiTBlock(dim, heads, attn_type="full3d")  # 全 3D 注意力（Sora）
            # 或 attn_type="space_then_time"        # 因式分解（CogVideoX, Open-Sora）
            for _ in range(depth)
        ])

    def forward(self, x_thw, t, text_emb):
        # 1) Patch：每 (1, 2, 2) 切块 → 时空 token
        x = patchify_3d(x_thw)
        # 2) 时空位置编码（3D-RoPE / NaViT-style）
        x = x + pos_3d(x.shape)
        for blk in self.blocks:
            x = blk(x, t, text_emb)
        return predicted_velocity        # Rectified Flow 输出 v(x_t, t)

# ─── (3) 训练 / 推理（Rectified Flow） ─────────────────────────
def train_step(video, prompt):
    z = vae.encode(video)
    t = sample_t()
    eps = torch.randn_like(z)
    x_t = (1-t) * eps + t * z
    target = z - eps
    pred = dit(x_t, t, text_encoder(prompt))
    loss = F.mse_loss(pred, target)

@torch.no_grad()
def sample(prompt, T=1, num_steps=50):
    x = torch.randn(noise_shape)
    for s in range(num_steps):
        t_now = s/num_steps
        v = dit(x, t_now, text_encoder(prompt))
        x = x + v / num_steps          # 欧拉法直线积分
    return vae.decode(x)
`;

    const long_code = `# ============================================================================
# 长视频生成：分钟 / 多镜头 / 一致性问题（前沿）
# ============================================================================

# 关键挑战：
#   (1) Token 爆炸：1 分钟 × 30fps × 100 token/帧 = 18 万 token
#   (2) 长程一致性：人物 / 物体不能突然消失或变样
#   (3) 镜头切换：从一个场景跳到另一个但保持故事

# 主流方案：
#   - 分块去噪 (Pyramid Flow, 2024) ：分辨率渐进生成
#   - Auto-Regressive on Chunks (CogVideoX-5B) ：分块 AR + 重叠帧条件
#   - Long-Context DiT + 稀疏 Attention (StreamingT2V)
#   - 一镜到底 → 多镜头：StoryDiffusion, MovieGen-Story (Meta 2024)

# Meta MovieGen (2024.10) 旗舰路线：
#   - 30B DiT，输出 16s 1080p 视频
#   - 个性化（参考图人物保留）
#   - 视频编辑 + 音效合成 + 配音
#   - 训练数据：1B+ 视频文本对

# Sora 关键特征（OpenAI 2024.02 → 2024.12 公开版）：
#   - 最长 1 分钟 1080p
#   - "Visual Patches"：把视频当成时空 token 序列
#   - Re-captioning：用 GPT-4 给视频生成详细 caption（DALL-E 3 同方法）
#   - Storyboard：可拼接多镜头
`;

    return `
      ${MCH.hero({
        icon: "🎬",
        name: "视频生成",
        en: "Video Generation · Sora / Kling / Veo / MovieGen / CogVideoX",
        tags: ["DiT", "3D-VAE", "Sora", "Kling", "MovieGen", "Open-Sora", "Rectified Flow"],
        meta: ["◈ 文/图生成 5-60s 高清视频", "⚡ Sora · 可灵 · Veo · 即梦 · Runway Gen-3"],
      })}

      <div class="section" style="background:linear-gradient(135deg,#dbeafe 0%,#bfdbfe 100%);border:1px solid #93c5fd;">
        <h2 style="color:#1e3a8a;border:none;padding:0;margin:0 0 6px 0;">🆕 视频生成的"GPT 时刻"：2024.02 Sora 发布 → 2024.06 可灵首发可用 → 2024.10 Meta MovieGen → 2024.12 Sora 开放 + Veo 2 + 可灵 1.6 → 2025 视频 Agent + 物理仿真融合</h2>
      </div>

      <div class="section">
        <h2>1. 发展史</h2>
        <div class="mermaid">
flowchart TB
    GAN[Video GAN 2016-2020<br/>MoCoGAN, DVD-GAN] --> NUW[NÜWA 2021<br/>微软]
    NUW --> CV[CogVideo 2022<br/>智谱]
    CV --> P[Phenaki 2022<br/>Google] --> Make[Make-A-Video 2022<br/>Meta]

    DDPM[DDPM 2020] --> VDM[Video Diffusion 2022<br/>Google] --> Imagen[Imagen Video 2022]
    Imagen --> SDV[Stable Video Diffusion 2023<br/>Stability]

    SDV --> Gen2[Runway Gen-2 2023]
    Gen2 --> Pika[Pika 1.0 2023]

    DiT[DiT 2023] --> Sora[🏆 Sora 2024.02<br/>OpenAI · 60s 1080p]
    Sora --> Veo[Veo 1/2 2024<br/>Google]
    Sora --> Kling[🏆 Kling 1.0/1.5/1.6 2024<br/>快手]
    Sora --> Hai[Hailuo 海螺 2024<br/>MiniMax]
    Sora --> Hunyuan[HunyuanVideo 2024<br/>腾讯 13B 开源]
    Sora --> Wan[Wanx 2.1 2025<br/>阿里]
    Sora --> Cog[CogVideoX 2024<br/>智谱开源]
    Sora --> Open[Open-Sora 1/2 2024<br/>Colossal-AI 开源]

    Sora --> Movie[Meta MovieGen 2024.10<br/>30B 多任务]
    Movie --> Goku[字节 Goku 2024]
    Movie --> Pyramid[Pyramid Flow 2024]

    Kling --> JM[即梦 Dreamina 2024<br/>字节]
    JM --> Pixel[PixVerse V3 2024]
        </div>
      </div>

      <div class="section">
        <h2>2. 核心技术栈对比</h2>
        <table class="table">
          <thead><tr><th>组件</th><th>典型方案</th><th>说明</th></tr></thead>
          <tbody>
            <tr><td><b>视频编码</b></td><td>3D-VAE / Causal-VAE / MagViT-2</td><td>把 (T,H,W) 压到 (T',h,w)，常见压缩 4×8×8 = 256 倍</td></tr>
            <tr><td><b>时空骨干</b></td><td>DiT-3D · ST-DiT · MMDiT-3D</td><td>Sora/Kling/Veo/MovieGen 全用 Transformer</td></tr>
            <tr><td><b>注意力</b></td><td>Full-3D · Space+Time 因式 · Sparse 时空</td><td>开销 vs 一致性的权衡</td></tr>
            <tr><td><b>条件</b></td><td>Cross-Attn / In-Context Token / ControlNet-3D</td><td>文本 / 首帧图 / 末帧图 / 相机轨迹</td></tr>
            <tr><td><b>训练目标</b></td><td>DDPM · Rectified Flow · Flow-Matching</td><td>2024 多数旗舰已切到 Rectified Flow</td></tr>
            <tr><td><b>采样</b></td><td>Euler · DPM-Solver++ · 一致性蒸馏</td><td>Distill 后 4-8 步出片</td></tr>
            <tr><td><b>音视频联合</b></td><td>MMAudio · MovieGen-Audio · V2A</td><td>视频 → 音效 / 配乐 / 配音</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码：DiT 骨架</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 长视频与一致性</h2>
        ${MCH.code(long_code, "python")}
      </div>

      <div class="section">
        <h2>5. 代表产品横评（2024-2025）</h2>
        <table class="table">
          <thead><tr><th>产品</th><th>厂商</th><th>开闭源</th><th>时长</th><th>分辨率</th><th>核心亮点</th></tr></thead>
          <tbody>
            <tr><td><b>Sora</b></td><td>OpenAI</td><td>闭源</td><td>5-60s</td><td>1080p</td><td>🏆 多镜头 · Storyboard · Remix</td></tr>
            <tr><td><b>Veo 2 / 3</b></td><td>Google DeepMind</td><td>闭源</td><td>8s</td><td>4K</td><td>真实物理 · 摄影机控制</td></tr>
            <tr><td><b>Kling 1.0 / 1.5 / 1.6</b></td><td>快手</td><td>闭源 SaaS</td><td>5-10s</td><td>1080p</td><td>🏆 中文 · 人物舞蹈 · 行业落地最快</td></tr>
            <tr><td><b>即梦 Dreamina</b></td><td>字节</td><td>闭源</td><td>5-10s</td><td>1080p</td><td>双语 · 抖音生态 · 一镜到底</td></tr>
            <tr><td><b>海螺 Hailuo</b></td><td>MiniMax</td><td>闭源</td><td>6s</td><td>720p+</td><td>价格优势 · API 开放</td></tr>
            <tr><td><b>HunyuanVideo</b></td><td>腾讯</td><td>🏆 开源 13B</td><td>5s</td><td>720p</td><td>当前开源 SOTA · 商业可用</td></tr>
            <tr><td><b>Wanx 2.1</b></td><td>阿里</td><td>开源 1.3B/14B</td><td>5s</td><td>1080p</td><td>双语 · 通义万相</td></tr>
            <tr><td><b>CogVideoX 2B/5B</b></td><td>智谱</td><td>开源</td><td>6s/10s</td><td>720p</td><td>首个真正可用开源 DiT 视频</td></tr>
            <tr><td><b>Open-Sora 1/2</b></td><td>Colossal-AI</td><td>开源</td><td>2-15s</td><td>720p</td><td>社区 Sora 复现</td></tr>
            <tr><td><b>Mochi-1</b></td><td>Genmo</td><td>开源</td><td>5s</td><td>480p</td><td>10B 开源 · Apache-2.0</td></tr>
            <tr><td><b>LTX-Video</b></td><td>Lightricks</td><td>开源 2B</td><td>4s</td><td>768p</td><td>实时（4090 上 < 4s 出片）</td></tr>
            <tr><td><b>MovieGen</b></td><td>Meta</td><td>未开源</td><td>16s</td><td>1080p</td><td>个性化 + 编辑 + 音效一体</td></tr>
            <tr><td><b>Runway Gen-3 / Gen-4</b></td><td>Runway</td><td>闭源</td><td>10s</td><td>720p</td><td>专业剪辑生态 · Camera Control</td></tr>
            <tr><td><b>Pika 1.5 / 2.0</b></td><td>Pika</td><td>闭源</td><td>5s</td><td>1080p</td><td>Pikaframes / Pika Effects 创意玩法</td></tr>
            <tr><td><b>PixVerse V3</b></td><td>爱诗科技</td><td>闭源</td><td>5-8s</td><td>1080p</td><td>Lipsync · 转绘风格化</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 关键工程难点</h2>
        <div class="grid-3">
          <div class="card">
            <h3>📺 长视频一致性</h3>
            <p style="font-size:13px;line-height:1.6">人物 / 场景 / 镜头切换后不漂移。当前 5-10s 是甜点；超过 30s 普遍出问题。需要 KV-Cache 视频版 + Memory Token + 重新参考首帧。</p>
          </div>
          <div class="card">
            <h3>🎬 摄影机控制</h3>
            <p style="font-size:13px;line-height:1.6">"推 / 拉 / 摇" 等专业镜头精准复现：CameraCtrl, MotionCtrl, MovieGen Camera Module。需要轨迹条件训练。</p>
          </div>
          <div class="card">
            <h3>🌍 物理 / 因果</h3>
            <p style="font-size:13px;line-height:1.6">"杯子掉地上会碎"是 SOTA 仍翻车的事。Veo 3 / Sora 已显著进步。Cosmos / 物理引擎闭环训练是路线。</p>
          </div>
          <div class="card">
            <h3>🗣 唇形同步</h3>
            <p style="font-size:13px;line-height:1.6">Lipsync (Lipsync.io / EMO / Hallo / V-Express) 已商业化。和 TTS、视频生成融合是数字人技术栈核心。</p>
          </div>
          <div class="card">
            <h3>⏱ 推理成本</h3>
            <p style="font-size:13px;line-height:1.6">单 5s 视频 H100 跑 30s-2min。LTX-Video 用因式 + 蒸馏做到实时。一致性蒸馏 (CD)、Step Distillation 是核心方向。</p>
          </div>
          <div class="card">
            <h3>🛡 安全 / 水印</h3>
            <p style="font-size:13px;line-height:1.6">SynthID（Google）、C2PA、Sora Watermark 等技术。监管层面（欧盟 AI Act / 中国生成内容标识规则）已强制。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>7. 主流应用场景</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>典型需求</th><th>代表落地</th></tr></thead>
          <tbody>
            <tr><td>📱 短视频 / 信息流</td><td>5-10s 创意片头 / 转场</td><td>抖音 / 快手 / TikTok 创作工具</td></tr>
            <tr><td>📢 广告营销</td><td>商品演示、TVC 草图</td><td>BGC 广告公司 · Cuebric · Dora</td></tr>
            <tr><td>🎬 影视前期</td><td>Pre-Vis 预演、分镜</td><td>Runway Gen-3 · Sora · 即梦</td></tr>
            <tr><td>🎮 游戏 / 元宇宙</td><td>NPC 动画、CG 概念</td><td>EmbodiedAI · 米哈游内部</td></tr>
            <tr><td>🛍 电商商品演示</td><td>360° 旋转 / 上身效果</td><td>淘宝 LeoVerse · 京东京 ME</td></tr>
            <tr><td>👔 数字人 / 直播</td><td>口播视频、虚拟主持人</td><td>Heygen · D-ID · 商汤如影</td></tr>
            <tr><td>📚 教育 / 科普</td><td>动态图解、动画讲解</td><td>Synthesia · Veed.io · 帆书</td></tr>
            <tr><td>🏛 文旅</td><td>古迹复原、城市宣传片</td><td>故宫博物院 · 各地文旅局</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>8. 未来发展方向</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🎞 长篇 / 多镜头叙事</h3>
            <p style="font-size:13px;line-height:1.6">从 5-10s "片段生成" 到 5-30 分钟"短片生成"。MovieGen / Vlogger / StoryDiffusion 是雏形。要解决：剧本 → 分镜 → 镜头一致性 → 配音/配乐 全链路。</p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🤖 视频 Agent</h3>
            <p style="font-size:13px;line-height:1.6">"做一个 30 秒电商主图视频" → Agent 自动写脚本、生成分镜、调用 Kling、剪辑。是视频版 ComfyUI + Cline。</p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🌐 World Model / 物理仿真</h3>
            <p style="font-size:13px;line-height:1.6">视频生成 = 隐式世界模型。Nvidia Cosmos / Genie 2 / V-JEPA-2 / Wayve GAIA-2 把视频当成驾驶 / 机器人 / 游戏的仿真器。</p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🎵 音视频联合生成</h3>
            <p style="font-size:13px;line-height:1.6">画面 + 音效 + 配乐 + 配音同步生成。MovieGen Audio · MMAudio · Veo 3 已开始。难点：跨模态时序对齐。</p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>👤 主体保持 / 个性化</h3>
            <p style="font-size:13px;line-height:1.6">同一人物、同一商品在不同视频保持一致：MovieGen Personalized / Hailuo I2V / EMO。商业化潜力最大方向之一。</p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>📺 视频编辑 / 局部重生成</h3>
            <p style="font-size:13px;line-height:1.6">"换衣服 / 换背景 / 时光倒流" 自然语言编辑视频。DiTCtrl, AnyV2V, Token-Flow 已起步。下一代 Premiere。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>9. 优缺点 & 选型</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3>✅ 优势</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>素材生产效率 100×</li>
              <li>免拍摄 / 免布景</li>
              <li>多语种本地化容易</li>
              <li>商业化模式清晰</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #ef4444;">
            <h3>❌ 局限</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>训练 H100 数千卡级别</li>
              <li>推理慢且贵（约 0.5-2 元/秒视频）</li>
              <li>5-10s 是甜点，长视频仍难</li>
              <li>版权 / Deepfake 隐患</li>
              <li>物理一致性仍弱</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>◎ 选型建议</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>中文商用：Kling / 即梦 / 海螺</li>
              <li>开源研究：HunyuanVideo / CogVideoX</li>
              <li>极致质量：Sora / Veo 2</li>
              <li>专业剪辑：Runway Gen-3 + Camera Ctrl</li>
              <li>实时端：LTX-Video / Mochi-1</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>10. 资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://github.com/hpcaitech/Open-Sora" target="_blank">Open-Sora · Colossal-AI</a></li>
          <li><a href="https://github.com/Tencent/HunyuanVideo" target="_blank">HunyuanVideo (腾讯)</a> · <a href="https://github.com/THUDM/CogVideo" target="_blank">CogVideoX (智谱)</a></li>
          <li><a href="https://openai.com/sora" target="_blank">Sora Technical Report</a> · <a href="https://ai.meta.com/research/movie-gen/" target="_blank">Meta MovieGen 论文</a></li>
          <li><a href="https://github.com/lichao-sun/SoraReview" target="_blank">Sora 综述</a> · <a href="https://github.com/showlab/Awesome-Video-Diffusion" target="_blank">Awesome Video Diffusion</a></li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
