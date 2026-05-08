/* 模块：案例 · 视频生成（行业落地） */
MCH.register("case_video_gen", {
  render() {
    const code = `# ============================================================================
# 工业视频生成完整生产管线（参考 Kling / 即梦 / Sora / Runway 内部架构）
# ============================================================================
# Stage 0：素材接入
#   · 文本 prompt + 可选首/末帧图 + 可选参考视频 + 风格 LoRA
# Stage 1：Prompt 工程
#   · LLM 重写 prompt（DALL-E 3 同款 re-captioning）：补镜头语言、灯光、镜头运动
#   · 安全审核：NSFW / 暴力 / 名人脸 / 商标拦截
# Stage 2：核心生成
#   · 文本 → DiT 视频生成（5-10s @ 1080p）
#   · 多 GPU 推理：H200 × 8 张并行 batch=1（一个用户一个 batch）
# Stage 3：后处理
#   · 时序补帧 (RIFE / FILM)：从 24fps → 60fps
#   · 超分 (Real-ESRGAN / VEAI)：1080p → 4K
#   · 时序去抖动 / 去闪烁 / 色彩稳定
#   · 音频生成（V2A）：MMAudio / MovieGen Audio 配音效
# Stage 4：交付
#   · 水印 (SynthID-Video / C2PA) + 元数据
#   · 转码：H.264 / H.265 / AV1，多分辨率切片
#   · CDN 分发

class IndustrialVideoPipeline:
    def __init__(self):
        self.prompt_rewriter = PromptLLM(qwen_or_gpt4o)
        self.safety   = SafetyClassifier()
        self.dit      = VideoDiT(weights="kling-1.6")  # 13B-30B 时空 Transformer
        self.refiner  = TemporalUpscaler(weights="rife")
        self.sr       = SpatialUpscaler(weights="real-esrgan-v3")
        self.v2a      = VideoToAudio(weights="mmaudio")
        self.watermark= SynthIDVideo()

    def generate(self, user_prompt, ref_image=None, duration=5, fps=24, resolution=(1080, 1920)):
        prompt = self.prompt_rewriter.rewrite(user_prompt, mode="cinematic")
        if not self.safety.allow(prompt):
            raise SafetyViolation()

        latent = self.dit.sample(prompt, ref_image=ref_image,
                                 duration=duration, fps=fps,
                                 num_inference_steps=30,
                                 guidance_scale=7.0)
        video = self.dit.vae_decode(latent)        # (T, H, W, C) ~ (120, 1080, 1920, 3)
        video = self.refiner.interpolate(video, target_fps=60)
        video = self.sr.upscale(video, scale=2)    # → 2160p
        audio = self.v2a.synthesize(video, prompt) # 配同步音效
        video = mux_audio(video, audio)
        video = self.watermark.embed(video)
        return encode_h265(video, bitrate="8M")`;

    return `
      ${MCH.hero({
        icon: "📹",
        name: "案例 · 视频生成",
        en: "Industrial Video Generation Pipeline",
        tags: ["DiT", "Kling", "Sora", "Runway", "MovieGen", "短视频", "广告", "影视"],
        meta: ["◈ 短视频 / 广告 / 影视前期 / 数字人 / 电商", "⚡ 快手 · 字节 · 阿里 · 腾讯 · 商汤"],
      })}

      <div class="section">
        <h2>1. 业务场景与核心挑战</h2>
        <div class="mermaid">
flowchart LR
    USER[创作者文本/图] --> PR[Prompt 重写<br/>+ 安全审核]
    PR --> DIT[DiT 视频生成<br/>5-10s 1080p]
    REF[参考图/视频<br/>风格 LoRA] --> DIT
    DIT --> POST[后处理<br/>补帧/超分/去抖]
    POST --> AUDIO[V2A 音效<br/>配乐/配音]
    AUDIO --> WM[水印 + 转码]
    WM --> CDN[CDN 分发]
        </div>
        ${MCH.info(`
          <b>工业视频生成的 5 个核心矛盾</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>质量 vs 成本</b>：H200 单块跑 5s 视频需 30-90s · 100 万 DAU 需千卡级集群</li>
            <li><b>时长 vs 一致性</b>：5-10s 是甜点 · 30s+ 主体易漂移</li>
            <li><b>开放 vs 安全</b>：deepfake / 名人 / 暴力 / 商标 / 政治敏感 全要拦</li>
            <li><b>定制 vs 通用</b>：电商商品 / 游戏角色 / 真人 IP 都需主体保持</li>
            <li><b>实时 vs 高清</b>：直播/短视频要实时；影视要 4K + 60fps</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 传统 vs 现代 视频生产链路</h2>
        <table class="table">
          <thead><tr><th>环节</th><th>🏛 传统</th><th>🚀 AI 生成</th><th>提效</th></tr></thead>
          <tbody>
            <tr><td>创意</td><td>团队头脑风暴 + 分镜手稿</td><td>LLM Prompt 编排 + 自动分镜</td><td>10×</td></tr>
            <tr><td>素材</td><td>实拍 / 库素材授权</td><td>Sora / Kling / Veo 直接生成</td><td>50×</td></tr>
            <tr><td>美术</td><td>原画师 / 3D 建模师</td><td>SD/FLUX + ControlNet 出概念图</td><td>30×</td></tr>
            <tr><td>配音</td><td>录音棚配音师</td><td>CosyVoice / ElevenLabs</td><td>100×</td></tr>
            <tr><td>剪辑</td><td>Premiere / FCP 人工剪</td><td>AI 自动剪辑 (剪映 AI)</td><td>5-10×</td></tr>
            <tr><td>特效 / 调色</td><td>AE / DaVinci 手工</td><td>SDE 视频编辑 + AI 调色 (TopazVEAI)</td><td>5-10×</td></tr>
            <tr><td>本地化</td><td>多语种重新拍摄/配音</td><td>HeyGen Translate / Hedra / 字节 Translate</td><td>50×</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 工业管线代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 主流商用方案对比</h2>
        <table class="table">
          <thead><tr><th>方案</th><th>厂商</th><th>核心优势</th><th>典型场景</th><th>定价</th></tr></thead>
          <tbody>
            <tr><td><b>Kling 1.6 / 2.0</b></td><td>快手</td><td>🏆 中文 SOTA · 人物舞蹈 · 行业落地最快</td><td>短视频 · 广告 · 服装</td><td>RMB 0.5-2 元/秒</td></tr>
            <tr><td><b>即梦 Dreamina</b></td><td>字节</td><td>抖音生态 · 素材库联动</td><td>抖音 / TikTok 创作</td><td>SaaS 订阅</td></tr>
            <tr><td><b>海螺 Hailuo Video</b></td><td>MiniMax</td><td>价格优势 · 双语</td><td>ToC API · 海外</td><td>低价 API</td></tr>
            <tr><td><b>通义万相 Wanx 2.1</b></td><td>阿里</td><td>开源 1.3B/14B · 双语</td><td>电商 · 私有化</td><td>开源 + API</td></tr>
            <tr><td><b>HunyuanVideo</b></td><td>腾讯</td><td>🏆 13B 开源 · 商业可用</td><td>研究 · 自建</td><td>开源</td></tr>
            <tr><td><b>Sora</b></td><td>OpenAI</td><td>🏆 多镜头 · Storyboard</td><td>影视前期 · 设计</td><td>$20-200/月 (Plus/Pro)</td></tr>
            <tr><td><b>Veo 2 / 3</b></td><td>Google</td><td>4K · 物理真实 · 镜头控制</td><td>专业短片 · 广告</td><td>$0.5/秒 (V3)</td></tr>
            <tr><td><b>Runway Gen-3 / Gen-4</b></td><td>Runway</td><td>专业剪辑生态 · Camera Control</td><td>影视专业用户</td><td>$15-95/月</td></tr>
            <tr><td><b>Pika 2.0</b></td><td>Pika</td><td>Pika Effects 创意玩法</td><td>个人 · 创意</td><td>$10-70/月</td></tr>
            <tr><td><b>HeyGen / Hedra</b></td><td>HeyGen / Hedra</td><td>🏆 数字人视频本地化</td><td>口播视频 · 多语种翻译</td><td>$30-300/月</td></tr>
            <tr><td><b>D-ID / Synthesia</b></td><td>D-ID / Synthesia</td><td>企业培训 · 数字人主播</td><td>企业内部视频</td><td>$30-1500/月</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 工业落地：6 个真实场景拆解</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🛍 电商商品视频</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：商品 360° 旋转、上身效果、场景化展示<br/>
              <b>方案</b>：Wanx + 商品图 + ControlNet（背景）+ 模特 LoRA<br/>
              <b>难点</b>：商品细节保留 · 衣服褶皱真实 · 卖点字幕同步<br/>
              <b>代表</b>：淘宝 LeoVerse · 京东京 ME · ZMO.ai
            </p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>📢 广告 TVC 制作</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：30s-60s 创意广告 · 多版本 A/B · 多语种<br/>
              <b>方案</b>：Sora / Kling 生成主镜头 + Runway 剪辑 + ElevenLabs 配音<br/>
              <b>难点</b>：品牌一致性 · 法律合规 · 受众洞察<br/>
              <b>代表</b>：可口可乐 / 丰田 / 阿迪达斯已上线 AI 广告
            </p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🎬 影视前期 / Pre-Vis</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：分镜可视化、概念预演、风格探索<br/>
              <b>方案</b>：Runway Gen-3 + Camera Control + Sora Storyboard<br/>
              <b>难点</b>：导演意图准确传达 · 多镜头连贯 · 角色一致性<br/>
              <b>代表</b>：BBC、Netflix、华纳已采用
            </p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>👤 数字人主播 / 直播</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：24/7 直播 · 多语种 · 嘴型同步 · 实时互动<br/>
              <b>方案</b>：HeyGen / 商汤如影 / 硅基智能 · 数字人 + LLM + TTS<br/>
              <b>难点</b>：直播长稳定 · 突发口播 · 跨语种唇形<br/>
              <b>代表</b>：李佳琦数字人 · 三只羊数字人 · 央视 AI 主播
            </p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>📺 短视频 / 信息流</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：5-15s 创意片 · 大量生产 · 风格多样<br/>
              <b>方案</b>：剪映 AI · 即梦 + 自动剪辑 + 自动配乐<br/>
              <b>难点</b>：素材个性化 · 时长卡点 · 平台合规<br/>
              <b>代表</b>：抖音 / 快手 / TikTok 创作工具
            </p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🌐 视频本地化</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：保留原音色翻译多国语言 · 唇形对齐<br/>
              <b>方案</b>：HeyGen Translate / Hedra · ASR + 翻译 + TTS + 唇形重生成<br/>
              <b>难点</b>：源/目标语言长度差 · 文化语境 · 版权<br/>
              <b>代表</b>：Mr.Beast 跨语种发布 · YouTube 多音轨
            </p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>6. 工程难点 + 实战经验</h2>
        <div class="grid-3">
          <div class="card">
            <h3>🎬 主体一致性</h3>
            <p style="font-size:13px;line-height:1.6">"同一个商品/IP在不同场景"是商业级核心。方案：参考图 IP-Adapter / DreamBooth-Video / 角色 LoRA · MovieGen Personalized 是 SOTA。</p>
          </div>
          <div class="card">
            <h3>📷 镜头控制</h3>
            <p style="font-size:13px;line-height:1.6">"推 / 拉 / 摇 / 跟" 等专业镜头：CameraCtrl / MotionCtrl / MovieGen Camera Module · 用户给轨迹条件。</p>
          </div>
          <div class="card">
            <h3>🌍 物理真实</h3>
            <p style="font-size:13px;line-height:1.6">"杯子掉地碎"是物理常识。Veo 3 / Sora 已显著进步 · Cosmos / 物理引擎闭环训练是路线。</p>
          </div>
          <div class="card">
            <h3>🗣 唇形同步</h3>
            <p style="font-size:13px;line-height:1.6">EMO / Hallo / V-Express / Hedra 已商业化 · 跨语种唇形是数字人核心难点。</p>
          </div>
          <div class="card">
            <h3>⏱ 推理成本</h3>
            <p style="font-size:13px;line-height:1.6">单 5s 1080p 视频 H100 跑 30s-2min。LTX-Video 用因式 + 蒸馏做到实时。一致性蒸馏 + Step Distillation 是核心方向。</p>
          </div>
          <div class="card">
            <h3>🛡 安全 / 水印 / 合规</h3>
            <p style="font-size:13px;line-height:1.6">SynthID-Video · C2PA · Sora Watermark · 中国生成内容标识规则强制 · 名人脸库 / NSFW 检测必备。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>7. 成本估算（2025）</h2>
        <table class="table">
          <thead><tr><th>方案</th><th>5s 1080p 视频成本</th><th>1 万条/天</th><th>说明</th></tr></thead>
          <tbody>
            <tr><td>Kling 1.6 Pro</td><td>RMB 5-10 元</td><td>5-10 万元/天</td><td>商业 SaaS · 中文最佳</td></tr>
            <tr><td>Sora (Pro)</td><td>$0.5-2 (含订阅摊薄)</td><td>$5k-20k</td><td>多镜头 · Storyboard</td></tr>
            <tr><td>Veo 3</td><td>$2-3 (按秒计费)</td><td>$20k-30k</td><td>4K · 物理真实</td></tr>
            <tr><td>HunyuanVideo 自托管</td><td>~RMB 1-2 元（电费 + 摊薄）</td><td>1-2 万元</td><td>需自建 H200 × 8 集群</td></tr>
            <tr><td>Hailuo / 海螺 API</td><td>RMB 0.5-1 元</td><td>0.5-1 万元</td><td>价格屠夫</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>8. 未来发展方向（2026 展望）</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🎞 长视频 / 多镜头叙事</h3>
            <p style="font-size:13px;line-height:1.6">从 5-10s 片段生成 → 5-30 分钟短片。需要：剧本 → 分镜 → 镜头一致性 → 配音/配乐全链路。MovieGen / Vlogger / StoryDiffusion 是雏形。</p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>🤖 视频 Agent / 自动化生产</h3>
            <p style="font-size:13px;line-height:1.6">"做一个 30 秒电商主图视频"→ Agent 自动写脚本、生成分镜、调用 Kling、剪辑。视频版 Cline / ComfyUI。</p>
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
            <h3>🎨 实时 / 端侧生成</h3>
            <p style="font-size:13px;line-height:1.6">M2/M3 Mac 跑 SDXL Turbo 实时；下一代 LTX-Video / Stable Video Lite 在 4090 实时出 5s 视频。元宇宙 / 游戏可用。</p>
          </div>
          <div class="card" style="border-left:3px solid #3b82f6;">
            <h3>📺 视频编辑 / 局部重生成</h3>
            <p style="font-size:13px;line-height:1.6">"换衣服 / 换背景 / 时光倒流" 自然语言编辑视频。DiTCtrl / AnyV2V / Token-Flow 已起步。下一代 Premiere。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>9. 选型建议矩阵</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3>🏢 ToB 私有化</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>开源主力：HunyuanVideo / CogVideoX / Wanx</li>
              <li>推理：vLLM / SGLang + H200 × 8</li>
              <li>定制：行业 LoRA + 主体保持模块</li>
              <li>合规：水印 + 内容审核 + 调用审计</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📱 ToC SaaS</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>中文：Kling / 即梦 / 海螺</li>
              <li>国际：Runway / Pika / Sora</li>
              <li>数字人：HeyGen / Hedra / 商汤如影</li>
              <li>本地化：HeyGen Translate</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #ec4899;">
            <h3>🔬 研究 / Hackathon</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>Open-Sora / CogVideoX (开源)</li>
              <li>LTX-Video / Mochi-1 (实时实验)</li>
              <li>ComfyUI 节点工作流</li>
              <li>HuggingFace Diffusers 集成</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>10. 资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://github.com/Tencent/HunyuanVideo" target="_blank">HunyuanVideo (腾讯)</a> · <a href="https://github.com/THUDM/CogVideo" target="_blank">CogVideoX</a></li>
          <li><a href="https://github.com/hpcaitech/Open-Sora" target="_blank">Open-Sora</a> · <a href="https://github.com/genmoai/models" target="_blank">Mochi-1</a></li>
          <li><a href="https://github.com/Lightricks/LTX-Video" target="_blank">LTX-Video (实时)</a> · <a href="https://github.com/Wan-Video/Wan2.1" target="_blank">Wan 2.1 (阿里)</a></li>
          <li><a href="https://comfyanonymous.github.io/ComfyUI_examples/" target="_blank">ComfyUI 视频工作流</a></li>
          <li><a href="https://github.com/showlab/Awesome-Video-Diffusion" target="_blank">Awesome Video Diffusion 综述</a></li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
