/* 模块：音频与歌曲生成（前沿） */
MCH.register("frontier_audio_gen", {
  render() {
    const code = `# ============================================================================
# 音频生成 · 三大表征 + 两大主流路线
# ============================================================================
#
# 表征三选一：
#   (1) Mel-Spectrogram：FastSpeech / Diffusion-TTS 的输入
#   (2) Codec Tokens (Neural Audio Codec)：Encodec / SoundStream / DAC / WavTokenizer
#       - 把 16/24/44 kHz 波形压到 50-150 token/秒
#       - 多 codebook（RVQ）可逐层重建
#   (3) Waveform：直接生成原始 16-bit PCM（HiFi-GAN/Vocos/BigVGAN 是声码器）
#
# 路线两选一：
#   (A) 自回归 LM (Decoder-only)：AudioLM / MusicLM / VALL-E / SongLM / Suno 类
#   (B) Diffusion / Flow：MusicGen-Melody (Flow), Stable Audio, AudioLDM, MusicFlow
#
# 现代趋势：A + B 混合（AR 抓粗结构 + Diffusion 精细化），如 Stable Audio Open

# ─── 神经音频编解码器（核心基础设施）─────────────────────────
class Encodec(nn.Module):
    """Meta Encodec：把 24kHz waveform → 75 Hz × 8 codebook tokens"""
    def encode(self, wav):
        z = self.conv_encoder(wav)              # (B, T_z, D)
        codes = self.rvq(z)                     # 残差矢量量化
        return codes                            # (B, T_z, n_q=8)
    def decode(self, codes):
        z = self.rvq.dequantize(codes)
        return self.conv_decoder(z)             # 还原 waveform

# ─── 文本到音乐 · MusicGen (Meta 2023) ─────────────────────────
class MusicGen(nn.Module):
    """单 LM + Encodec token + 文本/旋律条件"""
    def __init__(self):
        self.text_enc = T5Encoder()
        self.codec    = Encodec()
        self.lm       = TransformerDecoder(vocab=2048*8)  # 4 codebook 平铺
    def generate(self, text, duration=30):
        text_emb = self.text_enc(text)
        # delay pattern：每个 codebook 错位预测，便于并行训练
        codes = self.lm.generate(text_emb, max_len=duration * 50)
        return self.codec.decode(codes)

# ─── Suno / 歌曲生成（带歌词 + 人声） ─────────────────────────
# 核心多模态对齐：Lyrics (text) + Genre/Mood (text) + Vocal Style (speaker emb) → Codec tokens
# 训练数据：万级小时音乐 + 歌词时间对齐 + 风格标签
# 关键：人声 + 伴奏一体生成，避免分轨拼接的不自然
`;

    const tts_code = `# ============================================================================
# TTS 前沿 · LLM-Style + Flow Matching 是 2024-25 主流
# ============================================================================
# 演进：拼接合成 (HMM) → Tacotron 2 (2017) → FastSpeech 2 (2020)
#       → VITS (2021) → NaturalSpeech 3 (2024) → CosyVoice / Spark-TTS / VALL-E 2 / E2 / F5

# ─── VALL-E (Microsoft 2023)：In-context Prompt TTS ────────────
# 用 3 秒参考音 (zero-shot) 复刻音色
# 输入：prompt audio (codec tokens) + target text → 续写 codec tokens
# 路径：粗粒度 AR + 细粒度 NAR 两阶段

# ─── F5-TTS / E2-TTS (2024)：Flow Matching 范式 ───────────────
# Flow Matching：学习直线 ODE，10 步出片
# 不需要 phoneme/duration 显式建模，端到端 char→mel→wav
class F5TTS(nn.Module):
    def forward(self, text, ref_audio, ref_text):
        # 训练：x_t = (1-t)·noise + t·target_mel
        # 模型预测 v(x_t, t, condition) → 学习目标 = target_mel - noise
        return v_pred

# ─── CosyVoice / FunAudioLLM (阿里 2024)：LLM + Flow Matching ─
# Audio LLM 出 semantic tokens → CFM Vocoder 出 wav
# 优势：跨语种 zero-shot、副语言 (笑/哭/呼吸) 可控
`;

    return `
      ${MCH.hero({
        icon: "🎵",
        name: "音频/歌曲生成",
        en: "Audio & Music Generation · MusicGen / Suno / Udio / Stable Audio",
        tags: ["AudioLM", "MusicGen", "Suno", "Stable Audio", "Encodec", "Flow"],
        meta: ["◈ 文生音乐 · 文生歌曲 · 文生音效", "⚡ Suno · Udio · 海绵 · MusicGen · Stable Audio"],
      })}

      <div class="section" style="background:linear-gradient(135deg,#fef3c7 0%,#fde68a 100%);border:1px solid #fbbf24;">
        <h2 style="color:#854d0e;border:none;padding:0;margin:0 0 6px 0;">🆕 关键节点：WaveNet 2016 → Jukebox 2020 → AudioLM/MusicLM 2023 → MusicGen 2023 → Suno v3 2024 → Stable Audio 2 2024 → Suno v4/Udio 2024 → YuE 2024（首个开源全曲）→ <b>ACE-Step 1.0/1.5 2025（开源歌曲新 SOTA）</b> → Suno v5 2025</h2>
      </div>

      <div class="section">
        <h2>1. 发展史：从 PCM 直生成到 Codec-LM</h2>
        <div class="mermaid">
flowchart TB
    WN[WaveNet 2016<br/>Google · 自回归 PCM] --> WG[WaveRNN/WaveGAN]
    WN --> Tac[Tacotron 1/2 2017<br/>Mel-Spec TTS]
    Tac --> FS[FastSpeech 1/2 2019-20<br/>NAR TTS]
    FS --> VITS[VITS 2021<br/>端到端 TTS]

    Jx[Jukebox 2020<br/>OpenAI · 4096 step VQ-VAE] --> ALM[AudioLM 2022<br/>Google]
    ALM --> MusicLM[MusicLM 2023<br/>Google]
    ALM --> Bark[Bark 2023<br/>Suno 早期开源]

    Enc[Encodec 2022<br/>Meta] --> SS[SoundStream 2021] --> DAC[DAC 2023<br/>Descript]
    DAC --> WT[WavTokenizer 2024]

    Enc --> MusicGen[🏆 MusicGen 2023<br/>Meta · 33B token 训练]
    Enc --> SA1[Stable Audio 1 2023<br/>Stability]
    SA1 --> SA2[Stable Audio 2 2024<br/>3 分钟全曲]

    MusicLM --> SunoV1[Suno v1-v2 2023] --> SunoV3[🏆 Suno v3 2024.03<br/>2 分钟全人声歌曲]
    SunoV3 --> SunoV4[Suno v4 2024.11]
    SunoV3 --> Udio[Udio 2024.04<br/>前 Google Brain]

    SunoV3 --> Tian[天工 SkyMusic 2024<br/>昆仑万维]
    SunoV3 --> Hai[海绵 HaiMian 2024<br/>字节]
    SunoV3 --> YuE[YuE 2024<br/>港中文 + 多伦多]
    YuE --> ACE10[ACE-Step 1.0 2025<br/>ACE-Studio + StepFun]
    ACE10 --> ACE15[🏆 ACE-Step 1.5 2025<br/>2B DiT + LM Composer<br/>4 分钟曲 < 1s · MIT 开源]
    ACE15 --> SunoV5[Suno v5 2025]

    VITS --> NS3[NaturalSpeech 3 2024<br/>微软]
    Enc --> Cosy[CosyVoice / FunAudioLLM 2024<br/>阿里]
    Cosy --> F5[F5-TTS / E2-TTS 2024<br/>SJTU + 上海 AI Lab]
    F5 --> Spark[Spark-TTS 2025]
        </div>
      </div>

      <div class="section">
        <h2>2. 三大子任务对比</h2>
        <table class="table">
          <thead><tr><th>任务</th><th>典型时长</th><th>核心难点</th><th>主流方案</th><th>代表产品</th></tr></thead>
          <tbody>
            <tr><td><b>TTS（文生语音）</b></td><td>1-30s</td><td>音色复刻 / 情感 / 副语言</td><td>Flow Matching / Codec LM</td><td>CosyVoice, F5, ElevenLabs, OpenAI TTS</td></tr>
            <tr><td><b>音效 / SFX</b></td><td>1-10s</td><td>自然 + 文本对齐</td><td>AudioLDM / Stable Audio</td><td>Stable Audio, ElevenLabs SFX, AudioGen</td></tr>
            <tr><td><b>纯音乐生成</b></td><td>30s-3min</td><td>结构 / 旋律 / 节奏一致</td><td>Codec LM + Diffusion</td><td>MusicGen, Stable Audio 2, MusicFX</td></tr>
            <tr><td><b>歌曲（含人声）</b></td><td>30s-4min</td><td>歌词对齐 + 人声 + 伴奏一体</td><td>Codec LM (大数据)</td><td>🏆 Suno · Udio · 海绵 · 天工 · YuE</td></tr>
            <tr><td><b>语音克隆 (VC)</b></td><td>3s 参考</td><td>零样本相似度</td><td>VALL-E / CosyVoice / GPT-SoVITS</td><td>ElevenLabs Clone, OpenVoice</td></tr>
            <tr><td><b>语音转换 / 歌声合成</b></td><td>—</td><td>音高 / 颤音 / 共振峰</td><td>SVC / RVC / DiffSinger</td><td>RVC v2, NN-SVS, ACE Studio</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码：Codec-LM 范式</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section" style="background:linear-gradient(135deg,#ecfdf5 0%,#d1fae5 100%);border:1px solid #6ee7b7;">
        <h2 style="color:#065f46;">🆕 ACE-Step 1.5 — 2025 开源歌曲生成新 SOTA</h2>
        <p class="text-sm text-slate-700">
          ACE-Step 1.5 由 <b>ACE-Studio + StepFun 阶跃星辰</b>联合发布（2025），是目前
          <b>开源音乐基座模型的事实新标杆</b>：单卡 RTX 3090 上 10s 内生成 4 分钟整首歌、
          A100 不到 1s（200× 蒸馏加速），&lt; 4 GB 显存即可本地跑，MIT 开源 GitHub 10k+ 星。
          <b>Music Arena 主观盲评质量介于 Suno v4.5 与 Suno v5 之间</b>，全面超越所有现有开源方案。
        </p>
        <div class="grid-2 mt-3">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3 class="font-semibold">🧠 核心架构：Hybrid Reasoning-Diffusion</h3>
            <ul style="font-size:13px;line-height:1.65;padding-left:16px;list-style:disc inside;margin-top:6px;">
              <li><b>LM (Qwen3-0.6B) 当 Composer Agent</b>：把模糊用户 prompt 通过 CoT 转成结构化 YAML 蓝图（BPM / 调式 / 段落 / 时长 / 歌词配置）→ 再投喂 DiT</li>
              <li><b>DiT (~2B) 当声学渲染器</b>：Hybrid Attention（奇数层 Sliding Window 抓局部细节，偶数层 Global GQA 保整曲律动）</li>
              <li><b>FSQ Tokenizer</b>：把 25 Hz 连续潜变量压到 5 Hz 离散码（codebook ≈ 64k），打通 LM ↔ DiT</li>
              <li><b>Decoupled DMD2 + ConvNeXt-GAN 蒸馏</b>：50 步 → 8 步，A100 上 240s 曲子 ~ 1s 出片</li>
              <li><b>内禀 RL 对齐</b>：DiffusionNTF (DiT) + GRPO (LM)，无需外部 Reward Model</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #10b981;">
            <h3 class="font-semibold">🎼 4 大交互范式 + 6 大子任务</h3>
            <ul style="font-size:13px;line-height:1.65;padding-left:16px;list-style:disc inside;margin-top:6px;">
              <li><b>Planner Mode</b>：模糊 prompt → 完整结构蓝图</li>
              <li><b>Listener Mode</b>：从音频 codes 反向写歌词 / caption</li>
              <li><b>Co-Pilot Mode</b>：扩写简单创意为完整歌曲</li>
              <li><b>Refiner Mode</b>：标准化 / 优化用户 prompt</li>
              <li>统一 <b>Masked Generative Framework</b> 同时支持：① Text-to-Music ② Cover Generation 翻唱 ③ Repainting 局部重绘 ④ Track Extraction 抽轨 ⑤ Layering 加配器 ⑥ Completion 续写补全</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #10b981;">
            <h3 class="font-semibold">⚡ 性能 & 工程指标</h3>
            <ul style="font-size:13px;line-height:1.65;padding-left:16px;list-style:disc inside;margin-top:6px;">
              <li>4 分钟整首歌：A100 < 2s · RTX 3090 < 10s · &lt; 4 GB 显存</li>
              <li>支持 10s ~ 10 min 任意时长</li>
              <li><b>50+ 语种 + 2000+ 风格 + 1000+ 乐器</b></li>
              <li><b>10-120× 推理速度优势</b> vs 同类模型</li>
              <li>少样本 LoRA 微调（几首歌即可学个人风格）</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #10b981;">
            <h3 class="font-semibold">📊 训练范式（Alignment-First）</h3>
            <ul style="font-size:13px;line-height:1.65;padding-left:16px;list-style:disc inside;margin-top:6px;">
              <li>2700 万样本三阶段课程学习</li>
              <li>Stage 1：2000 万 T2M 对，铺底</li>
              <li>Stage 2：600 万 stem-separated 多轨，开放编辑能力</li>
              <li>Stage 3：200 万高质量精选 SFT，对齐 prompt 跟随</li>
              <li>ACE-Captioner / ACE-Transcriber 配套数据基建（基于 Qwen2.5-Omni + GRPO）</li>
            </ul>
          </div>
        </div>
        ${MCH.info(`
          <b>为什么 ACE-Step 1.5 是开源里程碑</b>：第一次让"商用级歌曲生成"在 <b>消费级 GPU + MIT 开源</b> 同时成立。
          v1.0（2025 Q1）就因为可消费级跑而出圈；v1.5 再叠加 200× 蒸馏 + Hybrid Reasoning 架构，
          让"几秒出整首带人声的歌"从 Suno SaaS 独占变成本地可控、可二创、可商用。
          它把 v1.0 的 mel-spec 表征短板换成 FSQ + DiT，并补齐了翻唱 / 重绘 / 抽轨等专业级编辑能力 —— 这一步基本对标 Suno v4.5。
        `, "tip")}
      </div>

      <div class="section">
        <h2>5. TTS 现代主流</h2>
        ${MCH.code(tts_code, "python")}
        ${MCH.info(`
          <b>2024 TTS 三大趋势</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>零样本音色克隆</b>：3-10s 参考即可。VALL-E 2、CosyVoice 2、F5、E2 都做到接近不可分辨。</li>
            <li><b>Flow Matching 替代 Diffusion</b>：训练稳定 + 推理 10 步内 + 无需 duration model。</li>
            <li><b>副语言 / 情感控制</b>：&lt;laughter&gt; &lt;breath&gt; &lt;sigh&gt; 等显式 token，CosyVoice / Spark-TTS 主推。</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>6. 代表产品横评（2024-2025）</h2>
        <table class="table">
          <thead><tr><th>产品</th><th>厂商</th><th>类别</th><th>开闭源</th><th>核心亮点</th></tr></thead>
          <tbody>
            <tr><td><b>Suno v3 / v4 / v4.5 / v5</b></td><td>Suno.ai</td><td>歌曲</td><td>闭源</td><td>🏆 闭源全曲 SOTA · 4 分钟人声+伴奏+歌词 · v5 加多语种 + 风格控制</td></tr>
            <tr><td><b>Udio</b></td><td>Udio</td><td>歌曲</td><td>闭源</td><td>前 Google Brain · 编曲质感细腻</td></tr>
            <tr><td><b>🏆 ACE-Step 1.5 / XL</b></td><td>ACE-Studio + StepFun</td><td>歌曲 / 全任务</td><td>🏆 开源 MIT</td><td>🆕 2B DiT + LM 规划 · A100 < 1s/4分钟 · 介于 Suno v4.5 与 v5 之间 · 翻唱/重绘/抽轨一体</td></tr>
            <tr><td><b>YuE</b></td><td>港中文 + Multimodal Art Projection</td><td>歌曲</td><td>开源 7B</td><td>首个开源全曲生成 · 已被 ACE-Step 反超</td></tr>
            <tr><td><b>天工 SkyMusic</b></td><td>昆仑万维</td><td>歌曲</td><td>闭源 SaaS</td><td>中文歌曲早期可用</td></tr>
            <tr><td><b>海绵 / 海贝</b></td><td>字节</td><td>歌曲 + TTS</td><td>闭源</td><td>抖音生态 · 改编同人</td></tr>
            <tr><td><b>MusicGen / MusicGen-Melody</b></td><td>Meta</td><td>音乐</td><td>开源</td><td>3.3B/7B · 旋律条件 · 30s</td></tr>
            <tr><td><b>Stable Audio 2.0 / Open</b></td><td>Stability</td><td>音乐 + SFX</td><td>开源 (Open)</td><td>3 分钟全曲 · Diffusion 路线</td></tr>
            <tr><td><b>AudioLDM 2 / AudioGen</b></td><td>Meta + 萨里大学</td><td>SFX</td><td>开源</td><td>文本 / 视觉条件音效</td></tr>
            <tr><td><b>MusicFX</b></td><td>Google</td><td>音乐</td><td>闭源</td><td>MusicLM 商业化</td></tr>
            <tr><td><b>ElevenLabs</b></td><td>ElevenLabs</td><td>TTS / SFX / Music</td><td>闭源</td><td>商业 TTS 标杆 · 2024 推出 SFX/Music</td></tr>
            <tr><td><b>OpenAI Voice / TTS / Realtime</b></td><td>OpenAI</td><td>TTS / 对话</td><td>闭源</td><td>GPT-4o 集成 · Realtime API</td></tr>
            <tr><td><b>CosyVoice 2 / FunAudioLLM</b></td><td>阿里</td><td>TTS</td><td>开源</td><td>🏆 中文零样本 · 双语 · 副语言</td></tr>
            <tr><td><b>F5-TTS / E2-TTS</b></td><td>SJTU + 上海 AI Lab</td><td>TTS</td><td>开源</td><td>Flow Matching · 高质量低门槛</td></tr>
            <tr><td><b>GPT-SoVITS</b></td><td>RVC 项目</td><td>TTS / VC</td><td>开源</td><td>个人少样本克隆神器</td></tr>
            <tr><td><b>RVC v2</b></td><td>开源社区</td><td>歌声转换</td><td>开源</td><td>"AI 孙燕姿" 现象级流行</td></tr>
            <tr><td><b>DiffSinger / NN-SVS</b></td><td>开源</td><td>歌声合成</td><td>开源</td><td>VOCALOID 替代</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 关键工程难点</h2>
        <div class="grid-3">
          <div class="card">
            <h3>🎼 长程结构</h3>
            <p style="font-size:13px;line-height:1.6">3 分钟一首歌需要"主歌-副歌-桥段"宏观结构。直接 AR 一镜到底易跑题。需要 Hierarchical Token / Layout Plan。</p>
          </div>
          <div class="card">
            <h3>🎤 人声 + 伴奏一体</h3>
            <p style="font-size:13px;line-height:1.6">分轨生成易"贴皮"。Suno / Udio 的优势是端到端混合训练，但训练数据极贵。歌词时间对齐数据是壁垒。</p>
          </div>
          <div class="card">
            <h3>📜 歌词 → 旋律对齐</h3>
            <p style="font-size:13px;line-height:1.6">字数 / 韵脚 / 重音对应节拍。Lyric-to-Melody 子任务、Phoneme + Note 联合建模。</p>
          </div>
          <div class="card">
            <h3>🎸 风格控制</h3>
            <p style="font-size:13px;line-height:1.6">"摇滚 / 古风 / Lofi"风格如何精确表达？文本 prompt + 参考音乐双条件 + Style Embedding。</p>
          </div>
          <div class="card">
            <h3>📦 数据 + 版权</h3>
            <p style="font-size:13px;line-height:1.6">高质量音乐多版权付费。Suno / Udio 已被 RIAA 起诉。Stable Audio 走"自有 + 授权"路线。</p>
          </div>
          <div class="card">
            <h3>🔊 高保真度</h3>
            <p style="font-size:13px;line-height:1.6">44.1 kHz 立体声不能有可闻 artifact。BigVGAN-V2、Vocos、APNet 是声码器主流。Codec 多 codebook 残差很关键。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>8. 主流应用场景</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>关键需求</th><th>代表落地</th></tr></thead>
          <tbody>
            <tr><td>📺 短视频 BGM</td><td>10-60s 风格音乐</td><td>抖音 / 快手 BGM 库（接 Suno / 海绵）</td></tr>
            <tr><td>🎮 游戏配乐</td><td>动态自适应 BGM</td><td>米哈游 · 网易 · 育碧实验</td></tr>
            <tr><td>📢 广告 / 营销</td><td>定制广告歌</td><td>Suno · Loudly · Mubert</td></tr>
            <tr><td>🎙 有声书 / 播客</td><td>多人对话 TTS</td><td>Audible · 微信读书 · Speechify</td></tr>
            <tr><td>📞 智能客服 TTS</td><td>多语种 / 情感 / 实时</td><td>科大讯飞 · 微软 Azure · ElevenLabs</td></tr>
            <tr><td>🎬 影视配音</td><td>多语种本地化</td><td>HeyGen · 网易 · 抖音 AI 配音</td></tr>
            <tr><td>👤 数字人 / 虚拟主播</td><td>实时对话 + 唱歌</td><td>Heygen · 商汤如影 · Live2D 联动</td></tr>
            <tr><td>🎼 音乐人辅助</td><td>编曲灵感 / Demo</td><td>Suno + DAW 工作流、AIVA</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>9. 未来发展方向</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>🎼 真长音频（5-10 分钟）</h3>
            <p style="font-size:13px;line-height:1.6">完整专辑级生成 + 歌单连贯性。需要 Memory Token + Section Plan + 蒸馏。Suno v4 / YuE 已推到 4-8 分钟。</p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>🎙 全双工对话音频</h3>
            <p style="font-size:13px;line-height:1.6">真正的"AI 主持人 / 主播"——边听边说还能唱。Moshi · GPT-4o Realtime · Qwen2.5-Omni。下一代 Siri / 小爱。</p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>🎸 可控编辑 / 局部重生成</h3>
            <p style="font-size:13px;line-height:1.6">"把第二段副歌换成钢琴版" 自然语言编辑。MusicLDM-Edit / Music ControlNet 已起步。下一代 Logic Pro。</p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>🎵 视频音乐联合生成</h3>
            <p style="font-size:13px;line-height:1.6">视频 → 配乐 / 配音同步生成（V2A）。Veo 3 + MMAudio + MovieGen Audio。下一代影视生产。</p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>🛡 水印 / 反 Deepfake</h3>
            <p style="font-size:13px;line-height:1.6">SynthID-Audio (Google)、AudioSeal (Meta)、Watermark Audio。监管层面已强制溯源。</p>
          </div>
          <div class="card" style="border-left:3px solid #f59e0b;">
            <h3>📱 端侧 TTS / Audio</h3>
            <p style="font-size:13px;line-height:1.6">手机 / 耳机端实时音频合成（Apple AFM、Distil-CosyVoice）。延迟 + 隐私 + 离线必备。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>10. 优缺点 & 选型</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3>✅ 优势</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>素材生产爆炸性提速</li>
              <li>音色克隆门槛低（3s 参考）</li>
              <li>多语种 / 跨风格切换</li>
              <li>开源生态可商用 (ACE-Step / CosyVoice / F5)</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #ef4444;">
            <h3>❌ 局限</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>真长曲目（10 分钟+）仍易跑题</li>
              <li>版权 / 数据合规复杂（Suno/Udio 已被 RIAA 起诉）</li>
              <li>声纹克隆带欺诈风险</li>
              <li>专业级混音 / 母带仍需人工</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>◎ 选型建议</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>🏆 开源歌曲首选：<b>ACE-Step 1.5 / XL</b></li>
              <li>歌曲商用闭源：Suno v5 / Udio / 天工</li>
              <li>开源备选：YuE / MusicGen-Melody</li>
              <li>音乐 / 音效：Stable Audio 2 / MusicGen</li>
              <li>中文 TTS：CosyVoice 2 / Spark-TTS</li>
              <li>音色克隆：F5-TTS / GPT-SoVITS</li>
              <li>实时对话：GPT-4o / Moshi / Qwen-Omni</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>11. 资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://github.com/ace-step/ACE-Step-1.5" target="_blank">🏆 ACE-Step 1.5 (ACE-Studio + StepFun, MIT)</a> · <a href="https://ace-step.github.io/ace-step-v1.5.github.io/" target="_blank">ACE-Step 1.5 Project Page</a></li>
          <li><a href="https://github.com/multimodal-art-projection/YuE" target="_blank">YuE (开源歌曲生成)</a></li>
          <li><a href="https://github.com/facebookresearch/audiocraft" target="_blank">AudioCraft (MusicGen / AudioGen)</a> · <a href="https://github.com/Stability-AI/stable-audio-tools" target="_blank">Stable Audio Tools</a></li>
          <li><a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">CosyVoice 2 (阿里)</a> · <a href="https://github.com/SWivid/F5-TTS" target="_blank">F5-TTS</a></li>
          <li><a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">GPT-SoVITS</a> · <a href="https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI" target="_blank">RVC v2</a></li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
