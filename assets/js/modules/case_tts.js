/* 模块：案例 · 语音合成 TTS（行业落地） */
MCH.register("case_tts", {
  render() {
    const code = `# ============================================================================
# 工业 TTS 完整管线（参考 ElevenLabs / 阿里 CosyVoice / 火山豆包语音）
# ============================================================================
# 阶段 1：文本前处理
#  · 文本清洗：URL/Emoji/HTML/Tag 处理
#  · TN (Text Normalization)：阿拉伯数字 → 中文读法 / 日期 / 单位 / 货币
#  · 多音字 + 韵律：基于规则 + BERT 级模型；声调 / 儿化音 / 轻声
#  · 句法切分：按标点 / 长度 / 语义切到 ~30s

# 阶段 2：声学建模（核心模型）
#  · 文本 → 音素 / 离散 token
#  · 音色 / 风格 / 情感条件
#  · 输出 Mel-Spec 或 Codec tokens

# 阶段 3：声码器（可选）
#  · 若输出 Mel-Spec，需 HiFi-GAN / BigVGAN-V2 / Vocos 还原波形
#  · 若直接输出 codec token，由 Codec decoder 还原（Encodec / DAC）

# 阶段 4：后处理
#  · 标点级停顿对齐 / 强制对齐时间戳（用于字幕）
#  · 响度归一化 (LUFS) / 静音 trim / 高通滤波
#  · 流式合成：Sentence-by-Sentence，TTFB < 300ms

# ─── 现代主流：CosyVoice 2 / F5 / E2 / VALL-E 2 通用骨架 ──────
class TTSPipeline:
    def __init__(self):
        self.tn       = TextNormalizer(rules="zh+en")
        self.encoder  = TextEncoder(qwen_or_bert)        # 文本理解
        self.codec    = AudioCodec(encodec_or_dac)       # 离散音频表征
        self.acoustic = FlowMatchingTTS(dim=1024, depth=22)  # 核心声学
        self.vocoder  = BigVGANV2()                      # 还原 24/44 kHz
        self.aligner  = ForcedAligner()                  # 时间戳

    def speak(self, text, ref_audio=None, ref_text=None, emotion=None):
        text = self.tn(text)
        sentences = split_by_prosody(text)
        wavs = []
        for s in sentences:
            cond = self.encoder(s, emotion)
            ref_codes = self.codec.encode(ref_audio) if ref_audio is not None else None
            mel = self.acoustic.flow_sample(cond, prompt_codes=ref_codes, n_steps=10)
            wav = self.vocoder(mel)
            wavs.append(wav)
        return concatenate(wavs)`;

    return `
      ${MCH.hero({
        icon: "🗣",
        name: "案例 · 语音合成 TTS",
        en: "Industrial Text-to-Speech Pipeline",
        tags: ["CosyVoice", "F5-TTS", "ElevenLabs", "实时流式", "音色克隆", "情感"],
        meta: ["◈ 客服 / 配音 / 数字人 / 智驾 / 教育", "⚡ 阿里 / 字节 / 科大讯飞 / ElevenLabs / OpenAI"],
      })}

      <div class="section">
        <h2>1. 业务场景与核心挑战</h2>
        <div class="mermaid">
flowchart LR
    INPUT[用户文本/SSML] --> TN[文本前处理<br/>TN/多音字/韵律]
    TN --> AM[声学模型<br/>Codec LM / Flow Matching]
    REF[参考音频 3s] --> AM
    EMO[情感/风格 token] --> AM
    AM --> VOC[声码器<br/>BigVGAN-V2 / Vocos]
    VOC --> POST[后处理<br/>Loudness/Trim/水印]
    POST --> OUT[音频流 24/44 kHz]
        </div>
        ${MCH.info(`
          <b>工业级 TTS 的 5 大核心挑战</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>实时性</b>：客服 / 直播首字延迟 (TTFB) &lt; 300ms · 可中断</li>
            <li><b>多语种 / 中英混读</b>：英文专有名词、数字、货币的自然读法</li>
            <li><b>音色一致 + 风格可控</b>：同 IP 角色跨剧集声音不漂移</li>
            <li><b>情感 + 副语言</b>：笑、叹气、低声、哽咽、儿化、轻声</li>
            <li><b>合规 + 反 Deepfake</b>：克隆需授权、强制水印、欺诈检测</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 传统 TTS vs 现代 TTS 技术对比</h2>
        <table class="table">
          <thead><tr><th>能力</th><th>🏛 传统（2020 之前）</th><th>🚀 现代（2024-2025）</th><th>关键差异</th></tr></thead>
          <tbody>
            <tr><td><b>建模范式</b></td><td>Tacotron 2 / FastSpeech 2 (Mel)</td><td>Codec LM / Flow Matching / VALL-E</td><td>音素+时长 → 端到端文本→codec</td></tr>
            <tr><td><b>音色定制</b></td><td>录音棚 ~10 小时数据 + 全量微调</td><td>3-10 秒参考音零样本克隆</td><td>需要专门录制 → in-context 提示</td></tr>
            <tr><td><b>多语种</b></td><td>每语种独立模型</td><td>单模型 30-50 语种 / 双语混读</td><td>共享表征 + 多语种数据</td></tr>
            <tr><td><b>情感</b></td><td>每情感独立模型</td><td>SSML / 情感 token / 自然语言</td><td>显式标签 → 文本指令</td></tr>
            <tr><td><b>实时</b></td><td>整段合成 1-3s 延迟</td><td>流式分段，TTFB &lt; 300ms</td><td>句切分 + KV-Cache + Chunk 解码</td></tr>
            <tr><td><b>标点/数字</b></td><td>规则 + 词表</td><td>BERT/Qwen 级 TN 模型</td><td>统计 → 神经化</td></tr>
            <tr><td><b>对话场景</b></td><td>独立合成不连贯</td><td>对话 LLM + TTS 端到端 (GPT-4o, Moshi)</td><td>分两阶段 → 一体化</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心管线代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 主流商用方案对比</h2>
        <table class="table">
          <thead><tr><th>方案</th><th>厂商</th><th>核心亮点</th><th>典型场景</th><th>价格档</th></tr></thead>
          <tbody>
            <tr><td><b>科大讯飞超拟人</b></td><td>科大讯飞</td><td>中文 SOTA · 多方言 · ToB 商用最早</td><td>政企 / 教育 / 车载</td><td>RMB 100-500/M 字符</td></tr>
            <tr><td><b>火山引擎/豆包语音</b></td><td>字节</td><td>抖音生态 · 价格屠夫 · 多语种</td><td>短视频 · ToC SaaS</td><td>RMB 5-50/M 字符</td></tr>
            <tr><td><b>阿里达摩 / CosyVoice 2</b></td><td>阿里</td><td>🏆 开源可商用 · 双语零样本 · 副语言</td><td>客服 · 自建私有化</td><td>开源免费 / API</td></tr>
            <tr><td><b>Microsoft Azure TTS</b></td><td>微软</td><td>140+ 语种 · Neural Voice · 自定义</td><td>跨国客服 · 本地化</td><td>$15-30/M 字符</td></tr>
            <tr><td><b>ElevenLabs</b></td><td>ElevenLabs</td><td>🏆 国际克隆事实标准 · 情感细腻</td><td>有声书 · 配音 · 海外 ToC</td><td>$0.18-0.30/M 字符</td></tr>
            <tr><td><b>OpenAI TTS / Realtime</b></td><td>OpenAI</td><td>低门槛 API · GPT-4o 一体对话</td><td>对话产品 · 海外</td><td>$15-30/M 字符</td></tr>
            <tr><td><b>Google Cloud TTS / WaveNet</b></td><td>Google</td><td>研究底蕴深 · WaveNet 系</td><td>Google Workspace 生态</td><td>$4-16/M 字符</td></tr>
            <tr><td><b>HeyGen / D-ID</b></td><td>HeyGen / D-ID</td><td>TTS + 数字人合一 · 多语种本地化</td><td>视频本地化 · 营销</td><td>$30-300/月订阅</td></tr>
            <tr><td><b>F5-TTS / E2-TTS / Spark-TTS</b></td><td>开源</td><td>Apache · 高质量低门槛</td><td>研究 / 自建私有化</td><td>免费</td></tr>
            <tr><td><b>GPT-SoVITS</b></td><td>开源</td><td>🏆 个人少样本克隆神器</td><td>同人 · 二创 · 个人项目</td><td>免费</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 工业落地：5 个真实场景拆解</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📞 智能客服外呼</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：千万级日呼叫 · 自然不机械 · 实时打断 · 情感识别<br/>
              <b>方案</b>：CosyVoice 2 在线流式 + 上下文 LLM 控制语速/情绪 + ASR 双工<br/>
              <b>难点</b>：用户突然挂断 / 反讽 / 投诉 → 模型实时调整语气
            </p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🎙 有声书 / 长音频</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：1-50 小时长读 · 多角色对话 · 一致音色 · 标点自然<br/>
              <b>方案</b>：F5-TTS / ElevenLabs Studio + 角色音色库 + 章节级强制对齐<br/>
              <b>难点</b>：连续 1 小时不漂移 · 角色切换无突兀 · 错字预校对
            </p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>👤 数字人 / 直播</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：嘴型同步 · 多语种 · 实时 · 情感<br/>
              <b>方案</b>：CosyVoice + EMO/Hallo 唇形 + Live2D/UE 5 渲染<br/>
              <b>难点</b>：跨语种唇音同步 · 突发口播 · 直播 24/7 稳定
            </p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🚗 车载语音</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：离线端侧 · 实时 · 噪声鲁棒 · 多角色<br/>
              <b>方案</b>：科大讯飞嵌入式 / Distil-TTS · 端侧 NPU 推理 · 多说话人切换<br/>
              <b>难点</b>：车载 NPU 算力受限 · 引擎噪声干扰 · 跨座位定向
            </p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>🎬 视频本地化 / 配音</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：保留原音色翻译多国语言 · 唇形对齐 · 情感保留<br/>
              <b>方案</b>：HeyGen Translate / 网易见外 / 字节 Translate Avatar<br/>
              <b>难点</b>：源 + 目标语言长度不一 · 文化语境差异 · 版权
            </p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📚 教育 / 朗读</h3>
            <p style="font-size:13px;line-height:1.6">
              <b>需求</b>：标准发音 · 多语种 · 朗读评测<br/>
              <b>方案</b>：FunASR + CosyVoice + 韵律打分 · 多邻国 / 流利说<br/>
              <b>难点</b>：发音准确度评分 · 中英文混读 · 课件 SSML 编排
            </p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>6. 工程难点 + 实战经验</h2>
        <div class="grid-3">
          <div class="card">
            <h3>🎯 文本规整 (TN)</h3>
            <p style="font-size:13px;line-height:1.6">"+86-13800138000 在 2025/3/14 14:30 充值 ¥1,234.56" 怎么读？规则 + 神经 TN 双轨：规则保覆盖率，神经修长尾。</p>
          </div>
          <div class="card">
            <h3>⏱ 实时流式</h3>
            <p style="font-size:13px;line-height:1.6">用户提问 → LLM token 流式吐 → 边吐边切句 → 边合成边播。需要 sentence-level 打断、KV-Cache 复用、首块降步数。</p>
          </div>
          <div class="card">
            <h3>🎤 音色稳定</h3>
            <p style="font-size:13px;line-height:1.6">同一 IP 跨剧集声音漂移：参考音冷启动 → 训练 LoRA / Speaker-ID embedding 固化 → 生产用静态音色。</p>
          </div>
          <div class="card">
            <h3>😊 情感控制</h3>
            <p style="font-size:13px;line-height:1.6">SSML &lt;prosody rate&gt; / &lt;emphasis&gt; 已不够，新方案：自然语言 prompt（"语气低落，慢一些"）+ 情感 token。</p>
          </div>
          <div class="card">
            <h3>🛡 反 Deepfake</h3>
            <p style="font-size:13px;line-height:1.6">SynthID-Audio / AudioSeal 强制水印；克隆需授权 + KYC；线上检测器（Resemble Detect）配套。</p>
          </div>
          <div class="card">
            <h3>📈 上线监控</h3>
            <p style="font-size:13px;line-height:1.6">指标：MOS / RTF (Real-Time Factor) / TTFB / 异常率（卡顿/重复/错音）。AB 测试 + 主观评测 + 自动机器评分（NISQA, UTMOS）。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>7. 成本估算（2025）</h2>
        <table class="table">
          <thead><tr><th>方案</th><th>合成 1 万小时音频成本</th><th>每秒成本</th><th>说明</th></tr></thead>
          <tbody>
            <tr><td>商用 SaaS（讯飞 / Azure）</td><td>10-50 万元</td><td>0.001-0.01 元</td><td>价格档差异大 · 商用支持完整</td></tr>
            <tr><td>API 极致价（豆包 / DeepSeek-TTS）</td><td>1-5 万元</td><td>~0.0003 元</td><td>互联网公司价格屠夫</td></tr>
            <tr><td>开源自托管（CosyVoice + L40S）</td><td>~3 万元 (电费 + 摊薄)</td><td>~0.0008 元</td><td>需自建运维 · 适合 1k 小时/天 起</td></tr>
            <tr><td>克隆音色定制</td><td>10-100 万元 (含数据)</td><td>—</td><td>录音 + 训练 + 上线 + 合规</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>8. 端到端 Demo Stack（2025 推荐）</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3>🏢 ToB 私有化最佳实践</h3>
            <p style="font-size:13px;line-height:1.6">
              · 模型：CosyVoice 2 / F5-TTS<br/>
              · 推理：vLLM / SGLang + L40S × 4<br/>
              · 框架：FastAPI + WebSocket 流式<br/>
              · 监控：Prometheus / Grafana / 自研 MOS 抽样<br/>
              · 合规：水印 + 调用审计 + Speaker-ID 白名单
            </p>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>📱 ToC 极致体验</h3>
            <p style="font-size:13px;line-height:1.6">
              · 模型：商用 API（ElevenLabs / OpenAI Realtime / 豆包）<br/>
              · 端到端：LLM + TTS 一体（GPT-4o · Qwen-Omni · Doubao）<br/>
              · 实时：WebRTC + Opus 编码 + JitterBuffer<br/>
              · 多端：iOS Speech Framework · Android Live Caption · 端侧 fallback<br/>
              · 商业模式：订阅 + Per-Voice 定价
            </p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>9. 资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://github.com/FunAudioLLM/CosyVoice" target="_blank">CosyVoice 2 (阿里 FunAudioLLM)</a></li>
          <li><a href="https://github.com/SWivid/F5-TTS" target="_blank">F5-TTS</a> · <a href="https://github.com/SparkAudio/Spark-TTS" target="_blank">Spark-TTS</a></li>
          <li><a href="https://github.com/RVC-Boss/GPT-SoVITS" target="_blank">GPT-SoVITS</a> · <a href="https://github.com/coqui-ai/TTS" target="_blank">Coqui TTS</a></li>
          <li><a href="https://elevenlabs.io/" target="_blank">ElevenLabs</a> · <a href="https://platform.openai.com/docs/guides/realtime" target="_blank">OpenAI Realtime</a></li>
          <li><a href="https://huggingface.co/spaces/TTS-AGI/TTS-Arena" target="_blank">TTS Arena (盲评)</a></li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
