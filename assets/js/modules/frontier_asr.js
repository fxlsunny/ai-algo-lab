/* 模块：语音识别 ASR（前沿） */
MCH.register("frontier_asr", {
  render() {
    const code = `# ============================================================================
# 现代 ASR：从 Wav2Vec → Whisper → SenseVoice / Paraformer / Qwen2-Audio
# ============================================================================
# 主流路线两类：
#   (A) Encoder-only + CTC / RNN-T : Conformer / Wav2Vec / WeNet / Paraformer
#   (B) Encoder-Decoder Seq2Seq    : Whisper / OWSM / SeamlessM4T / SenseVoice

# ─── 经典：CTC + Conformer Encoder ───────────────────────────────
class ConformerASR(nn.Module):
    """Conformer = Conv 局部 + Multi-Head Attention 全局 + Macaron FFN"""
    def __init__(self, n_mels=80, dim=512, depth=18, vocab=4233):
        self.frontend = nn.Sequential(
            MelSpectrogram(n_mels=n_mels),
            SpecAugment(p=0.5),                 # 时间/频率掩蔽
        )
        self.subsample = Conv2dSubsampling(in_dim=n_mels, out_dim=dim, factor=4)
        self.encoder   = nn.ModuleList([ConformerBlock(dim, heads=8) for _ in range(depth)])
        self.ctc_head  = nn.Linear(dim, vocab)

    def forward(self, wav, label):
        x = self.frontend(wav)
        x = self.subsample(x)               # T → T/4
        for blk in self.encoder:
            x = blk(x)
        log_p = self.ctc_head(x).log_softmax(-1)
        loss = F.ctc_loss(log_p, label, ...)
        return loss

# ─── 现代：Whisper-style Seq2Seq（端到端多语种 + 任务） ────────
class Whisper(nn.Module):
    """OpenAI Whisper：680k 小时弱监督训练，零样本多语种"""
    def __init__(self):
        self.encoder = ConformerEncoder(...)        # 30s 音频 → 1500 token
        self.decoder = TransformerDecoder(...)      # 自回归解码
        # 特殊 token：<|en|>, <|zh|>, <|transcribe|>, <|translate|>, <|notimestamps|>

    def transcribe(self, audio_30s):
        enc = self.encoder(log_mel(audio_30s))
        return self.decoder.generate(enc, prompt=[SOT, LANG, TASK])

# ─── 流式 ASR：U2++（双向蒸馏） / Cascaded Encoders ───────────
# WeNet U2++：训练共享 encoder，单向 + 双向两个 decoder，推理选模式
# RNN-T (Streaming)：Predictor + Joiner，逐块吐字
# Paraformer (NPU)：Non-Autoregressive 并行解码 → 实时性强

# ─── 2024 大模型 ASR：用 LLM 当 Decoder ────────────────────────
# Qwen2-Audio / Qwen2.5-Omni / SALMONN / Audio-Flamingo
# 架构：Audio Encoder + Adaptor + LLM
# 优势：跨语种 / 跨任务（识别 + 翻译 + 问答 + 情感） 一个模型搞定
class LLMBasedASR(nn.Module):
    def __init__(self, audio_enc, adaptor, llm):
        self.audio_enc = audio_enc          # Whisper-V3 Encoder / SenseVoice Enc
        self.adaptor   = adaptor            # 1-2 层 MLP / Q-Former
        self.llm       = llm                # Qwen2 / Llama-3 / Mistral

    def forward(self, audio, text_prompt):
        atok = self.adaptor(self.audio_enc(audio))   # 音频 token
        ttok = self.llm.embed(text_prompt)            # "请转写并翻译成中文"
        return self.llm.decode(torch.cat([atok, ttok], dim=1))
`;

    return `
      ${MCH.hero({
        icon: "🎙",
        name: "语音识别 ASR",
        en: "Speech Recognition · Whisper / Conformer / Paraformer / SenseVoice / Qwen-Audio",
        tags: ["Conformer", "Wav2Vec", "Whisper", "Paraformer", "U2++", "LLM-ASR"],
        meta: ["◈ 端到端语音转文本 + 多任务", "⚡ Whisper · Paraformer · SenseVoice · Qwen2-Audio"],
      })}

      <div class="section" style="background:linear-gradient(135deg,#cffafe 0%,#a5f3fc 100%);border:1px solid #67e8f9;">
        <h2 style="color:#155e75;border:none;padding:0;margin:0 0 6px 0;">🆕 关键演进：HMM-GMM (2010 前) → DNN-HMM 2012 → CTC/Listen-Attend-Spell 2014-16 → Conformer 2020 → Wav2Vec 2.0 2020 → Whisper 2022 → LLM-as-ASR 2024</h2>
      </div>

      <div class="section">
        <h2>1. 发展史</h2>
        <div class="mermaid">
flowchart TB
    HMM[HMM-GMM 1980-2010<br/>统计声学模型] --> DNN[DNN-HMM 2012<br/>Hinton/MS]
    DNN --> CTC[CTC 2006/2014<br/>Graves<br/>无强制对齐]
    DNN --> LAS[Listen-Attend-Spell 2015<br/>Google]
    CTC --> RNNT[RNN-T 2012/2017<br/>流式 ASR]

    LAS --> S2S[Seq2Seq Encoder-Decoder 2016+<br/>ESPnet]
    CTC --> Joint[CTC+Attn Hybrid 2017+<br/>Watanabe]

    S2S --> SP[SpecAugment 2019<br/>Google] --> Conf[Conformer 2020<br/>Google · 卷积+注意力]
    Conf --> WeNet[WeNet U2++ 2021<br/>出门问问]
    Conf --> Para[Paraformer 2022<br/>阿里达摩院]

    Wav2[Wav2Vec 2.0 2020<br/>Meta · 自监督] --> HuB[HuBERT 2021<br/>Meta]
    HuB --> WavLM[WavLM 2021<br/>微软]

    Wav2 --> Whisper[🏆 Whisper v1/v2/v3 2022-24<br/>OpenAI · 680k 小时]
    Whisper --> Distil[Distil-Whisper 2023]
    Whisper --> SenseVoice[SenseVoice 2024<br/>阿里 · 50 种语言]
    Whisper --> SeamlessM4T[SeamlessM4T 2023<br/>Meta]
    Whisper --> OWSM[OWSM 2024<br/>开源 Whisper]

    Whisper --> Qwen2A[Qwen2-Audio 2024<br/>阿里]
    Qwen2A --> Omni[🆕 Qwen2.5-Omni 2024]
    SenseVoice --> Salmonn[SALMONN 2024<br/>Spoken QA]
    Qwen2A --> GPT4o[GPT-4o Audio 2024<br/>OpenAI]
    Qwen2A --> Moshi[Moshi 2024<br/>Kyutai 全双工]
        </div>
      </div>

      <div class="section">
        <h2>2. 三大主流架构对比</h2>
        <table class="table">
          <thead><tr><th>架构</th><th>原理</th><th>代表</th><th>优势</th><th>劣势</th></tr></thead>
          <tbody>
            <tr><td><b>CTC + Encoder</b></td><td>Encoder + 帧级 softmax + CTC loss</td><td>Wav2Vec, WeNet, Conformer-CTC</td><td>极快 / 流式天然 / 单调对齐</td><td>独立预测、需外接 LM</td></tr>
            <tr><td><b>RNN-T (Transducer)</b></td><td>Encoder + Predictor + Joiner，流式</td><td>Google Pixel ASR, Apple Siri</td><td>纯流式 / 端到端 / 无 LM 也好</td><td>训练耗内存大</td></tr>
            <tr><td><b>Encoder-Decoder S2S</b></td><td>自回归解码生成 token</td><td>Whisper, SenseVoice, Paraformer</td><td>多任务通用 / 长上下文</td><td>非流式 / 解码慢</td></tr>
            <tr><td><b>NAR (Non-AR)</b></td><td>并行解码（CIF/Predictor）</td><td>Paraformer, SenseVoice-Small</td><td>推理快 5-10×</td><td>边界依赖 CIF 头</td></tr>
            <tr><td><b>LLM-Based ASR</b></td><td>Audio Enc + LLM 解码</td><td>Qwen2-Audio, SALMONN, Audio-Flamingo, GPT-4o</td><td>多任务统一 / 上下文/对话</td><td>大、慢、贵</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 代表模型横评（2024-2025）</h2>
        <table class="table">
          <thead><tr><th>模型</th><th>厂商</th><th>规模</th><th>语种</th><th>开闭源</th><th>核心亮点</th></tr></thead>
          <tbody>
            <tr><td><b>Whisper Large-v3 / Turbo</b></td><td>OpenAI</td><td>1.5B / 0.8B</td><td>99</td><td>开源</td><td>事实标准 · 多任务（识别/翻译/语种）</td></tr>
            <tr><td><b>Distil-Whisper</b></td><td>HF</td><td>756M</td><td>~10</td><td>开源</td><td>蒸馏 · 推理 6×</td></tr>
            <tr><td><b>OWSM v3.1</b></td><td>CMU</td><td>1B</td><td>150</td><td>开源</td><td>完全开放训练数据 · 复刻 Whisper</td></tr>
            <tr><td><b>SenseVoice (Small/Large)</b></td><td>阿里</td><td>234M / 1.6B</td><td>50</td><td>开源</td><td>🏆 中文 SOTA · 情感/事件检测</td></tr>
            <tr><td><b>Paraformer-zh-large</b></td><td>阿里</td><td>220M</td><td>中文</td><td>开源 (FunASR)</td><td>NAR 工业落地标杆</td></tr>
            <tr><td><b>FireRedASR</b></td><td>小红书</td><td>8.3B</td><td>中英</td><td>开源</td><td>🆕 多专家 LLM-ASR · 中文新 SOTA</td></tr>
            <tr><td><b>U2++ (WeNet 2.0)</b></td><td>出门问问</td><td>~100M</td><td>中英</td><td>开源</td><td>双向蒸馏 · 流式 + 离线统一</td></tr>
            <tr><td><b>Qwen2-Audio / Qwen2.5-Omni</b></td><td>阿里</td><td>7B+</td><td>30+</td><td>开源</td><td>LLM-ASR · 语音问答 · 双语</td></tr>
            <tr><td><b>SALMONN</b></td><td>清华+字节</td><td>13B</td><td>中英</td><td>开源</td><td>语音 + 音乐 + 环境声 多任务</td></tr>
            <tr><td><b>SeamlessM4T v2</b></td><td>Meta</td><td>2.3B</td><td>100+</td><td>开源</td><td>🏆 同时支持 ASR + S2T + S2S 翻译</td></tr>
            <tr><td><b>Voxtral</b></td><td>Mistral</td><td>3B / 24B</td><td>多语</td><td>开源</td><td>🆕 法系 LLM-ASR · 长音频 30 分钟</td></tr>
            <tr><td><b>GPT-4o Audio</b></td><td>OpenAI</td><td>未公开</td><td>多语</td><td>闭源</td><td>实时对话 ASR + TTS 一体</td></tr>
            <tr><td><b>Moshi</b></td><td>Kyutai</td><td>7B</td><td>主英</td><td>开源</td><td>🆕 全双工 / 边听边说</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 关键工程难点</h2>
        <div class="grid-3">
          <div class="card">
            <h3>🌍 噪声 + 口音</h3>
            <p style="font-size:13px;line-height:1.6">真实场景信噪比低（地铁 / 嘈杂会议）+ 口音重。SpecAugment、SimuGAN 增噪、强模型预训练 + 现场场景微调是必备。</p>
          </div>
          <div class="card">
            <h3>👥 多说话人 / 鸡尾酒会</h3>
            <p style="font-size:13px;line-height:1.6">会议 / 客服多说话人。Diarization (pyannote, NeMo) + ASR 联合是标准方案。Whisper-Diart, EEND-Vector 在前沿研究。</p>
          </div>
          <div class="card">
            <h3>🏷 时间戳 + 标点</h3>
            <p style="font-size:13px;line-height:1.6">字幕/会议纪要要求字级时间戳 + 自动标点。WhisperX (forced-align) / FunASR Punc 模型是流行方案。</p>
          </div>
          <div class="card">
            <h3>🔥 热词 / 命名实体</h3>
            <p style="font-size:13px;line-height:1.6">"产品名 / 人名"识别准确率关键。Hotword Boosting (CTC LM 加权)、Phrase Biasing、Prompt 注入是路径。</p>
          </div>
          <div class="card">
            <h3>⏱ 实时延迟</h3>
            <p style="font-size:13px;line-height:1.6">同传 / 直播字幕需 < 300ms。Streaming RNN-T / U2++ + Chunk Attention + 量化部署。WebRTC + GPU 推理流是基础。</p>
          </div>
          <div class="card">
            <h3>📦 长音频</h3>
            <p style="font-size:13px;line-height:1.6">Whisper 单次仅 30s。需要 VAD 切分 + 拼接 + 上下文转换，否则丢词。WhisperX / Pyannote.audio + chunk 是主流。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>6. 主流应用场景</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>关键需求</th><th>代表落地</th></tr></thead>
          <tbody>
            <tr><td>📞 智能客服</td><td>多轮对话、情感、意图</td><td>容联云、Microsoft Dynamics</td></tr>
            <tr><td>🎬 字幕生成</td><td>多语种、标点、时间戳</td><td>剪映 / 网易见外、Descript</td></tr>
            <tr><td>📋 会议纪要</td><td>多人 + 总结 + Action Item</td><td>飞书妙记、腾讯会议同传、Otter.ai</td></tr>
            <tr><td>🚗 车载语音</td><td>低延迟 + 噪声 + 离线</td><td>蔚来 NOMI、华为问界、Apple CarPlay</td></tr>
            <tr><td>🩺 医疗病历</td><td>专业术语、隐私、长录音</td><td>科大讯飞、Nuance DAX、Abridge</td></tr>
            <tr><td>📚 教育 / 学语言</td><td>发音评估、口语打分</td><td>多邻国、流利说</td></tr>
            <tr><td>📺 直播同传</td><td>实时翻译 + 字幕</td><td>抖音同传、腾讯会议、Zoom</td></tr>
            <tr><td>♿ 无障碍</td><td>实时字幕、聋哑助手</td><td>Live Caption (Android), Apple Live Speech</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 未来发展方向</h2>
        <div class="grid-2">
          <div class="card" style="border-left:3px solid #06b6d4;">
            <h3>🗣 全双工对话 ASR</h3>
            <p style="font-size:13px;line-height:1.6">边听边说、可被打断（GPT-4o, Moshi, Qwen-Omni）。延迟 < 300ms，不是"按下说话"。需要流式 + 端到端音频对齐。</p>
          </div>
          <div class="card" style="border-left:3px solid #06b6d4;">
            <h3>🌐 LLM-ASR 一体</h3>
            <p style="font-size:13px;line-height:1.6">音频 token + 文本 token 在同一个 LLM 中。识别 / 翻译 / 摘要 / 问答都通过 prompt 控制。FireRedASR、SALMONN、Voxtral 是先驱。</p>
          </div>
          <div class="card" style="border-left:3px solid #06b6d4;">
            <h3>🎶 ASR + 副语言</h3>
            <p style="font-size:13px;line-height:1.6">不只是字，还要识别情感、笑声、语速、停顿、说话人特征。SenseVoice、Audio-Flamingo 已尝试。情感客服核心。</p>
          </div>
          <div class="card" style="border-left:3px solid #06b6d4;">
            <h3>📞 端侧 ASR</h3>
            <p style="font-size:13px;line-height:1.6">手机 / 耳机 / 助听器本地实时识别。Apple AFM-on-device、Distil-Whisper-Tiny、Sherpa-Onnx。隐私 + 低延迟。</p>
          </div>
          <div class="card" style="border-left:3px solid #06b6d4;">
            <h3>🏞 鲁棒 / 长尾语种</h3>
            <p style="font-size:13px;line-height:1.6">从 Top 100 语种 → 7000+ 全人类语种。Meta MMS / SeamlessM4T / Google USM 在拓展。低资源数据 + 自监督是关键。</p>
          </div>
          <div class="card" style="border-left:3px solid #06b6d4;">
            <h3>🎯 个性化定制</h3>
            <p style="font-size:13px;line-height:1.6">企业内部专业术语自定义。Adapter / LoRA + 用户少样本数据。Whisper-Adapter、CrossLingual-LoRA 等。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>8. 优缺点 & 选型</h2>
        <div class="grid-3">
          <div class="card" style="border-left:3px solid #10b981;">
            <h3>✅ 优势</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>开源生态成熟 (Whisper / SenseVoice)</li>
              <li>多语种零样本能力强</li>
              <li>商业落地最早最广</li>
              <li>端侧/服务端方案完整</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #ef4444;">
            <h3>❌ 局限</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>嘈杂 / 强口音仍翻车</li>
              <li>低资源语种数据稀缺</li>
              <li>专业术语 / 命名实体易错</li>
              <li>30s+ 长音频拼接复杂</li>
            </ul>
          </div>
          <div class="card" style="border-left:3px solid #4f46e5;">
            <h3>◎ 选型建议</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li>中文工业：SenseVoice / Paraformer / FireRedASR</li>
              <li>多语种：Whisper Large-v3 / SeamlessM4T</li>
              <li>实时流：U2++ / WeNet / RNN-T</li>
              <li>端侧：Distil-Whisper / Sherpa-Onnx</li>
              <li>对话/Omni：Qwen2.5-Omni / GPT-4o</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>9. 资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://github.com/openai/whisper" target="_blank">OpenAI Whisper</a> · <a href="https://github.com/m-bain/whisperX" target="_blank">WhisperX (字级时间戳)</a></li>
          <li><a href="https://github.com/FunAudioLLM/SenseVoice" target="_blank">SenseVoice (阿里)</a> · <a href="https://github.com/modelscope/FunASR" target="_blank">FunASR (Paraformer)</a></li>
          <li><a href="https://github.com/wenet-e2e/wenet" target="_blank">WeNet U2++</a> · <a href="https://github.com/k2-fsa/sherpa-onnx" target="_blank">Sherpa-Onnx (端侧)</a></li>
          <li><a href="https://huggingface.co/spaces/hf-audio/open_asr_leaderboard" target="_blank">Open ASR Leaderboard</a></li>
        </ul>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
