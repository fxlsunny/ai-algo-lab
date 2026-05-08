/* 模块：Diffusion 扩散模型 */
MCH.register("frontier_diffusion", {
  render() {
    const code = `# DDPM 前向 + 反向过程核心
# Ho et al., NeurIPS 2020

# ----- 前向过程：渐进加噪 (固定，不可训) -----
# x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * ε,  ε ~ N(0, I)
# alpha_bar_t = ∏_{s=1}^t (1 - beta_s) ; beta 按 schedule 递增

# ----- 反向过程：渐进去噪 (可训神经网络) -----
# 训练：预测噪声 ε_θ(x_t, t) 最小化 MSE
# 推理：从 x_T ~ N(0, I) 开始 t=T..1 迭代去噪

class UNet(nn.Module):
    def __init__(self, ...):
        # UNet 架构：Encoder → Bottleneck → Decoder，跳跃连接
        # 附加时间步 t 的 Embedding（sinusoidal）
        ...
    def forward(self, x_t, t):
        return predicted_noise

def train_step(x_0, model):
    t = torch.randint(0, T, (batch_size,))
    noise = torch.randn_like(x_0)
    alpha_bar = schedule[t].reshape(-1, 1, 1, 1)
    x_t = alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * noise
    pred_noise = model(x_t, t)
    loss = F.mse_loss(pred_noise, noise)

@torch.no_grad()
def sample(model, n_samples, steps=1000):
    x = torch.randn(n_samples, 3, 64, 64)
    for t in reversed(range(steps)):
        pred_noise = model(x, t)
        x = (x - beta_t * pred_noise / sqrt(1 - alpha_bar_t)) / sqrt(1 - beta_t)
        if t > 0: x += sigma_t * torch.randn_like(x)
    return x


# 现代实践：Stable Diffusion = Latent Diffusion
#   1) VAE 编码：图 → 64×64 潜空间 (8× 下采样)
#   2) 扩散在潜空间做（大幅加速 + 省显存）
#   3) VAE 解码：潜空间 → 图
#   4) 文本 Condition：CLIP text encoder → Cross-Attention

# 采样器加速：DDIM (50 steps) → DPM-Solver (20 steps) → LCM (4 steps)
# ControlNet：冻结 SD + 加条件分支（线稿/深度/姿态）`;

    return `
      ${MCH.hero({
        icon: "🌫",
        name: "Diffusion 扩散模型",
        en: "Diffusion Models · DDPM / Stable Diffusion / DiT",
        tags: ["DDPM", "Stable Diffusion", "ControlNet", "LCM", "DiT", "Sora"],
        meta: ["◈ 生成 SOTA", "⚡ Stable Diffusion / DALL-E / MidJourney"],
      })}

      ${MCH.versionSection("frontier_diffusion")}

      <div class="section">
        <h2>1. Diffusion 核心思想</h2>
        <p class="text-sm text-slate-600">Diffusion 通过<b>逐步加噪</b>（前向过程）和<b>逐步去噪</b>（反向过程）学习数据分布：</p>
        <div class="mermaid">
flowchart LR
    X0[x₀ 清晰图] -->|加噪| X1[x₁] -->|加噪| X2[x₂] -->|...| XT[x_T 纯噪声]
    XT -.去噪.-> X_T_1[x_{T-1}] -.去噪.-> X_T_2[x_{T-2}] -.去噪.-> Xhat[x̂₀ 生成图]
        </div>
        <div class="formula-block">
          <b>前向过程</b>：$ q(x_t | x_{t-1}) = \\mathcal{N}(x_t; \\sqrt{1-\\beta_t} x_{t-1}, \\beta_t I) $<br/>
          <b>闭式采样</b>：$ x_t = \\sqrt{\\bar\\alpha_t} x_0 + \\sqrt{1-\\bar\\alpha_t} \\epsilon $<br/>
          <b>训练目标</b>：$ \\min_\\theta \\mathbb{E}_{t, x_0, \\epsilon} \\|\\epsilon - \\epsilon_\\theta(x_t, t)\\|^2 $（预测噪声）
        </div>
      </div>

      <div class="section">
        <h2>2. Stable Diffusion 架构（事实标准）</h2>
        <div class="mermaid">
flowchart LR
    P[Prompt: A cat ...] --> T[CLIP Text Encoder]
    I[Input Image 可选] --> VAE_E[VAE Encoder]
    VAE_E --> Z[潜变量 64×64]
    Z --> DM[UNet Diffusion<br/>1000 steps]
    T --> CA[Cross-Attention 条件]
    CA --> DM
    DM --> Zhat[生成潜变量]
    Zhat --> VAE_D[VAE Decoder]
    VAE_D --> OUT[生成图 512×512]
        </div>
        ${MCH.info(`
          <b>核心创新（Rombach 2022）</b>：把扩散放到 VAE 的 <b>8× 下采样潜空间</b>，显存 ÷ 64，训练速度 ×16。
          这让高分辨率生成首次在消费级 GPU 可行。
        `, "tip")}
      </div>

      <div class="section">
        <h2>3. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>4. 采样器加速演进（大幅影响推理速度）</h2>
        <table class="table">
          <thead><tr><th>采样器</th><th>步数</th><th>质量</th><th>发布</th></tr></thead>
          <tbody>
            <tr><td>DDPM (原始)</td><td>1000</td><td>标准</td><td>2020</td></tr>
            <tr><td><b>DDIM</b></td><td>50-100</td><td>接近 DDPM</td><td>2021 - 非马尔可夫采样</td></tr>
            <tr><td><b>PLMS / PNDM</b></td><td>25-50</td><td>SD v1.x 默认</td><td>2022</td></tr>
            <tr><td><b>DPM-Solver / DPM++</b></td><td>10-25</td><td>🏆 高质量少步</td><td>2022-23</td></tr>
            <tr><td><b>🆕 LCM (Consistency Model)</b></td><td>2-4</td><td>质量接近 50 步</td><td>2023</td></tr>
            <tr><td><b>🆕 SDXL Turbo</b></td><td>1-4</td><td>Adversarial Distill</td><td>2023</td></tr>
            <tr><td><b>🆕 Hyper-SD</b></td><td>1-8</td><td>ByteDance</td><td>2024</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 扩散模型生态（2024）</h2>
        <div class="grid-3">
          <div class="card"><h4 class="font-semibold text-pink-700">📷 图像生成</h4><p class="text-xs text-slate-600 mt-1">SD 1.x/2.x, SDXL, SD3<br/>🆕 FLUX.1 (12B 开源)<br/>DALL-E 3, MidJourney v6</p></div>
          <div class="card"><h4 class="font-semibold text-pink-700">🎬 视频生成</h4><p class="text-xs text-slate-600 mt-1">🆕 Sora (OpenAI, DiT)<br/>Veo (Google)<br/>Kling (快手)<br/>Open-Sora / CogVideoX 开源</p></div>
          <div class="card"><h4 class="font-semibold text-pink-700">🎛 条件控制</h4><p class="text-xs text-slate-600 mt-1">ControlNet (2023)<br/>IP-Adapter<br/>InstantID<br/>🆕 LoRA for Diffusion</p></div>
          <div class="card"><h4 class="font-semibold text-pink-700">🧬 生物/化学</h4><p class="text-xs text-slate-600 mt-1">AlphaFold 3<br/>分子生成 (MolDiff)<br/>🆕 RFdiffusion 蛋白质</p></div>
          <div class="card"><h4 class="font-semibold text-pink-700">🎨 3D / 场景</h4><p class="text-xs text-slate-600 mt-1">Stable Video 3D<br/>Zero123<br/>DreamFusion</p></div>
          <div class="card"><h4 class="font-semibold text-pink-700">🎵 音频</h4><p class="text-xs text-slate-600 mt-1">AudioLDM<br/>MusicGen<br/>Suno (歌曲)</p></div>
        </div>
      </div>

      <div class="section">
        <h2>6. DiT (Diffusion Transformer) — Sora 的秘密</h2>
        ${MCH.info(`
          <b>🆕 DiT (Peebles 2023)</b>：用 Transformer 替代传统 UNet 作为 Diffusion backbone。
          <br/>核心改进：
          <ul style="margin-top:6px;padding-left:20px;">
            <li>更好的 scaling law（同 Transformer 在 NLP）</li>
            <li>规模化后质量显著提升</li>
            <li><b>Sora、Stable Diffusion 3、FLUX.1 都基于 DiT</b></li>
          </ul>
          OpenAI 的 Sora 就是 DiT × 视频 patches × 海量数据的产物。
        `, "tip")}
      </div>

      <div class="section">
        <h2>7. 开源资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://huggingface.co/docs/diffusers/index" target="_blank">🏆 HuggingFace Diffusers</a> — 生态事实标准</li>
          <li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui" target="_blank">Automatic1111</a> — 最流行 SD Web UI</li>
          <li><a href="https://github.com/comfyanonymous/ComfyUI" target="_blank">ComfyUI</a> — 节点式工作流</li>
          <li><a href="https://github.com/black-forest-labs/flux" target="_blank">FLUX.1 官方</a> — 2024 开源 SOTA</li>
          <li><a href="https://arxiv.org/abs/2212.09748" target="_blank">DiT 论文</a></li>
          <li><a href="https://openai.com/sora" target="_blank">Sora Technical Report</a></li>
        </ul>
      </div>

      <div class="section">
        <h2>8. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("frontier_diffusion").pros, MCH.getById("frontier_diffusion").cons, MCH.getById("frontier_diffusion").best_for)}
      </div>
    `;
  },
});
