/* 模块：时序基座模型 */
MCH.register("ts_foundation", {
  render() {
    const code = `# Chronos (Amazon 2024) — 把时序当作"自然语言"
# 核心：量化 + Decoder-only Transformer (基于 T5)

from chronos import ChronosPipeline
import torch

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-large",    # 200M 参数基座
    device_map="cuda",
    torch_dtype=torch.bfloat16,
)

# 零样本预测：直接推理新的时序，不需要训练！
context = torch.tensor([23.1, 23.5, 24.1, ..., 25.3])  # 过去 512 步
forecast = pipeline.predict(
    context=context,
    prediction_length=64,           # 未来 64 步
    num_samples=20,                 # 概率采样
)
# forecast: (1, 20, 64) → 可取 median / quantile

# ============================================================
# TimesFM (Google 2024) 使用
# ============================================================
import timesfm
model = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend="gpu",
        horizon_len=128,                  # 预测长度
        num_layers=50,
        model_dims=1280,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m",
    ),
)
# 批量零样本预测
forecast_input = [np.array([series1]), np.array([series2])]
frequency_input = [0, 1]   # 0=高频, 1=月, 2=季/年
forecast, experimental_quantile_forecast = model.forecast(
    forecast_input, frequency_input)


# ============================================================
# Moirai (Salesforce 2024) - 多频率 + MoE
# ============================================================
from uni2ts.model.moirai import MoiraiForecast
model = MoiraiForecast.load_from_checkpoint(
    "Salesforce/moirai-1.0-R-large",
    prediction_length=96, target_dim=1,
)`;

    return `
      ${MCH.hero({
        icon: "🏔",
        name: "时序基座模型 — TimesFM / Chronos / Moirai",
        en: "Time Series Foundation Models (2024 New Era)",
        tags: ["TimesFM", "Chronos", "Moirai", "Lag-Llama", "零样本", "2024 新范式"],
        meta: ["◈ 零样本开箱即用", "⚡ 预训练 10 亿+ 时序点"],
      })}

      ${MCH.versionSection("ts_foundation")}

      <div class="section">
        <h2>1. 时序基座时代开启 (2024)</h2>
        ${MCH.info(`
          <b>范式转变</b>：2024 年谷歌、亚马逊、Salesforce、CMU 等先后发布时序基座模型，
          类似 NLP 的 GPT / BERT 时刻。<br/><br/>
          <b>核心变化</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li>🏛 <b>旧范式</b>：业务→数据收集→特征工程→单独训练模型→上线</li>
            <li>🚀 <b>新范式</b>：业务→直接调用基座模型（零样本）→上线</li>
          </ul>
          效果对比：在 Monash 等 30+ benchmark 上，基座模型<b>零样本</b>已接近或超过专门训练的 SOTA。
        `, "biz")}
      </div>

      <div class="section">
        <h2>2. 三大主流时序基座对比</h2>
        <table class="table">
          <thead>
            <tr><th>模型</th><th>发布方</th><th>参数</th><th>架构</th><th>亮点</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>🏆 TimesFM-1.0/2.0</b></td>
              <td>Google</td>
              <td>200M / 500M</td>
              <td>Decoder-only Transformer + Patching</td>
              <td>开源、100B 训练点、高频/月频通用</td>
            </tr>
            <tr>
              <td><b>🏆 Chronos-T5</b></td>
              <td>Amazon</td>
              <td>20M-710M (5 档)</td>
              <td>T5 Encoder-Decoder + 量化 tokenize</td>
              <td>🆕 把时序当文本，直接套 T5</td>
            </tr>
            <tr>
              <td><b>🏆 Moirai</b></td>
              <td>Salesforce</td>
              <td>14M-311M</td>
              <td>Encoder-only + MoE + Multi-frequency</td>
              <td>🆕 多频率统一，工业场景友好</td>
            </tr>
            <tr>
              <td>Lag-Llama</td>
              <td>ServiceNow/Mila</td>
              <td>200M</td>
              <td>Decoder-only + lag features</td>
              <td>首个开源概率预测基座</td>
            </tr>
            <tr>
              <td>MOMENT</td>
              <td>CMU</td>
              <td>40M-385M</td>
              <td>Encoder + patch + masking 预训练</td>
              <td>多任务：预测/分类/异常检测</td>
            </tr>
            <tr>
              <td>TimeGPT-1</td>
              <td>Nixtla</td>
              <td>商业闭源</td>
              <td>-</td>
              <td>首个商业化 API</td>
            </tr>
            <tr>
              <td>🆕 Toto</td>
              <td>Datadog</td>
              <td>151M</td>
              <td>Decoder-only</td>
              <td>专为运维指标训练</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 为什么时序基座可行？三大 insight</h2>
        <div class="grid-3">
          <div class="card" style="border-top:3px solid #06b6d4;">
            <h3 class="font-bold text-cyan-700">1️⃣ 跨领域共性</h3>
            <p class="text-xs text-slate-600 mt-2">任何领域的时序都有<b>趋势/周期/噪声/突变</b>这些共性模式。基座学会这些共性后可迁移。</p>
          </div>
          <div class="card" style="border-top:3px solid #06b6d4;">
            <h3 class="font-bold text-cyan-700">2️⃣ 数据量的 Scaling</h3>
            <p class="text-xs text-slate-600 mt-2">TimesFM 训练于 100B 时序点 + 合成数据；Chronos 用 8.4B + 合成；数据够多，模型就能总结规律。</p>
          </div>
          <div class="card" style="border-top:3px solid #06b6d4;">
            <h3 class="font-bold text-cyan-700">3️⃣ 统一表示</h3>
            <p class="text-xs text-slate-600 mt-2">Chronos 把时序量化成 token（类似 BPE），直接套 Transformer。Moirai 用 flattened patching 处理多频率。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>4. 核心使用代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. 零样本 vs 专门训练 的取舍</h2>
        <table class="table">
          <thead><tr><th>场景</th><th>推荐</th><th>理由</th></tr></thead>
          <tbody>
            <tr><td>新业务快速上线（&lt; 1 周）</td><td><b>基座零样本</b></td><td>TimesFM / Chronos，1 行代码出结果</td></tr>
            <tr><td>有丰富业务标注 + 追求极致精度</td><td>PatchTST 专训</td><td>针对数据调优仍有优势</td></tr>
            <tr><td>多业务统一预测平台</td><td><b>基座 + LoRA 微调</b></td><td>共享底座，各业务少量定制</td></tr>
            <tr><td>端侧 / 实时推理</td><td>经典/专训小模型</td><td>基座推理成本高</td></tr>
            <tr><td>强可解释要求</td><td>Prophet / ARIMA</td><td>基座黑盒</td></tr>
            <tr><td>多变量 + 协变量复杂</td><td>TFT / iTransformer</td><td>基座对协变量支持有限</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. 开源资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://github.com/google-research/timesfm" target="_blank">Google TimesFM GitHub</a> — Google 200M/500M 开源权重</li>
          <li><a href="https://github.com/amazon-science/chronos-forecasting" target="_blank">Amazon Chronos</a> — 5 种大小模型 HF Hub</li>
          <li><a href="https://github.com/SalesforceAIResearch/uni2ts" target="_blank">Salesforce Moirai</a> — 多频率通用基座</li>
          <li><a href="https://github.com/time-series-foundation-models/lag-llama" target="_blank">Lag-Llama</a></li>
          <li><a href="https://github.com/moment-timeseries-foundation-model/moment" target="_blank">CMU MOMENT</a> — 多任务</li>
          <li><a href="https://github.com/Nixtla/nixtla" target="_blank">Nixtla TimeGPT</a> — 商业 API</li>
          <li><a href="https://huggingface.co/collections/amazon/chronos-models-65f1791d630a8d57cb718444" target="_blank">HuggingFace Chronos 集合</a></li>
        </ul>
      </div>

      <div class="section">
        <h2>7. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("ts_foundation").pros, MCH.getById("ts_foundation").cons, MCH.getById("ts_foundation").best_for)}
      </div>
    `;
  },
});
