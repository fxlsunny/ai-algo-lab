/* 模块：辅助 Encoder */
MCH.register("aux_encoder", {
  render() {
    const code = `# 辅助 Encoder：APP 包名 + URL 域名 + 数值统计
# 来源：src/models/aux_encoder.py

class HashedBagEmbedding(nn.Module):
    """Feature Hashing + Mean Pooling：
       - 把任意离散字符串 (pkg, domain) 哈希到 2^N 个桶
       - 每个桶一个 embedding；同一商户的多个 ID mean pool 成一个向量
       - 优点：无需维护词表，新 APP / 新域名即时生效（冷启友好）"""
    def __init__(self, num_buckets, emb_dim=32):
        self.emb = nn.Embedding(num_buckets, emb_dim, padding_idx=0)

    def forward(self, ids, mask):
        e = self.emb(ids)                             # (B, L, emb_dim)
        m = mask.unsqueeze(-1).float()
        return (e * m).sum(1) / m.sum(1).clamp(min=1.0)


class AuxEncoder(nn.Module):
    def __init__(self, app_hash_buckets=262144, url_hash_buckets=524288,
                 emb_dim=32, dense_stat_dim=64, hidden_dim=512):
        self.app_bag = HashedBagEmbedding(app_hash_buckets, emb_dim)  # APP 2^18
        self.url_bag = HashedBagEmbedding(url_hash_buckets, emb_dim)  # URL 2^19
        self.dense = nn.Sequential(                                   # 64 维数值统计
            nn.Linear(dense_stat_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, 256), nn.GELU(), nn.Dropout(0.1),
        )
        self.proj = nn.Sequential(nn.Linear(emb_dim*2 + 256, hidden_dim),
                                  nn.LayerNorm(hidden_dim), nn.GELU())

    def forward(self, batch):
        app_vec   = self.app_bag(batch["app_ids"], batch["app_mask"])   # (B, 32)
        url_vec   = self.url_bag(batch["url_ids"], batch["url_mask"])   # (B, 32)
        dense_vec = self.dense(batch["dense_stat"])                      # (B, 256)
        return self.proj(torch.cat([app_vec, url_vec, dense_vec], -1))   # (B, 512)`;

    return `
      ${MCH.hero({
        icon: "A",
        name: "辅助 Encoder — Hash Bag + Dense MLP",
        en: "Auxiliary Encoder · feature hashing + dense stats",
        tags: ["APP 2^18 桶", "URL 2^19 桶", "64 维 dense 统计", "Feature Hashing"],
        meta: ["◈ emb_dim = 32", "⚡ 冷启友好", "◇ 输出 512-d"],
      })}

      <div class="section">
        <h2>1. 算法原理</h2>
        <p class="text-sm text-slate-600 leading-relaxed">
          辅助模态承载三类"弱信号"：
        </p>
        <div class="grid-3">
          <div class="card">
            <h4 class="font-semibold text-slate-800">📱 APP 包名</h4>
            <p class="text-xs text-slate-600 mt-2">商户绑定设备上安装的 APP 列表（<code>com.taobao</code> / <code>com.tencent.qqlive</code> ...）。<b>赌博类 APP</b> 聚簇是强风险信号。</p>
          </div>
          <div class="card">
            <h4 class="font-semibold text-slate-800">🌐 URL 域名</h4>
            <p class="text-xs text-slate-600 mt-2">商户近期访问域名三级（<code>ps1.ps2.path</code>）。<b>大量彩票/色情</b>类域名聚类触发灰名单。</p>
          </div>
          <div class="card">
            <h4 class="font-semibold text-slate-800">📊 Dense 统计</h4>
            <p class="text-xs text-slate-600 mt-2">64 维 <b>历史交易统计</b>：笔数、近 30d GMV、资金进出比、场景熵、日均笔数、夜间占比 ...</p>
          </div>
        </div>

        <h3>· 为什么用 Feature Hashing？</h3>
        <div class="formula-block">
          $$ h(\\text{pkg}) = \\text{MD5}(\\text{pkg}) \\bmod B, \\quad e_{\\text{pkg}} = \\text{Embedding}[h], \\quad v = \\frac{1}{|\\mathcal{P}|}\\sum_{p \\in \\mathcal{P}} e_p $$
        </div>
        ${MCH.info(`
          <b>Feature Hashing 的 3 个优势：</b>
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>冷启友好</b>：不需要词表，新 APP / 新域名即时映射到已存在桶；</li>
            <li><b>显存可控</b>：APP 2^18 × 32-d ≈ 8M 参数，URL 2^19 × 32-d ≈ 16M 参数；</li>
            <li><b>碰撞不致命</b>：mean pool 平均效应使偶发碰撞影响很小；桶数越大碰撞率越低。</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 核心代码（注释版）</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 交互可视化 — 哈希碰撞率</h2>
        <p class="text-sm text-slate-600">
          不同"独立 APP / URL 数量"下，桶大小 B 与哈希<b>碰撞率</b>的权衡（生日攻击近似
          $p \\approx 1 - e^{-N(N-1)/(2B)}$）。调节以观察桶数过小导致的碰撞问题。
        </p>

        <div class="grid-2 mt-4">
          <div>
            <div class="ctrl-panel">
              ${MCH.slider({ id: "hash-B", label: "桶大小 2^k（当前 2^）", min: 10, max: 22, step: 1, value: 18, format: (v) => v })}
              ${MCH.slider({ id: "hash-N-max", label: "独立 APP / URL 数上限 N_max", min: 1000, max: 200000, step: 1000, value: 50000, format: (v) => v.toLocaleString() })}
            </div>
            <div class="card mt-3">
              <h4 class="font-semibold text-slate-800 text-sm">· 当前配置</h4>
              <div id="hash-config" class="text-xs text-slate-600 mt-2 leading-relaxed"></div>
            </div>
          </div>
          <div id="chart-hash" style="height:320px;"></div>
        </div>
      </div>

      <div class="section">
        <h2>4. 64 维 Dense 统计特征示例</h2>
        <p class="text-sm text-slate-600">下面展示一个典型商户的 dense stat 分布（已 z-score 归一化）。</p>
        <div id="chart-dense" style="height:320px;margin-top:10px;"></div>
      </div>
    `;
  },

  mount() {
    // 哈希碰撞率
    const chart = MCH.echart(document.getElementById("chart-hash"), {
      tooltip: { trigger: "axis", valueFormatter: (v) => (v * 100).toFixed(2) + "%" },
      legend: { top: 0 },
      grid: { left: 60, right: 30, top: 40, bottom: 40 },
      xAxis: { type: "value", name: "N (独立元素数)" },
      yAxis: { type: "value", name: "碰撞率", min: 0, max: 1 },
      series: [
        { name: "2^10 (1K)", type: "line", showSymbol: false, smooth: true, data: [] },
        { name: "2^14 (16K)", type: "line", showSymbol: false, smooth: true, data: [] },
        { name: "2^18 (APP 默认)", type: "line", showSymbol: false, smooth: true, data: [], lineStyle: { width: 3 }, color: "#4f46e5" },
        { name: "2^19 (URL 默认)", type: "line", showSymbol: false, smooth: true, data: [], lineStyle: { width: 3 }, color: "#10b981" },
        { name: "当前", type: "line", showSymbol: false, smooth: true, data: [], lineStyle: { width: 4, type: "solid" }, color: "#f59e0b" },
      ],
    });
    const update = () => {
      const k = parseInt(document.getElementById("hash-B").value);
      const Nmax = parseInt(document.getElementById("hash-N-max").value);
      const B_current = 2 ** k;
      const Ns = MCH.linspace(100, Nmax, 60).map(Math.round);
      const compute = (B) => Ns.map(N => [N, 1 - Math.exp(-N * (N - 1) / (2 * B))]);
      chart.setOption({
        series: [
          { data: compute(1024) },
          { data: compute(16384) },
          { data: compute(262144) },
          { data: compute(524288) },
          { data: compute(B_current) },
        ],
      });
      // 当前碰撞
      const colAtNmax = 1 - Math.exp(-Nmax * (Nmax - 1) / (2 * B_current));
      document.getElementById("hash-config").innerHTML = `
        桶数 B = 2^${k} = <b>${B_current.toLocaleString()}</b><br/>
        假设 N = ${Nmax.toLocaleString()} 个独立元素<br/>
        预期碰撞率（至少一次碰撞）= <b style="color:${colAtNmax > 0.1 ? '#ef4444' : '#10b981'};">${(colAtNmax * 100).toFixed(2)}%</b><br/>
        总参数 ≈ <b>${(B_current * 32 / 1e6).toFixed(2)}M</b> 个 (embedding × 32-d)
      `;
    };
    ["hash-B", "hash-N-max"].forEach(id => {
      document.getElementById(id).addEventListener("input", (e) => {
        const fmt = id === "hash-B" ? ((v) => v) : ((v) => parseInt(v).toLocaleString());
        document.getElementById(id + "-val").textContent = fmt(e.target.value);
        update();
      });
    });
    update();

    // Dense stat example
    const labels = ["总笔数", "30d GMV", "日均笔数", "夜间占比", "资金入/出比", "场景熵", "对手方唯一度", "首笔天数", "小额占比", "大额占比", "退款率", "对私转账占比", "跨境占比", "活动日占比", "同设备商户数", "同IP商户数"];
    const values = [2.1, 3.5, 1.2, -0.3, 0.8, -1.2, 1.9, -0.8, -0.5, 2.3, 0.3, 1.1, -1.5, 0.4, -0.2, 0.7];
    MCH.echart(document.getElementById("chart-dense"), {
      tooltip: {},
      grid: { left: 100, right: 30, top: 30, bottom: 30 },
      xAxis: { type: "value", name: "z-score" },
      yAxis: { type: "category", data: labels, axisLabel: { fontSize: 11 } },
      series: [{
        type: "bar",
        data: values.map(v => ({ value: v, itemStyle: { color: v > 0 ? "#4f46e5" : "#ef4444" } })),
        barWidth: 12,
      }],
    });
  },
});
