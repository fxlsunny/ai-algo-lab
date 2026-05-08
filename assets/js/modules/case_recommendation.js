/* 模块：案例 · 推荐算法 */
MCH.register("case_recommendation", {
  render() {
    return `
      ${MCH.hero({
        icon: "🛒",
        name: "案例 · 推荐算法 Recommendation",
        en: "Recommendation Systems · 召回→粗排→精排→重排四级漏斗",
        tags: ["召回", "精排", "序列建模", "DIN", "SASRec", "LLM4Rec", "多目标"],
        meta: ["◈ 电商/短视频/信息流核心", "⚡ 阿里·字节·美团·小红书"],
      })}

      ${MCH.versionSection("case_recommendation")}

      <div class="section">
        <h2>1. 业务场景与核心挑战</h2>
        <p class="text-sm text-slate-600">推荐系统的本质是"在<b>千万级 item 池</b>中为每个用户实时挑选最相关的 Top-K"。分为四级漏斗：</p>
        <div class="mermaid">
flowchart LR
    U[用户请求] --> R1[召回 Recall<br/>千万 → 千级<br/>多路召回 / 向量检索]
    R1 --> R2[粗排 Pre-Rank<br/>千级 → 百级<br/>轻量模型 双塔]
    R2 --> R3[精排 Ranking<br/>百级 → 十级<br/>DIN/DIEN 重模型]
    R3 --> R4[重排 Re-Rank<br/>多样性 &amp; 业务规则<br/>MMR / DPP / LLM]
    R4 --> O[曝光 Top-K]
        </div>
        ${MCH.info(`
          <b>核心挑战</b>：
          <ul style="margin-top:6px;padding-left:20px;">
            <li><b>冷启动</b>：新用户/新物品几无历史行为</li>
            <li><b>长尾物品</b>：大部分物品曝光极少，Matthew Effect 明显</li>
            <li><b>多目标冲突</b>：CTR 高 ≠ CVR 高 ≠ 停留时长高</li>
            <li><b>Feedback Loop</b>：推荐影响用户行为，数据分布偏置</li>
            <li><b>实时性</b>：用户兴趣秒级变化，需在线学习</li>
          </ul>
        `, "tip")}
      </div>

      <div class="section">
        <h2>2. 传统方案 vs 前沿方案</h2>
        <table class="table">
          <thead>
            <tr><th>阶段</th><th>🏛 传统方案</th><th>🚀 前沿方案 (2024)</th><th>关键差异</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>召回</b></td>
              <td>协同过滤 (UserCF/ItemCF)<br/>Matrix Factorization (MF)<br/>关联规则 (Apriori)</td>
              <td>双塔 DSSM → YouTubeDNN<br/>Graph 召回 (GNN/Node2Vec)<br/>向量检索 FAISS / HNSW</td>
              <td>静态 → 语义向量 + ANN</td>
            </tr>
            <tr>
              <td><b>精排</b></td>
              <td>LR + 人工特征<br/>GBDT + 类别特征<br/>FM / FFM</td>
              <td>Wide &amp; Deep → DeepFM → DCN-V2<br/>DIN (2018) → DIEN → SIM<br/>SASRec / BERT4Rec (序列)<br/>LLM4Rec (2024 新)</td>
              <td>特征工程 → 端到端学习</td>
            </tr>
            <tr>
              <td><b>序列建模</b></td>
              <td>滑动窗口统计<br/>Markov Chain</td>
              <td>GRU4Rec (2016)<br/>SASRec (2018, Self-Attn)<br/>BERT4Rec (2019)<br/>SIM (Alibaba 2020)</td>
              <td>固定窗口 → 可变长注意力</td>
            </tr>
            <tr>
              <td><b>多目标</b></td>
              <td>业务加权公式<br/>多模型融合</td>
              <td>MMoE (2018) → PLE (2020)<br/>ESMM (CVR)<br/>PCGrad / CAGrad</td>
              <td>硬加权 → 软共享 + 梯度平衡</td>
            </tr>
            <tr>
              <td><b>冷启动</b></td>
              <td>热门推荐 baseline<br/>内容特征 + KNN</td>
              <td>Meta-Learning (MAML)<br/>LLM 生成 user/item 描述<br/>对比学习 (SimGCL)</td>
              <td>内容特征 → 跨域迁移</td>
            </tr>
            <tr>
              <td><b>强化学习</b></td>
              <td>-</td>
              <td>DQN / SlateQ<br/>RLHF for Ranking (2024)</td>
              <td>长期价值优化</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. 精排模型演进谱系</h2>
        <div class="mermaid">
flowchart TB
    LR[LR 2000s 前] --> FM[FM 2010<br/>低秩交互]
    FM --> FFM[FFM 2016<br/>Field 感知]
    FM --> WD[Wide&Deep 2016<br/>记忆+泛化]
    WD --> DeepFM[DeepFM 2017<br/>FM+DNN 端到端]
    WD --> DCN[DCN 2017<br/>交叉网络]
    DCN --> DCNV2[DCN-V2 2021<br/>矩阵分解]
    DeepFM --> DIN[DIN 2018<br/>目标 Attention]
    DIN --> DIEN[DIEN 2019<br/>兴趣演化 GRU]
    DIEN --> SIM[SIM 2020<br/>长序列 100k+]
    DIN --> SASRec[SASRec 2018<br/>Self-Attn 序列]
    SASRec --> BERT4Rec[BERT4Rec 2019]
    BERT4Rec --> LLM4Rec[LLM4Rec 2024<br/>生成式推荐]
        </div>
      </div>

      <div class="section">
        <h2>4. 代表模型亮点对比</h2>
        <table class="table">
          <thead><tr><th>模型</th><th>年份</th><th>核心贡献</th><th>适用场景</th></tr></thead>
          <tbody>
            <tr><td><b>Wide&amp;Deep</b></td><td>2016</td><td>"记忆"（LR） + "泛化"（DNN） 并联</td><td>Google Play 应用商店</td></tr>
            <tr><td><b>DeepFM</b></td><td>2017</td><td>DNN + FM 共享 embedding，端到端</td><td>华为应用市场，CTR</td></tr>
            <tr><td><b>DCN / DCN-V2</b></td><td>2017/2021</td><td>显式特征交叉（多阶多项式）</td><td>Google Cloud 推荐</td></tr>
            <tr><td><b>DIN</b> (Alibaba)</td><td>2018</td><td>🏆 局部 Attention：用户历史 vs 候选 item</td><td>阿里电商 CTR，广泛采用</td></tr>
            <tr><td><b>DIEN</b> (Alibaba)</td><td>2019</td><td>GRU + AUGRU 建模兴趣演化</td><td>阿里妈妈广告</td></tr>
            <tr><td><b>SASRec</b></td><td>2018</td><td>🏆 Self-Attention 替代 RNN 序列推荐</td><td>Amazon 商品序列</td></tr>
            <tr><td><b>BERT4Rec</b></td><td>2019</td><td>双向 Transformer + Masked LM 推荐</td><td>短视频/电商序列</td></tr>
            <tr><td><b>SIM</b> (Alibaba)</td><td>2020</td><td>🏆 超长序列 100k+ 点击历史</td><td>阿里 CVR</td></tr>
            <tr><td><b>DLRM</b> (Meta)</td><td>2019</td><td>Embedding + 交互 + MLP，工业落地标杆</td><td>Facebook Feed</td></tr>
            <tr><td><b>MMoE / PLE</b></td><td>2018/2020</td><td>🏆 多任务共享 + 私有专家</td><td>YouTube / 快手 / 抖音</td></tr>
            <tr><td><b>LLM4Rec</b></td><td>2023+</td><td>🆕 用 LLM 做 ranking/generation</td><td>冷启/零样本，Meta GR</td></tr>
            <tr><td><b>HSTU / GR</b> (Meta 2024)</td><td>2024</td><td>🆕 Generative Recommenders，万亿参数推荐</td><td>Meta 内部下一代</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 相关算法模块</h2>
        <div class="grid-3">
          ${["ml_logistic","nn_attention","nn_transformer","seq_encoder","graph_sage","mmoe","llm_foundation","ml_lightgbm","loss_multitask"].map(id => {
            const a = MCH.getById(id);
            if (!a) return "";
            const c = MCH.catColors[a.category];
            return `<a href="#/${a.route}" class="card block" style="text-decoration:none;color:inherit;border-left:3px solid ${c};">
              <div class="font-semibold text-sm" style="color:${c};">${a.icon} ${a.name}</div>
              <div class="text-xs text-slate-600 mt-1">${(a.tags || []).slice(0, 3).join(" · ")}</div>
            </a>`;
          }).join("")}
        </div>
      </div>

      <div class="section">
        <h2>6. 开源实现参考</h2>
        <table class="table">
          <thead><tr><th>框架</th><th>特点</th><th>链接</th></tr></thead>
          <tbody>
            <tr><td><b>DeepCTR</b></td><td>TensorFlow/Keras 实现 20+ CTR 模型</td><td><a href="https://github.com/shenweichen/DeepCTR" target="_blank">GitHub</a></td></tr>
            <tr><td><b>TorchRec</b> (Meta)</td><td>PyTorch 工业级推荐库</td><td><a href="https://github.com/pytorch/torchrec" target="_blank">GitHub</a></td></tr>
            <tr><td><b>DGL-KE</b> / <b>PyG</b></td><td>图神经网络推荐</td><td><a href="https://pytorch-geometric.readthedocs.io/" target="_blank">PyG Docs</a></td></tr>
            <tr><td><b>RecBole</b></td><td>中科院，70+ 算法统一框架</td><td><a href="https://github.com/RUCAIBox/RecBole" target="_blank">GitHub</a></td></tr>
            <tr><td><b>FuxiCTR</b></td><td>华为诺亚，100+ CTR 模型</td><td><a href="https://github.com/reczoo/FuxiCTR" target="_blank">GitHub</a></td></tr>
            <tr><td><b>Merlin</b> (NVIDIA)</td><td>GPU 加速推荐全流水线</td><td><a href="https://developer.nvidia.com/merlin" target="_blank">NVIDIA</a></td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>7. 选型建议</h2>
        ${MCH.prosCons(
          ["ABTest 可度量、业务价值直接", "数据充足 + 反馈明确", "技术栈成熟（Wide&Deep/DIN 等经过工业验证）"],
          ["冷启动问题几乎无法彻底解决", "多目标工程复杂（需要 PLE/GradNorm）", "在线工程/AB 平台要求高"],
          ["电商 / 短视频 / 信息流 / 新闻 / 音乐推荐", "千万级 item 池 + 亿级 DAU", "用户行为数据充足"],
        )}
      </div>
    `;
  },
});
