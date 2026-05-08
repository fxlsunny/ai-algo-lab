/* 模块：RAG 检索增强生成 */
MCH.register("llm_rag", {
  render() {
    const code = `# ========================================================================
# RAG (Retrieval-Augmented Generation) 典型流水线
# 核心思想：把外部知识"检索"出来，塞给 LLM 作为上下文，再让它生成答案
# ========================================================================

from sentence_transformers import SentenceTransformer
import faiss, numpy as np

# ─── 1. 离线索引阶段（Offline Indexing）─────────────────────────
class RAGIndex:
    def __init__(self, embed_model="BAAI/bge-large-zh-v1.5"):
        self.encoder = SentenceTransformer(embed_model)
        self.index = None
        self.chunks = []

    def add_docs(self, docs, chunk_size=500, overlap=50):
        """① 分块 chunk → ② 向量化 embed → ③ 存向量库"""
        for doc in docs:
            chunks = self._split(doc.text, chunk_size, overlap)
            for c in chunks:
                self.chunks.append({"text": c, "meta": doc.meta})

        vecs = self.encoder.encode([c["text"] for c in self.chunks], normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vecs.shape[1])  # 余弦相似 = 点积
        self.index.add(vecs.astype("float32"))

    def _split(self, text, size, overlap):
        # 按段落/语义切块，避免把一句话劈成两半
        sents = text.split("。")
        chunks, buf = [], ""
        for s in sents:
            if len(buf) + len(s) > size:
                chunks.append(buf); buf = buf[-overlap:] + s
            else:
                buf += s + "。"
        if buf: chunks.append(buf)
        return chunks

# ─── 2. 在线检索阶段（Online Retrieval）─────────────────────────
class RAGRetriever:
    def search(self, query, top_k=5):
        qv = self.encoder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(qv, top_k)
        return [(self.chunks[i], s) for i, s in zip(ids[0], scores[0])]

# ─── 3. 混合检索（BM25 + Dense）—— 工业级最稳───────────────────
from rank_bm25 import BM25Okapi
class HybridRetriever:
    def search(self, query, top_k=5, alpha=0.5):
        # 稠密向量检索
        dense_hits = self.dense.search(query, top_k=top_k*3)
        # 稀疏关键词检索
        sparse_hits = self.bm25.get_top_n(query.split(), self.chunks, n=top_k*3)
        # RRF 融合（Reciprocal Rank Fusion）
        scores = defaultdict(float)
        for rank, (c, _) in enumerate(dense_hits):
            scores[c["id"]] += alpha / (rank + 60)
        for rank, c in enumerate(sparse_hits):
            scores[c["id"]] += (1 - alpha) / (rank + 60)
        return sorted(scores.items(), key=lambda x: -x[1])[:top_k]

# ─── 4. Rerank 重排（精排阶段）─────────────────────────────────
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("BAAI/bge-reranker-large")
def rerank(query, candidates, top_n=3):
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)  # 直接对 (q, c) 打分，比 embedding 更准
    ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
    return [c for c, _ in ranked[:top_n]]

# ─── 5. 生成阶段 ─────────────────────────────────────────────
PROMPT = """你是专业问答助手。请严格依据下列"参考资料"回答用户问题。
如果资料中没有答案，直接说"未在知识库中找到"，不要编造。

【参考资料】
{context}

【用户问题】{question}

【回答】"""

def rag_answer(question):
    hits = retriever.search(question, top_k=10)
    hits = rerank(question, [h[0] for h in hits], top_n=3)
    context = "\\n\\n".join([f"[{i+1}] {h['text']}" for i, h in enumerate(hits)])
    return llm(PROMPT.format(context=context, question=question))`;

    const advanced = `# ========================================================================
# 高级 RAG 模式（2024 工业界主流）
# ========================================================================

# ① Query Rewriting —— 把用户口语化问题改写为更好检索的形式
def rewrite_query(user_q):
    prompt = f"请把下面问题改写成 3 个更适合检索的查询（去除代词、补充上下文）：\\n{user_q}"
    return llm(prompt).split("\\n")  # 多查询并发检索

# ② HyDE (Hypothetical Document Embeddings)
#    让 LLM 先"假装"回答问题，再用假答案去检索（通常比原问题更能匹配文档）
def hyde_search(question):
    hypothetical = llm(f"假设用一段话回答这个问题：\\n{question}")
    return retriever.search(hypothetical, top_k=5)

# ③ Self-RAG —— LLM 主动判断"我需要检索吗？检索到的有用吗？"
#    Asai et al. 2023 · 在生成过程中穿插 [Retrieve] / [Relevant] / [Supported] 等特殊 token
def self_rag_step(q, ctx=""):
    need = llm(f"{q}\\n是否需要额外知识？[YES/NO]")
    if "YES" in need:
        hits = retriever.search(q)
        relevant = [h for h in hits if llm(f"与问题相关吗？{h} / {q}") == "YES"]
        ctx += "\\n".join(relevant)
    return llm(f"{q}\\n上下文：{ctx}")

# ④ Contextual Chunks —— 把"文档上下文"嵌入每个块（Anthropic 2024）
#    每个 chunk 除了自身还带上 "这个 chunk 属于 X 文档的 Y 章节"
def contextualize(chunk, doc):
    ctx = llm(f"用一句话说明这段话在原文档中的位置/背景：\\n文档标题：{doc.title}\\n原文：{chunk}")
    return f"[{ctx}]\\n{chunk}"

# ⑤ GraphRAG (Microsoft 2024) —— 先从语料建知识图谱，检索时走图
#    适合：全局洞察类问题（"总结全部小说角色关系"）
#    实现：LLM 抽取 entity+relation → 构图 → Leiden 社区发现 → 社区级摘要

# ========================================================================
# 工程 Checklist（部署前对照）
# ========================================================================
# □ Embedding 模型选型：bge-large-zh / m3e / text-embedding-3-large
# □ 向量库：Milvus (1E+) · Qdrant (稳) · pgvector (简单) · FAISS (单机)
# □ Chunk Size：500-1000 token（中文按字），Overlap 10-15%
# □ Top-K：初筛 20-50，Rerank 到 3-5
# □ Metadata 过滤：时间/部门/权限（必须！否则跨部门数据泄露）
# □ 评估：RAGAS / TruLens —— Context Precision/Recall + Answer Faithfulness
# □ 监控：query 分布漂移、hit rate、人工标注样本定期回归`;

    return `
      ${MCH.hero({
        icon: "🔍",
        name: "RAG 检索增强生成",
        en: "Retrieval-Augmented Generation",
        tags: ["Embedding", "Dense/Sparse", "Rerank", "Hybrid Search", "HyDE", "Self-RAG", "GraphRAG"],
        meta: ["◈ Lewis et al. 2020 原始论文 · Microsoft GraphRAG 2024", "⚡ 大模型幻觉的终极解药"],
      })}

      ${MCH.versionSection ? MCH.versionSection("llm_rag") : ""}

      <div class="section">
        <h2>1. 为什么需要 RAG？</h2>
        <div class="grid-2">
          <div class="card">
            <h3>🚫 纯 LLM 的三大痛点</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li><b>知识截止</b>：GPT-4 不知道昨天的新闻</li>
              <li><b>幻觉</b>：编造"听起来对"的假事实</li>
              <li><b>领域盲</b>：公司内部数据 LLM 没见过</li>
              <li><b>无法追溯</b>：无法给出参考依据</li>
            </ul>
          </div>
          <div class="card">
            <h3>✅ RAG 的优势</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li><b>实时更新</b>：文档入库即可查询</li>
              <li><b>可溯源</b>：引用哪些文档一目了然</li>
              <li><b>隐私可控</b>：企业数据不出库</li>
              <li><b>成本低</b>：不用 fine-tune 大模型</li>
            </ul>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. 标准 RAG 流水线</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>3. 高级 RAG 模式</h2>
        ${MCH.code(advanced, "python")}
      </div>

      <div class="section">
        <h2>4. 向量库选型对比</h2>
        <table class="table">
          <thead><tr><th>产品</th><th>规模</th><th>部署</th><th>适合</th><th>价格</th></tr></thead>
          <tbody>
            <tr><td><b>FAISS</b></td><td>&lt;100W</td><td>单机</td><td>研究原型</td><td>免费</td></tr>
            <tr><td><b>pgvector</b></td><td>&lt;1000W</td><td>PostgreSQL 插件</td><td>已有 PG 的业务</td><td>免费</td></tr>
            <tr><td><b>Qdrant</b></td><td>1亿+</td><td>Docker/Cloud</td><td>中小团队生产</td><td>Cloud 起价 ¥150/月</td></tr>
            <tr><td><b>Milvus</b></td><td>10亿+</td><td>K8s 集群</td><td>大规模生产</td><td>免费（自建）</td></tr>
            <tr><td><b>Weaviate</b></td><td>1亿+</td><td>Cloud 友好</td><td>多模态</td><td>Cloud 起价 ¥210/月</td></tr>
            <tr><td><b>Pinecone</b></td><td>1亿+</td><td>纯托管</td><td>海外业务</td><td>¥420+/月</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>5. 评估指标（RAGAS）</h2>
        <div class="grid-2">
          <div class="card">
            <h3>🎯 检索质量</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li><b>Context Recall</b>：召回的文档是否覆盖了答案所需知识</li>
              <li><b>Context Precision</b>：召回中相关文档的比例</li>
              <li><b>Hit Rate@K</b>：TopK 中是否有正确文档</li>
            </ul>
          </div>
          <div class="card">
            <h3>✍ 生成质量</h3>
            <ul style="line-height:1.7;padding-left:18px">
              <li><b>Faithfulness</b>：回答是否忠实于检索上下文（不幻觉）</li>
              <li><b>Answer Relevancy</b>：答案是否切题</li>
              <li><b>Answer Correctness</b>：与 ground truth 比对（有标签时）</li>
            </ul>
          </div>
        </div>
      </div>
    `;
  },
  mount() { MCH.rerender && MCH.rerender(); },
});
