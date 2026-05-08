/* 模块：前沿 RAG */
MCH.register("frontier_rag", {
  render() {
    const code = `# 经典 RAG 流水线（Langchain / LlamaIndex 风格）
from llama_index.core import VectorStoreIndex, Document

# ① 文档分块（chunking）
docs = [Document(text=txt) for txt in docs_raw]
# chunk_size 与任务强相关：
# - 问答短文：256-512 tokens
# - 法律/医疗长文：1024-2048 tokens
# - 代码：按 function 边界分

# ② 向量化 + 索引
index = VectorStoreIndex.from_documents(docs, embed_model="text-embedding-3-large")

# ③ 检索 + 生成
query_engine = index.as_query_engine(similarity_top_k=5, llm="gpt-4o")
response = query_engine.query("What is Mamba?")


# ============================================================
# Advanced RAG 改进（2024 业界标准）
# ============================================================
# 1. Query Rewriting / HyDE：把 query 改写成适合检索的形式
# 2. Hybrid Retrieval：BM25 + Dense Vector 融合
# 3. Rerank：bge-reranker / Cohere Rerank 重排 Top-50 → Top-5
# 4. Contextual Retrieval (Anthropic 2024)：chunk 加上下文摘要后再嵌入

def advanced_rag(query):
    # Step 1: query 改写（HyDE）
    hypo_doc = llm(f"Generate a passage that answers: {query}")
    # Step 2: 混合检索
    bm25_results = bm25.search(query, k=50)
    vec_results = vector_db.search(encode(hypo_doc), k=50)
    merged = rrf_fusion([bm25_results, vec_results])
    # Step 3: Rerank
    top_5 = rerank(query, merged[:50], model="BAAI/bge-reranker-v2-m3")[:5]
    # Step 4: 生成
    context = "\\n".join(top_5)
    answer = llm(f"Context:\\n{context}\\n\\nQuestion: {query}\\nAnswer:")
    return answer


# ============================================================
# GraphRAG (Microsoft 2024) — 知识图谱增强
# ============================================================
# 1) 抽取实体 + 关系构建知识图谱
# 2) 社区发现（Louvain）得到层次总结
# 3) 查询时结合 query 的 community 做回答
# 适合：需要"全局理解"的问题（如"总结 X 主题"）


# ============================================================
# Agentic RAG (2024) — LLM 编排多步检索
# ============================================================
# Agent 可以：
#   - 判断是否需要检索
#   - 拆分复杂问题为子问题
#   - 迭代检索，链式推理
#   - 工具调用（SQL/API/Calculator）`;

    return `
      ${MCH.hero({
        icon: "🔍",
        name: "RAG 检索增强生成",
        en: "Retrieval-Augmented Generation (LLM 实际落地架构)",
        tags: ["RAG", "向量检索", "BM25", "GraphRAG", "Agentic RAG", "企业知识库"],
        meta: ["◈ LLM 落地事实标准", "⚡ 降低幻觉 + 可追溯"],
      })}

      ${MCH.versionSection("frontier_rag")}

      <div class="section">
        <h2>1. 为什么 RAG 是 LLM 落地第一选择？</h2>
        <p class="text-sm text-slate-600">纯 LLM 的三大痛点都被 RAG 解决：</p>
        <div class="grid-3">
          <div class="card" style="border-top:3px solid #d946ef;">
            <h3 class="font-bold text-fuchsia-700">🚫 知识截止</h3>
            <p class="text-xs text-slate-600 mt-2">LLM 训练数据到某个日期。<br/><b>RAG</b>：随时检索最新内部文档。</p>
          </div>
          <div class="card" style="border-top:3px solid #d946ef;">
            <h3 class="font-bold text-fuchsia-700">👻 幻觉</h3>
            <p class="text-xs text-slate-600 mt-2">LLM 会编造事实。<br/><b>RAG</b>：prompt 里带引用来源，可追溯可验证。</p>
          </div>
          <div class="card" style="border-top:3px solid #d946ef;">
            <h3 class="font-bold text-fuchsia-700">🔒 私有数据</h3>
            <p class="text-xs text-slate-600 mt-2">公司文档不能给 OpenAI 训练。<br/><b>RAG</b>：数据在自己的向量库里。</p>
          </div>
        </div>
      </div>

      <div class="section">
        <h2>2. RAG 演进三代</h2>
        <table class="table">
          <thead>
            <tr><th>代</th><th>方法</th><th>精度</th><th>复杂度</th></tr>
          </thead>
          <tbody>
            <tr>
              <td><b>1️⃣ Naive RAG</b> (2020-22)</td>
              <td>Chunk → Embedding → Top-k → 拼接 Prompt → LLM</td>
              <td>基线</td>
              <td>简单</td>
            </tr>
            <tr>
              <td><b>2️⃣ Advanced RAG</b> (2023)</td>
              <td>+ Query Rewriting (HyDE)<br/>+ Hybrid Retrieval (BM25 + Dense)<br/>+ Rerank (bge-reranker)<br/>+ Metadata Filter</td>
              <td>+5-20% 精度</td>
              <td>中等</td>
            </tr>
            <tr>
              <td><b>3️⃣ 🆕 Modular / Agentic RAG</b> (2024)</td>
              <td>+ GraphRAG (知识图谱)<br/>+ Self-RAG (自决定检索)<br/>+ CRAG (自纠错)<br/>+ Agentic (多步推理 + 工具)<br/>+ Contextual Retrieval (Anthropic)</td>
              <td>+10-30% 精度</td>
              <td>高</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>3. Advanced RAG 架构</h2>
        <div class="mermaid">
flowchart LR
    Q[用户 Query] --> QR[Query 改写<br/>HyDE 生成假答案]
    QR --> H[混合检索]
    H --> BM[BM25 词法]
    H --> VEC[Vector 语义]
    BM --> F[Fusion<br/>RRF 融合]
    VEC --> F
    F --> RR[Reranker<br/>bge-reranker]
    RR --> C[Top-5 Context]
    C --> L[LLM 生成]
    Q --> L
    L --> A[带引用的答案]
        </div>
      </div>

      <div class="section">
        <h2>4. 核心代码</h2>
        ${MCH.code(code, "python")}
      </div>

      <div class="section">
        <h2>5. RAG 关键组件选型</h2>
        <table class="table">
          <thead><tr><th>组件</th><th>主流选择</th></tr></thead>
          <tbody>
            <tr><td><b>Embedding 模型</b></td><td>OpenAI text-embedding-3-large; 🆕 BGE-M3 (2024); Jina v2; NV-Embed-v2 (MTEB 榜首)</td></tr>
            <tr><td><b>向量数据库</b></td><td>商用：Pinecone / Weaviate<br/>开源：Milvus / Chroma / Qdrant / FAISS</td></tr>
            <tr><td><b>词法检索</b></td><td>Elasticsearch / OpenSearch / Tantivy / BM25 (rank_bm25)</td></tr>
            <tr><td><b>Reranker</b></td><td>🆕 BAAI/bge-reranker-v2-m3 (开源 SOTA)<br/>Cohere Rerank API<br/>Jina Reranker v2</td></tr>
            <tr><td><b>LLM</b></td><td>GPT-4o / Claude 3.5 / Gemini / Qwen / LLaMA-3</td></tr>
            <tr><td><b>框架</b></td><td>🏆 LlamaIndex / LangChain / Haystack / DSPy</td></tr>
            <tr><td><b>🆕 评估</b></td><td>Ragas / TruLens / DeepEval</td></tr>
          </tbody>
        </table>
      </div>

      <div class="section">
        <h2>6. GraphRAG (Microsoft 2024) 亮点</h2>
        ${MCH.info(`
          <b>传统 RAG 的困境</b>：遇到"请总结 X 主题的整体趋势"这类<b>全局问题</b>，Top-k chunk 片段无法给出完整答案。
          <br/><br/>
          <b>GraphRAG 的解法</b>：
          <ol style="margin-top:6px;padding-left:20px;">
            <li>LLM 抽取 <b>实体 + 关系</b>构建知识图谱</li>
            <li>对图谱做 <b>Louvain 社区发现</b>，形成层次结构</li>
            <li>对每个社区用 LLM <b>摘要</b></li>
            <li>查询时：全局问题 → 用社区摘要；具体问题 → 用 chunk</li>
          </ol>
          <b>实测</b>：在金融报告、法律文档等长文理解上显著超越 Naive RAG。
        `, "tip")}
      </div>

      <div class="section">
        <h2>7. 开源资源</h2>
        <ul class="text-sm text-slate-700 list-disc pl-6">
          <li><a href="https://www.llamaindex.ai/" target="_blank">🏆 LlamaIndex</a> — RAG 专业框架</li>
          <li><a href="https://python.langchain.com/" target="_blank">LangChain</a> — LLM 应用最流行框架</li>
          <li><a href="https://github.com/microsoft/graphrag" target="_blank">🆕 Microsoft GraphRAG</a></li>
          <li><a href="https://github.com/infiniflow/ragflow" target="_blank">RAGFlow</a> — 企业级 RAG 开源</li>
          <li><a href="https://github.com/explodinggradients/ragas" target="_blank">Ragas</a> — RAG 评估</li>
          <li><a href="https://www.anthropic.com/news/contextual-retrieval" target="_blank">Anthropic Contextual Retrieval</a></li>
          <li><a href="https://github.com/stanfordnlp/dspy" target="_blank">DSPy</a> — 提示词 + 模块自动化</li>
        </ul>
      </div>

      <div class="section">
        <h2>8. 优缺点与场景</h2>
        ${MCH.prosCons(MCH.getById("frontier_rag").pros, MCH.getById("frontier_rag").cons, MCH.getById("frontier_rag").best_for)}
      </div>
    `;
  },
});
