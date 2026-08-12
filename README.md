# RAG + Agent Scaffold (LangGraph)

一个开箱即用的 **检索增强生成（RAG）+ 工具调用 Agent** 起步模板，基于 [LangGraph](https://github.com/langchain-ai/langgraph) 编排。

> 适用：内部文档问答、需要结合私有资料 + 外部工具的多步任务、作为更复杂多智能体系统的检索子模块。

## 特性

- **RAG**：`data/` 下的 `.txt` 文档 → 切分 → 向量库（`retrieve` 工具按需检索）
  - 默认 **FAISS** 本地向量库（零外部服务，开箱即跑）
  - 可一键切换 **SQL + Milvus**：向量存 Milvus、文本/元数据存 SQL，适合大规模生产检索
- **Agent**：LangGraph ReAct 循环，LLM 自主决定调用 `retrieve` / `calculator` 等工具
- **可编排**：`StateGraph` 显式串联 `agent` / `tools` 节点，便于扩展新节点
- **多模型**：基于 OpenAI 兼容接口，一行切换 DeepSeek 等国产模型
- **双入口**：CLI（`python -m rag_agent.cli`）+ FastAPI 服务（`/api/chat`）

## 目录结构

```
rag-agent-scaffold/
├── rag_agent/
│   ├── config.py        # 配置（LLM/嵌入/路径）
│   ├── llm.py           # get_chat_model() / get_embeddings()
│   ├── retriever.py     # 向量库抽象：build_index()/get_retriever() 按 VECTOR_BACKEND 切换（FAISS / SQL+Milvus）
│   ├── tools.py         # retrieve + calculator 工具（可扩展）
│   ├── graph.py         # LangGraph 状态图（agent ↔ tools 循环）
│   └── cli.py           # 命令行入口
├── app/main.py          # FastAPI 服务
├── data/sample.txt      # 示例知识库文档
├── tests/test_graph.py  # 图编译校验（无需 API key）
├── scripts/build_index.py
├── requirements.txt
├── .env.example
├── Dockerfile
└── README.md
```

## 快速开始

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # 填入 OPENAI_API_KEY（及可选的 OPENAI_BASE_URL）
python -m rag_agent.cli build-index   # 构建本地向量库
python -m rag_agent.cli run "这个脚手架支持哪些工具？"
```

启动 API 服务：

```bash
uvicorn app.main:app --reload --port 8000
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"如何扩展新的工具？"}'
```

## 它是怎么工作的

```
用户问题
   │
   ▼
[agent]  LLM 读取历史消息与已绑定工具，决定：直接回答 或 调用工具
   │
   ├─(无工具调用)──► END（返回最终回答）
   │
   └─(有工具调用)──► [tools] 执行 retrieve / calculator 等
                         │
                         └─► 回到 [agent]（带着工具结果继续思考）
```

`retrieve` 工具把私有知识注入上下文（RAG），`calculator` 演示通用工具调用；二者都是普通 LangChain `@tool`，可自由增删。

## 如何扩展

- **加工具**：在 `rag_agent/tools.py` 用 `@tool` 定义新函数，并加入 `TOOLS` 列表即可，无需改图。
- **换模型**：改 `.env` 的 `LLM_MODEL` / `EMBEDDING_MODEL`；接 DeepSeek 等只需把 `OPENAI_BASE_URL` 指向其兼容地址。
- **换向量库**：把 `retriever.py` 里的 `FAISS` 换成 Chroma / pgvector 等，接口保持一致即可。

## 切换到 SQL + Milvus（生产化向量检索）

默认后端是 FAISS（本地、零依赖）。当数据量变大、需要独立扩展检索服务时，可切换为 **SQL + Milvus** 组合：

- **Milvus** 存向量（支持 Milvus Lite 本地文件 `./milvus.db`，或独立服务 `http://host:19530`）
- **SQL** 存文本与元数据（MySQL / PostgreSQL / SQLite 均可，通过 SQLAlchemy）
- 检索时：Milvus 返回最相近的 `id` → 按 `id` 从 SQL 取回文本片段

两种后端共用同一套 `build_index()` / `get_retriever()` 接口，业务代码（graph / tools / cli）无需改动。

### 配置（.env）

```bash
VECTOR_BACKEND=milvus
MILVUS_URI=./milvus.db            # 或 http://localhost:19530
MILVUS_COLLECTION=rag_chunks
SQL_DOCSTORE_ENABLED=true
SQL_DOCSTORE_DSN=mysql+pymysql://user:pass@host:3306/rag   # 留空用本地 sqlite
SQL_DOCSTORE_TABLE=rag_chunks
```

### 安装与构建

```bash
pip install sqlalchemy pymilvus        # 默认 requirements.txt 已包含，但可按需精简
python -m rag_agent.cli build-index    # 按 VECTOR_BACKEND 构建（向量→Milvus，文本→SQL）
python -m rag_agent.cli run "这个脚手架支持哪些工具？"
```

> 注意：`VECTOR_BACKEND=milvus` **必须与** `SQL_DOCSTORE_ENABLED=true` 配合使用——Milvus 只存向量，文本由 SQL 提供，二者按 `id` 关联。若只设 milvus 不启用 SQL，`build_index()` / `get_retriever()` 会直接报错提示。

开箱即用：根目录 `.env.example` 仅含默认（FAISS）配置；`configs/.env.dual-backend.example` 把 **FAISS 与 SQL + Milvus 两种后端** 写在同一份文件里（各为一段完整可复制配置），复制为 `.env` 后按需保留一段即可。
- **加节点**：在 `graph.py` 的 `StateGraph` 中 `add_node` / `add_edge`，例如插入一个「重写查询」或「答案校验」节点。

## 测试

```bash
pytest -q
```

`tests/test_graph.py` 会真实构建并编译 LangGraph，**不需要任何 API key**，用于校验图结构正确。

## 生产化建议

- 将 API key 放入密钥管理，不要提交 `.env`
- FAISS 适合中小规模；大规模检索可切 `VECTOR_BACKEND=milvus`，配合 SQL 文档库（见上「切换到 SQL + Milvus」）
- 给 LLM 加超时与重试；对工具输入做校验
- CI 见 `.github/workflows/ci.yml`（推送需 PAT 带 `workflow` 权限）

## License

MIT
