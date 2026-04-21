# BioRAG 知识库检索系统

生信课题组的 RAG 知识服务，Skill 驱动 + 渐进式上下文注入 + 本地 Embedding。

## 快速开始

### 1. 安装依赖

```bash
cd A:\claudeworks\BioRAG
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 首次构建向量库

```bash
python scripts/build_index.py
```

首次构建需要 15-30 分钟（加载 Embedding 模型 + 编码所有文档）。

### 3. 启动 Engine 服务

```bash
python -m biorag.engine
# 或双击 start.bat
```

服务默认运行在 http://127.0.0.1:8765

### 4. 安装 Skill 到 Agent

```bash
# WorkBuddy / Claude Code
cp -r skill/ ~/.workbuddy/skills/biorag/

# OpenClaw
cp -r skill/ ~/.openclaw/skills/biorag/
```

### 5. 验证检索

```bash
python skill/scripts/search.py "DESeq2差异表达" --level 1
```

## 增量更新

每 3-5 天手动运行：

```bash
python skill/scripts/update.py
```

## 迁移

1. 复制整个 BioRAG/ 文件夹
2. 编辑 config.yaml，修改 `knowledge_base.path`
3. 重建虚拟环境并安装依赖
4. 运行 `python scripts/migrate.py --verify`
5. 启动服务

## 配置

编辑 `config.yaml`，主要配置项：

- `knowledge_base.path`: 知识库路径（迁移时改这里）
- `embedding.model`: Embedding 模型（bge-small-zh-v1.5 / bge-large-zh-v1.5）
- `embedding.device`: cpu / cuda
- `categories`: 文件夹 → 分类映射

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/search` | POST | 语义检索（Level 1/2/3） |
| `/search_kw` | POST | 关键词检索 |
| `/status` | GET | 知识库状态 |
| `/update` | POST | 增量更新 |
| `/stats/recent` | GET | 近期检索统计 |
| `/stats/hot` | GET | 热门文件 |
| `/vectors/umap` | GET | UMAP 坐标 |
| `/health` | GET | 健康检查 |

## 项目结构

```
BioRAG/
├── biorag/          # 核心引擎（config/loader/chunker/embedder/vectorstore/metadata/engine）
├── skill/           # Skill 文件（SKILL.md + scripts/）
├── scripts/         # 工具脚本（build_index.py）
├── data/            # 运行时数据（chroma_db/metadata.db/search_logs.db）
├── config.yaml      # 用户配置
└── start.bat        # Windows 启动
```

## 渐进式检索

| Level | 片段数 | 每块字数 | 总字数 | 适用场景 |
|-------|--------|---------|--------|----------|
| 1 | 3 | 200 | ~600 | 大多数问题 |
| 2 | 8 | 300 | ~2500 | 需要更多上下文 |
| 3 | 5 | 全文 | 不限 | 需要完整文档 |