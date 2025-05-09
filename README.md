# Meet-Pokémon - 基于大模型的知识库与知识图谱问答系统

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat&logo=python)
![Neo4j](https://img.shields.io/badge/Neo4j-5.0-blue?style=flat&logo=neo4j)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=ffffff)

---

## 📝 项目概述

**Meet-Pokémon** 是一个融合了 **大语言模型 (LLM)** 与 **知识图谱 (KG)**
的问答系统，灵感源于对宝可梦世界的深度挖掘，支持对宝可梦背景、属性、剧情等进行智能问答。项目采用多种技术组件构建，包括：

- **Neo4j** 知识图谱：快速存储与检索实体、关系
- **大模型**：支持本地模型/在线模型，用于对话和意图识别
- **自动化实体抽取**：结合 `roberta + TF-IDF + 规则匹配`
- **RAG Flow**：集成多模态文档解析（PDF、PPT、Excel、OCR 等）
- **API 集成**：可快速嵌入 [Dify](https://dify.ai/) 流程中
- **Agent 调度支持**：支持自定义任务链

---

## 🎯 核心功能

1. **知识图谱构建与问答**
    - 使用 `create_pokemon_json_data.py` 将实体导入 Neo4j
    - 使用 `chatbot_graph` 实现基于图谱的语义问答

2. **Dify 工作流对接**
    - 通过 `api.py` 提供统一问答接口，可直接集成 Dify 流程

3. **自动化抽取脚本**
    - 基于 `roberta + TF-IDF + 规则匹配` 实现实体与关系提取

4. **本地大模型意图识别**
    - 使用 ChatGLM、Qwen 等本地模型进行意图解析

5. **联网搜索与 OCR 插件**
    - 支持网页搜索与文档 OCR，提升问答信息丰富度

6. **RAGflow 整合**
    - 文档格式解析 + 分块 + 向量化 + 多模态问答

---

## 🚀 快速开始

以下为从图谱构建到问答服务启动的完整流程：

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 创建 Neo4j 图数据库

- 本地安装 Neo4j Desktop 或使用 Docker 方式：

  ```bash
  docker run -it \
    --name neo4j \
    -p 7474:7474 -p 7687:7687 \
    -e NEO4J_AUTH=neo4j/123456 \
    neo4j:5.3
  ```

- 确保用户名密码与 `KGsql` 里配置一致 (`neo4j/123456`)

### 3. 导入宝可梦实体与关系

在 `script/` 目录下，执行 Python 脚本：

```bash
python create_pokemon_json_data.py
```

脚本会读取 `data/` 下的实体、关系文件，并将其转换为 JSON 格式后写入 Neo4j。
如果数据库连接异常，可检查 `scr/kgsql/KGsql.py`（或对应配置）中的 Neo4j 地址、账号、密码是否正确。

### 4. 启动问答服务

以 `scr/server/nlp_pokemon_kg.py` 或者 `scr/qa/qagent.py` 为例：

```python
python nlp_pokemon_kg.py
```

如果有 **FastAPI** 或 **Flask** 接口服务，可以通过 `uvicorn` 启动：

```bash
uvicorn nlp_pokemon_kg:app --host 0.0.0.0 --port 8000
```

访问 `http://localhost:8000/docs` 即可看到 API 文档。

## 📃 License

本项目遵循 MIT License 协议，可免费用于商业或个人项目，二次开发需注明原作者与来源。