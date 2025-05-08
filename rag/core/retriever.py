
from src import config
from src.models.reranker_model import RerankerWrapper
from src.utils.logger import LogManager
from src.models import select_model
from rag.core.prompts import *
from agent.kg_agent import KGQueryAgent # 知识图谱
from src.stores import  KnowledgeBase
from rag.core.operators import HyDEOperator
knowledge_base = KnowledgeBase()

_log = LogManager()


class Retriever:

    def __init__(self):
        self._load_models()
        self.kg_agent = KGQueryAgent()
        self.default_distance_threshold = config.get("default_distance_threshold", 0.5)
        self.top_k = config.get("default_top_k", 10)

    def _load_models(self):
        if config.enable_reranker:
            self.reranker = RerankerWrapper(config)

        if config.enable_web_search:
            from api.websearch.websearcher import LiteBaseSearcher, TavilyBasicSearcher
            self.web_searcher = LiteBaseSearcher()

    def retrieval(self, query, history, meta):
        refs = {"query": query, "history": history, "meta": meta}
        refs["model_name"] = config.model_name
        refs["entities"] = self.reco_entities(query, history, refs)
        refs["knowledge_base"] = self.query_knowledgebase(query, history, refs)
        refs["graph_base"] = self.query_graph(query, history, refs) #知识库
        refs["web_search"] = self.query_web(query, history, refs)

        return refs

    def restart(self):
        """所有需要重启的模型"""
        self._load_models()

    def construct_query(self, query, refs, meta):
        _log.debug(f"{refs=}")
        if not refs or len(refs) == 0:
            return query

        external_parts = []

        # 解析知识库的结果
        kb_res = refs.get("knowledge_base", {}).get("results", [])
        if kb_res:
            kb_text = "\n".join(f"{r['id']}: {r['entity']['text']}" for r in kb_res)
            external_parts.extend(["知识库信息:", kb_text])

        # 解析图数据库的结果
        db_res = refs.get("graph_base", {}).get("results", {})
        if db_res.get("nodes") and len(db_res["nodes"]) > 0:
            db_text = "\n".join(
                [f"{edge['source_name']}和{edge['target_name']}的关系是{edge['type']}" for edge in
                 db_res.get("edges", [])]
            )
            external_parts.extend(["图数据库信息:", db_text])

        # 解析网络搜索的结果
        web_res = refs.get("web_search", {}).get("results", [])
        if web_res:
            web_text = "\n".join(f"{r['title']}: {r['content']}" for r in web_res)
            external_parts.extend(["网络搜索信息:", web_text])

        # 构造查询
        if external_parts and len(external_parts) > 0:
            external = "\n\n".join(external_parts)
            query = knowbase_qa_template.format(external=external, query=query)

        return query

    def query_classification(self, query):
        """判断是否需要查询
        - 对于完全基于用户给定信息的任务，称之为"足够""sufficient"，不需要检索；
        - 否则，称之为"不足""insufficient"，可能需要检索，
        """
        raise NotImplementedError

    def query_graph(self, query, history, refs):
        if refs["meta"].get("use_graph"):
            # 调用 KGQueryAgent.query，传hops参数
            result = self.kg_agent.query(
                query,
                hops=refs["meta"].get("graphHops", 2)
            )
            # result = {"answer": "...", "subgraph": {nodes,edges}}
            return result
        return {"answer": None, "subgraph": None}

    def query_knowledgebase(self, query, history, refs):
        response = {
            "results": [],
            "all_results": [],
            "rw_query": query,
            "message": "",
        }

        meta = refs["meta"]
        db_id = meta.get("db_id")
        if not db_id or not config.enable_knowledge_base:
            response["message"] = "知识库未启用、或未指定知识库、或知识库不存在"
            return response

        # 重写查询（如果有）
        rw_query = self.rewrite_query(query, history, refs)
        response["rw_query"] = rw_query

        try:
            kb_res = knowledge_base.search(
                query=rw_query,
                db_id=db_id,
                distance_threshold=meta.get("distanceThreshold", self.default_distance_threshold),
                rerank=True,
                top_k=meta.get("topK", self.top_k)
            )
            response["results"] = kb_res["results"]
            response["all_results"] = kb_res["all_results"]
        except Exception as e:
            response["message"] = f"检索出错: {e}"
        return response

    def query_web(self, query, history, refs):
        """查询网络"""

        if not (refs["meta"].get("use_web") and config.enable_web_search):
            return {"results": [], "message": "Web search is disabled"}

        try:
            search_results = self.web_searcher.search(query, top_k=5)
        except Exception as e:
            _log.error(f"Web search error: {str(e)}")
            return {"results": [], "message": "Web search error"}

        return {"results": search_results}

    def rewrite_query(self, query, history, refs):
        """重写查询"""
        model_provider = config.model_provider
        model_name = config.model_name
        model = select_model(config, model_provider=model_provider, model_name=model_name)
        if refs["meta"].get("mode") == "search":  # 如果是搜索模式，就使用 meta 的配置，否则就使用全局的配置
            rewrite_query_span = refs["meta"].get("use_rewrite_query", "off")
        else:
            rewrite_query_span = config.use_rewrite_query

        if rewrite_query_span == "off":
            rewritten_query = query
        else:

            history_query = [entry["content"] for entry in history if entry["role"] == "user"] if history else ""
            rewritten_query_prompt = rewritten_query_prompt_template.format(history=history_query, query=query)
            rewritten_query = model.predict(rewritten_query_prompt).content

        if rewrite_query_span == "hyde":
            res = HyDEOperator.call(model_callable=model.predict, query=query, context_str=history_query)
            rewritten_query = res.content

        return rewritten_query

    def reco_entities(self, query, history, refs):
        """识别句子中的实体"""
        query = refs.get("rewritten_query", query)
        model_provider = config.model_provider
        model_name = config.model_name
        model = select_model(config, model_provider=model_provider, model_name=model_name)

        entities = []
        if refs["meta"].get("use_graph"):
            entity_extraction_prompt = keywords_prompt_template.format(text=query)
            entities = model.predict(entity_extraction_prompt).content.split("<->")
            # entities = [entity for entity in entities if all(char.isalnum() or char in "汉字" for char in entity)]

        return entities

    def _extract_relationship_info(self, relationship, source_name=None, target_name=None, node_dict=None):
        """
        提取关系信息并返回格式化的节点和边信息
        """
        rel_id = relationship.element_id
        nodes = relationship.nodes
        if len(nodes) != 2:
            return None, None

        source, target = nodes
        source_id = source.element_id
        target_id = target.element_id

        source_name = node_dict[source_id]["name"] if source_name is None else source_name
        target_name = node_dict[target_id]["name"] if target_name is None else target_name

        relationship_type = relationship._properties.get("type", "unknown")
        if relationship_type == "unknown":
            relationship_type = relationship.type

        edge_info = {
            "id": rel_id,
            "type": relationship_type,
            "source_id": source_id,
            "target_id": target_id,
            "source_name": source_name,
            "target_name": target_name,
        }

        node_info = [
            {"id": source_id, "name": source_name},
            {"id": target_id, "name": target_name},
        ]

        return node_info, edge_info

    def format_general_results(self, results):
        formatted_results = {"nodes": [], "edges": []}

        for item in results:
            relationship = item[1]
            source_name = item[0]._properties.get("name", "unknown")
            target_name = item[2]._properties.get("name", "unknown") if len(item) > 2 else "unknown"

            node_info, edge_info = self._extract_relationship_info(relationship, source_name, target_name)
            if node_info is None or edge_info is None:
                continue

            for node in node_info:
                if node["id"] not in [n["id"] for n in formatted_results["nodes"]]:
                    formatted_results["nodes"].append(node)

            formatted_results["edges"].append(edge_info)

        return formatted_results

    def format_query_results(self, results):
        _log.debug(f"Graph Query Results: {results}")
        formatted_results = {"nodes": [], "edges": []}
        node_dict = {}

        for item in results:
            # 检查数据格式
            if len(item) < 2 or not isinstance(item[1], list):
                continue

            node_dict[item[0].element_id] = dict(id=item[0].element_id, name=item[0]._properties.get("name", "Unknown"))
            node_dict[item[2].element_id] = dict(id=item[2].element_id, name=item[2]._properties.get("name", "Unknown"))

            # 处理关系列表中的每个关系
            for i, relationship in enumerate(item[1]):
                try:
                    # 提取关系信息
                    node_info, edge_info = self._extract_relationship_info(relationship, node_dict=node_dict)
                    if node_info is None or edge_info is None:
                        continue

                    # 添加边
                    formatted_results["edges"].append(edge_info)
                except Exception as e:
                    _log.error(f"处理关系时出错: {e}, 关系: {relationship}")
                    continue

        # 将节点字典转换为列表
        formatted_results["nodes"] = list(node_dict.values())

        return formatted_results

    def __call__(self, query, history, meta):
        refs = self.retrieval(query, history, meta)
        query = self.construct_query(query, refs, meta)
        return query, refs
