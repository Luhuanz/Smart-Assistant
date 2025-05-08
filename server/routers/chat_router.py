import os
import json
import asyncio
import traceback
import uuid
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk

from src import executor, config, retriever
from rag.core import HistoryManager
from src.qa import chat_agent
from src.models import select_model
from src.utils.logger import LogManager
logger=LogManager()
from src.qa import PokemonKGChatAgent

kg_chat_agent = PokemonKGChatAgent()
chat = APIRouter(prefix="/chat")

@chat.get("/")
async def chat_get():
    return "Chat Get!"

@chat.post("/")
async def chat_post(
        query: str = Body(...),
        meta: dict = Body(None),
        history: list[dict] | None = Body(None),
        thread_id: str | None = Body(None)):
    """处理聊天请求的主要端点。
    Args:
        query: 用户的输入查询文本
        meta: 包含请求元数据的字典，可以包含以下字段：
            - use_web: 是否使用网络搜索
            - use_graph: 是否使用知识图谱
            - db_id: 数据库ID
            - history_round: 历史对话轮数限制
            - system_prompt: 系统提示词（str，不含变量）
        history: 对话历史记录列表
        thread_id: 对话线程ID
    Returns:
        StreamingResponse: 返回一个流式响应，包含以下状态：
            - searching: 正在搜索知识库
            - generating: 正在生成回答
            - reasoning: 正在推理
            - loading: 正在加载回答
            - finished: 回答完成
            - error: 发生错误
    Raises:
        HTTPException: 当检索器或模型发生错误时抛出
    """

    model = select_model()
    meta["server_model_name"] = model.model_name
    history_manager = HistoryManager(history, system_prompt=meta.get("system_prompt"))
    logger.debug(f"Received query: {query} with meta: {meta}")

    def make_chunk(content=None, **kwargs):
        return json.dumps({
            "response": content,
            "meta": meta,
            **kwargs
        }, ensure_ascii=False).encode('utf-8') + b"\n"

    def need_retrieve(meta):
        return meta.get("use_web") or meta.get("use_graph") or meta.get("db_id")

    def generate_response():
        modified_query = query
        refs = None

        # 处理知识库检索
        if meta and need_retrieve(meta):
            chunk = make_chunk(status="searching")
            yield chunk

            try:
                modified_query, refs = retriever(modified_query, history_manager.messages, meta)
            except Exception as e:
                logger.error(f"Retriever error: {e}, {traceback.format_exc()}")
                yield make_chunk(message=f"Retriever error: {e}", status="error")
                return

            yield make_chunk(status="generating")

        messages = history_manager.get_history_with_msg(modified_query, max_rounds=meta.get('history_round'))
        history_manager.add_user(query)  # 注意这里使用原始查询

        content = ""
        reasoning_content = ""
        try:
            for delta in model.predict(messages, stream=True):
                if not delta.content and hasattr(delta, 'reasoning_content'):
                    reasoning_content += delta.reasoning_content or ""
                    chunk = make_chunk(reasoning_content=reasoning_content, status="reasoning")
                    yield chunk
                    continue

                # 文心一言
                if hasattr(delta, 'is_full') and delta.is_full:
                    content = delta.content
                else:
                    content += delta.content or ""

                chunk = make_chunk(content=delta.content, status="loading")
                yield chunk

            logger.debug(f"Final response: {content}")
            logger.debug(f"Final reasoning response: {reasoning_content}")
            yield make_chunk(status="finished",
                            history=history_manager.update_ai(content),
                            refs=refs)
        except Exception as e:
            logger.error(f"Model error: {e}, {traceback.format_exc()}")
            yield make_chunk(message=f"Model error: {e}", status="error")
            return

    return StreamingResponse(generate_response(), media_type='application/json')

@chat.post("/call")
async def call(query: str = Body(...), meta: dict = Body(None)):
    meta = meta or {}
    model = select_model(model_provider=meta.get("model_provider"), model_name=meta.get("model_name"))
    async def predict_async(query):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, model.predict, query)

    response = await predict_async(query)
    logger.debug({"query": query, "response": response.content})

    return {"response": response.content}

@chat.post("/call_lite")
async def call_lite(query: str = Body(...), meta: dict = Body(None)):
    meta = meta or {}
    async def predict_async(query):
        loop = asyncio.get_event_loop()
        model_provider = meta.get("model_provider", config.model_provider_lite)
        model_name = meta.get("model_name", config.model_name_lite)
        model = select_model(model_provider=model_provider, model_name=model_name)
        return await loop.run_in_executor(executor, model.predict, query)

    response = await predict_async(query)
    logger.debug({"query": query, "response": response.content})

    return {"response": response.content}

@chat.post("/agent/{agent_name}")
async def chat_agent(
    agent_name: str,
    query: str = Body(...),
    history: list = Body([]),
    config: dict = Body({}),
    meta: dict = Body({})
):
    request_id = config.get("request_id", str(uuid.uuid4()))
    thread_id  = config.get("thread_id", request_id)
    meta.update({"query": query, "agent_name": agent_name, "thread_id": thread_id})

    def make_chunk(content="", status="", error=None, history=None):
        payload = {"request_id": request_id, "response": content,
                   "status": status, "meta": meta}
        if error:   payload["error"]   = error
        if history: payload["history"] = history
        return json.dumps(payload, ensure_ascii=False).encode() + b"\n"

    async def streamer():
        # 1) init
        yield make_chunk(status="init")

        # 2) 执行agent
        try:
            final_answer = ""
            # 假设它是 async generator of strings
            async for part in kg_chat_agent.query(query):
                final_answer += part
                yield make_chunk(content=part, status="loading")
        except Exception as e:
            logger.error(f"Agent error: {e}\n{traceback.format_exc()}")
            yield make_chunk(status="error", error=str(e))
            return

        # 3) finished，更新 history
        history_manager = HistoryManager(history)
        history_manager.add_user(query)
        history_manager.add_ai(final_answer)
        yield make_chunk(status="finished", content=final_answer,
                         history=history_manager.messages)

    return StreamingResponse(streamer(), media_type="application/json")

@chat.get("/models")
async def get_chat_models(model_provider: str):
    """获取指定模型提供商的模型列表"""
    model = select_model(model_provider=model_provider)
    return {"models": model.get_models()}

@chat.post("/models/update")
async def update_chat_models(model_provider: str, model_names: list[str]):
    """更新指定模型提供商的模型列表"""
    config.model_names[model_provider]["models"] = model_names
    config._save_models_to_file()
    return {"models": config.model_names[model_provider]["models"]}

