import json
import asyncio
import traceback
import uuid
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import StreamingResponse
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage

from src import executor, config, get_retriever
from rag.core import HistoryManager
from src.models import select_model
from src.utils.logger import LogManager
from src.qa import PokemonKGChatAgent
retriever = get_retriever()
logger = LogManager()
kg_chat_agent = PokemonKGChatAgent()

chat = APIRouter(prefix="/chat")

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def convert_messages_to_dicts(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert LangChain Message objects to plain dicts that can be JSON-encoded."""
    if not messages:
        return []
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }
    result: List[Dict[str, str]] = []

    for msg in messages:
        role = getattr(msg, "_type", None)          # '_type' is 'human' / 'ai' / 'system'
        result.append({
            "role": role_map.get(role, "user"),
            "content": msg.content,
        })
    return result


def make_chunk(meta: Dict[str, Any],
               content: Optional[str] = None,
               **kwargs) -> bytes:
    """统一的 SSE / chunk 打包函数（返回 bytes 行）"""
    def convert(obj):
        if isinstance(obj, BaseMessage):
            return {"role": getattr(obj, "_type", "user"), "content": obj.content}
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    payload = {
        "response": content,
        "meta": meta,
        **kwargs,
    }
    return json.dumps(payload, ensure_ascii=False, default=convert).encode("utf-8") + b"\n"



def need_retrieve(meta: Dict[str, Any]) -> bool:
    return meta.get("use_web") or meta.get("use_graph") or meta.get("db_id")

# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

@chat.get("/")
async def chat_get():
    return "Chat Get!"


@chat.post("/")
async def chat_post(
        query: str = Body(...),
        meta: Dict[str, Any] = Body({}),
        history: Optional[List[Dict[str, Any]]] = Body(None),
        thread_id: Optional[str] = Body(None),
):
    """主聊天接口，支持 **流式** 返回"""

    # 1. 选择模型 ----------------------------------------------------------------
    model = select_model()
    if model is None:
        raise HTTPException(status_code=500, detail="没有可用的模型，请检查模型配置")

    meta["server_model_name"] = model.model_name
    history_manager = HistoryManager(system_prompt=meta.get("system_prompt"))

    logger.debug(f"Received query: {query} with meta: {meta}")

    # ---------------------------------------------------------------------------
    async def generate_response():
        modified_query = query
        refs = None

        # 2. 检索阶段 -------------------------------------------------------------
        if meta and need_retrieve(meta):
            yield make_chunk(meta, status="searching")
            try:

                modified_query, refs = retriever(
                    modified_query,
                    history_manager.history.messages,
                    meta
                )
            except Exception as e:
                logger.error(f"Retriever error: {e}\n{traceback.format_exc()}")
                yield make_chunk(meta,
                                 message=f"Retriever error: {e}",
                                 status="error")
                return
            yield make_chunk(meta, status="generating")

        # 3. 构造 Prompt ----
        messages = history_manager.get_history_with_msg(
            modified_query,
            max_rounds=meta.get("history_round")
        )
        history_manager.add_user(query)                    # 把原始用户查询加入历史
        formatted_messages = convert_messages_to_dicts(messages)

        content = ""
        reasoning_content = ""
        try:
            for delta in model.predict(formatted_messages, stream=True):
                # 一些模型将「思考过程」放在 reasoning_content
                if not delta.content and hasattr(delta, "reasoning_content"):
                    reasoning_content += delta.reasoning_content or ""
                    yield make_chunk(meta,
                                     reasoning_content=reasoning_content,
                                     status="reasoning")
                    continue

                content += delta.content or ""
                yield make_chunk(meta, content=delta.content, status="loading")

            logger.debug(f"Final response: {content}")
            logger.debug(f"Final reasoning response: {reasoning_content}")

            # 4. 更新历史，发送最终块
            updated_history = history_manager.update_ai(content)
            history_serializable = convert_messages_to_dicts(updated_history)

            yield make_chunk(meta,
                             status="finished",
                             history=history_serializable,
                             refs=refs)
        except Exception as e:
            logger.error(f"Model error: {e}\n{traceback.format_exc()}")
            yield make_chunk(meta,
                             message=f"Model error: {e}",
                             status="error")

    return StreamingResponse(generate_response(),
                             media_type="application/json")


@chat.post("/call")
async def call(query: str = Body(...), meta: Dict[str, Any] = Body({})):
    """同步调用完整模型"""
    model = select_model(model_provider=meta.get("model_provider"), model_name=meta.get("model_name"))
    if model is None:
        raise HTTPException(status_code=500, detail="模型不可用")

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(executor, model.predict, query)
    logger.debug(f"query: {query}, response: {response.content}")
    return {"response": response.content}


@chat.post("/call_lite")
async def call_lite(query: str = Body(...), meta: Dict[str, Any] = Body({})):
    """使用 Lite 模型，同步调用"""
    async def _predict_async(q):
        model_provider = meta.get("model_provider", config.model_provider_lite)
        model_name = meta.get("model_name", config.model_name_lite)
        model = select_model(model_provider=model_provider, model_name=model_name)
        if model is None:
            raise HTTPException(status_code=500, detail="Lite 模型不可用")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, model.predict, q)

    response = await _predict_async(query)
    logger.debug(f"query: {query}, response: {response.content}")
    return {"response": response.content}


@chat.post("/agent/{agent_name}")
async def chat_agent(
    agent_name: str,
    query: str = Body(...),
    history: List[Dict[str, Any]] = Body([]),
    cfg: Dict[str, Any] = Body({}),
    meta: Dict[str, Any] = Body({}),
):
    request_id = cfg.get("request_id", str(uuid.uuid4()))
    thread_id = cfg.get("thread_id", request_id)
    meta.update({"query": query, "agent_name": agent_name, "thread_id": thread_id})

    def make_agent_chunk(content: str = "", status: str = "", error: str | None = None, history_resp=None):
        payload = {
            "request_id": request_id,
            "response": content,
            "status": status,
            "meta": meta,
        }
        if error:
            payload["error"] = error
        if history_resp:
            payload["history"] = history_resp
        return json.dumps(payload, ensure_ascii=False).encode() + b"\n"

    async def streamer():
        # init
        yield make_agent_chunk(status="init")
        try:
            final_answer = ""
            async for part in kg_chat_agent.query(query):
                final_answer += part
                yield make_agent_chunk(content=part, status="loading")
        except Exception as e:
            logger.error(f"Agent error: {e}\n{traceback.format_exc()}")
            yield make_agent_chunk(status="error", error=str(e))
            return

        # finished
        history_mgr = HistoryManager()
        history_mgr.add_user(query)
        history_mgr.add_ai(final_answer)
        yield make_agent_chunk(
            status="finished",
            content=final_answer,
            history_resp=convert_messages_to_dicts(history_mgr.history.messages)
        )

    return StreamingResponse(streamer(), media_type="application/json")


@chat.get("/models")
async def get_chat_models(model_provider: str):
    model = select_model(model_provider=model_provider)
    return {"models": model.get_models()}


@chat.post("/models/update")
async def update_chat_models(model_provider: str, model_names: List[str]):
    config.model_names[model_provider]["models"] = model_names
    config._save_models_to_file()
    return {"models": config.model_names[model_provider]["models"]}
