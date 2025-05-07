import os
import asyncio
import traceback
import uuid
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, File, UploadFile, HTTPException, Body, Query
from fastapi.responses import JSONResponse

from src.utils.logger import LogManager
from src import config
from src.stores import KnowledgeBase
from src.utils import hashstr

logger = LogManager()
data = APIRouter(prefix="/data")

# 单例 或者 在模块顶层初始化一次
kb = KnowledgeBase(
    milvus_uri=config.get("milvus_uri"),
    embedding_config={"enable_knowledge_base": True, **config}
)

@data.get("/")
async def list_databases():
    """列出所有知识库"""
    try:
        rows = kb.list_databases()
        return {"databases": rows}
    except Exception as e:
        logger.error(f"list_databases failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.post("/")
async def create_database(
    database_name: str = Body(...),
    description: str = Body(...),
    dimension: Optional[int] = Body(None)
):
    """创建一个新的知识库 Collection"""
    try:
        info = kb.create_database(database_name, description, dimension)
        return JSONResponse(info)
    except Exception as e:
        logger.error(f"create_database failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.delete("/")
async def delete_database(db_id: str = Query(...)):
    """删除一个 Collection"""
    try:
        kb.delete_database(db_id)
        return {"message": "删除成功"}
    except Exception as e:
        logger.error(f"delete_database failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.get("/info")
async def get_database_info(db_id: str = Query(...)):
    """获取单个 Collection 的详情（包含 Milvus stats）"""
    try:
        info = kb.get_collection_info(db_id)
        return info
    except Exception as e:
        logger.error(f"get_database_info failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.post("/ingest/file")
async def ingest_file(
    db_id: str = Body(...),
    path: str = Body(...),
    do_ocr: bool = Body(False),
    chunk_size: int = Body(1000),
    chunk_overlap: int = Body(100),
    ocr_threshold: float = Body(0.3)
):
    """把服务器路径下的单个文件导入到指定库"""
    try:
        file_id = kb.ingest_file(
            db_id=db_id,
            path=path,
            do_ocr=do_ocr,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            ocr_det_threshold=ocr_threshold
        )
        return {"file_id": file_id, "status": "success"}
    except Exception as e:
        logger.error(f"ingest_file failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.post("/ingest/dir")
async def ingest_directory(
    db_id: str = Body(...),
    folder: str = Body(...),
    suffixes: Optional[List[str]] = Body(None)
):
    """把服务器目录下所有支持后缀的文件批量导入"""
    try:
        ids = kb.ingest_directory(db_id, folder, suffixes)
        return {"file_ids": ids, "status": "success"}
    except Exception as e:
        logger.error(f"ingest_directory failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.get("/search")
async def search_kb(
    query: str = Query(...),
    db_id: str = Query(...),
    distance_threshold: float = Query(None),
    rerank: bool = Query(True),
    top_k: int = Query(None)
):
    """向量检索接口"""
    try:
        res = kb.search(
            query=query,
            db_id=db_id,
            distance_threshold=distance_threshold,
            rerank=rerank,
            top_k=top_k
        )
        return res
    except Exception as e:
        logger.error(f"search_kb failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))

@data.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    db_id: Optional[str] = Query(None)
):
    """前端上传文件到后端，再由你自己调 ingest_file 导入"""
    if not file.filename:
        raise HTTPException(400, "No file")
    # 存到临时 uploads 目录
    upload_dir = kb.work_dir
    if db_id:
        _, upload_dir = kb._ensure_directories(db_id)
    os.makedirs(upload_dir, exist_ok=True)

    name, ext = os.path.splitext(file.filename)
    fname = f"{name}_{hashstr(name, 4, True)}{ext}"
    path = os.path.join(upload_dir, fname)
    with open(path, "wb") as buf:
        buf.write(await file.read())
    return {"file_path": path, "db_id": db_id}

@data.delete("/document")
async def delete_document(
    db_id: str = Body(...),
    file_id: str = Body(...)
):
    """如果需要删某个文件对应的向量，顺便从 sqlite 里删记录"""
    try:
        # 如果你的 KB 里还没写 delete_file，可以自行实现：
        kb.client.delete(collection_name=db_id, filter=f"file_id=='{file_id}'")
        kb.db_manager.delete_file(file_id)
        return {"message": "删除成功"}
    except Exception as e:
        logger.error(f"delete_document failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(500, str(e))
