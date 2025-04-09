import os
from typing import List, Optional, Dict
from langchain.schema import Document
from rag.core.indexing import parse_file, chunk_file
from src.models.embedding import get_embedding_model
from rag.core.Milvus import MilvusStorage
from configs.settings import *


class DocumentIngestor:
    def __init__(
            self,
            milvus_config: Dict,
            embedding_config: Dict,
            chunk_size: int = 1000,
            chunk_overlap: int = 100,
            ocr_enabled: bool = False,
            ocr_det_threshold: float = 0.3,
    ):
        """
        文件 -> Chunk -> Embedding -> 存入 Milvus 的集成类

        Args:
            milvus_config: 初始化MilvusStorage的配置字典
            embedding_config: 初始化Embedding模型的配置字典
            chunk_size: 文本块的最大长度
            chunk_overlap: 文本块之间的重叠长度
            ocr_enabled: 是否对PDF启用OCR
            ocr_det_threshold: OCR检测阈值
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.ocr_enabled = ocr_enabled
        self.ocr_det_threshold = ocr_det_threshold

        self.embedder = get_embedding_model(embedding_config)
        self.store = MilvusStorage(**milvus_config)

    def ingest_single_file(self, file_path: str):
        """
        处理单个文件，并插入向量数据库。

        Args:
            file_path: 文件路径
        """
        print(f"📄 正在处理文件: {file_path}")
        chunks = chunk_file(
            file_path=file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            do_ocr=self.ocr_enabled,
            ocr_det_threshold=self.ocr_det_threshold
        )

        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedder.batch_encode(texts)

        for doc, embedding in zip(chunks, embeddings):
            doc.metadata["embedding"] = embedding

        self.store.insert(chunks)
        print(f"✅ 文件 {file_path} 处理完成并存入向量数据库！")

    def ingest_directory(self, directory_path: str, suffixes: Optional[List[str]] = None):
        """
        批量处理文件夹下的文件。
        Args:
            directory_path: 文件夹路径
            suffixes: 文件后缀列表，如[".pdf", ".docx", ".txt"]
        """
        suffixes = suffixes or [".pdf", ".docx", ".txt", ".md"]
        print(f"📁 扫描目录: {directory_path}")

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in suffixes):
                    file_path = os.path.join(root, file)
                    self.ingest_single_file(file_path)

    def close(self):
        """关闭 Milvus 连接"""
        self.store.close()
        print("🔌 向量数据库连接已关闭！")


if __name__ == "__main__":
    # 示例用法
    milvus_config = {
        "collection_name": "documents_collection",
        "overwrite": True,
        "dim": 1024,
        "host": "localhost",
        "port": "19530",
        "openai_base_url": MODEL_API_BASE,
        "openai_api_key": MODEL_API_KEY,
        "embedding_model": EMBEDDING_MODEL
    }

    embedding_config = {
        "enable_knowledge_base": True,
        "embed_model": "local/bge-large-zh-v1.5"
    }

    ingestor = DocumentIngestor(
        milvus_config=milvus_config,
        embedding_config=embedding_config,
        chunk_size=500,
        chunk_overlap=100,
        ocr_enabled=True
    )

    # 单文件导入示例
    ingestor.ingest_single_file("/data/Langagent/deepdoc/data/picture.pdf")

    # 整个文件夹导入示例
    # ingestor.ingest_directory("/data/docs")

    ingestor.close()
