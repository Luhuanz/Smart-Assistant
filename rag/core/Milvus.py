import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from typing import List, Optional
from langchain.schema import Document
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)
from langchain_openai import OpenAIEmbeddings
from configs.settings import *


class MilvusStorage:
    def __init__(
            self,
            collection_name: str = "default",
            dim: int = 1024,
            host: str = "localhost",
            port: str = "19530",
            overwrite: bool = False,
            openai_base_url: str = MODEL_API_BASE,
            openai_api_key: str = MODEL_API_KEY,
            embedding_model: str = EMBEDDING_MODEL,
    ):
        self.collection_name = collection_name
        self.dim = dim
        self.embedder = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_base=openai_base_url,
            openai_api_key=openai_api_key,
            chunk_size=32
        )

        connections.connect(host=host, port=port)

        self.fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="text_length", dtype=DataType.INT64),
        ]

        self.schema = CollectionSchema(
            fields=self.fields,
            description="Embedded documents",
            enable_dynamic_field=True
        )

        if utility.has_collection(collection_name):
            if overwrite:
                utility.drop_collection(collection_name)
                print(f"已覆盖现有集合: {collection_name}")
                self.collection = self._create_collection()
            else:
                self.collection = Collection(collection_name)
                print(f"已加载现有集合: {collection_name}")
        else:
            self.collection = self._create_collection()

        if not self.collection.has_index():
            self._create_index()

    def _create_collection(self) -> Collection:
        print(f"创建新集合: {self.collection_name}")
        return Collection(
            name=self.collection_name,
            schema=self.schema,
            consistency_level="Strong"
        )

    def _create_index(self, index_params: Optional[dict] = None):
        default_index = {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 16}
        }
        self.collection.create_index(
            field_name="embedding",
            index_params=index_params or default_index
        )
        self.collection.load()

    def insert(self, documents: List[Document], batch_size: int = 32):
        print(f"开始插入 {len(documents)} 个文档")
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [doc.page_content for doc in batch]
            embeddings = self.embedder.embed_documents(texts)
            metadata = [doc.metadata for doc in batch]
            text_lengths = [len(doc.page_content) for doc in batch]

            entities = [texts, embeddings, metadata, text_lengths]
            self.collection.insert(entities)
            print(f"已插入 {min(i + batch_size, len(documents))}/{len(documents)}")

        self.collection.flush()
        print(f"插入完成，总计 {self.collection.num_entities} 条数据")

    def close(self):
        connections.disconnect()
        print("Milvus 连接已关闭")
