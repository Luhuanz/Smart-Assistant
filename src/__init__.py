from dotenv import load_dotenv

load_dotenv("src/.env")
load_dotenv()  # 双保险，避免未命中

from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor()

from src.config import Config
config = Config()

from src.stores import  KnowledgeBase

knowledge_base = KnowledgeBase()

from agent.kg_agent import KGQueryAgent
graph_base = KGQueryAgent()

from rag.core.retriever import Retriever
retriever = Retriever()