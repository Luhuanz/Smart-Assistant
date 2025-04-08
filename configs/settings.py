import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# model
MODEL_RERANKER_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'bge-reranker-v2-m3')
MODEL_ROBERTA_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'chinese-roberta-wwm-ext')
EMBEDDING_MODEL = 'bge-m3-pro'

# api
MODEL_API_KEY = 'sk-36oMlDApF5Nlg0v23014A4B69e864000944151Cd75D82076'
MODEL_API_BASE = 'http://139.224.116.116:3000/v1'
MODEL_NAME = 'deepseek-ai/DeepSeek-V3'
TAVILY_API_KEY = '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'
SerperAPI = 'https://google.serper.dev/search'

# data
ENTITY_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'entity_data')

NER_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'ner_data')

RAW_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'raw_data')

RELATIONS_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'relations_data')

GRAPHRAG_RAW_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'graph_data', '精灵之沙暴天王.txt')

ARTIFACTS_DATA = os.path.join(BASE_DIR, 'rag', 'artifacts')

CACHE_BERTA_MODEL = os.path.join(BASE_DIR, 'resources', 'cache', 'roberta', 'best_roberta')

RAW_DATA = os.path.join(BASE_DIR, 'resources', 'data', 'raw_data')
