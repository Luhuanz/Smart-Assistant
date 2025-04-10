import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# model
MODEL_RERANKER_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'bge-reranker-v2-m3')
MODEL_ROBERTA_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'chinese-roberta-wwm-ext')
MODEL_EMBEDDING_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'bge-large-zh-v1.5')
MODEL_OCR_PATH = os.path.join(BASE_DIR, 'resources', 'models', 'ocr')
CACHE_BERTA_MODEL = os.path.join(BASE_DIR, 'resources', 'cache', 'roberta', 'best_roberta')
EMBEDDING_MODEL = 'bge-m3-pro'
EMBEDDING_MODEL_DIM = 1024
MODEL_BASE = r"/data/Langagent/resources/models/ocr"
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

DATA_PARSER_DATA = os.path.join(BASE_DIR, 'resources', 'data_parser')

LOG_DIR = os.path.join(BASE_DIR, 'logs')

EMBED_MODEL_INFO = {
    "local/bge-large-zh-v1.5": {
        "name": "bge-large-zh-v1.5",
        "dimension": 1024,
        "local_path": "/data/Langagent/resources/models/bge-large-zh-v1.5",
    },
    "ollama/bge-m3:latest": {
        "name": "bge-m3:latest",
        "dimension": 1024,
        "url": "http://localhost:11434/api/embed"
    },

}
