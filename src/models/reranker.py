import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import os
import json
import requests
import numpy as np
from FlagEmbedding import FlagReranker
import logging


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class RerankerWrapper:
    def __init__(self, reranker_key, model_name, local_path=None, device="cpu"):
        self.reranker_key = reranker_key
        self.model_name = model_name
        self.device = device
        self.local_path = local_path

        provider, model_short = reranker_key.split("/", 1)

        self.config = type("Config", (object,), {
            "device": self.device,
            "reranker": self.reranker_key,
            "reranker_names": {model_short: {"name": model_name}},
            "model_local_paths": {model_name: local_path or ""}
        })()

        if provider == "local":
            self.reranker = LocalReranker(self.config)
        elif provider == "siliconflow":
            self.reranker = SiliconFlowReranker(self.config)
        else:
            raise ValueError(f"Unknown reranker provider: {provider}")

    def run(self, query, docs, normalize=True):
        if isinstance(self.reranker, LocalReranker):
            pairs = [(query, doc) for doc in docs]
            return self.reranker.compute_score(pairs, normalize=normalize)
        else:
            return self.reranker.compute_score((query, docs), normalize=normalize)


class LocalReranker(FlagReranker):
    def __init__(self, config, **kwargs):
        provider, model_name = config.reranker.split('/', 1)
        model_info = config.reranker_names[model_name]
        model_name_or_path = config.model_local_paths.get(model_info["name"], model_info.get("local_path"))
        model_name_or_path = model_name_or_path or model_info["name"]
        logging.info(f"Loading Reranker model {config.reranker} from {model_name_or_path}")

        super().__init__(model_name_or_path, use_fp16=True, device=config.device, **kwargs)
        logging.info(f"Reranker model {config.reranker} loaded")


class SiliconFlowReranker:
    def __init__(self, config, **kwargs):
        self.url = "https://api.siliconflow.cn/v1/rerank"
        self.model = config.reranker_names[config.reranker.split("/")[1]]["name"]
        api_key = 'sk-airshplskaflsntrycgajclaomhovoycgmcckhkkqmdvtjfi'
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def compute_score(self, sentence_pairs, batch_size=256, max_length=512, normalize=False):
        query, sentences = sentence_pairs[0], sentence_pairs[1]
        payload = self.build_payload(query, sentences, max_length)
        response = requests.post(self.url, json=payload, headers=self.headers)

        try:
            response_json = response.json()
        except Exception:
            raise ValueError(f"JSON: {response.text}")

        if "results" not in response_json:
            raise ValueError(f"返回中未包含结果字段: {response_json}")

        results = sorted(response_json["results"], key=lambda x: x["index"])
        all_scores = [result["relevance_score"] for result in results]

        if normalize:
            all_scores = [sigmoid(score) for score in all_scores]

        return all_scores

    def build_payload(self, query, sentences, max_length=512):
        return {
            "model": self.model,
            "query": query,
            "documents": sentences,
            "max_chunks_per_doc": max_length,
        }


# 测试
if __name__ == '__main__':
    query = "皮卡丘的进化是什么？"
    docs = [
        "皮卡丘可以进化为雷丘。",
        "小火龙是初代宝可梦之一。",
        "天气真好，适合去散步。"
    ]

    reranker = RerankerWrapper(
        reranker_key="siliconflow/bge-reranker-v2-m3",
        model_name="BAAI/bge-reranker-v2-m3"
    )
    # reranker = RerankerWrapper(reranker_key='local/bge-reranker-v2-m3', model_name='bge-reranker-v2-m3',
    #                            local_path='/data/meet-Pok-mon-chat/resources/models/bge-reranker-v2-m3')
    scores = reranker.run(query, docs)
    for doc, score in zip(docs, scores):
        print(f"{doc}\nScore: {score:.4f}\n")
