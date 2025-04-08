import os
import numpy as np
from reranker import LocalReranker, SiliconFlowReranker, OllamaReranker, OneAPIReranker


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class DummyConfig:
    def __init__(self, reranker, model_name, local_path=None, device="cpu"):
        self.reranker = reranker
        self.device = device
        self.reranker_names = {
            model_name: {
                "name": model_name
            }
        }
        self.model_local_paths = {
            model_name: local_path or ""
        }


def test_reranker(reranker_type: str, model_name: str, key=None, local_path=None):
    print(f"\n--- Testing: {reranker_type}/{model_name} ---")

    # 环境变量（用于硅基 API）
    if reranker_type == "siliconflow":
        os.environ["SILICONFLOW_API_KEY"] = key or ""

    # 创建 config
    config = DummyConfig(f"{reranker_type}/{model_name}", model_name, local_path)

    # 初始化 reranker 类
    if reranker_type == "local":
        reranker = LocalReranker(config)
    elif reranker_type == "siliconflow":
        reranker = SiliconFlowReranker(config)
    elif reranker_type == "ollama":
        reranker = OllamaReranker(config)
    elif reranker_type == "oneapi":
        reranker = OneAPIReranker(config)
    else:
        raise ValueError("Unknown reranker type.")

    # 示例数据
    query = "皮卡丘的进化是什么？"
    docs = [
        "皮卡丘可以进化为雷丘。",
        "小火龙是初代宝可梦之一。",
        "天气真好，适合去散步。"
    ]

    scores = reranker.compute_score((query, docs), normalize=True)

    for i, (doc, score) in enumerate(zip(docs, scores)):
        print(f"\nDoc {i + 1}: {doc}\nScore: {score:.4f}")


if __name__ == '__main__':
    # 本地模型示例
    test_reranker(
        reranker_type="local",
        model_name='bge-reranker-v2-m3',
        local_path="/data/meet-Pok-mon-chat/resources/models/bge-reranker-v2-m3"
    )

    test_reranker(
        reranker_type="siliconflow",
        model_name="BAAI/bge-reranker-v2-m3",
        key='sk-airshplskaflsntrycgajclaomhovoycgmcckhkkqmdls'
    )
    # Ollama
    test_reranker(
        reranker_type="ollama",
        model_name="bge-reranker-v2-m3"
    )

    # # OneAPI 示例
    # test_reranker(
    #     reranker_type="oneapi",
    #     model_name="bge-reranker-base"
    # )
