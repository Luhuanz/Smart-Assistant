#!/usr/bin/env python
# coding: utf-8

import json
import math
import os
import time

import requests


def query_dify_brain(msg, call_reply, conversation_id=None, histories=None):
    """
    向 Dify API 发送请求并流式获取回答，逐行调用 call_reply 输出。
    :param msg: 用户输入的问题内容
    :param call_reply: 回调函数，用于处理每行返回结果
    :param conversation_id: 如果需要上下文对话，传入 Dify 返回的 conversation_id
    :param histories: 历史对话内容（字符串形式）
    """
    start_time = time.time()
    # 从环境变量获取必要的配置信息
    dify_api_key = os.environ.get("DIFY_API_KEY", "")
    nlp_pokemon_host = os.environ.get("NLP_POKEMON_HOST", "")

    if not dify_api_key or not nlp_pokemon_host:
        raise ValueError("请在环境变量中设置 DIFY_API_KEY 和 NLP_POKEMON_HOST。")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {dify_api_key}",
    }

    # 构造请求体
    # "histories" 可以是多轮对话的上下文，如果不需要可以留空字符串或不传
    payload = {
        "inputs": {
            "histories": histories if histories else "",
        },
        "query": msg,
        "response_mode": "blocking",  # 这里演示阻塞式或流式都可
        "conversation_id": conversation_id if conversation_id else "",
        "user": "robot-user",
        "files": [],
        "auto_generate_name": False,
    }

    # 使用session发起请求，设置 stream=True 以便实时读取数据
    session = requests.Session()
    response = session.post(
        url=f"{nlp_pokemon_host}/chat-messages",
        headers=headers,
        verify=False,  # 如果有证书错误，可以改为 True
        data=json.dumps(payload),
        stream=True,  # 开启流式读取
        timeout=30  # 可根据情况调整超时时间
    )

    # 如果状态码不是200，直接抛异常
    if response.status_code != 200:
        call_reply("[error]")
        raise Exception(f"请求失败，状态码：{response.status_code}")

    # 逐行读取输出
    json_buffer = ""  # 如果出现特殊标记需要聚合，可以用这个收集
    json_condition = False

    for line in response.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8", errors="ignore")

        # 替换 "data:" 前缀后再解析 JSON
        line_str = line_str.replace("data:", "").strip()
        try:
            chunk_data = json.loads(line_str)
        except json.JSONDecodeError:
            # 如果解析出错，可以忽略或直接传给回调看看
            call_reply(f"[json error]: {line_str}")
            continue
        # 如果返回了 event = "error"
        if chunk_data.get("event") == "error":
            call_reply("[error]")
            raise Exception(chunk_data.get("message", "未知错误"))
        # 如果返回了 event = "message"
        elif chunk_data.get("event") == "message":
            # 获取回答内容
            answer_str = chunk_data.get("answer", "")

            # 如果有特殊标记，
            # 如果 answer_str 或累计的结果以特殊符号开始，则开启特殊处理
            if "㊣" in answer_str:
                json_condition = True
            if json_condition:
                # 收集到 buffer
                json_buffer += answer_str
            else:
                # 正常流式返回
                call_reply(answer_str)

        # 其他事件可忽略或自行处理
        else:
            pass

    # 请求完毕后，打印耗时
    cost_ms = math.floor((time.time() - start_time) * 1000)
    print(f"[INFO] LLM处理耗时: {cost_ms} ms")

    if json_condition and json_buffer:
        call_reply(f"[SPECIAL JSON] {json_buffer}")

    # 最后标记回答结束
    call_reply("[DONE]")


# 回调函数
def example_call_reply(chunk):
    """
    每次获得一行回答，打印。
    """
    print(">>", chunk)


if __name__ == "__main__":
    # 例子：在环境变量中设置
    # 要问的问题
    user_msg = "皮卡丘进化是什么"

    # histories 字符串做上下文，比如：
    # histories = "Human:你好\nAssistant:你好，请问有什么可以帮您？\n"
    histories_example = ""

    # 发起请求
    print("[TEST调用DifyAPI，发送问题：", user_msg)
    try:
        query_dify_brain(
            msg=user_msg,
            call_reply=example_call_reply,
            conversation_id="",
            histories=histories_example
        )
    except Exception as e:
        print("[ERROR]", e)
