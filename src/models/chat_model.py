#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from typing import List, Dict, Union, Generator, Any

from openai import OpenAI
from src.utils import logger
from configs.settings import MODEL_API_KEY, MODEL_API_BASE, MODEL_NAME

_log = logger.LogManager()


class OpenAIBase:
    """
    OpenAI 模型调用基础类，统一封装 openai.ChatAPI的调用。
    """

    def __init__(self, api_key: str = MODEL_API_KEY,
                 base_url: str = MODEL_API_BASE,
                 model_name: str = MODEL_NAME) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        # 创建OpenAI客户端实例
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        _log.debug(f"Models: {self.get_models()}")

    def _prepare_messages(self, message: Union[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
        """将输入消息封装为消息列表格式"""
        if isinstance(message, str):
            return [{"role": "user", "content": message}]
        return message

    def predict(self,
                message: Union[str, List[Dict[str, str]]],
                stream: bool = False
                ) -> Union[Any, Generator[Any, None, None]]:
        """
        根据传入的消息调用模型接口，支持流式返回。
        :param message: 用户输入的消息，可以是字符串或消息列表
        :param stream: 是否启用流式返回
        :return: 模型返回的结果或生成器
        """
        messages = self._prepare_messages(message)
        if stream:
            return self._stream_response(messages)
        else:
            return self._get_response(messages)

    def _stream_response(self, messages: List[Dict[str, str]]) -> Generator[Any, None, None]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=True
        )
        for chunk in response:
            yield chunk.choices[0].delta

    def _get_response(self, messages: List[Dict[str, str]]) -> Any:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False
        )
        return response.choices[0].message

    def get_models(self) -> List[Any]:
        try:
            return self.client.models.list()
        except Exception as e:
            _log.error(f"Error getting models: {e}")
            return []


class OpenModel(OpenAIBase):
    """
    针对 OpenAI 模型的封装，默认使用 "gpt-4o-mini" 模型，可通过构造函数覆盖。
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        model_name = model_name or "gpt-4o-mini"
        # 使用环境变量中配置的API key 和base_url
        api_key = os.getenv("OPENAI_API_KEY", MODEL_API_KEY)
        base_url = os.getenv("OPENAI_API_BASE", MODEL_API_BASE)
        super().__init__(api_key=api_key, base_url=base_url, model_name=model_name)


class DashScope(OpenAIBase):
    """
    DashScope 模型接口，封装了 dashscope.Generation.call 方法。
    注意：DashScope 在调用时使用内部导入的 dashscope 模块，不依赖父类的客户端。
    """

    def __init__(self, model_name: str = "qwen-max-latest") -> None:
        self.model_name = model_name
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        _log.info(f"DashScope model: {self.model_name},using API key.")

    def _stream_response(self, messages: List[Dict[str, str]]) -> Generator[Any, None, None]:
        import dashscope
        response = dashscope.Generation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            result_format='message',
            stream=True
        )
        for chunk in response:
            message_obj = chunk.output.choices[0].message
            message_obj.is_full = False
            yield message_obj

    def _get_response(self, messages: List[Dict[str, str]]) -> Any:
        import dashscope
        response = dashscope.Generation.call(
            api_key=self.api_key,
            model=self.model_name,
            messages=messages,
            result_format='message',
            stream=False
        )
        return response.output.choices[0].message


if __name__ == "__main__":
    try:
        model = OpenModel()
        test_message = "请简单介绍一下人工智能的发展历史。"
        result = model.predict(test_message, stream=False)
        print(result)
        _log.info(f"模型返回: {result}")
    except Exception as e:
        _log.error(f"测试调用失败: {e}")
