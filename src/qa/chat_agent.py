import asyncio
import sys
from pathlib import Path
from typing import Dict, Optional

from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from configs.settings import *

# 设置项目路径
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))

# 本地模块导入
from agent.kg_agent import KGQueryAgent
from rag import GraphRAG
from api.websearch.LiteWebSearcher import WebSearcher
from api.websearch.websearcher import *


# 辅助类
class AgentState(MessagesState):
    next: str


class PokemonKGChatAgent:
    """宝可梦知识图谱聊天代理"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化代理
        :param config: 配置字典，包含以下可选键:
            - neo4j_auth: (user, password) 元组
            - llm_config: 语言模型配置
            - graph_rag_config: 图RAG配置
        """
        self.config = self._init_config(config)
        self._init_components()
        self._build_graph()

    def _init_config(self, config: Optional[Dict]) -> Dict:
        """初始化默认配置"""
        default_config = {
            "neo4j_auth": ("neo4j", "woshishamo630"),
            "llm_config": {
                "model": "Doubao-pro-256k-1.5",
                "base_url": "http://139.224.116.116:3000/v1",
                "api_key": "sk-36oMlDApF5Nlg0v23014A4B69e864000944151Cd75D82076"
            },
            "graph_rag_config": {
                "artifacts_path": "F:\\bigmodel\\meet-Pok-mon\\4.KGqa\\Pokemon-KGQA\\RAG\\artifacts",
                "llm_config": {
                    "model": "Doubao-pro-256k-1.5",
                    "base_url": "http://139.224.116.116:3000/v1",
                    "api_key": "sk-36oMlDApF5Nlg0v23014A4B69e864000944151Cd75D82076"
                },
                "community_level": 0
            }
        }
        if config:
            default_config.update(config)
        return default_config

    def _init_components(self):
        """初始化所有组件"""
        # 初始化LLM
        self.llm = ChatOpenAI(**self.config["llm_config"])

        # 初始化知识图谱查询代理 1
        self.kgsql_agent = KGQueryAgent(llm=self.llm)

        # 初始化图RAG 1
        self.graph_rag = GraphRAG(
            artifacts_path=self.config["graph_rag_config"]["artifacts_path"],
            llm_config=self.config["graph_rag_config"]["llm_config"],
            community_level=self.config["graph_rag_config"]["community_level"]
        )

        # 初始化网络搜索器 0
        self.searcher = WebSearcher()

    def _build_graph(self):
        """构建LangGraph状态图"""
        # 定义节点
        builder = StateGraph(AgentState)

        # 添加节点
        builder.add_node("supervisor", self._supervisor)
        builder.add_node("chat", self._chat)
        builder.add_node("kg_sqler", self._kgsql_node)
        builder.add_node("graph_rager", self._graph_rager)
        builder.add_node("web_searcher", RunnableLambda(self._web_searcher))

        # 定义成员列表
        members = ["chat", "kg_sqler", "graph_rager", "web_searcher"]

        # 添加边
        for member in members:
            builder.add_edge(member, "supervisor")

        # 添加条件边
        builder.add_conditional_edges("supervisor", lambda state: state["next"])
        builder.add_edge(START, "supervisor")

        # 编译图
        self.graph = builder.compile(checkpointer=MemorySaver())

    # 节点函数定义
    def _chat(self, state: AgentState):
        """自然语言聊天节点"""
        messages = state["messages"]
        model_response = self.llm.invoke(messages)
        return {"messages": model_response}

    def _kgsql_node(self, state: AgentState):
        """知识图谱查询节点"""
        result = self.kgsql_agent.agent.invoke(state)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="kg_sqler")
            ]
        }

    def _graph_rager(self, state: AgentState):
        """图RAG查询节点"""
        messages = state["messages"]
        response = self.graph_rag.query(messages)
        return {"messages": [HumanMessage(content=response.content, name="graph_rager")]}

    async def _web_searcher(self, state: AgentState):
        """网络搜索节点"""
        messages = state["messages"]
        response = await self.searcher.search_and_generate(messages[0].content)
        return {"messages": [HumanMessage(content=response, name="web_searcher")]}

    def _supervisor(self, state: AgentState):
        """监督员节点"""
        system_prompt = (
            "你被指定为对话监督员，负责协调以下工作模块的协作：{members}\n\n"
            "各模块职能划分：\n"
            "- chat：自然语言交互模块\n"
            "  • 直接处理用户输入的自然语言响应\n"
            "- kg_sqler：宝可梦知识图谱查询模块\n"
            "  • 属性数据（种族值/进化链/特性）\n"
            "  • 角色关系（训练师/劲敌/团队）\n"
            "  • 地域情报（地点/道馆/栖息地）\n"
            "- graph_rager：宝可梦相关知识库\n"
            "  • 人物介绍（如人物事迹等）\n"
            "  • 社群发现（如道馆派系识别）\n"
            "  • 路径分析（角色关联路径追踪）\n"
            "  • 时序关联（赛事参与时间轴分析）\n\n"
            "- web_searcher：实时联网搜索模块\n"
            "  • 当问题涉及最新资讯、新闻或时效性内容时使用\n"
            "  • 当其他知识库无法提供准确答案时使用\n"
            "  • 可获取官方公告、赛事结果等实时信息\n"
            "  • 能查询宝可梦相关社区讨论和玩家反馈\n"
            "  • 可验证其他模块提供信息的时效性和准确性\n\n"
            "模块调用原则：\n"
            "1. 优先使用本地知识库(kg_sqler/graph_rager)回答已知的宝可梦知识\n"
            "2. 当问题涉及实时信息或本地知识不足时，调用web_searcher\n"
            "3. 请根据用户请求指定下一个执行模块。"
            "4. 每个模块执行后将返回任务结果及状态。\n"
            "执行流程规范：\n"
            "1. chat模块最多能调用一次\n"
            "2. 可以链式调用多个模块（如先用kg_sqler查询，再用web_searcher验证）\n"
            "3. 你可以不断调用上述的模块，当某个模块的结果不足以回答用户的问题时（如未查询到相关结果），你可以继续调用其他模块，直到用户问题得到回答。"
            "4. 当你任务完成时，才能返回FINISH终止符"
        )

        prompt = ChatPromptTemplate.from_template("""
        请严格按以下JSON格式回复，只包含next字段:
        {{
            "next": "FINISH"
        }}
        输入：{input}
        """)

        messages = [{"role": "system", "content": system_prompt}] + state["messages"]
        chain = prompt | self.llm | JsonOutputParser()
        response = chain.invoke({"input": messages})

        next_ = response["next"]
        return {"next": END if next_ == "FINISH" else next_}

    # 公共接口
    async def query(self, question: str):
        """统一的查询接口"""
        input_message = {"messages": [HumanMessage(content=question)]}
        config = {"configurable": {"thread_id": "0"}}

        chunks = []
        async for chunk in self.graph.astream(input_message, config, stream_mode="values"):
            chunks.append(chunk["messages"][-1])

        yield chunks[-1].content if chunks else None


# 使用示例
if __name__ == "__main__":
    async def main():
        # 初始化代理
        agent = PokemonKGChatAgent()
        # 示例查询
        question = "拥有皮卡丘的角色中，有哪些是小刚的伙伴？"

        print(f"\n问题: {question}")
        print("回答:")
        async for chunk in agent.query(question):
            print(chunk)


    asyncio.run(main())
