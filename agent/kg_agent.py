import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Any, Iterator

import torch
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from py2neo import Graph
from pydantic import BaseModel
from transformers import BertTokenizer
from configs.settings import *

# 获取当前文件所在目录的父目录（即项目根目录）
current_dir = Path(__file__).parent
project_root = current_dir.parent.resolve()

# 将项目根目录添加到Python路径
sys.path.insert(0, str(project_root))
from src.ner.ner_model import *

# Neo4j 连接配置
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "tczaslw278")  # 替换为实际认证信息
g = Graph(NEO4J_URI, auth=NEO4J_AUTH)


class EntityRecognition:
    def __init__(self):
        # 初始化规则和 TF-IDF 对齐
        self.rule = rule_find()
        self.tfidf_r = tfidf_alignment()
        self.base_dir = BASE_DIR
        self.model_name = MODEL_ROBERTA_PATH
        self.pt_path = CACHE_BERTA_MODEL

        # 加载 tag2idx
        if os.path.exists(r'../resources/data/ner_data/tag2idx.npy'):
            with open(r'../resources/data/ner_data/tag2idx.npy', 'rb') as f:
                self.tag2idx = pickle.load(f)
                self.idx2tag = list(self.tag2idx)
        else:
            raise FileNotFoundError("tag2idx文件不存在！")

        # 初始化设备
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

        # 加载 tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name, cache_dir=MODEL_ROBERTA_PATH)

        # 初始化模型
        hidden_size = 128
        bi = True
        self.model = Bert_Model(self.model_name, hidden_size, len(self.tag2idx), bi)

        # 加载模型权重
        if os.path.exists(self.pt_path):
            print("加载已有模型")
            self.model.load_state_dict(torch.load(self.pt_path, map_location=self.device))
        else:
            raise FileNotFoundError("未找到模型权重文件!!")

        # 将模型移动到设备
        self.model = self.model.to(self.device)

        print('模型初始化完成 ......')

    def ner(self, question):
        # 调用 NER 方法
        return get_ner_result(self.model, self.tokenizer, question, self.rule, self.tfidf_r, self.device, self.idx2tag)


class KGQueryAgent:
    """宝可梦知识图谱查询代理"""

    def __init__(self, llm=None):
        """
        初始化查询代理
        :param llm: 可选的语言模型实例，默认使用ChatOpenAI
        """
        self.llm = llm or self._default_llm()
        self.ner = EntityRecognition()
        self.tools = self._init_tools()
        self.agent = self._create_agent()

    def _default_llm(self):
        """默认语言模型配置"""
        return ChatOpenAI(
            model="Doubao-pro-256k-1.5",
            base_url="http://139.224.116.116:3000/v1",
            api_key="sk-36oMlDApF5Nlg0v23014A4B69e864000944151Cd75D82076"
        )

    def _create_agent(self):
        """创建React代理"""
        return create_react_agent(
            self.llm,
            tools=self.tools,
            state_modifier="当用户询问关于宝可梦、人物、城镇、地区、属性的相关信息时，你将使用这些函数来查询neo4j数据库中的数据"
        )

    def query(self, question: str, stream: bool = False) -> Iterator[dict[str, Any] | Any] | dict[str, Any] | Any:
        """
        执行知识图谱查询
        :param question: 自然语言问题
        :param stream: 是否使用流式输出
        :return: 查询结果字典
        """
        input_message = {"messages": [HumanMessage(content=question)]}

        if stream:
            return self.agent.stream(input_message, stream_mode="updates")
        return self.agent.invoke(input_message)

    def _init_tools(self) -> List:
        """初始化所有查询工具"""

        # 定义查询模型
        class PokemonQuery(BaseModel):
            pokemon: str

        class PersonQuery(BaseModel):
            person: str

        class TownQuery(BaseModel):
            town: str

        class RegionQuery(BaseModel):
            region: str

        class IdentityQuery(BaseModel):
            identity: str

        class Entity(BaseModel):
            question: str

        # 宝可梦相关查询
        @tool(args_schema=PokemonQuery)
        def get_pokemon_chinese_name(pokemon: str):
            """查询宝可梦的中文名"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.name AS chinese_name;"
            return execute_query(sql, "chinese_name", f"未找到宝可梦: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_english_name(pokemon: str):
            """查询宝可梦的英文名"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.english_name AS english_name;"
            return execute_query(sql, "english_name", f"未找到宝可梦的英文名: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_ability(pokemon: str):
            """查询宝可梦的特性"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.ability AS ability;"
            return execute_query(sql, "ability", f"未找到宝可梦特性: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_hidden_ability(pokemon: str):
            """查询宝可梦的隐藏特性"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.hidden_ability AS hidden_ability;"
            return execute_query(sql, "hidden_ability", f"未找到宝可梦隐藏特性: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_height(pokemon: str):
            """查询宝可梦的身高"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.height AS height;"
            return execute_query(sql, "height", f"未找到宝可梦身高: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_weight(pokemon: str):
            """查询宝可梦的体重"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.weight AS weight;"
            return execute_query(sql, "weight", f"未找到宝可梦体重: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_evolution_level(pokemon: str):
            """查询宝可梦的进化等级"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.evolution_level AS evolution_level;"
            return execute_query(sql, "evolution_level", f"未找到宝可梦进化等级: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_attributes(pokemon: str):
            """查询宝可梦的属性抗性"""
            sql = f"MATCH (p:Pokémon) WHERE p.name = '{pokemon}' RETURN p.attr_ability AS attr_ability;"
            return execute_query(sql, "attr_ability", f"未找到宝可梦属性抗性: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_evolution(pokemon: str):
            """查询宝可梦的进化形态"""
            sql = f"MATCH (p1:Pokémon)-[:evolves_into]->(p2:Pokémon) WHERE p1.name = '{pokemon}' RETURN p2.name AS evolution_name;"
            return execute_query(sql, "evolution_name", f"未找到宝可梦进化形态: {pokemon}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_types(pokemon: str):
            """查询宝可梦的属性"""
            sql = f"MATCH (p:Pokémon)-[:has_type]->(i:identity) WHERE p.name = '{pokemon}' RETURN COLLECT(i.name) AS types;"
            return execute_query(sql, "types", f"未找到宝可梦属性: {pokemon}")

        # 人物相关查询
        @tool(args_schema=PersonQuery)
        def get_person_gender(person: str):
            """查询人物性别"""
            sql = f"MATCH (per:Person) WHERE per.name = '{person}' RETURN per.gender AS gender;"
            return execute_query(sql, "gender", f"未找到人物性别: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_english_name(person: str):
            """查询人物英文名"""
            sql = f"MATCH (per:Person) WHERE per.name = '{person}' RETURN per.english_name AS english_name;"
            return execute_query(sql, "english_name", f"未找到人物英文名: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_japanese_name(person: str):
            """查询人物日本名"""
            sql = f"MATCH (per:Person) WHERE per.name = '{person}' RETURN per.japanese_name AS japanese_name;"
            return execute_query(sql, "japanese_name", f"未找到人物日本名: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_challengers(person: str):
            """查询人物的挑战者"""
            sql = f"MATCH (per1:Person)-[:challenge]->(per2:Person) WHERE per1.name = '{person}' RETURN COLLECT(per2.name) AS challengers;"
            return execute_query(sql, "challengers", f"未找到人物挑战者: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_partners(person: str):
            """查询人物的伙伴"""
            sql = f"MATCH (per1:Person)-[:partner]->(per2:Person) WHERE per1.name = '{person}' RETURN COLLECT(per2.name) AS partners;"
            return execute_query(sql, "partners", f"未找到人物伙伴: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_enemies(person: str):
            """查询人物的敌对者"""
            sql = f"MATCH (per1:Person)-[:hostility]->(per2:Person) WHERE per1.name = '{person}' RETURN COLLECT(per2.name) AS enemies;"
            return execute_query(sql, "enemies", f"未找到人物敌对者: {person}")

        @tool(args_schema=PersonQuery)
        def get_person_relatives(person: str):
            """查询人物的亲戚"""
            sql = f"MATCH (per1:Person)-[:relative]->(per2:Person) WHERE per1.name = '{person}' RETURN COLLECT(per2.name) AS relatives;"
            return execute_query(sql, "relatives", f"未找到人物亲戚: {person}")

        # 地区、城镇相关查询
        @tool(args_schema=TownQuery)
        def get_town_region(town: str):
            """查询城镇所在的地区"""
            sql = f"MATCH (t:Town)-[:located_in]->(r:Region) WHERE t.name = '{town}' RETURN r.name AS region;"
            return execute_query(sql, "region", f"未找到城镇所在地区: {town}")

        @tool(args_schema=RegionQuery)
        def get_region_towns(region: str):
            """查询地区的城镇"""
            sql = f"MATCH (r:Region)<-[:located_in]-(t:Town) WHERE r.name = '{region}' RETURN COLLECT(t.name) AS towns;"
            return execute_query(sql, "towns", f"未找到地区城镇: {region}")

        @tool(args_schema=PersonQuery)
        def get_person_hometown(person: str):
            """查询人物来自哪个地区"""
            sql = f"MATCH (per:Person)-[:come_from]->(r:Region) WHERE per.name = '{person}' RETURN r.name AS region;"
            return execute_query(sql, "region", f"未找到人物家乡: {person}")

        @tool(args_schema=RegionQuery)
        def get_region_people(region: str):
            """查询地区有哪些人物"""
            sql = f"MATCH (per:Person)-[:come_from]->(r:Region) WHERE r.name = '{region}' RETURN COLLECT(per.name) AS people;"
            return execute_query(sql, "people", f"未找到地区人物: {region}")

        # 人物与宝可梦关系查询
        @tool(args_schema=PersonQuery)
        def get_person_pokemons(person: str):
            """查询人物拥有哪些宝可梦"""
            sql = f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE per.name = '{person}' RETURN COLLECT(p.name) AS pokemons;"
            return execute_query(sql, "pokemons", f"未找到人物拥有的宝可梦: {person}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_owners(pokemon: str):
            """查询拥有某个宝可梦的人物"""
            sql = f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE p.name = '{pokemon}' RETURN COLLECT(per.name) AS owners;"
            return execute_query(sql, "owners", f"未找到宝可梦的拥有者: {pokemon}")

        # 城镇与宝可梦关系查询
        @tool(args_schema=TownQuery)
        def get_town_pokemons(town: str):
            """查询城镇有哪些宝可梦"""
            sql = f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE t.name = '{town}' RETURN COLLECT(p.name) AS pokemons;"
            return execute_query(sql, "pokemons", f"未找到城镇的宝可梦: {town}")

        @tool(args_schema=PokemonQuery)
        def get_pokemon_towns(pokemon: str):
            """查询哪些城镇有某个宝可梦"""
            sql = f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE p.name = '{pokemon}' RETURN COLLECT(t.name) AS towns;"
            return execute_query(sql, "towns", f"未找到宝可梦出现的城镇: {pokemon}")

        # 城镇与人物关系查询
        @tool(args_schema=TownQuery)
        def get_town_people(town: str):
            """查询城镇有哪些人物"""
            sql = f"MATCH (t:Town)-[:has_celebrity]->(per:Person) WHERE t.name = '{town}' RETURN COLLECT(per.name) AS people;"
            return execute_query(sql, "people", f"未找到城镇人物: {town}")

        @tool(args_schema=PersonQuery)
        def get_person_town(person: str):
            """查询人物来自哪个城镇"""
            sql = f"MATCH (per:Person)-[:come_from]->(t:Town) WHERE per.name = '{person}' RETURN t.name AS town;"
            return execute_query(sql, "town", f"未找到人物所在城镇: {person}")

        # 统计查询
        @tool(args_schema=RegionQuery)
        def count_region_towns(region: str):
            """查询某个地区有多少城镇"""
            sql = f"MATCH (r:Region)<-[:located_in]-(t:Town) WHERE r.name = '{region}' RETURN COUNT(t) AS count;"
            return execute_query(sql, "count", f"未找到地区城镇数量: {region}")

        @tool(args_schema=TownQuery)
        def count_town_pokemons(town: str):
            """查询某个城镇有多少宝可梦"""
            sql = f"MATCH (t:Town)-[:location_pokemon]->(p:Pokémon) WHERE t.name = '{town}' RETURN COUNT(p) AS count;"
            return execute_query(sql, "count", f"未找到城镇宝可梦数量: {town}")

        @tool(args_schema=PersonQuery)
        def count_person_pokemons(person: str):
            """查询人物拥有多少宝可梦"""
            sql = f"MATCH (per:Person)-[:has_pokemon]->(p:Pokémon) WHERE per.name = '{person}' RETURN COUNT(p) AS count;"
            return execute_query(sql, "count", f"未找到人物宝可梦数量: {person}")

        @tool(args_schema=PokemonQuery)
        def count_pokemon_types(pokemon: str):
            """查询宝可梦有多少种属性"""
            sql = f"MATCH (p:Pokémon)-[:has_type]->(i:identity) WHERE p.name = '{pokemon}' RETURN COUNT(i) AS count;"
            return execute_query(sql, "count", f"未找到宝可梦属性数量: {pokemon}")

        # 属性相关查询
        @tool(args_schema=IdentityQuery)
        def get_pokemons_by_type(identity: str):
            """查询某个属性的宝可梦有哪些"""
            sql = f"MATCH (p:Pokémon)-[:has_type]->(i:identity) WHERE i.name = '{identity}' RETURN COLLECT(p.name) AS pokemons;"
            return execute_query(sql, "pokemons", f"未找到该属性的宝可梦: {identity}")

        # 实体匹配
        @tool(args_schema=Entity)
        def get_entity(question: str):
            """你必须调用这个工具，且只调用一次，对于用户的输入进行实体匹配，且后续查询的参数需在返回的实体中选择"""
            return self.ner.ner(question)

        def execute_query(sql: str, result_key: str, not_found_msg: str):
            """执行Neo4j查询并返回格式化结果"""
            try:
                result = g.run(sql).data()
                if result:
                    # 处理单条结果和列表结果
                    if isinstance(result[0].get(result_key), list):
                        return {result_key: result[0][result_key]}
                    return {result_key: result[0][result_key]}
                return {"message": not_found_msg}
            except Exception as e:
                return {"error": f"查询失败: {str(e)}", "sql": sql}

        return [
            get_pokemon_chinese_name,
            get_pokemon_english_name,
            get_pokemon_ability,
            get_pokemon_hidden_ability,
            get_pokemon_height,
            get_pokemon_weight,
            get_pokemon_evolution_level,
            get_pokemon_attributes,
            get_pokemon_evolution,
            get_pokemon_types,
            get_person_gender,
            get_person_english_name,
            get_person_japanese_name,
            get_person_challengers,
            get_person_partners,
            get_person_enemies,
            get_person_relatives,
            get_town_region,
            get_region_towns,
            get_person_hometown,
            get_region_people,
            get_person_pokemons,
            get_pokemon_owners,
            get_town_pokemons,
            get_pokemon_towns,
            get_town_people,
            get_person_town,
            count_region_towns,
            count_town_pokemons,
            count_person_pokemons,
            count_pokemon_types,
            get_pokemons_by_type,
            get_entity
        ]

    def _execute_query(self, sql: str, result_key: str, not_found_msg: str) -> Dict:
        """执行Neo4j查询"""
        try:
            result = g.run(sql).data()
            if result:
                return {result_key: result[0][result_key]}
            return {"message": not_found_msg}
        except Exception as e:
            return {"error": f"查询失败: {str(e)}"}


# 使用示例
if __name__ == "__main__":
    agent = KGQueryAgent()

    # 示例查询
    result = agent.query("拥有皮卡丘的角色中，有哪些是赤红的伙伴？")
    print(result)
