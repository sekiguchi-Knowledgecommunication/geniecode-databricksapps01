"""
マルチエージェントバックエンド

このモジュールは、SQLAnalysisAgent、RAGAgent、LangGraphルーターを提供します。
Gradioアプリ（app.py）から呼び出されます。
"""

import os
from typing import TypedDict, Annotated, Sequence, Generator, Dict, Any
from operator import add
from dataclasses import dataclass

from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks import sql

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatDatabricks
from langchain_core.prompts import ChatPromptTemplate


@dataclass
class AgentResponse:
    """エージェントからの構造化レスポンス"""
    route: str
    sql_result: str = ""
    rag_result: str = ""
    final_answer: str = ""


class AgentState(TypedDict):
    """LangGraphの状態定義"""
    question: str
    route: str
    sql_result: str
    rag_result: str
    final_answer: str
    messages: Annotated[Sequence[BaseMessage], add]


class SQLAnalysisAgent:
    """構造化データのSQL分析エージェント"""
    
    def __init__(self, catalog: str, schema: str, warehouse_id: str, llm_endpoint: str):
        self.catalog = catalog
        self.schema = schema
        self.warehouse_id = warehouse_id
        self.llm = ChatDatabricks(endpoint=llm_endpoint, temperature=0.1, max_tokens=2000)
    
    def _execute_sql(self, sql_query: str):
        """SQL Warehouseでクエリを実行"""
        # Databricks Apps環境では、認証情報が自動的に環境変数から取得される
        with sql.connect(
            http_path=f"/sql/1.0/warehouses/{self.warehouse_id}"
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                return cursor.fetchall(), [desc[0] for desc in cursor.description]
    
    def analyze(self, question: str) -> str:
        schema_info = f"""
利用可能なテーブル:
1. {self.catalog}.{self.schema}.survey_responses
   列: response_id, age, gender, region, purchase_freq_per_month, nps_score, monthly_spend_jpy, survey_date
2. {self.catalog}.{self.schema}.category_preferences
   列: response_id, category, purchase_freq_category, primary_usage_scene, affinity_score
3. {self.catalog}.{self.schema}.cross_purchase_matrix
   列: response_id, category_1, category_2, copurchase_freq_per_month, copurchase_reason, purchase_channel
"""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """あなたはSQL専門家です。{schema_info}

SELECT文のみを生成し、LIMIT句を含めてください。"""),
            ("user", "質問: {question}")
        ])
        
        chain = prompt | self.llm
        response = chain.invoke({"schema_info": schema_info, "question": question})
        sql_query = response.content.strip().replace("```sql", "").replace("```", "").strip()
        
        try:
            rows, headers = self._execute_sql(sql_query)
            
            if len(rows) == 0:
                return "結果が0件でした。"
            
            result_text = f"📊 分析結果（上位{len(rows)}件）:\n\n"
            result_text += " | ".join(headers) + "\n"
            result_text += "-" * 60 + "\n"
            
            for row in rows[:20]:
                result_text += " | ".join([str(val) for val in row]) + "\n"
            
            result_text += f"\n📈 総レコード数: {len(rows):,}\n"
            
            return result_text
        except Exception as e:
            return f"❌ SQLエラー: {str(e)}"
    
    def format_response(self, question: str, sql_result: str) -> str:
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """SQL分析結果を日本語でわかりやすく要約してください。マーケティング示唆を含めて3-5段落で。"""),
            ("user", "質問: {question}\n\n結果:\n{sql_result}")
        ])
        
        chain = summary_prompt | self.llm
        response = chain.invoke({"question": question, "sql_result": sql_result})
        return response.content


class RAGAgent:
    """Vector Search RAGエージェント"""
    
    def __init__(self, index_name: str, llm_endpoint: str, workspace_client: WorkspaceClient):
        self.index_name = index_name
        self.workspace_client = workspace_client
        
        # WorkspaceClientを使ってVector Searchクライアントを初期化
        self.vsc = VectorSearchClient(
            workspace_client=workspace_client,
            disable_notice=True
        )
        self.index = self.vsc.get_index(index_name=index_name)
        self.llm = ChatDatabricks(endpoint=llm_endpoint, temperature=0.3, max_tokens=2000)
    
    def search(self, question: str, num_results: int = 10) -> str:
        try:
            results = self.index.similarity_search(
                query_text=question,
                columns=["feedback_type", "category", "brand", "feedback_text", "sentiment"],
                num_results=num_results
            )
            
            docs = results.get('result', {}).get('data_array', [])
            
            if len(docs) == 0:
                return "関連フィードバックが見つかりませんでした。"
            
            context_text = "📝 関連する顧客フィードバック:\n\n"
            
            for i, doc in enumerate(docs, 1):
                feedback_type = doc[0] if len(doc) > 0 else "不明"
                category = doc[1] if len(doc) > 1 else "不明"
                brand = doc[2] if len(doc) > 2 else "不明"
                feedback_text = doc[3] if len(doc) > 3 else "不明"
                sentiment = doc[4] if len(doc) > 4 else "不明"
                
                context_text += f"{i}. [{feedback_type}] {feedback_text}\n"
                context_text += f"   カテゴリ: {category} | ブランド: {brand} | 感情: {sentiment}\n\n"
            
            return context_text
        except Exception as e:
            return f"❌ Vector Searchエラー: {str(e)}"
    
    def analyze(self, question: str) -> str:
        search_results = self.search(question, num_results=15)
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """顧客フィードバックを分析し、マーケティング示唆を提供してください。3-5段落で。"""),
            ("user", "質問: {question}\n\n{search_results}")
        ])
        
        chain = summary_prompt | self.llm
        response = chain.invoke({"question": question, "search_results": search_results})
        return response.content


def initialize_agents(catalog: str, schema: str, warehouse_id: str, llm_endpoint: str) -> Dict[str, Any]:
    """エージェントシステムを初期化"""
    
    # WorkspaceClientを作成（アプリの認証情報を自動取得）
    workspace_client = WorkspaceClient()
    
    # SQL分析エージェント
    sql_agent = SQLAnalysisAgent(catalog, schema, warehouse_id, llm_endpoint)
    
    # RAGエージェント（WorkspaceClientを渡す）
    index_name = f"{catalog}.{schema}.confectionery_feedback_index"
    rag_agent = RAGAgent(index_name, llm_endpoint, workspace_client)
    
    # LangGraphルーター構築
    def route_question(state: AgentState) -> AgentState:
        router_llm = ChatDatabricks(endpoint=llm_endpoint, temperature=0.1, max_tokens=500)
        router_prompt = ChatPromptTemplate.from_messages([
            ("system", """質問を分析してルーティング: 
1. "sql" - 数値・統計
2. "rag" - 意見・感想
3. "both" - ペルソナ・施策

1つのみ返してください。"""),
            ("user", "質問: {question}")
        ])
        
        chain = router_prompt | router_llm
        response = chain.invoke({"question": state["question"]})
        route = response.content.strip().lower()
        
        if route not in ["sql", "rag", "both"]:
            route = "both"
        
        state["route"] = route
        state["messages"] = [HumanMessage(content=state["question"])]
        return state
    
    def sql_analysis_node(state: AgentState) -> AgentState:
        sql_result = sql_agent.analyze(state["question"])
        formatted_result = sql_agent.format_response(state["question"], sql_result)
        state["sql_result"] = formatted_result
        state["messages"].append(AIMessage(content=f"[SQL完了]\n{formatted_result}"))
        return state
    
    def rag_analysis_node(state: AgentState) -> AgentState:
        rag_result = rag_agent.analyze(state["question"])
        state["rag_result"] = rag_result
        state["messages"].append(AIMessage(content=f"[RAG完了]\n{rag_result}"))
        return state
    
    def synthesize_node(state: AgentState) -> AgentState:
        synthesizer_llm = ChatDatabricks(endpoint=llm_endpoint, temperature=0.3, max_tokens=3000)
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", """SQL分析とRAG分析を統合し、包括的な回答を提供してください。5-8段落で。"""),
            ("user", "質問: {question}\n\nSQL: {sql_result}\n\nRAG: {rag_result}")
        ])
        
        chain = synthesis_prompt | synthesizer_llm
        response = chain.invoke({
            "question": state["question"],
            "sql_result": state.get("sql_result", ""),
            "rag_result": state.get("rag_result", "")
        })
        
        state["final_answer"] = response.content
        state["messages"].append(AIMessage(content=f"[統合完了]\n{response.content}"))
        return state
    
    def should_use_sql(state: AgentState) -> str:
        return "sql" if state["route"] in ["sql", "both"] else "skip_sql"
    
    def should_use_rag(state: AgentState) -> str:
        return "rag" if state["route"] in ["rag", "both"] else "skip_rag"
    
    def needs_synthesis(state: AgentState) -> str:
        if state["route"] == "both":
            return "synthesize"
        elif state["route"] == "sql":
            state["final_answer"] = state.get("sql_result", "")
            return "end"
        else:
            state["final_answer"] = state.get("rag_result", "")
            return "end"
    
    # グラフ構築
    workflow = StateGraph(AgentState)
    workflow.add_node("router", route_question)
    workflow.add_node("sql_analysis", sql_analysis_node)
    workflow.add_node("rag_analysis", rag_analysis_node)
    workflow.add_node("synthesize", synthesize_node)
    
    workflow.set_entry_point("router")
    workflow.add_conditional_edges("router", should_use_sql, {"sql": "sql_analysis", "skip_sql": "rag_analysis"})
    workflow.add_conditional_edges("sql_analysis", should_use_rag, {"rag": "rag_analysis", "skip_rag": "synthesize"})
    workflow.add_conditional_edges("rag_analysis", needs_synthesis, {"synthesize": "synthesize", "end": END})
    workflow.add_edge("synthesize", END)
    
    multi_agent_app = workflow.compile()
    
    return {
        "app": multi_agent_app,
        "sql_agent": sql_agent,
        "rag_agent": rag_agent
    }


def process_question(agent_system: Dict[str, Any], question: str) -> Generator[Any, None, None]:
    """質問を処理してストリーミングで結果を返す"""
    
    yield f"🎯 ルーティング分析中..."
    
    result = agent_system["app"].invoke({
        "question": question,
        "route": "",
        "sql_result": "",
        "rag_result": "",
        "final_answer": "",
        "messages": []
    })
    
    # 構造化レスポンスを返す
    response = AgentResponse(
        route=result.get("route", "both"),
        sql_result=result.get("sql_result", ""),
        rag_result=result.get("rag_result", ""),
        final_answer=result.get("final_answer", "")
    )
    
    yield response


def get_demo_questions() -> Dict[str, list]:
    """デモ質問を取得"""
    return {
        "pattern1": [
            {"label": "ヘビーユーザー特徴", "question": "ハイチュウを頻繁に購入しているヘビーユーザーの特徴を教えてください"},
            {"label": "味の不満点", "question": "ハイチュウの味について、顧客からの不満点や改善要望をまとめてください"},
            {"label": "30代女性ペルソナ", "question": "ハイチュウのターゲットとして、30代女性向けの詳細なペルソナを作成してください"}
        ],
        "pattern2": [
            {"label": "クロスセル機会", "question": "チョコレート購入者の他カテゴリ併買パターンを分析してください"},
            {"label": "健康志向層", "question": "健康志向の顧客層の特徴と意見を教えてください"},
            {"label": "若年層施策", "question": "20代の若年層向けのクロスカテゴリ施策を提案してください"}
        ]
    }
