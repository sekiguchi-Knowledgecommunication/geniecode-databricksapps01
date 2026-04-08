import os
import gradio as gr
from typing import Generator
import time

from agent_backend import (
    initialize_agents,
    process_question,
    get_demo_questions,
    AgentResponse
)

agent_system = None

def initialize_app():
    global agent_system
    if agent_system is None:
        print("🚀 エージェント初期化中...")
        agent_system = initialize_agents(
            catalog="workspace",
            schema="confectionery_survey_demo",
            vector_search_endpoint="confectionery_survey_vs_endpoint",
            llm_endpoint="databricks-meta-llama-3-3-70b-instruct"
        )
        print("✅ 完了")

def chat_with_agent(message: str, history: list) -> Generator[str, None, None]:
    if not message.strip():
        yield "質問を入力してください。"
        return
    
    if agent_system is None:
        yield "⏳ 初期化中..."
        initialize_app()
    
    yield f"**質問:** {message}\n\n🎯 分析中..."
    time.sleep(0.5)
    
    response_parts = []
    for response_chunk in process_question(agent_system, message):
        if isinstance(response_chunk, AgentResponse):
            final_output = format_response(response_chunk)
            yield final_output
        else:
            response_parts.append(response_chunk)
            yield "\n\n".join(response_parts)

def format_response(response: AgentResponse) -> str:
    output = []
    
    route_emoji = {"sql": "📊", "rag": "💬", "both": "🔄"}
    emoji = route_emoji.get(response.route, "🤖")
    output.append(f"{emoji} **ルーティング:** {response.route.upper()}")
    output.append("")
    
    if response.sql_result:
        output.append("---")
        output.append("### 📊 SQL分析結果")
        output.append(response.sql_result)
        output.append("")
    
    if response.rag_result:
        output.append("---")
        output.append("### 💬 顧客の声（RAG分析）")
        output.append(response.rag_result)
        output.append("")
    
    if response.final_answer:
        output.append("---")
        output.append("### 💡 統合分析結果")
        output.append(response.final_answer)
    
    return "\n".join(output)

def create_interface():
    demo_questions = get_demo_questions()
    
    custom_css = """
    .gradio-container {
        max-width: 900px !important;
    }
    footer {visibility: hidden}
    """
    
    with gr.Blocks(title="製菓アンケート分析エージェント", css=custom_css, theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🍫 製菓メーカー顧客アンケート分析
            ## マルチエージェント分析システム
            
            このシステムは**SQL分析エージェント**（構造化データ）と**RAGエージェント**（自由記述テキスト）を
            自動的にルーティングして、包括的なマーケティング示唆を提供します。
            """
        )
        
        chatbot = gr.Chatbot(
            label="エージェント応答",
            height=500,
            show_label=True,
            show_copy_button=True
        )
        
        msg = gr.Textbox(
            label="質問を入力",
            placeholder="例: ハイチュウのヘビーユーザーの特徴を教えて",
            lines=2,
            show_label=True
        )
        
        with gr.Row():
            submit_btn = gr.Button("送信", variant="primary", scale=2)
            clear_btn = gr.Button("クリア", scale=1)
        
        gr.Markdown("### 📋 デモ質問サンプル")
        gr.Markdown("クリックすると質問が入力されます：")
        
        with gr.Accordion("パターン1: ブランド特化型", open=False):
            with gr.Row():
                for q in demo_questions["pattern1"]:
                    gr.Button(q["label"]).click(
                        lambda x=q["question"]: x,
                        outputs=msg
                    )
        
        with gr.Accordion("パターン2: カテゴリ横断型", open=False):
            with gr.Row():
                for q in demo_questions["pattern2"]:
                    gr.Button(q["label"]).click(
                        lambda x=q["question"]: x,
                        outputs=msg
                    )
        
        with gr.Accordion("ℹ️ システム情報", open=False):
            gr.Markdown(
                """
                **データソース:**
                - 構造化データ: workspace.confectionery_survey_demo (3テーブル、68,358レコード)
                - テキストデータ: Vector Search Index (4,800フィードバック)
                
                **エージェント:**
                1. **SQL分析エージェント**: 年齢・性別・地域・購買頻度・NPSなどの統計分析
                2. **RAGエージェント**: 顧客の好きな点・改善要望・利用シーンの抽出
                3. **ルーター**: 質問に応じて最適なエージェントを自動選択
                
                **モデル:** Databricks Meta Llama 3.3 70B Instruct
                """
            )
        
        def respond(message, chat_history):
            bot_message = ""
            for chunk in chat_with_agent(message, chat_history):
                bot_message = chunk
                yield chat_history + [[message, bot_message]]
        
        msg.submit(respond, [msg, chatbot], [chatbot]).then(
            lambda: "", None, [msg]
        )
        submit_btn.click(respond, [msg, chatbot], [chatbot]).then(
            lambda: "", None, [msg]
        )
        clear_btn.click(lambda: None, None, [chatbot], queue=False)
    
    return demo

if __name__ == "__main__":
    initialize_app()
    app = create_interface()
    port = int(os.getenv("DATABRICKS_APP_PORT", "8000"))
    
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False
    )
