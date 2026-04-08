# 製菓メーカー顧客アンケート分析 - マルチエージェントApp

## 📋 概要

このアプリケーションは、Databricks上でマルチエージェントシステムを使用して
顧客アンケートデータを分析する対話型インターフェースです。

### 機能

- **SQL分析エージェント**: 構造化データ（年齢・性別・地域・購買パターン等）の統計分析
- **RAGエージェント**: Vector Searchを使用した自由記述テキストの検索・要約
- **ルーター**: 質問内容に応じて最適なエージェントを自動選択

### データソース

- **構造化データ**: `workspace.confectionery_survey_demo` (3テーブル)
  - survey_responses: 8,000回答者
  - category_preferences: 25,987レコード
  - cross_purchase_matrix: 34,371併買パターン

- **テキストデータ**: Vector Search Index (4,800フィードバック)
  - Index: `workspace.confectionery_survey_demo.confectionery_feedback_index`
  - Endpoint: `confectionery_survey_vs_endpoint`

## 🚀 デプロイ方法

### 前提条件

1. ✅ Vector Search Endpointが作成済み
2. ✅ Vector Search Indexが同期完了済み
3. ✅ Databricks CLI がインストール済み

### ステップ1: ファイルのダウンロード

1. Databricks UIでVolume `/Volumes/workspace/confectionery_survey_demo/survey_documents/app_deploy` を開く
2. 以下のファイルをローカルにダウンロード:
   - `app.py`
   - `agent_backend.py`
   - `requirements.txt`
   - `app.yaml`

### ステップ2: Databricks Appの作成

ローカルのターミナルで以下を実行:

```bash
# ダウンロードしたファイルがあるディレクトリに移動
cd /path/to/downloaded/files

# Databricks Appを作成
databricks apps create confectionery-survey-agent --source-code-path .
```

### ステップ3: 動作確認

1. Databricks UIの左サイドバーから **Apps** を選択
2. `confectionery-survey-agent` を開く
3. アプリが起動するまで数分待機
4. サンプル質問ボタンをクリックして動作確認

## 📝 使用例

### パターン1: ブランド特化型分析

- 「ハイチュウのヘビーユーザーの特徴を教えて」
- 「味についての不満点をまとめて」
- 「30代女性向けのペルソナを作成」

### パターン2: カテゴリ横断型分析

- 「チョコレート購入者の他カテゴリ併買パターンを分析して」
- 「健康志向層の特徴と意見を教えて」
- 「20代若年層向けのクロスカテゴリ施策を提案して」

## 🔧 トラブルシューティング

### アプリが起動しない

```bash
# ログを確認
databricks apps logs confectionery-survey-agent
```

### Vector Search接続エラー

- Vector Search Endpointが`ONLINE`状態か確認
- Vector Search Indexが同期完了しているか確認

### LLMエンドポイントエラー

- `databricks-meta-llama-3-3-70b-instruct` エンドポイントが利用可能か確認
- 別のエンドポイントを使用する場合は `agent_backend.py` の `llm_endpoint` を変更

## 📚 技術スタック

- **UI Framework**: Gradio 4.44+
- **Agent Orchestration**: LangGraph
- **Vector Search**: Databricks Vector Search
- **LLM**: Databricks Meta Llama 3.3 70B Instruct
- **Data Processing**: PySpark

## 🎯 次のステップ

1. **カスタマイズ**: デモ質問を追加・変更
2. **拡張**: 新しいエージェント（価格最適化、プロモーション設計等）を追加
3. **統合**: 他のデータソース（POS、CRM等）との連携

---

生成日: 2026-04-08 14:35:16
