#!/bin/bash

# PSM分析ツール（AI版）起動スクリプト

echo "=================================="
echo "PSM分析ツール（AI版）を起動します"
echo "=================================="
echo ""

# 環境変数の読み込み
if [ -f .env ]; then
    echo "✅ .env ファイルを読み込みました"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  .env ファイルが見つかりません"
    echo "   必要に応じて env.template をコピーして .env を作成してください"
fi

echo ""

# 依存パッケージのチェック
echo "📦 依存パッケージを確認中..."
if ! pip show streamlit &> /dev/null; then
    echo "❌ Streamlitがインストールされていません"
    echo "   以下のコマンドでインストールしてください："
    echo "   pip install -r requirements_ai.txt"
    exit 1
fi

echo "✅ 依存パッケージOK"
echo ""

# アプリケーション起動
echo "🚀 アプリケーションを起動します..."
echo "   ブラウザで http://localhost:8501 が開きます"
echo ""
echo "   終了するには Ctrl+C を押してください"
echo ""

streamlit run psm_analyzer_ai.py --server.port 8501 --server.headless false
