# PSM分析自動レポートツール（AI版）

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Claude AI](https://img.shields.io/badge/Claude_AI-Sonnet_4.5-6B4FBB?style=for-the-badge)](https://www.anthropic.com/)

Claude APIを活用した高度なPSM（Price Sensitivity Meter）分析ツールです。CSVファイルをアップロードするだけで、PSM分析結果とAIによる深いビジネスインサイトを含む包括的なレポートを自動生成します。

## ✨ 主な機能

### 📊 自動PSM分析
- **OPP（最適価格点）**: 「高い」と「安い」の交点
- **IDP（妥協価格点）**: 「高すぎる」と「お買い得」の交点
- **PMC（最小抵抗価格点）**: 「高い」と「お買い得」の交点
- **PME（最大ストレス価格点）**: 「高すぎる」と「安い」の交点

### 🤖 AI分析（Claude Sonnet 4.5）
- 分析結果のサマリー
- 具体的な価格設定の推奨事項
- 顧客セグメント分析
- 実装のためのアクションプラン
- 追加調査の提案

### 📈 インタラクティブグラフ
- Plotlyによる美しい可視化
- 4本の累積分布曲線
- 主要指標の交点マーカー
- ズーム、パン、ホバー機能

### 📑 Excelレポート自動生成
- **AIインサイトシート**: Claude分析の全文
- **分析サマリーシート**: PSM主要指標と統計
- **カテゴリ別分析シート**: セグメント別の詳細分析
- **生データシート**: クリーニング済みデータ

### 🎯 カテゴリ別分析
- 性別、年齢、利用頻度などでセグメント分析
- 各セグメントの価格感覚の違いを可視化
- セグメント別の推奨事項を提供

## 🚀 デモ

![PSM Analysis Demo](https://via.placeholder.com/800x400?text=Demo+Screenshot)

## 📋 必要要件

- Python 3.8以上
- Anthropic Claude APIキー

## 💻 インストール

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/psm-analyzer.git
cd psm-analyzer
```

### 2. 依存パッケージのインストール

```bash
pip install -r requirements_ai.txt
```

### 3. 環境変数の設定

```bash
cp env.template .env
```

`.env`ファイルを編集してAPIキーを設定：

```
ANTHROPIC_API_KEY=sk-ant-api03-your-api-key-here
```

## 🎯 使い方

### クイックスタート

```bash
# アプリケーションを起動
streamlit run psm_analyzer_ai.py

# または起動スクリプトを使用
chmod +x run_ai_app.sh
./run_ai_app.sh
```

ブラウザで `http://localhost:8501` が自動的に開きます。

### 基本的な使用手順

1. **ログイン**
   - パスワードを入力してログイン

2. **CSVファイルをアップロード**
   - PSMアンケートデータのCSVファイルを選択
   - サンプルデータ（`sample_data.csv`）も利用可能

3. **列を選択**
   - 必須：4つの価格列を選択
     - 「安い」と感じる価格
     - 「お買い得」と感じる価格
     - 「高い」と感じる価格
     - 「高すぎる」と感じる価格
   - オプション：カテゴリ列を選択（性別、年齢など）

4. **分析を実行**
   - 「分析を実行」ボタンをクリック
   - AI分析完了まで30秒程度待機

5. **結果を確認**
   - AIインサイトを確認
   - グラフと統計を確認
   - カテゴリ別分析を確認

6. **レポートをダウンロード**
   - Excelレポートをダウンロード

## 📊 データ形式

### 必須列

CSVファイルには以下の4つの数値列が必要です：

| 列名（例） | 内容 |
|-----------|------|
| 安い価格 | この価格だと「安い」と感じる価格 |
| お買い得価格 | この価格だと「お買い得」と感じる価格 |
| 高い価格 | この価格だと「高い」と感じる価格 |
| 高すぎる価格 | この価格だと「高すぎる」と感じる価格 |

### オプション列

セグメント分析のため、以下のような列を追加できます：

- 性別（男性/女性）
- 年齢層（20代/30代/40代など）
- 利用頻度（週1回/月2-3回など）
- 地域、職業、収入層など

### サンプルデータ

```csv
ID,性別,年齢,利用頻度,安い価格,お買い得価格,高い価格,高すぎる価格
1,男性,25,週1回,3000,4000,6000,8000
2,女性,32,月2-3回,3500,4500,7000,9000
...
```

## 🛠️ 技術スタック

- **Framework**: Streamlit 1.30.0+
- **Language**: Python 3.8+
- **Data Processing**: pandas, numpy
- **Visualization**: Plotly
- **AI**: Anthropic Claude API (Sonnet 4.5)
- **Report Generation**: openpyxl

## 📁 ファイル構成

```
psm-analyzer/
├── psm_analyzer_ai.py          # メインアプリケーション
├── requirements_ai.txt         # 依存パッケージ
├── run_ai_app.sh              # 起動スクリプト
├── env.template               # 環境変数テンプレート
├── README.md                  # このファイル
├── README_AI.md               # 詳細ドキュメント
├── QUICKSTART_AI.md           # クイックスタート
├── sample_data.csv            # サンプルデータ
└── .gitignore                 # Git除外設定
```

## 💰 料金について

### Anthropic API料金

- **Claude Sonnet 4.5**: 入力 $3 / 1M tokens、出力 $15 / 1M tokens
- **1回のPSM分析**: 約$0.01-0.05（1〜5円程度）

$10のクレジットで、約200〜1000回の分析が可能です。

## 🔒 セキュリティ

- APIキーは`.env`ファイルで管理（Gitにコミットされない）
- パスワード認証機能
- アップロードデータはメモリ上のみで処理
- セッション終了時に自動破棄

## 📚 ドキュメント

- [詳細ドキュメント](README_AI.md)
- [クイックスタート](QUICKSTART_AI.md)

## 🤝 コントリビューション

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

---

**開発**: 2025年10月
**バージョン**: 1.0.0
**AI Model**: Claude Sonnet 4.5 (claude-sonnet-4-20250514)
