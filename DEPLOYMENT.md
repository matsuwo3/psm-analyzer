# デプロイメント情報

## 🌐 デプロイ先

### Streamlit Cloud
- **URL**: https://share.streamlit.io/
- **アプリURL**: `https://[your-app-url].streamlit.app/`
  - ダッシュボードから確認してください

### GitHubリポジトリ
- **リポジトリURL**: https://github.com/matsuwo3/psm-analyzer
- **ユーザー名**: matsuwo3
- **リポジトリ名**: psm-analyzer

---

## ⚙️ Streamlit Cloud設定

### デプロイ設定
- **Repository**: `matsuwo3/psm-analyzer`
- **Branch**: `main`
- **Main file path**: `psm_analyzer_ai.py`

### Secrets設定（重要）
Streamlit Cloud > Settings > Secrets に以下を設定：

```toml
ANTHROPIC_API_KEY = "your-api-key-here"
```

**注意**: 実際のAPIキーは別途管理してください（`.env`ファイルまたはパスワードマネージャーに保存）

---

## 🔐 認証情報

### アプリパスワード
- **パスワード**: `matsuo1234`
- **設定場所**: `psm_analyzer_ai.py` の `APP_PASSWORD` 変数

### Claude API
- **APIキー**: Streamlit Cloud Secretsに設定済み
- **管理画面**: https://console.anthropic.com/

---

## 🔄 更新手順

### コードを更新してデプロイ

```bash
# 変更をコミット
git add .
git commit -m "更新内容"

# GitHubにプッシュ
git push

# Streamlit Cloudが自動的に再デプロイ（2-3分）
```

### 手動で再デプロイ

1. Streamlit Cloudダッシュボードを開く
2. アプリの「⋮」メニュー → Settings
3. 「Reboot app」をクリック

---

## 📝 重要なファイル

- `psm_analyzer_ai.py` - メインアプリケーション
- `requirements.txt` - 依存パッケージ（Streamlit Cloud用）
- `requirements_ai.txt` - 依存パッケージ（ローカル用）
- `.env` - ローカル環境変数（Gitにコミットされない）
- `sample_data.csv` - サンプルデータ

---

## 🚨 トラブルシューティング

### アプリが起動しない
1. Streamlit Cloud > Manage app > Logs を確認
2. Secrets設定を確認
3. requirements.txt が最新か確認

### APIキーエラー
1. Streamlit Cloud > Settings > Secrets を確認
2. APIキーが正しく設定されているか確認
3. Anthropic Console でクレジット残高を確認

### デプロイが反映されない
1. GitHubに正しくプッシュされているか確認
2. Streamlit Cloudで手動リブート

---

## 📊 使用状況

### Streamlit Cloud
- **プラン**: Community（無料）
- **制限**:
  - 1アプリまで無料
  - 月間実行時間制限あり

### Claude API
- **モデル**: claude-sonnet-4-20250514
- **料金**: 従量課金
- **管理**: https://console.anthropic.com/settings/billing

---

## 📅 デプロイ履歴

- **初回デプロイ**: 2025年10月30日
- **最終更新**: Git履歴で確認 (`git log`)

---

## 🔗 関連リンク

- [Streamlit Cloud Dashboard](https://share.streamlit.io/)
- [GitHubリポジトリ](https://github.com/matsuwo3/psm-analyzer)
- [Anthropic Console](https://console.anthropic.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
