以下是依照你提供的風格，為 **Autogen 圖片檢索 Demo** 撰寫的正式版 `README.md`：

---

# Autogen x Image Retrieval Demo

本專案示範如何使用 **Microsoft Autogen** 建立一個多代理（multi-agent）系統，
結合 **BM25** 與 **Dense Retrieval 向量檢索**，進行圖片描述資料的語意搜尋與自動問答。

* Microsoft Autogen: [https://github.com/microsoft/autogen](https://github.com/microsoft/autogen)
* Ollama (本地 LLM 推理): [https://ollama.ai/](https://ollama.ai/)

---

### 環境設定 與 資料集準備

#### 建立 Conda 環境

```
conda create -n autogen-demo python=3.11 -y
conda activate autogen-demo
pip install -r requirements.txt
```

#### 安裝 LLM 模型（使用本地 Ollama）

Ollama ([https://ollama.ai/](https://ollama.ai/))

```
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3:14b
ollama pull bge-m3
```

> `qwen3:14b` 用於代理對話
> `bge-m3` 用於文字向量生成（Dense Retrieval）

#### 準備資料集

請先將原始資料 `img_index.csv` 放入 `data/` 資料夾下。
其內容應包含兩個欄位：

| img_filename | descriptions |
| ------------ | ------------ |
| img001.jpg   | 預製數據中心的建設流程  |
| img002.jpg   | 雲端伺服器佈線架構    |

#### 建立檢索資料庫

建立 BM25 Token Corpus

```
python preprocess_p1_bm25_corpus.py
```

建立 Dense 向量資料庫

```
python preprocess_p1_vector_corpus.py
```

若您已有 `.parquet` 檔案，可直接放入 `data/` 目錄，無需重新生成。
範例資料結構：

```
├── data
│   ├── embeddings_ollama.parquet
│   └── image_descriptions_bm25.parquet
```

---

### 🤖 Autogen 多代理運行流程

運行主程式：

```
python autogen_demo.py
```

系統將啟動四個代理（agents）：

| 代理名稱                | 職責                                      |
| ------------------- | --------------------------------------- |
| **planner**         | 任務規劃與協調代理流程                             |
| **query_reasoning** | 改寫與優化查詢語句                               |
| **search_agent**    | 呼叫 `search_bm25` 或 `search_vector` 進行檢索 |
| **writer**          | 根據檢索結果生成最終回答                            |

#### 自動代理對話流程

```
使用者輸入查詢
   │
   ▼
query_reasoning → 改寫查詢
   │
   ▼
search_agent → 檢索 (優先 BM25，可補 vector)
   │
   ▼
writer → 撰寫最終回答
   │
   ▼
TERMINATE → 流程結束
```
#### 搜尋工具模組

`search_tool.py`
負責整合兩種檢索演算法：

| 函式                           | 功能說明                   |
| ---------------------------- | ---------------------- |
| `local_search_bm25(query)`   | 使用 BM25 演算法進行關鍵字檢索     |
| `local_search_vector(query)` | 使用 `bge-m3` 向量模型進行語意檢索 |

可同時載入 `.parquet` 檔案進行快速查詢。

#### 🧩 範例輸入與輸出

執行：

```
python autogen_demo.py
```

輸入：

```
Query：預製數據中心的建設流程
```

輸出（範例）：

```
[planner] → 呼叫 query_reasoning 改寫查詢
[query_reasoning] → 生成優化查詢：「預製數據中心 建設 流程」
[search_agent] → 呼叫 search_bm25 檢索
[search_agent] → 呼叫 search_vector 補充語意資料
[writer] → 撰寫最終回覆
TERMINATE
```

---

### 架構

```
.
├── autogen_demo.py                 # 主程式：多代理自動協作
├── search_tool.py                  # 檢索工具（BM25 + 向量搜尋）
├── preprocess_p1_bm25_corpus.py    # 建立 BM25 corpus
├── preprocess_p1_vector_corpus.py  # 建立向量資料庫
├── data/
│   ├── img_index.csv
│   ├── embeddings_ollama.parquet
│   └── image_descriptions_bm25.parquet
├── requirements.txt
└── readme.md
```

### Reference

* [Microsoft Autogen](https://github.com/microsoft/autogen)
* [Ollama](https://ollama.ai/)
* [rank-bm25](https://pypi.org/project/rank-bm25/)
* [CKIP Transformers](https://github.com/ckiplab/ckip-transformers)

---

更新日期：2025-10-06
