# search_tool.py
import re
import pandas as pd
import numpy as np
import ollama
from opencc import OpenCC
from ckip_transformers.nlp import CkipWordSegmenter
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from tqdm import tqdm

# ==============================
# 初始化工具
# ==============================
tqdm.pandas()
cc = OpenCC('s2t')
ws_driver = CkipWordSegmenter(model="bert-base")

# 停用詞準備
stop_words_simplified = set(stopwords.words('chinese'))
stop_words_traditional = set([cc.convert(word) for word in stop_words_simplified])
stop_words_all = list(stop_words_traditional.union(set(stopwords.words('english'))))

# ==============================
# 檔案設定
# ==============================
bm25_parquet = 'data/image_descriptions_bm25.parquet'
embedding_parquet = 'data/embeddings_ollama.parquet'

# ==============================
# 載入 BM25 corpus
# ==============================
print(f"載入 BM25 Parquet: {bm25_parquet}")
df_bm25 = pd.read_parquet(bm25_parquet)
bm25_corpus = df_bm25['tokens'].tolist()
bm25 = BM25Okapi(bm25_corpus)

# ==============================
# 載入 Dense 向量資料
# ==============================
print(f"載入 Embedding Parquet: {embedding_parquet}")
df_vec = pd.read_parquet(embedding_parquet)
df_vec['embedding'] = df_vec['embedding'].apply(lambda x: np.array(x, dtype=np.float32))

# ==============================
# 共用前處理函式
# ==============================
def remove_punctuation(text):
    """移除中英文標點符號"""
    pattern = r'[^\w\s]'
    cleaned_text = re.sub(pattern, ' ', text)
    while '  ' in cleaned_text:
        cleaned_text = cleaned_text.replace('  ', ' ')
    return cleaned_text


def preprocess_bm25(text):
    """文字清理 + 斷詞 + 去除停用詞"""
    tokens = remove_punctuation(text)
    tokens = ws_driver([tokens])
    res = []
    for word_ws in tokens[0]:
        if word_ws not in stop_words_all and word_ws.strip():
            res.append(word_ws)
    return res


def get_embedding(text):
    """使用 Ollama 產生文字向量"""
    response = ollama.embeddings(
        model='bge-m3',
        prompt=text
    )
    return np.array(response['embedding'], dtype=np.float32)


# ==============================
# 搜尋功能：BM25
# ==============================
def local_search_bm25(query, top_k=3):
    query_tokens = preprocess_bm25(query)
    scores = bm25.get_scores(query_tokens)
    ranked_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

    ll = "BM25搜尋結果:\n"
    for idx, score in ranked_results:
        desc = df_bm25.loc[idx, 'descriptions']
        img = df_bm25.loc[idx, 'img_filename']
        ll += f"[Score: {score:.4f}] file: {img} - {desc}...\n"
    return ll


# ==============================
# 搜尋功能：Dense 向量搜尋
# ==============================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def local_search_vector(query, top_k=3):
    query_vec = get_embedding(query)
    similarities = []

    for emb in df_vec['embedding']:
        sim = cosine_similarity(query_vec, emb)
        similarities.append(sim)

    df_vec['similarity'] = similarities
    results = df_vec.sort_values('similarity', ascending=False).head(top_k)

    ll = "Vector搜尋結果:\n"
    for _, row in results.iterrows():
        ll += f"[Score: {row['similarity']:.4f}] file: {row['img_filename']} - {row['descriptions']}...\n"
    return ll


# ==============================
# 測試區
# ==============================
if __name__ == "__main__":
    q = "預製數據中心的建設流程"
    print(local_search_bm25(q, top_k=3))
    print(local_search_vector(q, top_k=3))
