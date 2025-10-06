import os
import re
import pandas as pd
from rank_bm25 import BM25Okapi
from opencc import OpenCC
from ckip_transformers.nlp import CkipWordSegmenter
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# 初始化 tqdm
tqdm.pandas()
# 繁簡轉換、斷詞模型
cc = OpenCC('s2t')
ws_driver = CkipWordSegmenter(model="bert-base")

# 準備停用詞
stop_words_simplified = set(stopwords.words('chinese'))
stop_words_traditional = set([cc.convert(word) for word in stop_words_simplified])
stop_words_all = list(stop_words_traditional.union(set(stopwords.words('english'))))

def remove_punctuation(text):
    """移除中英文標點符號"""
    pattern = r'[^\w\s]'  # 非字母、數字、下劃線、空格
    cleaned_text = re.sub(pattern, ' ', text)
    while '  ' in cleaned_text:
        cleaned_text = cleaned_text.replace('  ', ' ')
    return cleaned_text

def preprocess(text):
    """文字清理 + 斷詞 + 去除停用詞"""
    tokens = remove_punctuation(text)
    tokens = ws_driver([tokens])
    res = []
    for word_ws in tokens[0]:
        if word_ws not in stop_words_all and word_ws.strip():
            res.append(word_ws)
    return res


# === 設定檔案名稱 ===
csv_file = 'data/img_index.csv'
parquet_file = 'data/image_descriptions_bm25.parquet'

if os.path.exists(parquet_file):
    print(f"Load Parquet：{parquet_file}")
    df = pd.read_parquet(parquet_file)
    corpus = df['tokens'].tolist()
else:
    print(f"Make Corpus")
    df = pd.read_csv(csv_file)
    df['descriptions'] = df['descriptions'].fillna("")

    df['tokens'] = df['descriptions'].progress_apply(preprocess)
    corpus = df['tokens'].tolist()

    df.to_parquet(parquet_file, index=False)
    print(f"done {parquet_file}")

print(f"load {len(corpus)} data。")

# bm25 = BM25Okapi(corpus)
# def search(query, top_k=3):
#     query_tokens = preprocess(query)
#     scores = bm25.get_scores(query_tokens)
#     ranked_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    
#     print(f"\n query: {query}\n")
#     for idx, score in ranked_results[:top_k]:
#         # print(f"[Score: {score:.4f}] file: {df['img_filename'][idx]} - {df['descriptions'][idx]}")
#         print(f"[Score: {score:.4f}] file: {df['img_filename'][idx]}")

# # 測試查詢
# search("預製數據中心的建設流程")
