import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import ollama

# === 設定檔案名稱 ===
csv_file = 'data/img_index.csv'
parquet_file = 'data/embeddings_ollama.parquet'

# === 使用 Ollama 的 embedding 模型 ===
def get_embedding(text):
    """使用 Ollama 產生文字向量"""
    response = ollama.embeddings(
        model='bge-m3',
        prompt=text
    )
    return np.array(response['embedding'], dtype=np.float32)

# === 如果 Parquet 已存在就直接載入，否則重新生成 ===
if os.path.exists(parquet_file):
    print(f"Load Parquet：{parquet_file}")
    df = pd.read_parquet(parquet_file)
else:
    print(f"Make Corpus")
    df = pd.read_csv(csv_file)
    df = df[['img_filename', 'descriptions']].dropna(subset=['descriptions'])
    df['descriptions'] = df['descriptions'].fillna("")

    embeddings = []
    for text in tqdm(df['descriptions'], desc="Embedding with Ollama"):
        embeddings.append(get_embedding(text))
    
    df['embedding'] = embeddings
    df.to_parquet(parquet_file, index=False)
    print(f"done {parquet_file}")

print(f"load {len(df)} data。")

# # === 查詢功能 ===
# def cosine_similarity(a, b):
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def search(query, top_k=3):
#     print(f"\n query: {query}\n")
#     query_vec = get_embedding(query)
    
#     # 計算所有相似度
#     similarities = []
#     for emb in df['embedding']:
#         sim = cosine_similarity(query_vec, np.array(emb))
#         similarities.append(sim)
    
#     df['similarity'] = similarities
#     results = df.sort_values('similarity', ascending=False).head(top_k)
    
#     for _, row in results.iterrows():
#         # print(f"[Score: {row['similarity']:.4f}] {row['img_filename']} - {row['content_new'][:80]}...")
#         print(f"[Score: {row['similarity']:.4f}] {row['img_filename']}")
    
#     return results

# # === 測試查詢 ===
# search("預製數據中心的建設流程")
