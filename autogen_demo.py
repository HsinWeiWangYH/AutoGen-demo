import asyncio
from datetime import datetime
from typing import Any, Dict, List, Tuple
import os
import re
import search_tool

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import Swarm
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient


# ====================================================
# 模型設定
# ====================================================
model_client = OpenAIChatCompletionClient(
    model="qwen3:14b",
    base_url="http://localhost:11434/v1",
    api_key="placeholder",
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": False,
        "family": "unknown",
    },
)


# ====================================================
# 工具（Tool Functions）
# ====================================================
async def search_bm25(query: str):
    """以 BM25 執行檢索"""
    try:
        res = search_tool.local_search_bm25(query)
        return ("bm25", res)
    except Exception as e:
        return ("bm25_error", f"BM25 搜尋發生錯誤: {str(e)}")


async def search_vector(query: str):
    """以向量模型執行檢索"""
    try:
        res = search_tool.local_search_vector(query)
        return ("vector", res)
    except Exception as e:
        return ("vector_error", f"向量搜尋發生錯誤: {str(e)}")


# ====================================================
# Agent 定義
# ====================================================

planner = AssistantAgent(
    name="planner",
    model_client=model_client,
    handoffs=["query_reasoning", "search_agent", "writer"],
    system_message="""您是一名智慧助理。  
        負責通過委派專業代理人來回答問題，代理人包括：  
        - **query_reasoning**：負責改寫與優化查詢。  
        - **search_agent**：負責根據查詢進行資料檢索。  
        - **writer**：負責根據檢索結果撰寫最終回覆。  

        ---

        ### 工作流程說明  

        1. **查詢推理階段（query_reasoning）**  
        - 任務開始時，請先將使用者的原始查詢交給 `query_reasoning`。  
        - `query_reasoning` 會負責改寫查詢，使其更具檢索效率。  
        - 當收到改寫後的查詢結果後，立即將該查詢交給 `search_agent` 進行資料檢索。  

        2. **資料檢索階段（search_agent）**  
        - `search_agent` 根據改寫查詢執行資料搜尋。  
        - 若 `search_agent` 回傳結果顯示資料不足、無法回答或為 `'unknown_search'`，請重新將改寫查詢任務交回 `query_reasoning`，要求進一步修正查詢內容。  
        - 當再次收到改寫查詢後，再次交由 `search_agent` 執行第二次檢索。  

        3. **結果撰寫階段（writer）**  
        - 當 `search_agent` 回傳的檢索結果包含足夠資料時，請將結果交給 `writer`。  
        - `writer` 負責根據搜尋內容進行分析與撰寫最終回覆。  
        - 撰寫完成後，請輸出「TERMINATE」以結束整個任務流程。  

        ---

        ### 總結流程順序  
        **query_reasoning → search_agent →（必要時回 query_reasoning 再 search_agent）→ writer → TERMINATE**
        """
    )

query_reasoning = AssistantAgent(
    name="query_reasoning",
    model_client=model_client,
    handoffs=["planner"],
    system_message=(
        "### 角色定位：查詢推理專員\n"
        "負責根據使用者輸入，將查詢改寫為更準確、可檢索的版本。\n\n"
        "### 指引\n"
        "- 不改變查詢意圖，只改善可檢索性。\n"
        "- 結果輸出格式：`改寫查詢：<新查詢>`\n"
        "- 完成後立即回傳給 `planner`。"
    ),
)

search_agent = AssistantAgent(
    name="search_agent",
    model_client=model_client,
    handoffs=["planner"],
    tools=[search_bm25, search_vector],
    system_message=(
        "### 角色定位：資料檢索專員\n"
        "您負責根據使用者查詢，從內部知識庫中檢索最相關的內容，確保結果完整且準確。\n\n"
        "### 可用工具\n"
        "您擁有兩個可使用的檢索工具：\n"
        "1. **search_bm25**：關鍵字匹配為主，適合結構化或明確的查詢。\n"
        "2. **search_vector**：語意相似度為主，適合語義較抽象或描述性查詢。\n\n"
        "### 檢索指引\n"
        "- **請優先執行 `search_bm25`** 來獲取初步結果。\n"
        "- 若您判斷 `search_bm25` 的結果不足以完整回答問題，請再執行 `search_vector` 進行補充。\n"
        "- 每個工具僅能呼叫一次。\n"
        "- 當兩者皆執行時，請合併結果後回傳。\n\n"
        "### 回傳格式\n"
        "整合後的回覆請遵循以下格式：\n"
        "```\n"
        "[資料來源結果]\n"
        "BM25結果：...\n"
        "向量結果：...\n"
        "```\n"
        "若只使用了其中一個工具，也請保留對應欄位並標示「無」或「未執行」。\n\n"
        "請將最終檢索結果交回給 `planner`，以便後續進行撰寫或決策。"
    ),
)

writer = AssistantAgent(
    name="writer",
    model_client=model_client,
    handoffs=["planner"],
    system_message=(
        "### 角色定位：報告撰寫專員\n"
        "根據使用者查詢與檢索結果，產生最終答覆。\n\n"
        "### 指引\n"
        "- 僅根據檢索內容撰寫。\n"
        "- 不可加入主觀臆測。\n"
        "- 若內容不足，請明確說明資料不足。\n"
        "- 撰寫完成後回傳結果，並包含「TERMINATE」以結束流程。"
    ),
)


# ====================================================
# 任務終止條件
# ====================================================
termination = TextMentionTermination("TERMINATE")


# ====================================================
# 建立 Team
# ====================================================
research_team = Swarm(
    participants=[planner, query_reasoning, search_agent, writer],
    termination_condition=termination,
)


# ====================================================
# 主程式（互動式）
# ====================================================
async def main():
    print("🚀 Autogen Demo 啟動中...\n")
    print("輸入問題（輸入 exit 離開）\n")
    while True:
        task = input("🧠 問題：")
        if task.lower().strip() in ["exit", "quit"]:
            print("👋 再見！")
            break
        await Console(research_team.run_stream(task=task))


if __name__ == "__main__":
    asyncio.run(main())
