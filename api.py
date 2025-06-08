import uvicorn
from fastapi import FastAPI, Body
from pydantic import BaseModel
from rag import generate_answer  # 复用刚才的 RAG 逻辑

# 实例化 FastAPI 应用
app = FastAPI(
    title="RAG System API",
    description="一个基于FAISS和本地LLM的问答系统",
    version="1.0.0"
)

"""
请求体的数据模型，用于接收前端/客户端发来的问题。
"""
class QueryRequest(BaseModel):
    question: str

"""
向 RAG 系统提问，并返回生成的回答。
参数：
    req (QueryRequest): 包含 question 字段的请求体。
返回：
    dict: JSON 格式的回答信息，若 question 为空则返回错误提示。
"""
@app.post("/ask")
def ask_rag(req: QueryRequest):
    question = req.question.strip()
    if not question:
        return {"error": "问题为空，请输入问题"}

    answer = generate_answer(question)
    return {"answer": answer}

"""
健康检查接口，用于确认服务是否正常运行。

返回：
    dict: 包含运行状态及提示信息的 JSON。
"""
@app.get("/")
def health_check():
    return {"status": "ok", "message": "RAG API is running"}

if __name__ == "__main__":
    # 以 uvicorn 的方式启动服务，监听地址为 127.0.0.1，端口为 8000，reload=True 表示代码变动可自动重载
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
