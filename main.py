# main.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
from loader import load_documents  # 使用改进后的 loader 模块
from config import config
import logging

# ========================
# 模块初始化配置
# 配置日志输出
# ========================
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 加载预训练的嵌入模型（Embedding Model）
embedding_model_path = os.path.abspath(config.EMBEDDING_MODEL)
embedding_model = SentenceTransformer(embedding_model_path)


def rebuild_index():
    """重新加载所有文档，并重建 FAISS 索引"""
    print("🔄 开始重建 FAISS 索引...")

    # 使用配置中的知识库目录
    knowledge_dir = config.REFERENCE_FOLDER  # 这里改为 `REFERENCE_FOLDER`

    # 通过 loader 并发加载所有文档
    docs = list(load_documents(knowledge_dir))
    if not docs:
        print("⚠️ 没有找到文档，索引未更新。")
        return "⚠️ 没有找到文档，索引未更新。"

    print(f"📂 加载了 {len(docs)} 个文档")  # 打印文档数量

    # 提取每篇文档的内容和文件名
    texts = [doc["content"] for doc in docs]
    filenames = [doc["filename"] for doc in docs]

    # 将文档内容转化为向量表示
    embeddings = np.array(embedding_model.encode(texts))  # 确保是 NumPy 数组

    # 使用 FAISS 建立索引
    dim = embeddings.shape[1]  # 向量维度
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # 保存 FAISS 索引和文档文件名列表
    faiss.write_index(index, str(config.FAISS_CACHE / "docs.index"))

    with open(str(config.FAISS_CACHE / "filenames.pkl"), "wb") as f:
        pickle.dump(filenames, f)

    print("✅ FAISS 索引已成功重建！")
    return "✅ FAISS 索引已成功重建！"


# 如果 `main.py` 直接运行，则自动创建索引
if __name__ == "__main__":
    rebuild_index()
