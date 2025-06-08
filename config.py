# config.py
from pathlib import Path
import os

# 基础目录，默认为当前文件所在目录
BASE_DIR = Path(__file__).parent


class Config:
    # 嵌入模型路径：你可以替换成更适合中文的模型，例如 GanymedeNil/text2vec-base-chinese
    EMBEDDING_MODEL = str(BASE_DIR / "model" / "BAAI" / "bge-m3")

    # 默认模型相关参数
    DEFAULT_MAX_LENGTH = 4096
    CHUNK_SIZE = 1000
    OVERLAP = 200

    # FAISS 索引缓存目录
    FAISS_CACHE = BASE_DIR / "cache" / "faiss_index"

    # 可用的 LLM 模型路径，存放在 MODEL_PATHS 字典中
    MODEL_PATHS = {
        'DeepSeek-R1-1.5B': os.path.abspath(BASE_DIR / "model" / "DeepSeek-R1-Distill-Qwen-1.5B"),
        'ChatGLM3-6B': os.path.abspath(BASE_DIR / "model" / "ChatGlm3-6B")
    }

    # 默认使用的 LLM 模型（可以在此更改为其他键）
    DEFAULT_LLM_MODEL = 'DeepSeek-R1-1.5B'
    # 从 MODEL_PATHS 中取出默认模型路径，并转换为字符串
    LLM_MODEL_PATH = str(MODEL_PATHS[DEFAULT_LLM_MODEL])

    # 参考文档文件夹，用于存放系统的身份设定、常见文档等
    REFERENCE_FOLDER = BASE_DIR / "knowledge_base"
    # 确保目录存在
    REFERENCE_FOLDER.mkdir(parents=True, exist_ok=True)
    # 系统身份设定文件（例如 identity.md），放在参考文档文件夹中
    IDENTITY_FILE = REFERENCE_FOLDER / "identity.md"

    # 其他参数设置
    MAX_HISTORY = 5
    STREAM_SEGMENT_SIZE = 5
    STREAM_DELAY = 0.1
    ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "Maddie")

    def __init__(self):
        # 确保缓存目录和参考文档文件夹存在
        self.FAISS_CACHE.mkdir(parents=True, exist_ok=True)
        self.REFERENCE_FOLDER.mkdir(parents=True, exist_ok=True)


# 实例化配置对象，后续代码中直接导入 cfg 使用
config = Config()
