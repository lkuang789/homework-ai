# app.py
import gradio as gr
import os
import torch
import psutil  # 用于系统监控
from rag import generate_answer
from config import config
from loader import load_documents
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import json
from main import rebuild_index  # ✅ 直接导入 `rebuild_index`
import shutil
import time

# 头像路径
USER_ICON_PATH = "icon/user.png"
BOT_ICON_PATH = "icon/bot.png"

# 1️ 加载嵌入模型（SentenceTransformer）
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)


# 2️ 加载 LLM 模型
def load_llm_model(model_name):
    """动态加载 LLM"""
    model_path = str(config.MODEL_PATHS.get(model_name, config.LLM_MODEL_PATH))

    # ✅ 确保 `trust_remote_code=True`
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16,
                                                 device_map="auto")

    return tokenizer, model


# 预加载默认 LLM
current_model_name = config.DEFAULT_LLM_MODEL
tokenizer, model = load_llm_model(current_model_name)


# 3️ 处理文件上传
def upload_files(files, chatbot):
    """上传文件，更新 Chatbot 并确保返回符合 Gradio Chatbot 格式"""
    if not isinstance(files, list):
        files = [files]

    saved_files = []
    failed_files = []

    for file in files:
        try:
            original_filename = file.orig_name if hasattr(file, "orig_name") else os.path.basename(file.name)
            dest_path = os.path.join(config.REFERENCE_FOLDER, original_filename)

            # ✅ 将临时路径文件移动到 `knowledge_base` 目录
            shutil.move(file.name, dest_path)

            saved_files.append(original_filename)
        except Exception as e:
            failed_files.append(original_filename)
            print(f"❌ 上传失败: {original_filename}, 错误: {e}")

    # ✅ 只有至少有一个文件上传成功，才重建索引
    if saved_files:
        print("🔄 至少一个文件上传成功，开始重建索引...")
        index_message = rebuild_index()
    else:
        index_message = "⚠️ 所有文件上传失败，索引未更新。"

    # ✅ 构建返回消息
    message = f"📂 上传成功 {len(saved_files)} 个文件: {', '.join(saved_files)}"
    if failed_files:
        message += f"\n❌ 上传失败 {len(failed_files)} 个文件: {', '.join(failed_files)}"

    message += f"\n{index_message}"
    print(message)

    # ✅ 让 Chatbot 显示消息（格式必须是 `[(用户输入, 机器人回复)]`）
    chatbot.append(("📁 文件上传", message))
    return chatbot


# 4️ 处理对话
def chat_with_rag(question, chatbot, max_tokens, temperature, top_p, show_debug, topk_retrieval, dist_threshold):
    chatbot.append((question, ""))
    start_time = time.time()

    stream = generate_answer(question)
    bot_response = ""

    for current_output in stream:
        if "</think>" in current_output:
            parts = current_output.split("</think>")
            if len(parts) >= 2:
                bot_response = parts[1].strip()  # 模型的回答
                debug_info = parts[2].strip() if len(parts) > 2 else "暂无推理过程"
            else:
                bot_response = parts[0].strip()
                debug_info = "暂无推理过程"
        else:
            bot_response = current_output.strip()

        chatbot[-1] = (question, bot_response)
        yield "", chatbot

    # 推理时间计算
    elapsed_time = time.time() - start_time
    elapsed_str = f"🔍 点击查看推理过程，耗时 {elapsed_time:.2f} 秒 ⌄"

    # ✅ 仅对推理过程的字体进行淡化（去除对 bot_response 的额外颜色修改）
    if show_debug and debug_info:
        bot_response = (
            f"<details>"
            f"<summary style='color:#888;font-size:12px;'>{elapsed_str}</summary>"
            f"<div style='color:#ccc;background:#f5f5f5;padding:10px;border-radius:5px;'>\n\n"
            f"{debug_info}\n"
            f"</div></details>\n\n"
            f"{bot_response}"  # ✅ 去掉了 bot_response 外部的额外颜色设置
        )
        chatbot[-1] = (question, bot_response)
        yield "", chatbot


# 5️ 切换 LLM 模型
def switch_model(new_model):
    """切换 LLM 模型"""
    global tokenizer, model, current_model_name
    tokenizer, model = load_llm_model(new_model)
    current_model_name = new_model
    return f"✅ 已切换到 {new_model} 模型"


# 6️ 系统监控
def system_diagnosis():
    """返回 CPU、RAM、GPU 资源使用情况"""
    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    gpu_usage = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    return {"CPU 使用率": f"{cpu_usage}%", "内存使用率": f"{ram_usage}%", "GPU 占用": f"{gpu_usage:.2f}GB"}


# 7️ 导出 & 导入对话历史
def export_chat_history(chatbot):
    """导出对话历史为 JSON 文件"""
    history = [{"用户": msg[0], "AI": msg[1]} for msg in chatbot]
    file_path = "chat_history.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return file_path


def import_chat_history(file):
    """导入对话历史"""
    if file is None:
        return []

    with open(file.name, "r", encoding="utf-8") as f:
        history = json.load(f)

    return [(msg["用户"], msg["AI"]) for msg in history]


# ========================
# 🚀 Gradio 界面构建
# ========================
def create_gradio_interface():
    """构建交互界面"""
    theme = gr.themes.Default(
        primary_hue="orange",
        secondary_hue="blue"
    )

    with gr.Blocks(theme=theme, title="DeepSeek RAG System 2.0") as interface:
        gr.Markdown("# 🔍 DeepSeek RAG 知识管理系统 (改进版)")

        # 对话区
        chatbot = gr.Chatbot(
            value=[(None, "您好！我是 Theodore（西奥-多尔），您的智能助手 🚀")],
            height=680,
            avatar_images=(USER_ICON_PATH, BOT_ICON_PATH)
        )
        msg_input = gr.Textbox(placeholder="输入您的问题...", lines=3)

        with gr.Row():
            submit_btn = gr.Button("💬 发送", variant="primary")
            os.environ["GRADIO_MAX_FILE_SIZE"] = "100mb"
            upload_btn = gr.UploadButton(
                "📁 上传文档",
                file_types=[".pdf", ".docx", ".txt", ".md", ".pptx", ".xlsx"],
                file_count="multiple"
            )
            clear_btn = gr.Button("🔄 清空对话")

        # 💻 系统监控
        with gr.Accordion("💻 系统监控", open=False):
            gr.Markdown("### 实时系统指标")
            diagnose_btn = gr.Button("🔄 刷新状态")
            status_panel = gr.JSON(label="系统状态", value={"状态": "正在获取..."})

        interface.load(system_diagnosis, inputs=None, outputs=status_panel)

        # 🤖 模型管理
        with gr.Accordion("🤖 模型管理", open=False):
            model_selector = gr.Dropdown(
                label="选择模型",
                choices=list(config.MODEL_PATHS.keys()),
                value=current_model_name
            )
            model_status = gr.Textbox(label="模型状态", interactive=False, value="正在初始化模型...")

        interface.load(lambda: switch_model(current_model_name), inputs=None, outputs=model_status)

        # 💬 对话历史
        with gr.Accordion("💬 对话历史", open=False):
            export_btn = gr.Button("导出历史")
            import_btn = gr.UploadButton("导入历史", file_types=[".json"])
            export_btn.click(export_chat_history, inputs=chatbot, outputs=gr.File())
            import_btn.upload(import_chat_history, inputs=import_btn, outputs=chatbot)

        # 📊 生成参数
        with gr.Accordion("📊 生成参数", open=False):
            max_tokens = gr.Slider(128, 4096, value=512, label="生成长度限制")
            temperature = gr.Slider(0.1, 1.0, value=0.7, label="创造性")
            top_p = gr.Slider(0.1, 1.0, value=0.9, label="核心采样")
            topk_retrieval = gr.Slider(1, 10, value=3, step=1, label="检索文档数量 top_k")
            dist_threshold = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="检索距离阈值")
            show_debug = gr.Checkbox(label="显示推理过程", value=True)

        # 绑定事件
        msg_input.submit(
            chat_with_rag,
            inputs=[msg_input, chatbot, max_tokens, temperature, top_p, show_debug, topk_retrieval, dist_threshold],
            outputs=[msg_input, chatbot]
        )
        submit_btn.click(
            chat_with_rag,
            inputs=[msg_input, chatbot, max_tokens, temperature, top_p, show_debug, topk_retrieval, dist_threshold],
            outputs=[msg_input, chatbot]
        )

        clear_btn.click(lambda: [(None, "对话已清空")], outputs=chatbot)

        upload_btn.upload(upload_files, inputs=[upload_btn, chatbot], outputs=[chatbot])

        diagnose_btn.click(system_diagnosis, outputs=status_panel)
        model_selector.change(switch_model, inputs=model_selector, outputs=model_status)

    return interface


# 启动 Gradio
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(server_name="127.0.0.1", server_port=7860, share=True)
