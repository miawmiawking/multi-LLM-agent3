import requests
import streamlit as st
from langchain.tools import DuckDuckGoSearchRun
import PyPDF2
from docx import Document
import chardet
import base64
from openai import OpenAI

# 初始化会话状态
def initialize_session_state():
    state_defaults = {
        "messages": [],
        "search_enabled": False,
        "file_analyzed": False,
        "file_content": "",
        "file_summary": "",
        "selected_model": "豆包",
        "selected_function": "智能问答",
        "api_keys": {}
    }
    for key, value in state_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# 页面配置
st.set_page_config(page_title="多模型智能助手", layout="wide")
initialize_session_state()

# ====================
# 核心功能实现
# ====================
def call_model_api(prompt, model_type, uploaded_file=None):
    """统一模型调用接口"""
    headers = {"Content-Type": "application/json"}
    params = {}

    try:
        if model_type == "DeepSeek":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['DeepSeek']}"
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "你是DeepSeek，由杭州深度求索人工智能基础技术研究有限公司开发的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。DeepSeek为专有名词，不可翻译成其他语言。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            return response.json()["choices"][0]["message"]["content"]

        elif model_type == "豆包":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['豆包']}"
            response = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                json={
                    "model": "ep-20250128163906-p4tb5",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            try:
                response_json = response.json()
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    st.error(f"豆包 API 返回格式异常: {response_json}")
                    return None
            except Exception as e:
                st.error(f"解析豆包 API 响应失败: {str(e)}")
                return None

        elif model_type == "通义千问":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['通义千问']}"
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                json={
                    "model": "qwen-plus",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            response_json = response.json()
            try:
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    st.error(f"通义千问 API 返回格式异常: {response_json}")
                    return None
            except Exception as e:
                st.error(f"解析通义千问 API 响应失败: {str(e)}")
                return None

        elif model_type == "文心一言":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['文心一言']}"
            response = requests.post(
                "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            response_json = response.json()
            try:
                if "result" in response_json:
                    return response_json["result"]
                else:
                    st.error(f"文心一言 API 返回格式异常: {response_json}")
                    return None
            except Exception as e:
                st.error(f"解析文心一言 API 响应失败: {str(e)}")
                return None

        elif model_type == "智谱清言":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['智谱清言']}"
            response = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                json={
                    "model": "glm-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            return response.json()["choices"][0]["message"]["content"]

        elif model_type == "MiniMax":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['MiniMax']}"
            response = requests.post(
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
                json={
                    "model": "abab5.5-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            return response.json()["choices"][0]["message"]["content"]

        elif model_type == "Kimi(视觉理解)":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['Kimi(视觉理解)']}"
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json={
                    "model": "moonshot-v1-8k",
                    "messages": [
                        {"role": "system", "content": "你是Kimi，由Moonshot AI提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            return response.json()["choices"][0]["message"]["content"]

        elif model_type == "GPTs(聊天、语音识别)":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['OpenAI']}"
            try:
                response = requests.post(
                    "https://api.openai.com/v1/chat/completions",
                    json={
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    },
                    headers={"Authorization": f"Bearer {st.session_state.api_keys['OpenAI']}"}
                )
                response_json = response.json()
                if "error" in response_json:
                    st.error(f"API 返回错误: {response_json['error']['message']}")
                    return None
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    return response_json["choices"][0]["message"]["content"]
                else:
                    st.error(f"API 返回格式异常: {response_json}")
                    return None
            except Exception as e:
                st.error(f"API 调用失败: {str(e)}")
                return None

        elif model_type == "DALL-E(文生图)":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['OpenAI']}"
            response = requests.post(
                "https://api.openai.com/v1/images/generations",
                json={
                    "prompt": prompt,
                    "n": 1,
                    "size": "512x512"
                },
                headers=headers
            )
            response_json = response.json()
            if "data" in response_json and len(response_json["data"]) > 0:
                image_url = response_json["data"][0]["url"]
                return image_url
            else:
                st.error(f"DALL-E API 返回格式异常: {response_json}")
                return None

        elif model_type == "moonshot-v1-8k-vision-preview":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['Kimi']}"
            # 将图片内容转换为Base64编码
            encoded_string = base64.b64encode(prompt).decode("utf-8")
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json={
                    "model": "moonshot-v1-8k-vision-preview",
                    "messages": [
                        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}]}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                },
                headers=headers
            )
            return response.json()["choices"][0]["message"]["content"]

        elif model_type == "o1(深度推理)":
            headers["Authorization"] = f"Bearer {st.session_state.api_keys['OpenAI']}"
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "o1-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": max_tokens  # 保留 max_completion_tokens 参数
                },
                headers=headers
            )
            response_json = response.json()
            if "choices" in response_json:
                return response_json["choices"][0]["message"]["content"]
            else:
                st.error(f"o1 API 返回格式异常: {response_json}")
                return None

    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return None

def handle_file_upload(uploaded_file):
    """处理上传文件并返回内容"""
    file_type = uploaded_file.name.split(".")[-1].lower()
    try:
        if file_type in ["txt", "pdf", "docx"]:
            # 文本类文件处理
            if file_type == "txt":
                raw_data = uploaded_file.getvalue()
                encoding = chardet.detect(raw_data)["encoding"]
                return raw_data.decode(encoding)

            elif file_type == "pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                return "\n".join([page.extract_text() for page in pdf_reader.pages])

            elif file_type == "docx":
                doc = Document(uploaded_file)
                return "\n".join([para.text for para in doc.paragraphs])

        elif file_type in ["jpg", "jpeg", "png"]:
            # 图片文件直接返回文件内容
            return uploaded_file.getvalue()

        elif file_type in ["mp3", "wav", "m4a", "mp4", "webm"]:  # 支持的音频格式
            return uploaded_file

    except Exception as e:
        st.error(f"文件处理失败: {str(e)}")
        return None

def perform_visual_analysis(image_content):
    """使用 moonshot-v1-8k-vision-preview 模型进行视觉分析"""
    try:
        # 调用 moonshot-v1-8k-vision-preview 模型进行视觉分析
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.session_state.api_keys['Kimi']}"}
        # 将图片内容转换为Base64编码
        encoded_string = base64.b64encode(image_content).decode("utf-8")
        response = requests.post(
            "https://api.moonshot.cn/v1/chat/completions",
            json={
                "model": "moonshot-v1-8k-vision-preview",
                "messages": [
                    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}]}
                ],
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            headers=headers
        )
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            return response_json["choices"][0]["message"]["content"]
        else:
            st.error(f"moonshot-v1-8k-vision-preview API 返回格式异常: {response_json}")
            return None
    except Exception as e:
        st.error(f"视觉分析失败: {str(e)}")
        return None

# ====================
# 侧边栏配置
# ====================
with st.sidebar:
    st.header("⚙️ 系统设置")

    # API 密钥管理
    st.subheader("API密钥管理")
    api_key_input = st.text_input(
        "输入 API 密钥",
        help="输入一个API密钥，用于访问所选模型",
        type="password"
    )
    if api_key_input:
        st.session_state.api_keys = {
            "豆包": api_key_input,
            "Kimi(视觉理解)": api_key_input,
            "DeepSeek": api_key_input,
            "通义千问": api_key_input,
            "文心一言": api_key_input,
            "智谱清言": api_key_input,
            "MiniMax": api_key_input,
            "OpenAI": api_key_input,
        }
        st.success("API 密钥已保存！")

    # 模型选择
    model_options = {
        "豆包": ["ep-20250128163906-p4tb5"],

        "DeepSeek": ["deepseek-chat", "deepseek-reasoner"],
        "通义千问": ["qwen-plus"],
        "文心一言": ["ERNIE-Bot"],
        "智谱清言": ["glm-4"],
        "MiniMax": ["abab5.5-chat"],
        "DALL-E(文生图)": ["dall-e-3"],
        "o1(深度推理)": ["o1-mini"],
        "Kimi(视觉理解)": ["moonshot-v1-8k", "moonshot-v1-8k-vision-preview"],
        "GPTs(聊天、语音识别)": ["gpt-4o"]
    }

    st.session_state.selected_model = st.selectbox(
        "选择大模型",
        list(model_options.keys()),
        index=0
    )

    # 功能选择
    function_options = [
        "智能问答",
        "文本翻译",
        "文本总结",
        "文生图",
        "深度推理",
        "视觉理解",
        "语音识别"
    ]
    st.session_state.selected_function = st.selectbox(
        "选择功能",
        function_options,
        index=0
    )

    # 通用参数
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("创意度", 0.0, 2.0, 0.7, 0.1)
    with col2:
        max_tokens = st.slider("响应长度", 100, 2048, 1024, 100)

    # API 测试功能
    st.subheader("API 测试")
    if st.button("🔍 测试 API 连接"):
        if not st.session_state.api_keys:
            st.error("请先输入 API 密钥！")
        else:
            with st.spinner("正在测试 API 连接..."):
                try:
                    test_prompt = "你好，请回复'连接成功'。"
                    response = call_model_api(test_prompt, st.session_state.selected_model)
                    if response:
                        st.success(f"API 连接成功！模型回复：{response}")
                    else:
                        st.error("API 连接失败，请检查密钥和网络设置。")
                except Exception as e:
                    st.error(f"API 测试失败：{str(e)}")

    if st.button("🧹 清空对话历史"):
        st.session_state.messages = []
        st.rerun()

# ====================
# 主界面布局
# ====================
st.title("🤖 多模型智能助手")

# 文件上传区域
uploaded_file = st.file_uploader(
    "📁 上传文件（支持文本/PDF/Word/图片/音频）",
    type=["txt", "pdf", "docx", "doc", "jpg", "jpeg", "png", "mp3", "wav", "m4a", "mp4", "webm"],
    key="file_uploader"
)

# 将上传的文件保存到 session_state 中
if uploaded_file:
    st.session_state.uploaded_file = uploaded_file
    st.session_state.file_content = handle_file_upload(uploaded_file)
    if st.session_state.file_content:
        st.session_state.file_analyzed = True

        # 自动执行语音识别功能
        if uploaded_file.name.split(".")[-1].lower() in ["mp3", "wav", "m4a", "mp4", "webm"]:
            try:
                # 初始化 OpenAI 客户端
                client = OpenAI(api_key=st.session_state.api_keys["OpenAI"])
                # 调用 Whisper 模型进行语音识别
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=st.session_state.file_content
                )
                st.write("语音识别结果：")
                st.write(transcription.text)
                st.session_state.file_summary = transcription.text
            except Exception as e:
                st.error(f"语音识别失败: {str(e)}")
        elif uploaded_file.name.split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
            analysis_result = perform_visual_analysis(st.session_state.file_content)
            st.write("视觉分析结果：")
            st.write(analysis_result)
        else:
            summary_prompt = f"请对以下内容进行总结和梳理，提取核心内容：\n{st.session_state.file_content}"
            summary_response = call_model_api(summary_prompt, st.session_state.selected_model)
            if summary_response:
                st.session_state.file_summary = summary_response
                st.write("文件核心内容总结：")
                st.write(st.session_state.file_summary)

# 功能操作区
with st.container():
    col1, col2 = st.columns([1, 1])
    with col1:
        search_toggled = st.button(
            f"🌐 联网搜索 {'(on)' if st.session_state.search_enabled else '(off)'}",
            help="启用/禁用实时网络搜索",
            use_container_width=True
        )
        if search_toggled:
            st.session_state.search_enabled = not st.session_state.search_enabled

# 用户问题输入栏
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        user_input = st.chat_input("请输入您的问题或指令...", key="user_input")

# 初始化 DuckDuckGo 搜索工具
search_tool = DuckDuckGoSearchRun()
def perform_web_search(query):
    """执行网络搜索并返回结果"""
    try:
        search_results = search_tool.run(query)
        return search_results
    except Exception as e:
        st.error(f"网络搜索失败: {str(e)}")
        return "无法获取网络搜索结果"

# ====================
# 交互处理逻辑
# ====================
if uploaded_file:
    st.session_state.file_content = handle_file_upload(uploaded_file)
    if st.session_state.file_content:
        st.session_state.file_analyzed = True

        # 调用大模型进行文件内容的总结和梳理
        if uploaded_file.name.split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
            analysis_result = perform_visual_analysis(st.session_state.file_content)
            st.write("视觉分析结果：")
            st.write(analysis_result)
        elif uploaded_file.name.split(".")[-1].lower() in ["mp3", "wav", "m4a", "mp4", "webm"]:  # 语音文件处理
            if st.session_state.selected_function == "语音识别":
                try:
                    # 直接调用 Whisper 模型进行语音识别
                    client = OpenAI(api_key=st.session_state.api_keys["OpenAI"])
                    transcription = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=st.session_state.file_content
                    )
                    st.write("语音识别结果：")
                    st.write(transcription.text)
                except Exception as e:
                    st.error(f"语音识别失败: {str(e)}")
        else:
            summary_prompt = f"请对以下内容进行总结和梳理，提取核心内容：\n{st.session_state.file_content}"
            summary_response = call_model_api(summary_prompt, st.session_state.selected_model)
            if summary_response:
                st.session_state.file_summary = summary_response
                st.write("文件核心内容总结：")
                st.write(st.session_state.file_summary)

if user_input:
    full_prompt = f"{user_input}\n{st.session_state.file_summary}"

    with st.spinner("🧠 正在处理请求..."):
        # 根据功能路由处理
        if st.session_state.selected_function == "文生图":
            image_url = call_model_api(full_prompt, "DALL-E")
            st.image(image_url, caption="生成结果")

        elif st.session_state.selected_function == "视觉理解":
            if uploaded_file and uploaded_file.name.split(".")[-1].lower() in ["jpg", "jpeg", "png"]:
                analysis_result = perform_visual_analysis(st.session_state.file_content)
                st.write("视觉分析结果：")
                st.write(analysis_result)
            else:
                st.error("请上传图片文件进行视觉理解分析。")

        elif st.session_state.selected_function == "深度推理":
            if st.session_state.selected_model == "o1":
                response_text = call_model_api(full_prompt, "o1")
                st.write(response_text)

        else:  # 智能问答/翻译/总结
            if st.session_state.search_enabled:
                search_results = perform_web_search(user_input)
                full_prompt = f"{full_prompt}\n【网络搜索结果】\n{search_results}"

            response = call_model_api(full_prompt, st.session_state.selected_model)
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "type": "text"
            })

# ====================
# 对话历史展示
# ====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"])
        else:
            st.write(msg["content"])

# 初始提示
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.write("您好！我是多模型智能助手，请选择模型和功能开始交互。")

