import requests
import streamlit as st
from langchain_community.tools import DuckDuckGoSearchRun
import PyPDF2
from docx import Document
import chardet
import base64
import io
from langchain.docstore.document import Document as LC_Document # 新增 langchain 相关依赖
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
import tempfile
import os
from pydub import AudioSegment
from openai import OpenAI
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import re
from urllib.parse import urlparse
import csv

# 全局变量定义
CHROMADB_PATH = None
COLLECTION_NAME = "rag_collection"

# ChromaDB 配置函数
def configure_chromadb():
    st.divider()
    with st.expander("🗄️ RAG知识库设置", expanded=not bool(st.session_state.get("chromadb_path"))):
        st.markdown("### ChromaDB存储路径")
        
        # 显示当前路径
        current_path = st.session_state.get("chromadb_path", "")
        if current_path:
            st.info(f"当前路径：{current_path}")
            
            # 添加清空数据库按钮
            if st.button("🗑️ 清空知识库", key="clear_db"):
                try:
                    # 获取向量库实例
                    vectorstore = get_vector_store()
                    if vectorstore:
                        # 删除所有文档
                        vectorstore.delete_collection()
                        vectorstore = None
                        st.session_state.vector_store = None
                        st.session_state.rag_data = []
                        st.success("✅ 知识库已清空")
                        st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ 清空知识库失败：{str(e)}")
        
        # 路径输入
        default_path = os.path.join(os.path.expanduser("~"), "chromadb_data")
        new_path = st.text_input(
            "设置存储路径",
            value=current_path or default_path,
            placeholder="例如：C:/Users/YourName/Documents/chromadb",
            key="chromadb_path_input"
        )
        
        # 确认按钮
        if st.button("✅ 确认路径", key="confirm_chromadb_path"):
            try:
                # 确保路径存在
                os.makedirs(new_path, exist_ok=True)
                
                # 测试路径是否可写
                test_file = os.path.join(new_path, "test_write.txt")
                try:
                    with open(test_file, "w") as f:
                        f.write("test")
                    os.remove(test_file)
                except Exception as e:
                    st.error(f"路径无写入权限：{str(e)}")
                    return
                
                # 更新路径
                st.session_state.chromadb_path = new_path
                # 更新全局变量
                global CHROMADB_PATH
                CHROMADB_PATH = new_path
                
                st.success("✅ ChromaDB路径设置成功！")
                st.session_state.vector_store = None  # 重置向量库实例
                
            except Exception as e:
                st.error(f"❌ 路径设置失败：{str(e)}")
        
        st.markdown("""
        **说明：**
        1. 首次使用请设置存储路径
        2. 路径需要有写入权限
        3. 建议选择本地固定位置
        4. 确保有足够存储空间
        """)

# 初始化会话状态
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_enabled" not in st.session_state:
    st.session_state.search_enabled = False
if "file_analyzed" not in st.session_state:
    st.session_state.file_analyzed = False
if "file_content" not in st.session_state:
    st.session_state.file_content = ""
if "file_summary" not in st.session_state:
    st.session_state.file_summary = ""
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "豆包"
if "selected_function" not in st.session_state:
    st.session_state.selected_function = "智能问答"
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}
if "rag_enabled" not in st.session_state:
    st.session_state.rag_enabled = False
if "rag_data" not in st.session_state:
    st.session_state.rag_data = []
if "temperature" not in st.session_state:
    st.session_state.temperature = 0.5
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = 2048
if "chromadb_path" not in st.session_state:
    st.session_state.chromadb_path = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# 页面配置
st.set_page_config(page_title="多模型智能助手2.0", layout="wide")

# 初始化/加载 langchain 封装的 Chroma 向量库
def get_vector_store():
    """获取或创建向量数据库实例"""
    # 检查是否已设置路径
    if not st.session_state.get("chromadb_path"):
        st.error("⚠️ 请先在侧边栏设置 ChromaDB 存储路径！")
        return None
    
    try:
        # 初始化 embeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 创建向量库实例
        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=st.session_state.chromadb_path
        )
        
        # 检查集合是否为空
        if vectorstore._collection.count() == 0:
            st.warning("⚠️ 知识库为空，请先上传文件或网址。")
            return None
            
        return vectorstore
        
    except Exception as e:
        st.error(f"❌ 初始化向量库失败：{str(e)}")
        return None

# 初始化 DuckDuckGo 搜索工具
search_tool = DuckDuckGoSearchRun()

# 核心功能实现

def handle_web_search(query):
    """联网搜索功能，使用 DuckDuckGo API"""
    if not st.session_state.search_enabled:
        return None
    try:
        search = DuckDuckGoSearchRun()
        results = search.run(query)
        return results
    except Exception as e:
        st.error(f"联网搜索失败: {str(e)}")
        return None

def call_model_api(prompt, model_type, rag_data=None):
    """调用除 RAG 部分外的其他接口"""
    headers = {"Content-Type": "application/json"}
    try:
        if model_type == "豆包":
            api_key = st.session_state.api_keys.get("豆包", "")
            if not api_key:
                st.error("请提供豆包 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                json={
                    "model": "ep-20250128163906-p4tb5",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        elif model_type == "DeepSeek-V3":
            api_key = st.session_state.api_keys.get("DeepSeek", "")
            if not api_key:
                st.error("请提供 DeepSeek API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        elif model_type == "通义千问":
            api_key = st.session_state.api_keys.get("通义千问", "")
            if not api_key:
                st.error("请提供 通义千问 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                json={
                    "model": "qwen-plus",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        elif model_type == "文心一言":
            api_key = st.session_state.api_keys.get("文心一言", "")
            if not api_key:
                st.error("请提供 文心一言 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions",
                json={
                    "model": "ERNIE-Bot",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        elif model_type == "智谱清言":
            api_key = st.session_state.api_keys.get("智谱清言", "")
            if not api_key:
                st.error("请提供 智谱清言 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions",
                json={
                    "model": "glm-4",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        elif model_type == "MiniMax":
            api_key = st.session_state.api_keys.get("MiniMax", "")
            if not api_key:
                st.error("请提供 MiniMax API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
                json={
                    "model": "abab5.5-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        elif model_type == "DALL-E(文生图)":
            api_key = st.session_state.api_keys.get("OpenAI", "")
            if not api_key:
                st.error("请提供 DALL-E(文生图) API 密钥！")
                return None
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
        elif model_type == "DeepSeek-R1(深度推理)":
            api_key = st.session_state.api_keys.get("DeepSeek", "")
            if not api_key:
                st.error("请提供 DeepSeek API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",  # 修改为正确的接口地址
                json={
                    "model": "deepseek-reasoner",  # 修改为正确的模型名称
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        elif model_type == "o1(深度推理)":
            api_key = st.session_state.api_keys.get("OpenAI", "")
            if not api_key:
                st.error("请提供 o1 API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "o1-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": st.session_state.max_tokens  # 修改参数名称
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        elif model_type == "Kimi(视觉理解)":
            api_key = st.session_state.api_keys.get("Kimi(视觉理解)", "")
            if not api_key:
                st.error("请提供 Kimi(视觉理解) API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            
            # 简单的文本测试
            response = requests.post(
                "https://api.moonshot.cn/v1/chat/completions",
                json={
                    "model": "moonshot-v1-8k-vision-preview",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "请回复'API连接正常'"
                                }
                            ]
                        }
                    ]
                },
                headers=headers
            )
            return handle_response(response)
        elif model_type == "GPTs(聊天、语音识别)":
            api_key = st.session_state.api_keys.get("OpenAI", "")
            if not api_key:
                st.error("请提供 OpenAI API 密钥！")
                return None
            headers["Authorization"] = f"Bearer {api_key}"
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                json={
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": st.session_state.temperature,
                    "max_tokens": st.session_state.max_tokens
                },
                headers=headers
            )
            return handle_response(response, rag_data)
        else:
            # 默认调用使用 RAG 生成答案（下文使用 langchain 实现）
            return rag_generate_response(prompt)
    except Exception as e:
        st.error(f"API调用失败: {str(e)}")
        return None

def handle_response(response, rag_data=None):
    """处理 API 响应"""
    try:
        if response.status_code == 200:
            response_json = response.json()
            if "choices" in response_json and len(response_json["choices"]) > 0:
                answer = response_json["choices"][0]["message"]["content"]
            elif "result" in response_json:
                # 针对文心一言返回格式处理
                answer = response_json["result"]
            elif "data" in response_json and isinstance(response_json["data"], list) and len(response_json["data"]) > 0:
                # 针对 DALL-E 返回格式处理
                if "url" in response_json["data"][0]:
                    answer = response_json["data"][0]["url"]
                else:
                    st.error(f"API 返回格式异常: {response_json}")
                    return None
            else:
                st.error(f"API 返回格式异常: {response_json}")
                return None

            if rag_data and isinstance(answer, str):  # 确保是文本才添加引用
                answer += "\n\n引用来源：\n" + "\n".join([f"- {source}" for source in rag_data])
            return answer
        elif response.status_code == 503:
            st.error("服务器繁忙，请稍后再试。")
            return None
        else:
            st.error(f"API 请求失败，错误码：{response.status_code}")
            return None
    except ValueError as e:
        st.error(f"响应解析失败: {str(e)}")
        return None

# 使用 langchain 实现 RAG：加载文档、分割、嵌入、索引
def get_embeddings():
    """获取 embeddings 实例"""
    try:
        return HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        st.error(f"初始化 embeddings 失败：{str(e)}")
        return None

def rag_index_document(content, source):
    """将文档添加到向量数据库"""
    try:
        # 检查内容和路径
        if not content or not st.session_state.chromadb_path:
            st.error("⚠️ 内容为空或未设置存储路径")
            return False
            
        # 文本分割
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(content)
        
        if not texts:
            st.error("⚠️ 文本分割后为空")
            return False
            
        # 获取 embeddings
        embeddings = get_embeddings()
        if not embeddings:
            return False
        
        # 创建或获取向量库实例
        try:
            vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embeddings,
                persist_directory=st.session_state.chromadb_path
            )
        except Exception as e:
            st.error(f"创建向量库实例失败：{str(e)}")
            return False
        
        # 为每个文本块添加源信息
        metadatas = [{"source": source} for _ in texts]
        
        # 添加文档
        try:
            vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas
            )
            vectorstore.persist()
            st.session_state.vector_store = vectorstore
            st.info(f"✅ 成功添加 {len(texts)} 个文本块到知识库")
            return True
        except Exception as e:
            st.error(f"添加文本到向量库失败：{str(e)}")
            return False
            
    except Exception as e:
        st.error(f"❌ 添加文档到向量库失败：{str(e)}")
        import traceback
        st.error(f"详细错误：{traceback.format_exc()}")
        return False

def rag_generate_response(query):
    """生成 RAG 响应"""
    # 获取向量库实例
    vectorstore = get_vector_store()
    if not vectorstore:
        return "请先上传文件或网址到知识库。"
    
    try:
        # 执行相似性搜索
        docs = vectorstore.similarity_search(query, k=3)
        
        if not docs:
            return "未找到相关信息。请尝试调整问题或添加更多相关文档。"
        
        # 构建提示词
        context = "\n\n".join([doc.page_content for doc in docs])
        sources = "\n".join([f"- {doc.metadata.get('source', '未知来源')}" for doc in docs])
        
        prompt = f"""基于以下参考信息回答问题：

参考信息：
{context}

问题：{query}

请提供准确、相关的回答。如果参考信息不足以回答问题，请明确说明。
"""
        # 调用模型生成回答
        response = call_model_api(prompt, st.session_state.selected_model)
        if response:
            return f"{response}\n\n来源：\n{sources}"
        return "生成回答失败，请重试。"
    
    except Exception as e:
        st.error(f"❌ 生成回答失败：{str(e)}")
        return None

def handle_file_upload(uploaded_files):
    """处理上传文件，根据 RAG 状态及文件类型执行不同操作：
       - RAG 模式下：文本、表格类文件加入知识库；
       - 非 RAG 模式下：
           图片文件 -> 视觉分析
           语音文件 -> 语音识别
           文本文件 -> 文本总结
    """
    if not uploaded_files:
        return

    if not isinstance(uploaded_files, list):
        uploaded_files = [uploaded_files]

    for uploaded_file in uploaded_files:
        if not hasattr(uploaded_file, "name"):
            st.error("上传的文件格式错误，缺少名称属性。")
            continue

        file_name = uploaded_file.name
        file_type = uploaded_file.type.split("/")[-1].lower()
        try:
            if st.session_state.rag_enabled:
                # RAG 模式下，仅处理文本、表格类文件加入知识库
                if file_type in ["txt", "pdf", "docx", "doc", "csv", "xlsx", "xls"]:
                    content = extract_text_from_file(uploaded_file)
                    if content:
                        if rag_index_document(content, file_name):
                            st.session_state.rag_data.append(file_name)
                            st.success(f"文件 {file_name} 已成功加入 RAG 知识库")
                else:
                    st.warning(f"RAG 模式下，文件 {file_name} 的类型（{file_type}）不支持加入知识库。")
            else:
                # 非 RAG 模式，根据文件类型调用对应功能
                if file_type in ["jpg", "jpeg", "png"]:
                    st.write(f"正在分析图片：{file_name}")
                    analysis_result = perform_visual_analysis(uploaded_file.getvalue())
                    if analysis_result:
                        st.write("视觉分析结果：")
                        st.write(analysis_result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"图片分析结果：\n{analysis_result}",
                            "type": "text"
                        })
                elif file_type in ["mp3", "wav", "m4a", "mpeg"]:
                    st.write(f"正在进行语音识别：{file_name}")
                    speech_result = perform_speech_recognition(uploaded_file.getvalue())
                    if speech_result:
                        st.write("语音识别结果：")
                        st.write(speech_result)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"语音识别结果：\n{speech_result}",
                            "type": "text"
                        })
                elif file_type in ["txt", "pdf", "docx", "doc"]:
                    content = extract_text_from_file(uploaded_file)
                    if content:
                        st.write(f"正在总结文本：{file_name}")
                        summary_result = perform_text_summary(content)
                        if summary_result:
                            st.write("文本总结结果：")
                            st.write(summary_result)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"文本总结结果：\n{summary_result}",
                                "type": "text"
                            })
                else:
                    st.warning(f"文件 {file_name} 的类型（{file_type}）不支持处理。")
        except Exception as e:
            st.error(f"文件处理失败 ({file_name}): {str(e)}")

def extract_text_from_file(file):
    """从不同类型的文件中提取文本"""
    try:
        file_type = file.name.split('.')[-1].lower()
        
        if file_type == 'txt':
            return file.getvalue().decode('utf-8')
            
        elif file_type == 'pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() + '\n'
            return text
            
        elif file_type in ['docx', 'doc']:
            doc = Document(file)
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
        elif file_type in ['csv']:
            return process_csv_file(file)
            
        elif file_type in ['xlsx', 'xls']:
            # 如果需要处理 Excel 文件，可以使用 openpyxl
            import openpyxl
            wb = openpyxl.load_workbook(file)
            text = []
            for sheet in wb.sheetnames:
                ws = wb[sheet]
                for row in ws.rows:
                    text.append(' '.join(str(cell.value) for cell in row if cell.value))
            return '\n'.join(text)
            
        else:
            st.error(f"不支持的文件类型：{file_type}")
            return None
            
    except Exception as e:
        st.error(f"处理文件失败：{str(e)}")
        return None

def process_csv_file(file):
    content = []
    csv_data = file.read().decode('utf-8').splitlines()
    csv_reader = csv.reader(csv_data)
    for row in csv_reader:
        content.append(' '.join(row))
    return '\n'.join(content)

def perform_visual_analysis(image_content):
    """使用 moonshot-v1-8k-vision-preview 模型进行视觉分析"""
    try:
        api_key = st.session_state.api_keys.get("Kimi(视觉理解)")
        if not api_key:
            st.error("请提供 Kimi(视觉理解) API 密钥！")
            return None

        # 将图片内容编码为 base64 字符串
        encoded_string = base64.b64encode(image_content).decode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": "moonshot-v1-8k-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "请描述这张图片的内容。"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_string}"
                            }
                        }
                    ]
                }
            ],
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens
        }

        response = requests.post(
            "https://api.moonshot.cn/v1/chat/completions",
            json=payload,
            headers=headers
        )
        return handle_response(response)
    except Exception as e:
        st.error(f"视觉分析失败: {str(e)}")
        return None

def perform_speech_recognition(audio_bytes):
    """
    使用当前选择的模型进行语音识别
    """
    api_key = st.session_state.api_keys.get("OpenAI", "")
    if not api_key:
        st.error("请提供 OpenAI API 密钥以进行语音识别！")
        return None
    
    try:
        # 创建 OpenAI 客户端
        client = OpenAI(api_key=api_key)
        
        # 将音频数据转换为临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(audio_bytes)
            temp_file_path = temp_file.name
        
        with open(temp_file_path, 'rb') as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
        
        os.unlink(temp_file_path)  # 删除临时文件
        return transcript.text
        
    except Exception as e:
        st.error(f"语音识别失败：{str(e)}")
        return None

def perform_text_summary(text):
    """
    使用当前选择的模型对文本进行总结
    """
    try:
        summary_prompt = f"请对以下文本进行简明扼要的总结：\n\n{text}"
        response = call_model_api(summary_prompt, st.session_state.selected_model)
        return response
    except Exception as e:
        st.error(f"文本总结失败：{str(e)}")
        return None

def retrieve_relevant_content(query):
    """
    利用 langchain 封装的向量库检索与查询相关的文档，
    返回包含来源信息的列表。
    """
    vectorstore = get_vector_store()
    try:
        results = vectorstore.similarity_search(query, k=3)
    except Exception as e:
        st.error(f"检索时出现错误: {str(e)}")
        return []
    # 提取文档 metadata 中的 "source" 信息；如果不存在则返回 "未知来源"
    return [doc.metadata.get("source", "未知来源") for doc in results]

def fetch_url_content(url):
    """获取网页内容并提取有效文本"""
    try:
        # 添加请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, verify=False, timeout=10)
        response.raise_for_status()
        
        # 使用 BeautifulSoup 提取文本内容
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # 移除脚本和样式元素
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 获取文本并处理
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        st.error(f"获取网页内容失败：{str(e)}")
        return None

def clear_vector_store():
    """清除向量数据库中的所有数据"""
    try:
        # 获取向量存储实例
        vectorstore = get_vector_store()
        if vectorstore:
            # 删除集合中的所有数据
            vectorstore.delete_collection()
            # 重新创建空集合
            vectorstore = get_vector_store()
            # 清空会话状态中的数据记录
            st.session_state.rag_data = []
            return True
    except Exception as e:
        st.error(f"清除向量数据库失败：{str(e)}")
        return False

def clean_text(text):
    """清理文本内容"""
    if not text:
        return ""
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text).strip()
    # 移除特殊字符，保留货币符号
    text = re.sub(r'[^\w\s\u4e00-\u9fff,.?!，。？！:：;；""''()（）《》<>¥$€£%]', '', text)
    return text

def is_financial_domain(url):
    """判断是否为财经金融相关的高质量域名"""
    try:
        domain = urlparse(url).netloc.lower()
        
        # 财经金融网站优先级
        financial_domains = {
            # 官方机构
            'pbc.gov.cn': 10,     # 中国人民银行
            'csrc.gov.cn': 10,    # 中国证监会
            'safe.gov.cn': 10,    # 外汇管理局
            'stats.gov.cn': 10,   # 国家统计局
            'mof.gov.cn': 10,     # 财政部
            
            # 交易所
            'sse.com.cn': 9,      # 上海证券交易所
            'szse.cn': 9,         # 深圳证券交易所
            'cffex.com.cn': 9,    # 中国金融期货交易所
            
            # 金融门户网站
            'eastmoney.com': 8,   # 东方财富
            'finance.sina.com.cn': 8,  # 新浪财经
            'caixin.com': 8,      # 财新网
            'yicai.com': 8,       # 第一财经
            'stcn.com': 8,        # 证券时报网
            'cnstock.com': 8,     # 中国证券网
            '21jingji.com': 8,    # 21世纪经济网
            
            # 财经媒体
            'bloomberg.cn': 8,     # 彭博
            'ftchinese.com': 8,   # FT中文网
            'nbd.com.cn': 7,      # 每日经济新闻
            'ce.cn': 7,           # 中国经济网
            'jrj.com.cn': 7,      # 金融界
            'hexun.com': 7,       # 和讯网
            
            # 研究机构
            'cfets.org.cn': 7,    # 中国外汇交易中心
            'chinabond.com.cn': 7, # 中国债券信息网
            'shibor.org': 7,      # Shibor官网
            
            # 国际金融网站
            'reuters.com': 8,      # 路透社
            'bloomberg.com': 8,    # 彭博
            'wsj.com': 8,         # 华尔街日报
            'ft.com': 8,          # 金融时报
            'economist.com': 8,    # 经济学人
            
            # 其他相关网站
            'investing.com': 7,    # 英为财情
            'marketwatch.com': 7,  # 市场观察
            'cnfol.com': 6,       # 中金在线
            'stockstar.com': 6,   # 证券之星
            '10jqka.com.cn': 6,   # 同花顺财经
        }
        
        # 检查域名优先级
        for known_domain, priority in financial_domains.items():
            if known_domain in domain:
                return priority
                
        return 0  # 非金融网站返回0优先级
    except:
        return 0

def perform_web_search(query, max_results=10):
    """执行优化的财经金融搜索"""
    try:
        # 优化搜索查询
        financial_keywords = ['金融', '财经', '经济', '股市', '基金', '债券', '外汇', 
                            '期货', '理财', '投资', '证券', '银行', '保险', '金价']
        
        # 检查是否需要添加财经关键词
        if not any(keyword in query for keyword in financial_keywords):
            # 添加财经相关关键词以提高相关性
            optimized_query = query + ' 财经'
        else:
            optimized_query = query
        
        # 使用 DuckDuckGoSearchRun 进行主搜索
        search_tool = DuckDuckGoSearchRun()
        initial_results = search_tool.run(optimized_query)
        
        # 使用 DDGS 进行补充搜索
        with DDGS() as ddgs:
            detailed_results = list(ddgs.text(
                optimized_query,
                max_results=max_results,
                region='cn-zh',
                safesearch='moderate',
                timelimit='m'  # 限制最近一个月的结果，保证信息时效性
            ))
        
        # 结果处理和排序
        processed_results = []
        seen_content = set()
        
        if detailed_results:
            for result in detailed_results:
                title = clean_text(result.get('title', ''))
                snippet = clean_text(result.get('body', ''))
                link = result.get('link', '')
                
                # 内容去重检查
                content_hash = f"{title}_{snippet}"
                if content_hash in seen_content:
                    continue
                seen_content.add(content_hash)
                
                # 计算域名质量分数
                domain_score = is_financial_domain(link)
                
                # 计算内容相关性分数
                relevance_score = sum(1 for word in query.lower().split() 
                                    if word in title.lower() or word in snippet.lower())
                
                # 检查是否包含财经关键词
                financial_relevance = sum(1 for keyword in financial_keywords 
                                        if keyword in title or keyword in snippet)
                
                # 综合评分
                total_score = domain_score * 3 + relevance_score * 2 + financial_relevance * 2
                
                if domain_score > 0 or financial_relevance > 0:  # 只保留金融相关网站的内容
                    processed_results.append({
                        'title': title,
                        'snippet': snippet,
                        'link': link,
                        'score': total_score
                    })
        
        # 按综合评分排序
        processed_results.sort(key=lambda x: x['score'], reverse=True)
        
        # 构建最终响应
        final_response = "财经相关搜索结果：\n\n"
        
        # 添加初步搜索结果
        if initial_results and any(keyword in initial_results.lower() for keyword in financial_keywords):
            final_response += f"{initial_results}\n\n"
        
        # 添加高质量补充结果
        if processed_results:
            final_response += "补充信息：\n"
            for idx, result in enumerate(processed_results[:5], 1):
                if result['score'] > 4:  # 提高显示阈值，确保高质量结果
                    final_response += f"{idx}. **{result['title']}**\n"
                    final_response += f"   {result['snippet']}\n"
                    final_response += f"   来源：[{urlparse(result['link']).netloc}]({result['link']})\n\n"
        
        return final_response.strip()
    
    except Exception as e:
        st.error(f"财经信息搜索失败: {str(e)}")
        return None

def get_search_response(query):
    """生成优化的财经搜索响应，并由大模型总结"""
    try:
        # 获取搜索结果
        search_results = perform_web_search(query)
        if not search_results:
            return "抱歉，没有找到相关的财经信息。"
        
        # 构建提示词，让大模型进行总结
        summary_prompt = f"""
请针对以下用户问题和搜索结果，进行专业的总结分析：

用户问题：{query}

搜索结果：
{search_results}

请你作为金融专家：
1. 提取要点，直接回答用户的核心问题
2. 确保信息的准确性和时效性
3. 如有必要，给出专业的建议或风险提示
4. 保持简洁清晰，突出重点

请以专业、客观的口吻回答。
"""
        # 调用大模型进行总结
        summary = call_model_api(summary_prompt, st.session_state.selected_model)
        
        # 构建最终响应
        response = "📊 **核心回答：**\n\n"
        response += f"{summary}\n\n"
        response += "---\n"
        response += "🔍 **详细搜索结果：**\n\n"
        response += f"{search_results}\n\n"
        response += "---\n"
        response += "*以上信息来自权威财经金融网站，并经AI分析整理。请注意信息时效性，建议进一步核实具体数据。*"
        
        return response

    except Exception as e:
        st.error(f"生成回答失败：{str(e)}")
        return None

def process_urls(urls_input):
    """处理输入的网址，提取内容并添加到 RAG 知识库"""
    urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
    
    for url in urls:
        with st.spinner(f"正在处理网址：{url}"):
            try:
                # 发送 HTTP 请求获取网页内容
                response = requests.get(url, timeout=10)
                response.raise_for_status()  # 检查请求是否成功
                
                # 使用 BeautifulSoup 解析网页内容
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # 移除脚本和样式元素
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # 提取文本内容
                text = soup.get_text()
                
                # 清理文本（移除多余的空白字符）
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                if text:
                    # 将网页内容添加到 RAG 知识库
                    if rag_index_document(text, url):
                        st.session_state.rag_data.append(url)
                        st.success(f"✅ 网址 {url} 已成功加入知识库")
                else:
                    st.warning(f"⚠️ 网址 {url} 未提取到有效内容")
                    
            except requests.RequestException as e:
                st.error(f"❌ 访问网址 {url} 失败：{str(e)}")
            except Exception as e:
                st.error(f"❌ 处理网址 {url} 时出错：{str(e)}")

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
    api_keys_to_set = {
        "豆包": api_key_input,
        "Kimi(视觉理解)": api_key_input,
        "DeepSeek": api_key_input,
        "通义千问": api_key_input,
        "文心一言": api_key_input,
        "智谱清言": api_key_input,
        "MiniMax": api_key_input,
        "OpenAI": api_key_input
    }
    if api_key_input:
        for key, value in api_keys_to_set.items():
            st.session_state.api_keys[key] = value
        st.success("API 密钥已保存！")

    # 模型选择
    model_options = {
        "豆包": ["ep-20250128163906-p4tb5"],
        "DeepSeek-V3": ["deepseek-chat"],
        "通义千问": ["qwen-plus"],
        "文心一言": ["ERNIE-Bot"],
        "智谱清言": ["glm-4"],
        "MiniMax": ["abab5.5-chat"],
        "DALL-E(文生图)": ["dall-e-3"],
        "DeepSeek-R1(深度推理)": ["deepseek-reasoner"],
        "o1(深度推理)": ["o1-mini"],
        "Kimi(视觉理解)": ["moonshot-v1-8k-vision-preview"],
        "GPTs(聊天、语音识别)": ["gpt-4"]
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
        st.session_state.temperature = st.slider("创意度", 0.0, 1.0, 0.5, 0.1)
    with col2:
        st.session_state.max_tokens = st.slider("响应长度", 100, 4096, 2048, 100)

    # 联网搜索功能按钮
    if st.button(
        f"🌏 联网搜索[{('on' if st.session_state.search_enabled else 'off')}]",
        use_container_width=True
    ):
        st.session_state.search_enabled = not st.session_state.search_enabled
        st.rerun()

    # RAG 功能按钮
    if st.button(
        f"📚 RAG 功能[{('on' if st.session_state.rag_enabled else 'off')}]",
        use_container_width=True
    ):
        st.session_state.rag_enabled = not st.session_state.rag_enabled
        st.rerun()

    # API 测试功能
    st.subheader("API 测试")
    if st.button("🔍 测试 API 连接"):
        if not st.session_state.api_keys:
            st.error("请先输入 API 密钥！")
        else:
            with st.spinner("正在测试 API 连接..."):
                try:
                    test_prompt = "您好，请回复'连接成功'。"
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

    # 添加清空知识库按钮
    if st.button("🗑️ 清空知识库", help="清除所有已上传的文件和网址数据"):
        if st.session_state.rag_enabled:
            with st.spinner("正在清空知识库..."):
                if clear_vector_store():
                    st.success("✅ 知识库已清空")
                    st.session_state.rag_data = []
                    st.rerun()
                else:
                    st.error("❌ 清空知识库失败")
        else:
            st.warning("请先开启 RAG 功能")

    # 更新说明
    st.subheader("更新说明")
    st.write("- 新增：构建私人知识库(RAG)功能")
    st.write("- 预告：后续将增加 AI agent 等功能")

    # 在主界面的侧边栏添加 ChromaDB 配置
    if st.session_state.rag_enabled:
        configure_chromadb()

# ====================
# 主界面布局
# ====================
st.title("🤖 多模型智能助手2.0")

# 文件和网址上传区域
st.markdown("### 📁 文件上传")

# RAG 模式：多文件上传和网址输入
if st.session_state.rag_enabled:
    # 文件上传
    uploaded_files = st.file_uploader(
        "支持多个文件上传（建议不超过5个）",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "doc", "csv", "xlsx", "xls"],
        key="multi_file_uploader"
    )
    
    # 网址输入
    st.markdown("### 🔗 网址上传")
    urls_input = st.text_area(
        "每行输入一个网址（建议不超过5个）",
        height=100,
        key="urls_input",
        placeholder="https://example1.com\nhttps://example2.com"
    )
    
    # 提交按钮
    if st.button("📤 提交文件和网址"):
        if not uploaded_files and not urls_input.strip():
            st.warning("请至少上传一个文件或输入一个网址。")
        else:
            success_count = 0
            # 处理文件
            if uploaded_files:
                if len(uploaded_files) > 5:
                    st.warning("⚠️ 文件数量超过5个，建议减少文件数量以获得更好的处理效果。")
                
                for file in uploaded_files:
                    with st.spinner(f"正在处理文件：{file.name}"):
                        try:
                            content = extract_text_from_file(file)
                            if content:
                                if rag_index_document(content, file.name):
                                    success_count += 1
                                    st.session_state.rag_data.append(file.name)
                                    st.success(f"✅ 文件 {file.name} 已成功加入知识库")
                            else:
                                st.error(f"❌ 无法提取文件内容：{file.name}")
                        except Exception as e:
                            st.error(f"❌ 处理文件失败：{str(e)}")
            
            # 处理网址
            if urls_input.strip():
                process_urls(urls_input)

            if success_count > 0:
                st.success(f"✅ 共成功处理 {success_count} 个文件/网址")
                # 强制刷新向量库实例
                st.session_state.vector_store = None
                # 重新加载向量库
                get_vector_store()
            else:
                st.error("❌ 未能成功处理任何文件或网址")

# 非 RAG 模式：单文件上传并立即处理
else:
    uploaded_file = st.file_uploader(
        "上传单个文件进行分析",
        accept_multiple_files=False,
        type=["txt", "pdf", "docx", "doc", "jpg", "jpeg", "png", "mp3", "wav", "m4a"],
        key="single_file_uploader"
    )
    
    if uploaded_file:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        try:
            # 1. 语音识别（GPTs）
            if file_type in ["mp3", "wav", "m4a"]:
                with st.spinner("🎵 正在进行语音识别..."):
                    if "OpenAI" not in st.session_state.api_keys:
                        st.error("请先配置 OpenAI API 密钥")
                    else:
                        client = OpenAI(api_key=st.session_state.api_keys["OpenAI"])
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file.flush()
                            
                            with open(tmp_file.name, "rb") as audio_file:
                                transcription = client.audio.transcriptions.create(
                                    model="whisper-1",
                                    file=audio_file,
                                    language="zh"
                                )
                        
                        st.success("✅ 语音识别完成")
                        with st.chat_message("assistant"):
                            st.markdown(f"**语音识别结果：**\n\n{transcription.text}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"语音文件 {uploaded_file.name} 的识别结果：\n\n{transcription.text}",
                            "type": "text"
                        })
            
            # 2. 图片分析（moonshot-v1-8k-vision-preview）
            elif file_type in ["jpg", "jpeg", "png"]:
                with st.spinner("🖼️ 正在分析图片..."):
                    if "Kimi(视觉理解)" not in st.session_state.api_keys:
                        st.error("请先配置 Kimi(视觉理解) API 密钥")
                    else:
                        image_content = uploaded_file.getvalue()
                        encoded_image = base64.b64encode(image_content).decode('utf-8')
                        
                        headers = {
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {st.session_state.api_keys['Kimi(视觉理解)']}"
                        }
                        
                        payload = {
                            "model": "moonshot-v1-8k-vision-preview",
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": "请详细分析这张图片的内容，包括主要对象、场景、细节等方面。"
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/jpeg;base64,{encoded_image}"
                                            }
                                        }
                                    ]
                                }
                            ]
                        }
                        
                        response = requests.post(
                            "https://api.moonshot.cn/v1/chat/completions",
                            json=payload,
                            headers=headers
                        )
                        
                        if response.status_code == 200:
                            result = response.json()["choices"][0]["message"]["content"]
                            st.success("✅ 图片分析完成")
                            with st.chat_message("assistant"):
                                st.markdown(f"**图片分析结果：**\n\n{result}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"图片 {uploaded_file.name} 的分析结果：\n\n{result}",
                                "type": "text"
                            })
                        else:
                            st.error(f"❌ 图片分析失败：{response.text}")
            
            # 3. 文档总结
            elif file_type in ["txt", "pdf", "docx", "doc"]:
                with st.spinner("📄 正在总结文档..."):
                    content = extract_text_from_file(uploaded_file)
                    if content:
                        summary_prompt = f"""请对以下文本进行专业的总结分析：

文本内容：
{content}

请从以下几个方面进行总结：
1. 核心要点（最重要的2-3个关键信息）
2. 主要内容概述
3. 重要结论或发现
4. 相关建议（如果适用）

请用清晰、专业的语言组织回答。"""

                        summary = call_model_api(summary_prompt, st.session_state.selected_model)
                        if summary:
                            st.success("✅ 文档总结完成")
                            with st.chat_message("assistant"):
                                st.markdown(f"**文档总结结果：**\n\n{summary}")
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"文档 {uploaded_file.name} 的总结：\n\n{summary}",
                                "type": "text"
                            })
            
            else:
                st.warning(f"⚠️ 不支持的文件类型：{file_type}")
        
        except Exception as e:
            st.error(f"❌ 处理文件失败：{str(e)}")
            import traceback
            st.error(f"详细错误：{traceback.format_exc()}")

# ====================
# 用户问题输入区域
with st.container():
    # 初始提示（仅在对话记录为空时显示）
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.write("您好！我是多模型智能助手，请选择模型和功能开始交互。")
            
    user_input = st.chat_input("请输入您的问题或指令...", key="user_input")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        st.session_state.messages.append({"role": "user", "content": user_input, "type": "text"})
        
        with st.spinner("🧠 正在思考..."):
            combined_response = ""
            
            # 联网搜索部分
            if st.session_state.search_enabled:
                try:
                    search_response = get_search_response(user_input)
                    if search_response:
                        combined_response += search_response + "\n\n"
                except Exception as e:
                    st.error(f"搜索过程出错：{str(e)}")
            
            # RAG 检索部分
            if st.session_state.rag_enabled:
                try:
                    rag_response = rag_generate_response(user_input)
                    if rag_response:
                        combined_response += "📚 **知识库检索结果：**\n\n" + rag_response + "\n\n"
                except Exception as e:
                    st.error(f"RAG 检索出错：{str(e)}")
            
            # 如果两个功能都未开启，使用普通对话模式
            if not (st.session_state.search_enabled or st.session_state.rag_enabled):
                response = call_model_api(user_input, st.session_state.selected_model)
                if response:
                    combined_response = response
            
            # 显示组合后的回答
            if combined_response:
                with st.chat_message("assistant"):
                    st.markdown(combined_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": combined_response,
                    "type": "text"
                })
            else:
                st.error("未能获取到任何结果，请重试。")

# ====================
# 显示历史对话记录
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"])
        else:
            st.write(msg["content"])



