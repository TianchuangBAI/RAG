import json
import os
import numpy as np
from typing import List, Dict, Any
import faiss
import pickle
from pathlib import Path
import re
import torch

# PDF处理
import PyPDF2
from io import BytesIO

# 模型相关
from openai import OpenAI
from modelscope import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import CrossEncoder

# 文本处理
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGSystem:
    def __init__(self, 
                 chunk_size=500,
                 chunk_overlap=50,
                 top_k=10,
                 rerank_top_k=3):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        
        # 固定使用Qwen3-14B模型和缓存目录
        model_name = "Qwen/Qwen3-14B"
        model_cache_dir = "autodl-tmp/qwen"
        
        # 创建模型缓存目录
        os.makedirs(model_cache_dir, exist_ok=True)
        print(f"模型缓存目录: {model_cache_dir}")
        
        # 初始化本地LLM
        print("正在加载本地LLM模型...")
        print(f"使用模型: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=model_cache_dir,  # 设置缓存目录
                local_files_only=False  # 允许从网络下载
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=model_cache_dir,  # 设置缓存目录
                torch_dtype="auto",
                device_map="auto",
                low_cpu_mem_usage=True,  # 添加低内存使用选项
                local_files_only=False  # 允许从网络下载
            )
            print("本地LLM模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请检查网络连接或模型名称")
            raise e
        
        # 初始化embedding客户端
        self.embedding_client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.embedding_model = "text-embedding-v4"
        
        # 初始化rerank模型
        print("正在加载重排序模型...")
        try:
            self.reranker = CrossEncoder("BAAI/bge-reranker-base")
            print("重排序模型加载完成")
        except:
            print("重排序模型加载失败，将使用余弦相似度作为备选")
            self.reranker = None
        
        # 向量库相关
        self.index = None
        self.chunks = []
        self.chunk_embeddings = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """从PDF文件中提取文本"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"读取PDF文件 {pdf_path} 时出错: {e}")
        return text
    
    def chunk_text(self, text: str) -> List[str]:
        """将文本分割成块"""
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            if end > text_length:
                end = text_length
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start >= text_length:
                break
        
        return chunks
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本的embedding"""
        try:
            response = self.embedding_client.embeddings.create(
                model=self.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取embedding失败: {e}")
            return []
    
    def build_knowledge_base(self, pdf_folder: str):
        """构建知识库"""
        print(f"正在构建知识库，PDF文件夹: {pdf_folder}")
        
        # 检查PDF文件夹是否存在
        if not os.path.exists(pdf_folder):
            print(f"PDF文件夹不存在: {pdf_folder}")
            return
        
        # 获取所有PDF文件
        pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
        if not pdf_files:
            print(f"在 {pdf_folder} 中未找到PDF文件")
            return
        
        print(f"找到 {len(pdf_files)} 个PDF文件")
        
        # 处理每个PDF文件
        for pdf_file in pdf_files:
            pdf_path = os.path.join(pdf_folder, pdf_file)
            print(f"正在处理: {pdf_file}")
            
            # 提取文本
            text = self.extract_text_from_pdf(pdf_path)
            if not text.strip():
                print(f"  {pdf_file} 未提取到文本内容")
                continue
            
            # 分割文本
            file_chunks = self.chunk_text(text)
            print(f"  分割成 {len(file_chunks)} 个文本块")
            
            # 获取embedding
            for chunk in file_chunks:
                embedding = self.get_embedding(chunk)
                if embedding:
                    self.chunks.append(chunk)
                    self.chunk_embeddings.append(embedding)
        
        if not self.chunks:
            print("未成功提取任何文本块")
            return
        
        print(f"总共提取了 {len(self.chunks)} 个文本块")
        
        # 构建FAISS索引
        print("正在构建FAISS索引...")
        embeddings_array = np.array(self.chunk_embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(embeddings_array.shape[1])
        self.index.add(embeddings_array)
        
        print("知识库构建完成！")
    
    def retrieve_documents(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """检索相关文档"""
        if not self.index or not self.chunks:
            return []
        
        if top_k is None:
            top_k = self.top_k
        
        # 获取查询的embedding
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        
        # 搜索相似文档
        query_embedding_array = np.array([query_embedding]).astype('float32')
        scores, indices = self.index.search(query_embedding_array, top_k)
        
        # 构建结果
        retrieved_docs = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                retrieved_docs.append({
                    'text': self.chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return retrieved_docs
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
        """重排序文档"""
        if not documents:
            return []
        
        if top_k is None:
            top_k = self.rerank_top_k
        
        if self.reranker:
            # 使用CrossEncoder重排序
            try:
                pairs = [(query, doc['text']) for doc in documents]
                scores = self.reranker.predict(pairs)
                
                # 将分数添加到文档中
                for doc, score in zip(documents, scores):
                    doc['rerank_score'] = float(score)
                
                # 按重排序分数排序
                reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
                return reranked_docs[:top_k]
            except Exception as e:
                print(f"CrossEncoder重排序失败: {e}")
                # 回退到余弦相似度
                pass
        
        # 使用余弦相似度重排序
        try:
            # 获取查询的TF-IDF向量
            query_tokens = list(jieba.cut(query))
            query_text = ' '.join(query_tokens)
            
            # 获取文档的TF-IDF向量
            doc_texts = [doc['text'] for doc in documents]
            doc_tokens = [list(jieba.cut(doc)) for doc in doc_texts]
            doc_texts_processed = [' '.join(tokens) for tokens in doc_tokens]
            
            # 计算TF-IDF
            all_texts = [query_text] + doc_texts_processed
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            
            # 计算余弦相似度
            query_vector = tfidf_matrix[0:1]
            doc_vectors = tfidf_matrix[1:]
            similarities = cosine_similarity(query_vector, doc_vectors).flatten()
            
            # 将相似度分数添加到文档中
            for doc, similarity in zip(documents, similarities):
                doc['rerank_score'] = float(similarity)
            
            # 按相似度排序
            reranked_docs = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            return reranked_docs[:top_k]
            
        except Exception as e:
            print(f"TF-IDF重排序也失败: {e}")
            # 返回原始排序的文档
            return documents[:top_k]
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成答案"""
        try:
            # 构建上下文
            context = "\n\n".join([doc['text'] for doc in context_docs])
            
            # 构建提示词
            prompt = f"""基于以下上下文信息，回答用户的问题。请提供准确、详细的答案。

上下文信息：
{context}

用户问题：{query}

请回答："""

            # 生成答案
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取生成的答案部分
            answer = answer.replace(prompt, "").strip()
            
            return {
                'answer': answer,
                'thinking': f"基于 {len(context_docs)} 个相关文档生成答案",
                'success': True
            }
            
        except Exception as e:
            return {
                'answer': f"生成答案时出错: {e}",
                'thinking': "",
                'success': False
            }
    
    def process_single_query(self, query: str) -> Dict[str, Any]:
        """处理单个查询"""
        # 检索文档
        retrieved_docs = self.retrieve_documents(query)
        if not retrieved_docs:
            return {
                'answer': "未找到相关文档",
                'thinking': "检索失败",
                'success': False,
                'retrieved_docs': [],
                'reranked_docs': []
            }
        
        # 重排序文档
        reranked_docs = self.rerank_documents(query, retrieved_docs)
        
        # 生成答案
        result = self.generate_answer(query, reranked_docs)
        
        return {
            'retrieved_docs': retrieved_docs,
            'reranked_docs': reranked_docs,
            'answer': result['answer'],
            'thinking': result['thinking'],
            'success': result['success']
        }
    
    def batch_inference(self, input_json_path: str, output_json_path: str):
        """批量推理"""
        print(f"开始批量推理，输入文件: {input_json_path}")
        
        # 读取输入数据
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        
        results = []
        
        for i, item in enumerate(input_data):
            query = item.get('input', '')
            expected_output = item.get('output', '')
            
            print(f"\n处理第 {i+1}/{len(input_data)} 个查询...")
            
            # 处理查询
            result = self.process_single_query(query)
            
            # 添加原始数据和期望输出
            result.update({
                'original_instruction': item.get('instruction', ''),
                'original_input': item.get('input', ''),
                'expected_output': expected_output,
                'index': i
            })
            
            results.append(result)
            
            # 定期保存中间结果
            if (i + 1) % 10 == 0:
                temp_output_path = output_json_path.replace('.json', f'_temp_{i+1}.json')
                with open(temp_output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"已保存中间结果到: {temp_output_path}")
        
        # 保存最终结果
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"\n批量推理完成，结果已保存到: {output_json_path}")
        print(f"总共处理了 {len(results)} 个查询")


def interactive_mode(rag_system):
    """交互式推理模式"""
    print("\n" + "="*60)
    print("RAG + LLM 交互式问答系统")
    print("="*60)
    print("系统已准备就绪！您可以开始提问了。")
    print("输入 'quit' 或 'exit' 退出系统")
    print("输入 'help' 查看帮助信息")
    print("-"*60)
    
    # 交互式问答循环
    while True:
        try:
            # 获取用户输入
            query = input("\n请输入您的问题: ").strip()
            
            # 检查退出命令
            if query.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用，再见！")
                break
            
            # 检查帮助命令
            if query.lower() in ['help', '帮助', 'h']:
                print("\n帮助信息:")
                print("- 直接输入问题即可获得答案")
                print("- 系统会自动检索相关文档并生成回答")
                print("- 输入 'quit' 或 'exit' 退出系统")
                print("- 输入 'help' 查看此帮助信息")
                continue
            
            # 检查空输入
            if not query:
                print("请输入有效的问题。")
                continue
            
            print(f"\n正在处理您的问题: {query}")
            print("-"*40)
            
            # 处理查询
            result = rag_system.process_single_query(query)
            
            # 显示检索到的相关文档
            print("\n📚 检索到的相关文档:")
            print("-"*40)
            if result['reranked_docs']:
                for i, doc in enumerate(result['reranked_docs']):
                    print(f"文档 {i+1} (相似度: {doc.get('rerank_score', doc.get('score', 0)):.4f}):")
                    # 截取文档内容的前200个字符
                    doc_preview = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                    print(f"  {doc_preview}")
                    print()
            else:
                print("未找到相关文档")
            
            # 显示生成的答案
            print("\n🤖 AI 生成的答案:")
            print("-"*40)
            if result['success']:
                print(result['answer'])
                
                # 如果启用了thinking，显示推理过程
                if result.get('thinking'):
                    print(f"\n💭 推理过程:")
                    print("-"*20)
                    print(result['thinking'])
            else:
                print(f"❌ 生成答案时出现错误: {result['answer']}")
            
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n检测到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 处理过程中出现错误: {e}")
            print("请重试或联系技术支持")


def batch_mode(rag_system):
    """批量推理模式"""
    print("\n" + "="*60)
    print("RAG + LLM 批量推理系统")
    print("="*60)
    
    # 获取输入和输出文件路径
    input_json_path = input("请输入JSON文件路径: ").strip()
    if not input_json_path:
        print("❌ 请输入有效的文件路径")
        return
    
    if not os.path.exists(input_json_path):
        print(f"❌ 输入文件不存在: {input_json_path}")
        return
    
    output_json_path = input("请输入输出JSON文件路径: ").strip()
    if not output_json_path:
        print("❌ 请输入有效的输出文件路径")
        return
    
    print(f"开始批量推理...")
    print(f"输入文件: {input_json_path}")
    print(f"输出文件: {output_json_path}")
    
    # 执行批量推理
    rag_system.batch_inference(input_json_path, output_json_path)


def main():
    # 配置参数
    PDF_FOLDER = "knowledge_base_TCM"  # PDF文件夹路径,需修改此处代码更新结果
    
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("请设置环境变量 DASHSCOPE_API_KEY")
        return
    
    print("RAG + LLM 系统配置:")
    print("- 使用模型: Qwen3-14B")
    print("- 模型缓存目录: autodl-tmp/qwen")
    print("- 文本块大小: 500字符")
    print("- 检索数量: 10个文档")
    print("- 重排序后保留: 3个文档")
    print()
    
    # 初始化RAG系统
    rag_system = RAGSystem(
        chunk_size=500,
        chunk_overlap=50,
        top_k=10,
        rerank_top_k=3
    )
    
    # 构建知识库
    rag_system.build_knowledge_base(PDF_FOLDER)
    
    print("\n" + "="*60)
    print("RAG + LLM 系统")
    print("="*60)
    print("系统已准备就绪！请选择运行模式：")
    print("1. 交互式推理模式")
    print("2. 批量推理模式")
    print("-"*60)
    
    while True:
        try:
            choice = input("请输入您的选择 (1 或 2): ").strip()
            
            if choice == "1":
                interactive_mode(rag_system)
                break
            elif choice == "2":
                batch_mode(rag_system)
                break
            else:
                print("❌ 无效选择，请输入 1 或 2")
                continue
                
        except KeyboardInterrupt:
            print("\n\n检测到中断信号，正在退出...")
            break
        except Exception as e:
            print(f"\n❌ 选择过程中出现错误: {e}")
            print("请重试")


if __name__ == "__main__":
    main()
