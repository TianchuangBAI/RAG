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
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from sentence_transformers import CrossEncoder

# 文本处理
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class LoRARAGSystem:
    def __init__(self, 
                 base_model_path="./Qwen/Qwen3-14B",
                 lora_checkpoint_path="checkpoint",
                 chunk_size=500,
                 chunk_overlap=50,
                 top_k=10,
                 rerank_top_k=3,
                 use_fast=False,
                 trust_remote_code=True,
                 knowledge_base_type="auto"):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.rerank_top_k = rerank_top_k
        self.base_model_path = base_model_path
        self.lora_checkpoint_path = lora_checkpoint_path
        self.knowledge_base_type = knowledge_base_type
        
        # 初始化本地LLM
        print("正在加载LoRA微调模型...")
        print(f"基础模型路径: {base_model_path}")
        print(f"LoRA checkpoint路径: {lora_checkpoint_path}")
        
        try:
            # 加载原下载路径的tokenizer和model
            print("📥 加载基础模型...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path, 
                use_fast=use_fast, 
                trust_remote_code=trust_remote_code
            )
            
            self.base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, 
                device_map="auto", 
                torch_dtype=torch.bfloat16
            )
            print("✅ 基础模型加载完成")
            
            # 加载LoRA模型
            print("📥 加载LoRA微调权重...")
            self.model = PeftModel.from_pretrained(
                self.base_model, 
                model_id=lora_checkpoint_path
            )
            print("✅ LoRA微调模型加载完成")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            print("请检查模型路径和文件完整性")
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
            print("✅ ")
        except:
            print("将使用余弦相似度作为排序")
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
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """将文本分割成chunks"""
        # 简单的文本分割策略
        sentences = re.split(r'[。！？\n]', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + "。"
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "。"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """获取文本的embeddings"""
        embeddings = []
        batch_size = 10  # 批量处理以避免API限制
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            try:
                response = self.embedding_client.embeddings.create(
                    model=self.embedding_model,
                    input=batch_texts,
                    dimensions=1024,
                    encoding_format="float"
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
                print(f"已处理 {min(i+batch_size, len(texts))}/{len(texts)} 条文本的embedding")
                
            except Exception as e:
                print(f"获取embedding时出错: {e}")
                # 如果出错，添加零向量作为占位符
                for _ in batch_texts:
                    embeddings.append([0.0] * 1024)
        
        return np.array(embeddings)
    
    def build_knowledge_base(self, pdf_folder: str = None, save_path: str = None):
        """构建知识库"""
        print("正在构建知识库...")
        
        # 自动确定知识库类型和路径
        if self.knowledge_base_type == "auto":
            # 根据checkpoint路径自动检测
            if "TCM" in self.lora_checkpoint_path:
                self.knowledge_base_type = "TCM"
            elif "MM" in self.lora_checkpoint_path:
                self.knowledge_base_type = "MM"
            else:
                print("⚠️ 无法自动检测知识库类型")
                return False
        
        # 根据类型确定默认路径
        if pdf_folder is None:
            if self.knowledge_base_type == "TCM":
                pdf_folder = "knowledge_base_TCM"
            elif self.knowledge_base_type == "MM":
                pdf_folder = "knowledge_base_MM"
            else:
                print(f"❌ 不支持的知识库类型: {self.knowledge_base_type}")
                return False
        
        if save_path is None:
            if self.knowledge_base_type == "TCM":
                save_path = "knowledge_base_TCM.pkl"
            elif self.knowledge_base_type == "MM":
                save_path = "knowledge_base_MM.pkl"
            else:
                save_path = "knowledge_base.pkl"
        
        print(f"🔍 构建 {self.knowledge_base_type} 知识库")
        print(f"📁 PDF文件夹: {pdf_folder}")
        print(f"💾 保存路径: {save_path}")
        
        # 检查是否已存在知识库文件
        if os.path.exists(save_path):
            print("发现已存在的知识库文件，正在加载...")
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_embeddings = data['embeddings']
                self.index = data['index']
            print("✅ 知识库加载完成")
            return True
        
        # 读取所有PDF文件
        all_text = ""
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))
        
        if not pdf_files:
            print(f"❌ 在 {pdf_folder} 目录中未找到PDF文件")
            return
        
        for pdf_file in pdf_files:
            print(f"正在处理: {pdf_file}")
            text = self.extract_text_from_pdf(str(pdf_file))
            all_text += f"\n\n=== {pdf_file.name} ===\n\n" + text
        
        # 分割文本
        print("正在分割文本...")
        self.chunks = self.split_text_into_chunks(all_text)
        print(f"共生成 {len(self.chunks)} 个文本块")
        
        # 生成embeddings
        print("正在生成embeddings...")
        embeddings = self.get_embeddings(self.chunks)
        self.chunk_embeddings = embeddings
        
        # 构建FAISS索引
        print("正在构建FAISS索引...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # 使用内积相似度
        
        # 归一化embeddings以使用内积计算余弦相似度
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.index.add(normalized_embeddings.astype(np.float32))
        
        # 保存知识库
        print("正在保存知识库...")
        with open(save_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'embeddings': self.chunk_embeddings,
                'index': self.index
            }, f)
        
        print("✅ 知识库构建完成")
        return True
    
    def load_knowledge_base(self, knowledge_base_path: str = None):
        """加载预构建的知识库"""
        # 如果没有指定路径，自动检测
        if knowledge_base_path is None:
            if self.knowledge_base_type == "TCM":
                knowledge_base_path = "knowledge_base_TCM.pkl"
            elif self.knowledge_base_type == "MM":
                knowledge_base_path = "knowledge_base_MM.pkl"
            else:
                print("❌ 无法确定知识库路径")
                return False
        
        print(f"正在加载知识库: {knowledge_base_path}")
        
        if not os.path.exists(knowledge_base_path):
            print(f"❌ 知识库文件不存在: {knowledge_base_path}")
            print("系统将自动构建新的知识库")
            return False
        
        try:
            with open(knowledge_base_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data['chunks']
                self.chunk_embeddings = data['embeddings']
                self.index = data['index']
            print("✅ 知识库加载完成")
            return True
        except Exception as e:
            print(f"❌ 加载知识库失败: {e}")
            return False
    
    def auto_detect_knowledge_base(self):
        """自动检测并加载对应的知识库"""
        print("🔍 自动检测知识库类型...")
        
        # 直接调用build_knowledge_base，它会自动处理所有逻辑
        return self.build_knowledge_base()
    
    def retrieve_relevant_docs(self, query: str) -> List[Dict[str, Any]]:
        """检索相关文档"""
        # 获取查询的embedding
        query_embedding = self.get_embeddings([query])[0]
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # 检索
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32), 
            self.top_k
        )
        
        # 整理检索结果
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # 确保索引有效
                results.append({
                    'text': self.chunks[idx],
                    'score': float(score),
                    'index': int(idx)
                })
        
        return results
    
    def rerank_documents(self, query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """重排序文档"""
        if not documents:
            return documents
        
        if self.reranker is not None:
            try:
                # 使用CrossEncoder进行重排序
                pairs = [(query, doc['text']) for doc in documents]
                scores = self.reranker.predict(pairs)
                
                # 更新分数并排序
                for i, score in enumerate(scores):
                    documents[i]['rerank_score'] = float(score)
                
                documents = sorted(documents, key=lambda x: x['rerank_score'], reverse=True)
            except Exception as e:
                print(f"重排序失败，使用原始分数: {e}")
        
        # 返回top_k个结果
        return documents[:self.rerank_top_k]
    
    def generate_answer(self, query: str, context_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成答案"""
        # 构建上下文
        context = "\n\n".join([f"参考文档{i+1}:\n{doc['text']}" for i, doc in enumerate(context_docs)])
        
        # 构建prompt
        prompt = f"""基于以下参考文档回答问题，如果参考文档中没有相关信息，请说明无法从提供的文档中找到答案。

参考文档:
{context}

问题: {query}

请提供准确、详细的答案:"""

        try:
            # 准备模型输入
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成答案
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,  # 使用更大的token数量，适合14B模型
                temperature=0.7,      # 适中的温度
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # 解析thinking内容
            try:
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            
            return {
                'answer': content,
                'thinking': thinking_content,
                'context_docs': context_docs,
                'success': True
            }
            
        except Exception as e:
            print(f"生成答案时出错: {e}")
            return {
                'answer': f"生成答案时出现错误: {e}",
                'thinking': "",
                'context_docs': context_docs,
                'success': False
            }
    
    def process_single_query(self, query: str) -> Dict[str, Any]:
        """处理单个查询"""
        print(f"正在处理查询: {query}")
        
        # 检索相关文档
        retrieved_docs = self.retrieve_relevant_docs(query)
        
        # 重排序
        reranked_docs = self.rerank_documents(query, retrieved_docs)
        
        # 生成答案
        result = self.generate_answer(query, reranked_docs)
        
        return {
            'query': query,
            'retrieved_docs': retrieved_docs,
            'reranked_docs': reranked_docs,
            'answer': result['answer'],
            'thinking': result['thinking'],
            'success': result['success']
        }
