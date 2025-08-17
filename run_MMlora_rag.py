import os
from rag_system_lora import LoRARAGSystem


def main():
    """主函数"""
    # 配置参数 - 直接使用您指定的路径
    PDF_FOLDER = "knowledge_base_MM"  # PDF文件夹路径
    BASE_MODEL_PATH = "./Qwen/Qwen3-14B"  # 基础模型路径
    LORA_CHECKPOINT_PATH = "./output/Qwen3-14B-en/checkpoint"   # LoRA checkpoint路径
    
    # 检查环境变量
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("❌ 请设置环境变量 DASHSCOPE_API_KEY")
        print("设置命令: export DASHSCOPE_API_KEY='your_api_key'")
        return
    
    # 验证路径
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"❌ 基础模型路径不存在: {BASE_MODEL_PATH}")
        print("请检查基础模型路径是否正确")
        return
    
    if not os.path.exists(LORA_CHECKPOINT_PATH):
        print(f"❌ LoRA checkpoint路径不存在: {LORA_CHECKPOINT_PATH}")
        print("请检查checkpoint路径是否正确")
        return
    
    print("🚀 LoRA微调模型RAG系统快速启动")
    print("="*50)
    print(f"✅ 基础模型: {BASE_MODEL_PATH}")
    print(f"✅ LoRA checkpoint: {LORA_CHECKPOINT_PATH}")
    print(f"✅ PDF文件夹: {PDF_FOLDER}")
    print()
    
    try:
        # 初始化LoRA RAG系统
        print("正在初始化系统...")
        rag_system = LoRARAGSystem(
            base_model_path=BASE_MODEL_PATH,
            lora_checkpoint_path=LORA_CHECKPOINT_PATH,
            chunk_size=500,
            chunk_overlap=50,
            top_k=10,
            rerank_top_k=3,
            use_fast=False,
            trust_remote_code=True
        )
        
        # 构建知识库
        rag_system.build_knowledge_base(PDF_FOLDER)
        
        print("\n" + "="*60)
        print("LoRA微调模型RAG系统")
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
                    print("- 使用LoRA微调后的模型，具有更好的领域适应性")
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
                        print(f"文档 {i+1} (相似度: {doc.get('score', 0):.4f}):")
                        # 截取文档内容的前200个字符
                        doc_preview = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                        print(f"  {doc_preview}")
                        print()
                else:
                    print("未找到相关文档")
                
                # 显示生成的答案
                print("\n🤖 LoRA微调模型生成的答案:")
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
    
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查配置和依赖")


if __name__ == "__main__":
    main()
