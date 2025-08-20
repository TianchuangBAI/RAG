import os
from rag_system_lora import LoRARAGSystem


def interactive_mode(rag_system):
    """交互式问答模式"""
    print("\n" + "="*60)
    print("LoRA微调模型RAG系统 - 交互模式")
    print("="*60)
    print("系统已准备就绪！您可以开始提问了。")
    print("输入 'quit' 或 'exit' 退出系统")
    print("输入 'help' 查看帮助信息")
    print("-"*60)
    
    # 交互式问答循环
    while True:
        try:
            # 获取用户输入
            query = input("\n请输入您的西医问题: ").strip()
            
            # 检查退出命令
            if query.lower() in ['quit', 'exit', '退出', 'q']:
                print("感谢使用，再见！")
                break
            
            # 检查帮助命令
            if query.lower() in ['help', '帮助', 'h']:
                print("\n帮助信息:")
                print("- 直接输入西医相关问题即可获得答案")
                print("- 系统会自动检索相关西医文档并生成回答")
                print("- 使用LoRA微调后的模型，具有更好的西医领域适应性")
                print("- 输入 'quit' 或 'exit' 退出系统")
                print("- 输入 'help' 查看此帮助信息")
                continue
            
            # 检查空输入
            if not query:
                print("请输入有效的西医问题。")
                continue
            
            print(f"\n正在处理您的西医问题: {query}")
            print("-"*40)
            
            # 处理查询
            result = rag_system.process_single_query(query)
            
            # 显示检索到的相关文档
            print("\n📚 检索到的相关西医文档:")
            print("-"*40)
            if result['reranked_docs']:
                for i, doc in enumerate(result['reranked_docs']):
                    print(f"文档 {i+1} (相似度: {doc.get('score', 0):.4f}):")
                    # 截取文档内容的前200个字符
                    doc_preview = doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
                    print(f"  {doc_preview}")
                    print()
            else:
                print("未找到相关MM文档")
            
            # 显示生成的答案
            print("\n🤖 LoRA微调模型生成的西医答案:")
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
    """批量处理模式"""
    QUERY_FILE = "query/query_MM.txt"  # 查询文件路径
    OUTPUT_FILE = "output.txt"  # 输出文件路径
    
    # 检查查询文件是否存在
    if not os.path.exists(QUERY_FILE):
        print(f"❌ 查询文件不存在: {QUERY_FILE}")
        print("请检查查询文件路径是否正确")
        return
    
    print("\n" + "="*60)
    print("MM LoRA微调模型RAG系统 - 批量处理模式")
    print("="*60)
    print("系统已准备就绪！开始批量处理查询...")
    print("-"*60)
    
    # 读取查询文件
    try:
        with open(QUERY_FILE, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"❌ 读取查询文件失败: {e}")
        return
    
    print(f"📖 共读取到 {len(queries)} 个查询")
    print()
    
    # 批量处理查询
    results = []
    for i, query in enumerate(queries, 1):
        print(f"🔄 正在处理第 {i}/{len(queries)} 个MM查询: {query[:50]}{'...' if len(query) > 50 else ''}")
        
        try:
            # 处理查询
            result = rag_system.process_single_query(query)
            
            # 准备结果
            query_result = {
                'query': query,
                'success': result['success'],
                'answer': result['answer'] if result['success'] else f"错误: {result['answer']}",
                'thinking': result.get('thinking', ''),
                'reranked_docs': result.get('reranked_docs', [])
            }
            
            results.append(query_result)
            
            # 显示进度
            if result['success']:
                print(f"   ✅ 成功生成答案")
            else:
                print(f"   ❌ 生成答案失败: {result['answer']}")
            
        except Exception as e:
            print(f"   ❌ 处理查询时出现错误: {e}")
            query_result = {
                'query': query,
                'success': False,
                'answer': f"处理错误: {e}",
                'thinking': '',
                'reranked_docs': []
            }
            results.append(query_result)
    
    print("\n" + "="*60)
    print("MM批量处理完成！正在保存结果...")
    print("-"*60)
    
    # 保存结果到输出文件
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write("LoRA微调模型RAG系统批量处理结果\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                f.write(f"MM查询 {i}:\n")
                f.write(f"问题: {result['query']}\n")
                f.write(f"状态: {'成功' if result['success'] else '失败'}\n")
                f.write(f"答案: {result['answer']}\n")
                
                if result['thinking']:
                    f.write(f"推理过程: {result['thinking']}\n")
                
                if result['reranked_docs']:
                    f.write("相关文档:\n")
                    for j, doc in enumerate(result['reranked_docs']):
                        f.write(f"  文档 {j+1} (相似度: {doc.get('score', 0):.4f}): {doc['text'][:200]}...\n")
                
                f.write("\n" + "="*80 + "\n\n")
        
        print(f"✅ 结果已保存到: {OUTPUT_FILE}")
        
        # 统计结果
        success_count = sum(1 for r in results if r['success'])
        print(f"📊 MM处理统计:")
        print(f"   总查询数: {len(results)}")
        print(f"   成功数: {success_count}")
        print(f"   失败数: {len(results) - success_count}")
        print(f"   成功率: {success_count/len(results)*100:.1f}%")
        
    except Exception as e:
        print(f"❌ 保存结果文件失败: {e}")
        return
    
    print("\n🎉 批量处理完成！")


def main():
    """主函数 - 用户选择模式"""
    # 配置参数 - 使用MM相关的路径
    PDF_FOLDER = "knowledge_base_MM"  # PDF文件夹路径
    BASE_MODEL_PATH = "./Qwen/Qwen3-14B"  # 基础模型路径
    LORA_CHECKPOINT_PATH = "output/checkpoint/checkpoint"   # LoRA checkpoint路径
    
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
    
    print("🚀 MM LoRA微调模型RAG系统")
    print("="*50)
    print(f"✅ 基础模型: {BASE_MODEL_PATH}")
    print(f"✅ LoRA checkpoint: {LORA_CHECKPOINT_PATH}")
    print(f"✅ PDF文件夹: {PDF_FOLDER}")
    print()
    
    try:
        # 初始化LoRA RAG系统
        print("正在初始化MM RAG系统...")
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
        print("MM LoRA微调模型RAG系统")
        print("="*60)
        print("系统已准备就绪！请选择运行模式：")
        print("1. 交互式问答模式")
        print("2. 批量处理模式")
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
    
    except Exception as e:
        print(f"❌ MM系统初始化失败: {e}")
        print("请检查配置和依赖")


if __name__ == "__main__":
    main()
