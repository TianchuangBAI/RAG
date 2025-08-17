import os
from rag_system_lora import LoRARAGSystem


def main():
    """主函数"""
    # 配置参数 - 直接使用您指定的路径
    PDF_FOLDER = "knowledge_base_MM"  # PDF文件夹路径
    BASE_MODEL_PATH = "./Qwen/Qwen3-14B"  # 基础模型路径
    LORA_CHECKPOINT_PATH = "./output/Qwen3-14B-en/checkpoint"   # LoRA checkpoint路径
    QUERY_FILE = "query/query_MM.txt"  # 查询文件路径
    OUTPUT_FILE = "output.txt"  # 输出文件路径
    
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
    
    # 检查查询文件是否存在
    if not os.path.exists(QUERY_FILE):
        print(f"❌ 查询文件不存在: {QUERY_FILE}")
        print("请检查查询文件路径是否正确")
        return
    
    print("🚀 LoRA微调模型RAG系统批量处理模式")
    print("="*50)
    print(f"✅ 基础模型: {BASE_MODEL_PATH}")
    print(f"✅ LoRA checkpoint: {LORA_CHECKPOINT_PATH}")
    print(f"✅ PDF文件夹: {PDF_FOLDER}")
    print(f"✅ 查询文件: {QUERY_FILE}")
    print(f"✅ 输出文件: {OUTPUT_FILE}")
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
            print(f"🔄 正在处理第 {i}/{len(queries)} 个查询: {query[:50]}{'...' if len(query) > 50 else ''}")
            
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
        print("批量处理完成！正在保存结果...")
        print("-"*60)
        
        # 保存结果到输出文件
        try:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                for i, result in enumerate(results, 1):
                    f.write(f"查询 {i}:\n")
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
            print(f"📊 处理统计:")
            print(f"   总查询数: {len(results)}")
            print(f"   成功数: {success_count}")
            print(f"   失败数: {len(results) - success_count}")
            print(f"   成功率: {success_count/len(results)*100:.1f}%")
            
        except Exception as e:
            print(f"❌ 保存结果文件失败: {e}")
            return
        
        print("\n🎉 批量处理完成！")
    
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        print("请检查配置和依赖")


if __name__ == "__main__":
    main()
