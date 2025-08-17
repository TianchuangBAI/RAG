import json
from bert_score import score
import torch


def calculate_file_similarity(file1_path: str, file2_path: str, 
                            model_type: str = 'bert-base-chinese') -> dict:
    """
    计算两个文件的BERTScore相似度
    
    Args:
        file1_path (str): 第一个文件路径
        file2_path (str): 第二个文件路径
        model_type (str): BERT模型类型
        
    Returns:
        dict: 包含相似度分数的结果字典
    """
    try:
        # 读取文件内容
        with open(file1_path, 'r', encoding='utf-8') as f:
            content1 = f.read().strip()
        
        with open(file2_path, 'r', encoding='utf-8') as f:
            content2 = f.read().strip()
        
        # 简单的句子分割（按句号、问号、感叹号分割）
        sentences1 = [s.strip() for s in content1.split('。') if s.strip()]
        sentences2 = [s.strip() for s in content2.split('。') if s.strip()]
        
        # 如果句子为空，尝试其他分割方式
        if not sentences1:
            sentences1 = [s.strip() for s in content1.split('.') if s.strip()]
        if not sentences2:
            sentences2 = [s.strip() for s in content2.split('.') if s.strip()]
        
        print(f"文件1包含 {len(sentences1)} 个句子")
        print(f"文件2包含 {len(sentences2)} 个句子")
        
        if not sentences1 or not sentences2:
            return {"error": "无法从文件中提取有效句子"}
        
        # 计算BERTScore
        print("正在计算BERTScore相似度...")
        P, R, F1 = score(sentences1, sentences2, 
                         model_type=model_type, 
                         verbose=True,
                         device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 计算平均分数
        avg_precision = P.mean().item()
        avg_recall = R.mean().item()
        avg_f1 = F1.mean().item()
        
        results = {
            "file1": file1_path,
            "file2": file2_path,
            "model_type": model_type,
            "sentence_count_1": len(sentences1),
            "sentence_count_2": len(sentences2),
            "average_precision": round(avg_precision, 4),
            "average_recall": round(avg_recall, 4),
            "average_f1": round(avg_f1, 4),
            "precision_scores": [round(x.item(), 4) for x in P],
            "recall_scores": [round(x.item(), 4) for x in R],
            "f1_scores": [round(x.item(), 4) for x in F1]
        }
        
        return results
        
    except Exception as e:
        return {"error": f"计算过程中出错: {str(e)}"}


def main():
    """主函数 - 示例用法"""
    # 示例用法
    print("BERTScore文件相似度计算工具")
    print("=" * 40)
    
    # 这里可以修改为您的实际文件路径
    file1 = "reslut_example/file_1.txt"  # 替换为您的第一个文件路径
    file2 = "reslut_example/file_2.txt"  # 替换为您的第二个文件路径
    
    print(f"计算文件相似度:")
    print(f"文件1: {file1}")
    print(f"文件2: {file2}")
    print("=" * 40)
    
    # 计算相似度
    results = calculate_file_similarity(file1, file2)
    
    if "error" not in results:
        print("\n计算结果:")
        print(f"平均精确度 (Precision): {results['average_precision']}")
        print(f"平均召回率 (Recall): {results['average_recall']}")
        print(f"平均F1分数: {results['average_f1']}")
        print(f"句子数量: {results['sentence_count_1']} vs {results['sentence_count_2']}")
        
        # 保存结果到JSON文件
        with open('bertscore_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n结果已保存到 bertscore_results.json")
        
        # 显示前几个句子的分数
        print("\n前5个句子的F1分数:")
        for i, score in enumerate(results['f1_scores'][:5]):
            print(f"句子 {i+1}: {score}")
            
    else:
        print(f"错误: {results['error']}")


if __name__ == "__main__":
    main()

