import json
from bert_score import score
import torch


def calculate_line_similarity(file1_path: str, file2_path: str, 
                            model_type: str = 'bert-base-chinese') -> dict:
    """
    计算两个文件中每一行的BERTScore相似度
    
    Args:
        file1_path (str): 第一个文件路径
        file2_path (str): 第二个文件路径
        model_type (str): BERT模型类型
        
    Returns:
        dict: 包含每行相似度分数的结果字典
    """
    try:
        # 读取文件内容，按行分割
        with open(file1_path, 'r', encoding='utf-8') as f:
            lines1 = [line.strip() for line in f.readlines() if line.strip()]
        
        with open(file2_path, 'r', encoding='utf-8') as f:
            lines2 = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"文件1包含 {len(lines1)} 行")
        print(f"文件2包含 {len(lines2)} 行")
        
        if not lines1 or not lines2:
            return {"error": "无法从文件中提取有效行"}
        
        # 计算BERTScore
        print("正在计算每行的BERTScore相似度...")
        P, R, F1 = score(lines1, lines2, 
                         model_type=model_type, 
                         verbose=True,
                         device='cuda' if torch.cuda.is_available() else 'cpu')
        
        # 计算平均分数
        avg_precision = P.mean().item()
        avg_recall = R.mean().item()
        avg_f1 = F1.mean().item()
        
        # 创建每行的详细结果
        line_results = []
        for i in range(len(lines1)):
            line_result = {
                "line_number": i + 1,
                "file1_content": lines1[i],
                "file2_content": lines2[i] if i < len(lines2) else "",
                "precision": round(P[i].item(), 4),
                "recall": round(R[i].item(), 4),
                "f1_score": round(F1[i].item(), 4)
            }
            line_results.append(line_result)
        
        results = {
            "file1": file1_path,
            "file2": file2_path,
            "model_type": model_type,
            "line_count_1": len(lines1),
            "line_count_2": len(lines2),
            "average_precision": round(avg_precision, 4),
            "average_recall": round(avg_recall, 4),
            "average_f1": round(avg_f1, 4),
            "precision_scores": [round(x.item(), 4) for x in P],
            "recall_scores": [round(x.item(), 4) for x in R],
            "f1_scores": [round(x.item(), 4) for x in F1],
            "line_details": line_results
        }
        
        return results
        
    except Exception as e:
        return {"error": f"计算过程中出错: {str(e)}"}


def main():
    """主函数 - 示例用法"""
    # 示例用法
    print("BERTScore行级相似度计算工具")
    print("=" * 40)
    
    # 这里可以修改为您的实际文件路径
    file1 = "result_example/file_1.txt"  # 替换为您的第一个文件路径
    file2 = "result_example/file_2.txt"  # 替换为您的第二个文件路径
    
    print(f"计算文件相似度:")
    print(f"文件1: {file1}")
    print(f"文件2: {file2}")
    print("=" * 40)
    
    # 计算相似度
    results = calculate_line_similarity(file1, file2)
    
    if "error" not in results:
        print("\n计算结果:")
        print(f"平均精确度 (Precision): {results['average_precision']}")
        print(f"平均召回率 (Recall): {results['average_recall']}")
        print(f"平均F1分数: {results['average_f1']}")
        print(f"行数: {results['line_count_1']} vs {results['line_count_2']}")
        
        # 保存结果到JSON文件
        with open('bertscore_line_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n结果已保存到 bertscore_line_results.json")
        
        # 显示前几行的分数
        print("\n前5行的F1分数:")
        for i, score in enumerate(results['f1_scores'][:5]):
            print(f"行 {i+1}: {score}")
            
        # 显示相似度最高的前5行
        print("\n相似度最高的前5行:")
        sorted_lines = sorted(results['line_details'], key=lambda x: x['f1_score'], reverse=True)
        for i, line_result in enumerate(sorted_lines[:5]):
            print(f"第{i+1}名 - 行 {line_result['line_number']}: F1={line_result['f1_score']}")
            print(f"  文件1: {line_result['file1_content'][:50]}...")
            print(f"  文件2: {line_result['file2_content'][:50]}...")
            print()
            
    else:
        print(f"错误: {results['error']}")


if __name__ == "__main__":
    main()


