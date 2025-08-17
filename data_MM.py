import json
import random
from pathlib import Path

def load_jsonl_with_error_handling(file_path):
    """加载JSONL文件，跳过格式错误的行并记录错误"""
    data = []
    error_lines = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                error_lines.append({
                    'line_number': line_no,
                    'error': str(e),
                    'content': line[:100] + '...' if len(line) > 100 else line  # 截断过长的行
                })
    
    return data, error_lines

def save_jsonl(data, file_path):
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():
    # 配置参数
    random.seed(42)
    input_file = 'train_en.json'  # 替换为你的输入文件路径
    train_output = 'train_en.jsonl'
    val_output = 'val_en.jsonl'
    train_ratio = 0.9  # 训练集比例

    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误：输入文件 {input_file} 不存在！")
        return

    # 加载数据并处理错误
    print("正在加载数据...")
    data_list, errors = load_jsonl_with_error_handling(input_file)
    
    # 输出错误信息
    if errors:
        print(f"\n发现 {len(errors)} 行格式错误：")
        for err in errors[:5]:  # 最多显示前5个错误
            print(f"行 {err['line_number']}: {err['error']}")
            print(f"内容: {err['content']}\n")
        print(f"已跳过这些错误行，完整错误日志请查看程序输出。")
    
    if not data_list:
        print("错误：没有加载到有效数据！")
        return

    # 分割数据
    random.shuffle(data_list)
    split_idx = int(len(data_list) * train_ratio)
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]

    # 保存数据
    save_jsonl(train_data, train_output)
    save_jsonl(val_data, val_output)

    # 输出结果
    print("\n数据处理完成：")
    print(f"输入文件: {input_file}")
    print(f"总有效行数: {len(data_list)}")
    print(f"训练集已保存到: {train_output} ({len(train_data)} 条)")
    print(f"验证集已保存到: {val_output} ({len(val_data)} 条)")
    
    if errors:
        print(f"\n警告：跳过了 {len(errors)} 行错误数据")

if __name__ == '__main__':
    main()
