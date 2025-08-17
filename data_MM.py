import json
import random
from pathlib import Path
import os

def load_json_or_jsonl_with_error_handling(file_path):
    """加载JSON或JSONL文件，自动识别格式并处理可能的格式错误"""
    data = []
    errors = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # 尝试作为标准JSON文件读取
            try:
                f.seek(0)  # 重置文件指针
                content = f.read()
                parsed_data = json.loads(content)
                
                # 如果成功解析为标准JSON
                if isinstance(parsed_data, list):
                    data = parsed_data
                    print(f"检测到标准JSON格式，包含 {len(data)} 条数据")
                elif isinstance(parsed_data, dict):
                    data = [parsed_data]
                    print(f"检测到标准JSON格式（单个对象），已包装为列表")
                else:
                    print(f"警告：文件包含未知数据类型: {type(parsed_data)}")
                    return [], []
                    
            except json.JSONDecodeError:
                # 如果标准JSON解析失败，尝试作为JSONL文件读取
                print("检测到JSONL格式，正在逐行解析...")
                f.seek(0)  # 重置文件指针
                
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue
                        
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        errors.append({
                            'line_number': line_no,
                            'error': str(e),
                            'content': line[:100] + '...' if len(line) > 100 else line
                        })
                
                if data:
                    print(f"成功解析JSONL格式，包含 {len(data)} 条数据")
                if errors:
                    print(f"发现 {len(errors)} 行格式错误")
                    
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return [], []
    except Exception as e:
        print(f"错误：读取文件 {file_path} 时发生未知错误: {e}")
        return [], []
    
    return data, errors

def save_jsonl(data, file_path):
    """保存数据到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def main():
    # 配置参数
    random.seed(42)
    
    print("=" * 50)
    print("ChatDoctor 训练数据处理工具")
    print("=" * 50)
    
    # 使用train_en.json文件作为输入
    input_file = 'train_en.json'  # 训练数据文件
    print(f"使用训练数据文件: {input_file}")
    
    train_output = 'train_en.jsonl'
    val_output = 'val_en.jsonl'
    train_ratio = 0.9  # 训练集比例

    # 检查输入文件是否存在
    if not Path(input_file).exists():
        print(f"错误：输入文件 {input_file} 不存在！")
        print("请确保 train_en.json 文件存在于当前目录")
        print("\n说明：")
        print("当前使用训练数据文件: train_en.json")
        print("请确保该文件包含需要处理的训练数据")
        return

    # 加载数据并处理错误
    print("正在加载数据...")
    data_list, errors = load_json_or_jsonl_with_error_handling(input_file)
    
    # 输出错误信息
    if errors:
        print(f"\n发现 {len(errors)} 行格式错误：")
        for err in errors[:5]:  # 最多显示前5个错误
            print(f"行 {err['line_number']}: {err['error']}")
            print(f"内容: {err['content']}\n")
        if len(errors) > 5:
            print(f"... 还有 {len(errors) - 5} 个错误未显示")
    
    if not data_list:
        print("错误：没有加载到有效数据！")
        print("\n可能的原因：")
        print("1. 文件格式不正确")
        print("2. 文件为空")
        print("3. 所有行都包含格式错误")
        return

    print(f"成功加载 {len(data_list)} 条数据")

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
    print(f"总有效数据条数: {len(data_list)}")
    print(f"训练集已保存到: {train_output} ({len(train_data)} 条)")
    print(f"验证集已保存到: {val_output} ({len(val_data)} 条)")
    
    if errors:
        print(f"\n警告：跳过了 {len(errors)} 行错误数据")

if __name__ == '__main__':
    main()
