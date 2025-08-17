import json
import os


def process_test_MM_json():
    """处理test_MM.json文件，提取所有input字段里的文字内容并保存到query/query_MM.txt"""
    
    # 文件路径
    input_file = "test data/test_MM.json"
    output_dir = "query"
    output_file = os.path.join(output_dir, "query_MM.txt")
    
    print("🚀 开始处理test_MM.json文件中的input字段...")
    print("="*50)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    
    try:
        # 读取JSON文件
        print(f"📖 正在读取 {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 成功读取JSON文件")
        
        # 提取input字段里的文字内容
        input_texts = []
        
        # 递归查找所有input字段
        def extract_input_texts(data_item, path=""):
            texts = []
            if isinstance(data_item, dict):
                # 如果是字典，检查是否有input字段
                if 'input' in data_item and isinstance(data_item['input'], str):
                    text = data_item['input'].strip()
                    if text:  # 只添加非空文字
                        texts.append(text)
                # 递归遍历所有值
                for key, value in data_item.items():
                    current_path = f"{path}.{key}" if path else key
                    texts.extend(extract_input_texts(value, current_path))
            elif isinstance(data_item, list):
                # 如果是列表，遍历所有元素
                for i, item in enumerate(data_item):
                    current_path = f"{path}[{i}]"
                    texts.extend(extract_input_texts(item, current_path))
            return texts
        
        # 提取所有input字段的文字
        input_texts = extract_input_texts(data)
        
        print(f"🔍 找到 {len(input_texts)} 个input字段的文字内容")
        
        if not input_texts:
            print("❌ 未找到任何input字段的文字内容")
            return
        
        # 去重并保存到文件
        unique_texts = list(dict.fromkeys(input_texts))  # 保持顺序的去重
        print(f"🔄 去重后剩余 {len(unique_texts)} 个唯一文字内容")
        
        # 保存到输出文件
        print(f"💾 正在保存到 {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in unique_texts:
                f.write(text + '\n')
        
        print(f"✅ 成功保存 {len(unique_texts)} 个input文字内容到 {output_file}")
        
        # 显示前几个文字内容作为示例
        print("\n📝 前5个input文字内容示例:")
        print("-"*40)
        for i, text in enumerate(unique_texts[:5], 1):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"{i}. {preview}")
        
        if len(unique_texts) > 5:
            print(f"... 还有 {len(unique_texts) - 5} 个input文字内容")
        
        print("\n🎉 input字段处理完成！")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        print("请检查JSON文件格式是否正确")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        print("请检查文件内容和权限")


def process_output_MM_json():
    """处理test_MM.json文件，提取所有output字段里的文字内容并保存到result_example/groundtruth_MM.txt"""
    
    # 文件路径
    input_file = "test data/test_MM.json"
    output_dir = "result_example"
    output_file = os.path.join(output_dir, "groundtruth_MM.txt")
    
    print("\n🚀 开始处理test_MM.json文件中的output字段...")
    print("="*50)
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return
    
    # 创建输出目录（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✅ 创建输出目录: {output_dir}")
    
    try:
        # 读取JSON文件
        print(f"📖 正在读取 {input_file}...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"✅ 成功读取JSON文件")
        
        # 提取output字段里的文字内容
        output_texts = []
        
        # 递归查找所有output字段
        def extract_output_texts(data_item, path=""):
            texts = []
            if isinstance(data_item, dict):
                # 如果是字典，检查是否有output字段
                if 'output' in data_item and isinstance(data_item['output'], str):
                    text = data_item['output'].strip()
                    if text:  # 只添加非空文字
                        texts.append(text)
                # 递归遍历所有值
                for key, value in data_item.items():
                    current_path = f"{path}.{key}" if path else key
                    texts.extend(extract_output_texts(value, current_path))
            elif isinstance(data_item, list):
                # 如果是列表，遍历所有元素
                for i, item in enumerate(data_item):
                    current_path = f"{path}[{i}]"
                    texts.extend(extract_output_texts(item, current_path))
            return texts
        
        # 提取所有output字段的文字
        output_texts = extract_output_texts(data)
        
        print(f"🔍 找到 {len(output_texts)} 个output字段的文字内容")
        
        if not output_texts:
            print("❌ 未找到任何output字段的文字内容")
            return
        
        # 去重并保存到文件
        unique_texts = list(dict.fromkeys(output_texts))  # 保持顺序的去重
        print(f"🔄 去重后剩余 {len(unique_texts)} 个唯一文字内容")
        
        # 保存到输出文件
        print(f"💾 正在保存到 {output_file}...")
        with open(output_file, 'w', encoding='utf-8') as f:
            for text in unique_texts:
                f.write(text + '\n')
        
        print(f"✅ 成功保存 {len(unique_texts)} 个output文字内容到 {output_file}")
        
        # 显示前几个文字内容作为示例
        print("\n📝 前5个output文字内容示例:")
        print("-"*40)
        for i, text in enumerate(unique_texts[:5], 1):
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"{i}. {preview}")
        
        if len(unique_texts) > 5:
            print(f"... 还有 {len(unique_texts) - 5} 个output文字内容")
        
        print("\n🎉 output字段处理完成！")
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON解析错误: {e}")
        print("请检查JSON文件格式是否正确")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        print("请检查文件内容和权限")


if __name__ == "__main__":
    # 处理input字段
    process_test_MM_json()
    
    # 处理output字段
    process_output_MM_json()
    
    print("\n" + "="*60)
    print("🎉 所有处理任务完成！")
    print("="*60)
