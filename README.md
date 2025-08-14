# RAG

# RAG系统与LoRA微调项目

## 项目概述

本项目实现了一个基于检索增强生成（RAG）的问答系统，结合LoRA微调技术来提升模型性能。系统能够利用外部知识库来增强回答的准确性和相关性。

## 环境要求

- Python 3.8+
- pip
- Git

## 安装步骤

### 1. 安装依赖包

```bash
pip install -r requirements.txt
```

### 2. 克隆BLEURT评估工具

```bash
# 确保pip是最新版本
pip install --upgrade pip

# 克隆BLEURT仓库
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .

# 下载BLEURT模型
wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip .
unzip BLEURT-20.zip
```

## 使用流程

### 1. 数据预处理

运行数据处理脚本来加工原生input-output数据：

```bash
python dataprocess.py
```

处理后的数据将保存在 `processed_data` 目录下。

### 2. 数据分割和处理

针对训练数据进行分割和处理：

```bash
python data.py
```

### 3. LoRA微调训练

启动LoRA微调训练，checkpoint将保存在 `output/checkpoint` 目录下：

```bash
python train_lora.py
```

### 4. 知识库准备

将外部知识库文件放入 `knowledge_base` 目录中。

### 5. 配置API密钥

设置DashScope API密钥：

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

**注意：** 请将 `your_api_key_here` 替换为您的实际API密钥。

### 6. 运行RAG系统

启动RAG问答系统：

```bash
python run_TCMlora_rag.py
python run_baseline_rag.py
python run_MMlora_rag.py

```

三个不同的命令行分别代表着不同的模型，TCM代表微调后的中医模型，baseline代表qwen3-14b模型，MM代表微调后的西医模型。用户可以根据诉求选择对应的模型，输入查询后即可获得相应的输出结果。
## 结果评估

### 方法1：BLEURT评估

BLEURT是一种基于BERT的评估指标，用于评估生成文本的质量。

**使用步骤：**

1. 准备两个需要对比的文本文件
2. 运行评估命令：

```bash
python -m bleurt.score_files \
  -candidate_file=bleurt/test_data/candidates \
  -reference_file=bleurt/test_data/references \
  -bleurt_checkpoint=BLEURT-20
```

**参数说明：**
- `-candidate_file`: 候选文本文件路径
- `-reference_file`: 参考文本文件路径  
- `-bleurt_checkpoint`: BLEURT模型检查点路径

### 方法2：BERTScore评估

BERTScore使用BERT的上下文嵌入来计算文本相似度。

**使用步骤：**

1. 将需要对比的两个文本分别命名为 `file_1.txt` 和 `file_2.txt`
2. 运行评估脚本：

```bash
python simple_compare.py
```
