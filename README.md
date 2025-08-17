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

### 1. 数据分割和处理

针对训练数据进行分割和处理：

```bash
python data_TCM.py
python data_MM.py
```
data_TCM.py代表对TCM训练数据进行处理，data_MM.py代表对西医数据的处理。

### 2. LoRA微调训练

启动LoRA微调训练，checkpoint将保存在 `output/checkpoint` 目录下：

```bash
python train_lora_TCM.py
python train_lora_MM.py
```
TCM代表启动中医模型微调，MM代表启动西医模型微调。

### 3. 知识库准备

将外部知识库文件放入 `knowledge_base` 目录中。
knowledge_base_TCM为TCM知识库
knowledge_base_MM为西医知识库

### 4. 配置API密钥

设置DashScope API密钥：

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

**注意：** 请将 `your_api_key_here` 替换为您的实际API密钥。

### 5. 运行RAG系统

启动RAG问答系统：

```bash
python run_TCMlora_rag.py
python run_baseline_rag.py
python run_MMlora_rag.py

```

三个不同的命令行分别代表着不同的模型，TCM代表微调后的中医模型，baseline代表qwen3-14b模型，MM代表微调后的西医模型。用户可以根据诉求选择对应的模型，并更新代码中的checkpoint地址，输入查询后即可获得相应的输出结果。
## 结果评估

### 方法1：BLEURT评估

BLEURT是一种基于BERT的评估指标，用于评估生成文本的质量。

**使用步骤：**

1. 准备两个需要对比的文本文件
2. 运行评估命令：

```bash
python -m bleurt.score_files \
  -candidate_file=result_example/output_TCM.txt \
  -reference_file=result_example/groundtruth_TCM.txt \
  -bleurt_checkpoint=BLEURT-20
```

**参数说明：**
- `-candidate_file`: 候选文本文件路径
- `-reference_file`: 参考文本文件路径  
- `-bleurt_checkpoint`: BLEURT模型检查点路径

### 方法2：BERTScore评估

BERTScore使用BERT的上下文嵌入来计算文本相似度。

**使用步骤：**

1. 将需要对比的两个文本分别命名为 `groundtruth_TCM.txt` 和 `output_TCM.txt`
2. 运行评估脚本：

```bash
python bertscore.py
```
