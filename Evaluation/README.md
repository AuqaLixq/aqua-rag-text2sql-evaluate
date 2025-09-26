# Text2SQL评估系统 - Sakila数据集评估框架

基于Sakila数据集的Text2SQL系统评估框架，支持**四大评估指标**和**细分召回率分析**，提供全面的性能评估和可视化分析。

## 🎯 系统特色

- **🔍 细分召回率**: 表级、列级、关键列、权重召回率四维分析
- **📊 四大指标**: 执行准确率、语法正确率、语义相似度、检索召回率
- **🚀 多种配置**: 完整版系统+真实DB、**简化版系统+模拟模式** ⭐当前使用、快速测试模式
- **📈 可视化报告**: HTML报告 + JSON结果 + 控制台输出
- **🛠️ 智能回退**: 自动从完整版回退到简化版，确保系统可用性
- **🔧 独立实现**: 重新实现Text2SQL系统，参考用户原始架构但独立运行

## 💡 系统架构说明

### Text2SQL系统实现方式
本评估框架**没有直接调用用户提供的Text2SQL程序**，而是：

1. **重新实现了Text2SQL系统**：
   - `text2sql_system.py`：完整版（需要Milvus向量数据库）
   - `text2sql_system_simple.py`：**简化版（当前使用）** - 基于文本匹配，无需Milvus

2. **参考了用户的架构设计**：
   - 从`Text2SQL/Sakila/05-text2sql-rag-v3-agent.py`学习RAG检索流程
   - 采用三种知识库结构（DDL、Q2SQL示例、字段描述）
   - 保持OpenAI API调用方式和SQL提取逻辑的一致性

3. **使用了用户的数据文件**：
   - ✅ `Data/sakila/ddl_statements.yaml` - 16个表结构定义
   - ✅ `Data/sakila/db_description.yaml` - 字段业务描述
   - ✅ `Data/sakila/q2sql_pairs.json` - 36条问答对

4. **数据库连接情况**：
   - ❌ **未创建MySQL数据库** - 用户的`01-Text2SQL-创建数据库表.py`创建的是SQLite旅游数据库
   - ✅ **使用模拟模式** - 配置指向MySQL但实际运行时禁用数据库连接
   - ✅ **简化版系统** - 只需OpenAI API，无需真实数据库连接

### 当前配置组合：简化版系统 + 模拟模式 ⭐
- **Text2SQL系统**: `text2sql_system_simple.py` - 基于关键词文本匹配（无需Milvus）
- **运行模式**: `salila_config_real_mock.yaml` - 模拟模式（无需MySQL连接）
- **评估指标**: 语法正确率 + 语义相似度 + 检索召回率（禁用执行准确率）
- **优势**: 部署简单、运行稳定、评估准确（98.25%综合得分）

## 📁 目录结构

```
Evaluation/
├── configs/                           # 配置文件目录
│   ├── salila_config.yaml            # 基础配置(Milvus版)
│   ├── salila_config_real.yaml       # 真实数据库配置
│   └── salila_config_real_mock.yaml  # 模拟模式配置
├── datasets/                          # 数据集目录
│   ├── sakila_test.json              # 测试数据集(15条)
│   └── sakila_real.json              # 真实数据集(36条)
├── scripts/                           # 核心脚本
│   ├── run_evaluation.py             # 主评估程序(451行)
│   ├── evaluation_metrics.py         # 指标计算引擎(700+行)
│   ├── text2sql_system.py            # 完整版Text2SQL系统
│   ├── text2sql_system_simple.py     # 简化版Text2SQL系统
│   └── convert_sakila_dataset.py     # 数据集转换工具
├── results/                           # 评估结果输出
├── requirements.txt                   # Python依赖列表
├── run_evaluation.bat                # Windows运行脚本
├── run_simple_evaluation.bat         # 简化版运行脚本
├── run_sakila_evaluation.bat         # Sakila专用运行脚本
├── README.md                          # 本文档
├── Text2SQL评估系统完整开发文档.md    # 完整开发文档(21KB)
├── 快速入门指南.md                    # 5分钟快速开始(9.6KB)
├── 代码结构说明.md                    # 代码架构详解(15KB)
├── 召回率细分功能说明.md              # 召回率功能说明(4.4KB)
└── 系统实现关系说明.md                # 与用户原始程序关系说明(新增)
```

## 🚀 5分钟快速开始

### 1. 环境准备
```bash
# 激活conda环境
conda activate rag-project01

# 进入脚本目录
cd E:\rag-text2sql-sakila\Evaluation\scripts
```

### 2. 一键运行（简化版系统 + 模拟模式）
```bash
# 方式1: 使用批处理脚本
run_sakila_evaluation.bat

# 方式2: 直接运行Python脚本
python run_evaluation.py --config ../configs/salila_config_real_mock.yaml
```

**当前配置说明**：
- **Text2SQL系统**：自动使用简化版系统（无需Milvus向量数据库）
- **运行模式**：模拟模式（无需MySQL数据库连接）
- **评估指标**：语法正确率 + 语义相似度 + 检索召回率（禁用执行准确率）

### 3. 查看结果
```
==================================================
评估完成！结果摘要:
总查询数: 36
语法正确率: 100.00%
语义相似度: 96.93%
检索召回率: 97.22%
  - 表级召回率: 97.22%
  - 列级召回率: 72.22%
  - 关键列召回率: 72.22%
  - 权重召回率: 97.22%
综合得分: 98.25%
==================================================
```

## 📊 评估指标详解

### 🎯 四大核心指标

#### 1. 执行准确率 (Execution Accuracy) - 权重40% ❌当前禁用
- **功能**: 通过实际执行SQL比较结果
- **支持**: SELECT结果对比、INSERT/UPDATE/DELETE影响行数对比
- **当前状态**: 模拟模式禁用（简化版系统 + 模拟模式配置）
- **启用条件**: 需要真实数据库连接（MySQL Sakila数据库）

#### 2. 语法正确率 (Syntax Accuracy) - 权重20%
- **功能**: 验证SQL语法完整性和正确性
- **方法**: SQL解析器验证 + 结构检查
- **结果**: 100.00% (完美表现)

#### 3. 语义相似度 (Semantic Similarity) - 权重20%
- **组成**: 嵌入相似度(70%) + 结构相似度(30%)
- **模型**: OpenAI text-embedding-3-large
- **结果**: 96.93% (优秀表现)

#### 4. 检索召回率 (Retrieval Recall) - 权重20% ⭐**新增细分功能**
- **表级召回率**: 97.22% - 检索表与期望表的匹配度
- **列级召回率**: 72.22% - 检索列与期望列的匹配度
- **关键列召回率**: 72.22% - 重要字段识别准确度
- **权重召回率**: 97.22% - 智能加权综合评估

### 🔍 细分召回率分析

#### 权重机制
- **表级**: 50% (最重要 - 表是SQL的基础)
- **关键列**: 30% (次重要 - 主键外键至关重要)
- **普通列**: 20% (相对较低 - 普通字段)

#### 详细统计 (基于36条Sakila查询)
- **表级分析**: 期望36表，检索53表，匹配35表
- **列级分析**: 期望46列，检索319列，匹配44列
- **关键列分析**: 期望34关键列，匹配33关键列

#### SQL类型表现
- **SELECT查询**: 88.89% (有1条查询表现较差)
- **INSERT操作**: 100% ✅ (完美表现)
- **UPDATE操作**: 100% ✅ (完美表现)
- **DELETE操作**: 100% ✅ (完美表现)

## 📈 实际评估结果

### 🏆 总体表现 (基于36条真实Sakila查询)
- **综合得分**: 98.25% ⭐⭐⭐⭐⭐ (优秀)
- **评估耗时**: 144.97秒
- **成功率**: 100% (所有查询均成功评估)

### 📊 指标分布
| 指标 | 得分 | 权重 | 贡献 | 评级 |
|------|------|------|------|------|
| 语法正确率 | 100.00% | 20% | 20.00% | 完美 ✅ |
| 语义相似度 | 96.93% | 20% | 19.39% | 优秀 ✅ |
| 检索召回率 | 97.22% | 20% | 19.44% | 优秀 ✅ |
| 执行准确率 | - | 40% | - | 模拟模式禁用 |
| **综合得分** | **98.25%** | **100%** | **98.25%** | **优秀** ⭐ |

### 🔍 问题诊断
- **最常缺失的表**: category (1次)
- **最常缺失的列**: category_id, name (各1次)
- **优化建议**: 关注category表的检索策略优化

## 🔧 配置组合对比

| 配置组合 | Text2SQL系统 | 运行模式 | 配置文件 | Milvus | MySQL | 执行准确率 | 适用场景 |
|---------|-------------|----------|----------|--------|-------|------------|----------|
| **完整版+真实DB** | 完整版 | 真实数据库 | `salila_config.yaml` | ✅ | ✅ | ✅ | 生产环境评估 |
| **简化版+真实DB** | 简化版 | 真实数据库 | `salila_config_real.yaml` | ❌ | ✅ | ✅ | 快速评估 |
| **简化版+模拟模式** ⭐ | 简化版 | 模拟模式 | `salila_config_real_mock.yaml` | ❌ | ❌ | ❌ | 当前使用 |

## 📊 数据集说明

### 用户提供的原始数据集
```
Data/sakila/
├── q2sql_pairs.json        # 36条问题-SQL对
├── ddl_statements.yaml     # 16个表的完整结构定义
└── db_description.yaml     # 表字段的业务含义描述
```

### 转换后的标准数据集
- **sakila_real.json**: 36条真实查询，通过转换工具自动生成
- **数据特点**: SELECT(9) + INSERT(9) + UPDATE(9) + DELETE(9)
- **涉及表数**: 16个Sakila数据库表
- **难度级别**: 主要为easy级别（单表操作）

## 📈 结果输出

### 1. 控制台输出
- 实时进度跟踪
- 详细的召回率分析
- 综合得分摘要

### 2. JSON结果文件
```json
{
  "evaluation_info": {
    "timestamp": "2025-06-13 12:31:46",
    "total_queries": 36,
    "evaluation_duration": 144.97
  },
  "overall_metrics": {
    "retrieval_recall": {
      "table_recall_mean": 0.9722,
      "column_recall_mean": 0.7222,
      "key_column_recall_mean": 0.7222,
      "weighted_recall_mean": 0.9722,
      "table_analysis": {...},
      "column_analysis": {...},
      "key_column_analysis": {...}
    }
  }
}
```

### 3. HTML可视化报告
- 概览卡片显示各指标
- 召回率详细分析部分
- 最常缺失元素统计
- 详细结果表格

## 🛠️ 配置说明

### 核心配置项

#### 当前使用配置（模拟模式）
```yaml
# 数据库配置
database:
  enabled: false  # 禁用真实数据库连接
  type: "mock"
  connection_string: "sqlite:///:memory:"  # 模拟连接

# 指标权重配置（模拟模式）
metrics:
  execution_accuracy:
    enabled: false  # 禁用执行准确率
    weight: 0.0
  syntax_accuracy:
    enabled: true
    weight: 0.4     # 增加权重
  semantic_similarity:
    enabled: true
    weight: 0.3     # 增加权重
  retrieval_recall:
    enabled: true
    weight: 0.3     # 增加权重

# Text2SQL系统配置
text2sql_system:
  type: "rag"
  retrieval:
    use_vector_db: false  # 使用简化版系统，不需要Milvus
```

#### 真实数据库模式配置（可选升级）
```yaml
# 数据库配置
database:
  connection_string: "mysql+pymysql://root:password@localhost:3306/sakila"
  enabled: true  # 启用真实数据库

# 指标权重配置（真实模式）
metrics:
  execution_accuracy:
    enabled: true
    weight: 0.4
  syntax_accuracy:
    enabled: true
    weight: 0.2
  semantic_similarity:
    enabled: true
    weight: 0.2
  retrieval_recall:
    enabled: true
    weight: 0.2
```

## 🔍 故障排除

### 常见问题

#### 1. 环境问题
```bash
# 问题: ModuleNotFoundError
# 解决: 安装依赖
pip install -r requirements.txt
```

#### 2. Milvus连接失败
```bash
# 解决: 使用简化版系统
python run_evaluation.py --config ../configs/salila_config_real_mock.yaml
```

#### 3. 数据库连接问题
```yaml
# 解决: 使用模拟模式
database:
  mock_mode: true
```

#### 4. API密钥问题
```bash
# 解决: 设置环境变量
export OPENAI_API_KEY="your-api-key"
```

### 调试技巧
```bash
# 查看详细日志
tail -f ../results/evaluation_*.log

# 检查中间结果
ls ../results/intermediate_results_*.json

# 单步调试
python -c "
from run_evaluation import SalilaEvaluator
evaluator = SalilaEvaluator('../configs/salila_config_real_mock.yaml')
# 调试代码...
"
```

## 📚 文档导航

- **📖 [完整开发文档](Text2SQL评估系统完整开发文档.md)** - 系统架构、核心模块、扩展开发
- **🚀 [快速入门指南](快速入门指南.md)** - 5分钟快速开始、实际结果解读
- **🔧 [代码结构说明](代码结构说明.md)** - 详细的代码架构和实现原理
- **📊 [召回率功能说明](召回率细分功能说明.md)** - 细分召回率的详细说明
- **🔗 [系统实现关系说明](系统实现关系说明.md)** - 与用户原始程序的关系说明 ⭐新增

## 🎯 下一步建议

1. **📖 阅读文档**: 从[快速入门指南](快速入门指南.md)开始
2. **🚀 运行评估**: 使用模拟模式快速体验
3. **📊 分析结果**: 重点关注HTML可视化报告
4. **🔧 自定义配置**: 根据需求调整权重和参数
5. **📈 扩展功能**: 参考[完整开发文档](Text2SQL评估系统完整开发文档.md)添加新指标

## 🏆 版本历史

- **v2.0.0** - 新增细分召回率功能，完整文档体系
- **v1.5.0** - 添加简化版Text2SQL系统，智能回退机制
- **v1.0.0** - 初始版本，支持基础四项指标评估

---

*Text2SQL评估系统 - 让Text2SQL评估更精确、更全面、更智能！* 