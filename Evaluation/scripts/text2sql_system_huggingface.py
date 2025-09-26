"""
基于Hugging Face免费模型的Text2SQL系统
使用开源模型替代OpenAI API
"""

import os
import sys
import logging
import re
import json
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

try:
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
except ImportError as e:
    logging.error(f"导入依赖失败: {e}")
    raise

class HuggingFaceText2SQLSystem:
    """基于Hugging Face的Text2SQL系统"""
    
    def __init__(self, config=None):
        """初始化Text2SQL系统"""
        self.config = config or {}
        self.setup_environment()
        self.setup_models()
        self.setup_clients()
        self.load_knowledge_base()
        
    def setup_environment(self):
        """设置环境变量和配置"""
        load_dotenv()
        
        # 模型配置
        self.embedding_model_name = self.config.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        self.text2sql_model_name = self.config.get('text2sql_model', 'microsoft/DialoGPT-medium')
        
        # 数据库配置
        self.db_url = os.getenv(
            "SAKILA_DB_URL", 
            "mysql+pymysql://root:password@localhost:3306/sakila"
        )
        
    def setup_models(self):
        """初始化Hugging Face模型"""
        try:
            logging.info("正在加载嵌入模型...")
            # 使用轻量级的sentence-transformers模型
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            logging.info("正在加载Text2SQL生成模型...")
            # 使用适合的生成模型
            # 注意：这里使用一个通用的对话模型，实际项目中建议使用专门的Text2SQL模型
            self.tokenizer = AutoTokenizer.from_pretrained(self.text2sql_model_name)
            self.text2sql_model = AutoModelForCausalLM.from_pretrained(
                self.text2sql_model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            # 生成长度参数（避免超过模型上限）
            self.max_new_tokens = int(self.config.get('max_new_tokens', 96))
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logging.info("Hugging Face模型加载完成")
            
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            # 如果模型加载失败，使用更简单的方案
            self._setup_fallback_models()
    
    def _setup_fallback_models(self):
        """设置备用模型（更轻量级）"""
        try:
            logging.info("使用备用模型...")
            # 使用更小的嵌入模型
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            # 使用简单的文本生成pipeline
            self.text_generator = pipeline(
                "text-generation",
                model="gpt2",  # 使用GPT-2作为备用
                tokenizer="gpt2",
                max_length=200,
                do_sample=True,
                temperature=0.7
            )
            
            logging.info("备用模型加载完成")
            
        except Exception as e:
            logging.error(f"备用模型加载也失败: {e}")
            raise
    
    def setup_clients(self):
        """初始化客户端连接"""
        try:
            # 初始化数据库连接
            self.db_engine = create_engine(self.db_url)
            logging.info("Hugging Face Text2SQL系统初始化成功")
            
        except Exception as e:
            logging.error(f"Text2SQL系统初始化失败: {e}")
            raise
    
    def load_knowledge_base(self):
        """加载知识库文件"""
        try:
            # 获取数据文件路径
            base_path = Path(__file__).parent.parent.parent / "Data" / "sakila"
            
            # 加载DDL数据
            ddl_file = base_path / "ddl_statements.yaml"
            if ddl_file.exists():
                with open(ddl_file, 'r', encoding='utf-8') as f:
                    self.ddl_data = yaml.safe_load(f)
            else:
                self.ddl_data = {}
                logging.warning("DDL文件不存在")
            
            # 加载数据库描述
            desc_file = base_path / "db_description.yaml"
            if desc_file.exists():
                with open(desc_file, 'r', encoding='utf-8') as f:
                    self.db_descriptions = yaml.safe_load(f)
            else:
                self.db_descriptions = {}
                logging.warning("数据库描述文件不存在")
            
            # 加载Q2SQL示例
            q2sql_file = base_path / "q2sql_pairs.json"
            if q2sql_file.exists():
                with open(q2sql_file, 'r', encoding='utf-8') as f:
                    self.q2sql_examples = json.load(f)
            else:
                self.q2sql_examples = []
                logging.warning("Q2SQL示例文件不存在")
            
            logging.info("知识库加载完成")
            
        except Exception as e:
            logging.error(f"知识库加载失败: {e}")
            self.ddl_data = {}
            self.db_descriptions = {}
            self.q2sql_examples = []
    
    def generate_sql(self, question, return_context=False):
        """
        生成SQL查询
        
        Args:
            question (str): 自然语言问题
            return_context (bool): 是否返回检索上下文
            
        Returns:
            str or tuple: 生成的SQL语句，如果return_context=True则返回(sql, context)
        """
        try:
            # 1. 使用向量检索获取相关上下文
            context = self._retrieve_context_with_embeddings(question)
            
            # 2. 构建prompt
            prompt = self._build_prompt(question, context)
            
            # 3. 使用Hugging Face模型生成SQL
            sql = self._generate_sql_with_hf_model(prompt)
            # 失败时使用基于示例/规则的回退
            if not sql:
                sql = self._fallback_rule_based_sql(question, context)
            
            if return_context:
                return sql, context
            return sql
            
        except Exception as e:
            logging.error(f"SQL生成失败: {e}")
            return None
    
    def _retrieve_context_with_embeddings(self, question):
        """使用嵌入模型进行上下文检索"""
        try:
            # 对问题进行向量化
            question_embedding = self.embedding_model.encode([question])
            
            context = {
                'ddl': [],
                'examples': [],
                'descriptions': []
            }
            
            # 检索相关的DDL信息
            if self.ddl_data:
                ddl_texts = []
                ddl_items = []
                for table_name, table_info in self.ddl_data.items():
                    # table_info 可能是dict或str
                    if isinstance(table_info, dict):
                        desc = table_info.get('description', '')
                    else:
                        desc = str(table_info)
                    ddl_text = f"Table {table_name}: {desc}"
                    ddl_texts.append(ddl_text)
                    ddl_items.append((table_name, table_info))
                
                if ddl_texts:
                    ddl_embeddings = self.embedding_model.encode(ddl_texts)
                    similarities = np.dot(question_embedding, ddl_embeddings.T)[0]
                    top_indices = np.argsort(similarities)[-3:][::-1]  # 取前3个
                    
                    for idx in top_indices:
                        if similarities[idx] > 0.3:  # 相似度阈值
                            table_name, table_info = ddl_items[idx]
                            context['ddl'].append((table_name, table_info))
                            # 同步补充字段描述，便于列级与关键列召回
                            try:
                                if isinstance(self.db_descriptions, dict) and table_name in self.db_descriptions:
                                    fields = self.db_descriptions[table_name]
                                    if isinstance(fields, dict):
                                        for field_name, description in fields.items():
                                            context['descriptions'].append((table_name, field_name, description))
                            except Exception:
                                pass
            
            # 检索相关的示例
            if self.q2sql_examples:
                example_texts = []
                example_items = []
                for example in self.q2sql_examples:
                    if isinstance(example, dict):
                        q = example.get('question', '')
                        sql = example.get('expected_sql', example.get('sql', ''))
                    else:
                        q = str(example)
                        sql = ''
                    example_text = f"{q} {sql}"
                    example_texts.append(example_text)
                    example_items.append(example)
                
                if example_texts:
                    example_embeddings = self.embedding_model.encode(example_texts)
                    similarities = np.dot(question_embedding, example_embeddings.T)[0]
                    top_indices = np.argsort(similarities)[-3:][::-1]  # 取前3个
                    
                    for idx in top_indices:
                        if similarities[idx] > 0.3:  # 相似度阈值
                            example = example_items[idx]
                            if isinstance(example, dict):
                                context['examples'].append((
                                    example.get('question', ''),
                                    example.get('expected_sql', example.get('sql', ''))
                                ))
                            else:
                                context['examples'].append((str(example), ''))
            
            return context
            
        except Exception as e:
            logging.error(f"上下文检索失败: {e}")
            return {'ddl': [], 'examples': [], 'descriptions': []}
    
    def _build_prompt(self, question, context):
        """构建提示词"""
        # DDL上下文（兼容table_info为str/dict）
        ddl_items = context.get('ddl', []) if isinstance(context, dict) else []
        ddl_lines = []
        for table_name, table_info in ddl_items:
            if isinstance(table_info, dict):
                desc = table_info.get('description', '')
            else:
                desc = str(table_info)
            ddl_lines.append(f"Table {table_name}: {desc}")
        ddl_context = "\n".join(ddl_lines)
        
        # 示例上下文
        examples = context.get('examples', []) if isinstance(context, dict) else []
        example_context = "\n".join([
            f"Question: \"{q}\"\nSQL: \"{sql}\"" 
            for q, sql in examples
            if isinstance(q, str)
        ])
        
        # 截断上下文，避免超过模型上下文长度
        def truncate(text: str, max_chars: int) -> str:
            return text[:max_chars]

        ddl_context = truncate(ddl_context, 2000)
        example_context = truncate(example_context, 1500)

        prompt = f"""You are a SQL expert. Generate SQL queries based on the given schema and examples.

### Database Schema:
{ddl_context}

### Examples:
{example_context}

### Question:
"{question}"

Please generate only the SQL query without any explanations or markdown formatting:"""
        
        return prompt
    
    def _generate_sql_with_hf_model(self, prompt):
        """使用Hugging Face模型生成SQL"""
        try:
            if hasattr(self, 'text_generator'):
                # 使用pipeline方式
                result = self.text_generator(
                    prompt,
                    max_length=min(len(prompt.split()) + 64, 256),
                    num_return_sequences=1,
                    do_sample=False,
                    pad_token_id=self.text_generator.tokenizer.eos_token_id
                )
                generated_text = result[0]['generated_text']
                sql = generated_text[len(prompt):].strip()
            else:
                # 使用模型直接生成
                encoded = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=1024
                )
                
                with torch.no_grad():
                    outputs = self.text2sql_model.generate(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded.get("attention_mask"),
                        max_new_tokens=self.max_new_tokens,
                        num_return_sequences=1,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                sql = generated_text[len(prompt):].strip()
            
            # 清理和提取SQL
            sql = self._extract_sql(sql)
            return sql
            
        except Exception as e:
            logging.error(f"Hugging Face模型生成SQL失败: {e}")
            return None

    def _fallback_rule_based_sql(self, question: str, context: Dict[str, Any]) -> str:
        """基于简单规则/示例的回退生成（保证调试阶段有输出）"""
        q = (question or "").lower()
        # 1) 直接用最相近的示例
        examples = context.get('examples', []) if isinstance(context, dict) else []
        for ex_q, ex_sql in examples:
            if isinstance(ex_q, str) and isinstance(ex_sql, str):
                if any(tok in ex_q.lower() for tok in q.split()):
                    if ex_sql.strip():
                        return ex_sql.strip()
        # 2) 常见模式回退
        if "actor" in q and ("id" in q or "name" in q):
            return "SELECT actor_id, first_name, last_name FROM actor;"
        if "film" in q and ("title" in q or "description" in q):
            return "SELECT film_id, title, description FROM film;"
        if "customer" in q:
            return "SELECT customer_id, first_name, last_name, email FROM customer;"
        # 3) 兜底：选择任何检索到的表的所有字段
        ddl_items = context.get('ddl', []) if isinstance(context, dict) else []
        if ddl_items:
            table = ddl_items[0][0] if isinstance(ddl_items[0], (list, tuple)) else None
            if isinstance(table, str) and table:
                return f"SELECT * FROM {table};"
        return ""
    
    def _extract_sql(self, text):
        """从文本中提取SQL语句"""
        # 尝试匹配SQL代码块
        sql_blocks = re.findall(r'```sql\n(.*?)\n```', text, re.DOTALL)
        if sql_blocks:
            return sql_blocks[0].strip()
        
        # 尝试匹配各种SQL语句
        patterns = [
            r'(SELECT.*?;)',
            r'(INSERT.*?;)',
            r'(UPDATE.*?;)',
            r'(DELETE.*?;)',
            r'(CREATE.*?;)',
            r'(ALTER.*?;)',
            r'(DROP.*?;)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # 如果没有找到标准SQL，返回清理后的文本
        return text.strip()
    
    def get_retrieval_info(self, question):
        """获取检索信息（用于评估）"""
        context = self._retrieve_context_with_embeddings(question)
        # 日志打印部分上下文，便于排查召回率
        try:
            # 打印检索到的表（最多5个）
            ddl_items_log = []
            for item in (context.get('ddl', []) if isinstance(context, dict) else [])[:5]:
                try:
                    tbl_name = item[0] if isinstance(item, (list, tuple)) and item else str(item)
                    ddl_items_log.append(str(tbl_name))
                except Exception:
                    pass
            logging.info(f"[RAG] Retrieved tables (top): {ddl_items_log}")

            # 打印字段描述（最多5条）
            desc_samples = []
            for entry in (context.get('descriptions', []) if isinstance(context, dict) else [])[:5]:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    desc_samples.append(f"{entry[0]}.{entry[1]}")
            logging.info(f"[RAG] Description samples (top): {desc_samples}")
        except Exception:
            pass
        # 提取表名
        tables = [item[0] for item in context.get('ddl', [])] if isinstance(context, dict) else []
        # 解析列名（来自字段描述与DDL元信息）
        columns_set = set()
        key_columns_set = set()
        try:
            # 从字段描述中提取列（优先，因为较规范）
            if isinstance(context, dict):
                for entry in context.get('descriptions', []) or []:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                        table, column = entry[0], entry[1]
                        desc = entry[2] if len(entry) > 2 else ""
                        if isinstance(table, str) and isinstance(column, str):
                            col_full = f"{table}.{column}"
                            columns_set.add(col_full)
                            # 简单关键列启发式
                            desc_l = str(desc).lower()
                            if 'primary' in desc_l or '主键' in desc_l or column.lower().endswith('id'):
                                key_columns_set.add(col_full)

            # 从DDL信息中提取列（table_info 可能是dict或字符串）
            for table_name, table_info in context.get('ddl', []) if isinstance(context, dict) else []:
                # dict风格
                if isinstance(table_info, dict):
                    # 常见字段容器键名
                    for key in ['columns', 'fields', 'schema', 'columns_def']:
                        if key in table_info:
                            cols = table_info.get(key)
                            # 形如 {col_name: meta}
                            if isinstance(cols, dict):
                                for col_name, meta in cols.items():
                                    col_full = f"{table_name}.{col_name}"
                                    columns_set.add(col_full)
                                    meta_str = str(meta).lower()
                                    if 'primary' in meta_str or '主键' in meta_str or str(col_name).lower().endswith('id'):
                                        key_columns_set.add(col_full)
                            # 形如 [ {name: ..., description: ...}, ... ]
                            elif isinstance(cols, list):
                                for item in cols:
                                    if isinstance(item, dict):
                                        col_name = item.get('name') or item.get('field') or item.get('column')
                                        if col_name:
                                            col_full = f"{table_name}.{col_name}"
                                            columns_set.add(col_full)
                                            meta_str = (item.get('description') or item.get('desc') or '').lower()
                                            if 'primary' in meta_str or '主键' in meta_str or str(col_name).lower().endswith('id'):
                                                key_columns_set.add(col_full)
                # 文本DDL风格（粗略正则）
                elif isinstance(table_info, str):
                    import re
                    # 提取括号内的列定义行，匹配以标识符开头的列名
                    for m in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]*)\s+\w+', table_info):
                        col_name = m.group(1)
                        # 排除CREATE/TABLE/PRIMARY/FOREIGN等关键字误匹配
                        if col_name.lower() not in {'create', 'table', 'primary', 'key', 'foreign', 'constraint'}:
                            col_full = f"{table_name}.{col_name}"
                            columns_set.add(col_full)
                            if col_name.lower().endswith('id'):
                                key_columns_set.add(col_full)
        except Exception:
            pass

        columns = list(columns_set)
        info = {
            'tables': tables,
            'columns': columns,
            'context': context
        }
        # 兼容旧键名，避免其他地方引用失败
        info['retrieved_tables'] = info['tables']
        info['retrieved_examples'] = len(context.get('examples', [])) if isinstance(context, dict) else 0
        # 附带关键列用于调试（评估器用不到此键，但便于开发观察）
        info['key_columns_detected'] = list(key_columns_set)
        return info
