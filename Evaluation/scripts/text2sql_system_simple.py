"""
简化版Text2SQL系统封装类
不依赖Milvus向量数据库，使用文件直接匹配
"""

import os
import sys
import logging
import re
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    import openai
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
except ImportError as e:
    logging.error(f"导入依赖失败: {e}")
    raise

class SimpleText2SQLSystem:
    """简化版Text2SQL系统封装类（无需Milvus）"""
    
    def __init__(self, config=None):
        """初始化Text2SQL系统"""
        self.config = config or {}
        self.setup_environment()
        self.setup_clients()
        self.load_knowledge_base()
        
    def setup_environment(self):
        """设置环境变量和配置"""
        load_dotenv()
        
        # OpenAI配置
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # 优先使用配置文件中的模型，然后是环境变量，最后是默认值
        self.model_name = self.config.get('llm_model') or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        # 数据库配置
        self.db_url = os.getenv(
            "SAKILA_DB_URL", 
            "mysql+pymysql://root:password@localhost:3306/sakila"
        )
        
    def setup_clients(self):
        """初始化客户端连接"""
        try:
            # 初始化数据库连接
            self.db_engine = create_engine(self.db_url)
            logging.info("简化版Text2SQL系统初始化成功")
            
        except Exception as e:
            logging.error(f"Text2SQL系统初始化失败: {e}")
            raise
    
    def load_knowledge_base(self):
        """加载知识库文件"""
        try:
            # 获取数据路径
            base_path = Path(__file__).parent.parent.parent / "Data" / "sakila"
            
            # 加载DDL语句
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
            # 1. 简单文本匹配检索
            context = self._retrieve_context_simple(question)
            
            # 2. 构建prompt
            prompt = self._build_prompt(question, context)
            
            # 3. 生成SQL
            sql = self._generate_sql_with_llm(prompt)
            
            if return_context:
                return sql, context
            return sql
            
        except Exception as e:
            logging.error(f"SQL生成失败: {e}")
            return None
    
    def _retrieve_context_simple(self, question):
        """使用简单文本匹配检索相关上下文"""
        context = {
            'ddl': [],
            'examples': [],
            'descriptions': []
        }
        
        question_lower = question.lower()
        
        try:
            # 1. 检索相关DDL（基于关键词匹配）
            for table_name, ddl_content in self.ddl_data.items():
                if table_name.lower() in question_lower or any(
                    keyword in question_lower 
                    for keyword in ['table', 'create', 'structure']
                ):
                    context['ddl'].append(str(ddl_content))
            
            # 如果没有匹配到，添加一些常用表的DDL
            if not context['ddl']:
                common_tables = ['actor', 'film', 'customer', 'rental', 'payment']
                for table in common_tables:
                    if table in self.ddl_data:
                        context['ddl'].append(str(self.ddl_data[table]))
                        if len(context['ddl']) >= 3:  # 限制数量
                            break
            
            # 2. 检索相似的Q2SQL示例
            for example in self.q2sql_examples:
                # 兼容：示例既可能是dict也可能是字符串
                if isinstance(example, dict):
                    example_question = str(example.get('question', '')).lower()
                    example_sql = example.get('sql', example.get('expected_sql', ''))
                else:
                    example_question = str(example).lower()
                    example_sql = ''

                # 简单的关键词匹配
                if any(word in example_question for word in question_lower.split()):
                    context['examples'].append((
                        example.get('question', example_question) if isinstance(example, dict) else example_question,
                        example_sql
                    ))
                    if len(context['examples']) >= 3:  # 限制数量
                        break
            
            # 3. 检索相关字段描述
            for table_name, fields in self.db_descriptions.items():
                table_key = str(table_name).lower()
                if table_key in question_lower:
                    if isinstance(fields, dict):
                        for field_name, description in fields.items():
                            context['descriptions'].append((
                                str(table_name), str(field_name), description
                            ))
                    elif isinstance(fields, list):
                        # 兼容列表形式
                        for field in fields:
                            if isinstance(field, dict):
                                fname = field.get('name') or field.get('field') or ''
                                desc = field.get('description') or field.get('desc') or ''
                                if fname:
                                    context['descriptions'].append((str(table_name), str(fname), desc))
            
            # 如果没有匹配到描述，添加一些常用的
            if not context['descriptions']:
                common_tables = ['actor', 'film', 'customer']
                for table in common_tables:
                    if table in self.db_descriptions:
                        fields = self.db_descriptions[table]
                        if isinstance(fields, dict):
                            for field_name, description in list(fields.items())[:3]:
                                context['descriptions'].append((
                                    table, field_name, description
                                ))
            
        except Exception as e:
            logging.error(f"上下文检索失败: {e}")
        
        return context
    
    def _build_prompt(self, question, context):
        """构建LLM提示"""
        # DDL上下文
        ddl_list = context.get('ddl', []) if isinstance(context, dict) else []
        ddl_context = "\n".join([str(item) for item in ddl_list])
        
        # 示例上下文
        examples = context.get('examples', []) if isinstance(context, dict) else []
        example_context = "\n".join([
            f"NL: \"{q}\"\nSQL: \"{sql}\"" 
            for q, sql in examples if isinstance(q, str)
        ])
        
        # 字段描述上下文
        descriptions = context.get('descriptions', []) if isinstance(context, dict) else []
        desc_context = "\n".join([
            f"{table}.{column}: {desc}" 
            for table, column, desc in descriptions if isinstance(table, str) and isinstance(column, str)
        ])
        
        prompt = (
            f"### Schema Definitions:\n{ddl_context}\n"
            f"### Field Descriptions:\n{desc_context}\n"
            f"### Examples:\n{example_context}\n"
            f"### Query:\n\"{question}\"\n"
            "请只返回SQL语句，不要包含任何解释或说明。"
        )
        
        return prompt
    
    def _generate_sql_with_llm(self, prompt):
        """使用LLM生成SQL"""
        try:
            response = openai.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            
            raw_sql = response.choices[0].message.content.strip()
            sql = self._extract_sql(raw_sql)
            
            return sql
            
        except Exception as e:
            logging.error(f"LLM生成SQL失败: {e}")
            return None
    
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
            r'(DELETE.*?;)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # 如果都没有找到，返回原始文本
        return text.strip()
    
    def execute_sql(self, sql):
        """
        执行SQL语句
        
        Args:
            sql (str): SQL语句
            
        Returns:
            tuple: (success, columns, rows, error_msg)
        """
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text(sql))
                
                # 对于SELECT语句，获取结果
                if sql.strip().upper().startswith('SELECT'):
                    columns = list(result.keys())
                    rows = result.fetchall()
                    return True, columns, rows, None
                else:
                    # 对于INSERT/UPDATE/DELETE，返回影响的行数
                    return True, None, result.rowcount, None
                    
        except Exception as e:
            error_msg = str(e)
            logging.error(f"SQL执行失败: {error_msg}")
            return False, None, None, error_msg
    
    def get_retrieval_info(self, question):
        """
        获取检索信息，用于评估检索召回率
        
        Args:
            question (str): 自然语言问题
            
        Returns:
            dict: 检索到的表、列等信息
        """
        try:
            context = self._retrieve_context_simple(question)
            
            # 提取表名和列名
            retrieved_tables = set()
            retrieved_columns = set()
            
            # 从DDL中提取表名
            for ddl in context.get('ddl', []):
                tables = re.findall(r'CREATE TABLE\s+(\w+)', str(ddl), re.IGNORECASE)
                retrieved_tables.update(tables)
            
            # 从字段描述中提取表名和列名
            for table, column, _ in context.get('descriptions', []):
                if table:
                    retrieved_tables.add(table)
                if column:
                    retrieved_columns.add(column)
            
            return {
                'tables': list(retrieved_tables),
                'columns': list(retrieved_columns),
                'context': context
            }
            
        except Exception as e:
            logging.error(f"获取检索信息失败: {e}")
            return {'tables': [], 'columns': [], 'context': {}}


if __name__ == "__main__":
    # 测试代码
    system = SimpleText2SQLSystem()
    
    test_question = "List all actors with their IDs and names."
    sql = system.generate_sql(test_question)
    print(f"问题: {test_question}")
    print(f"生成的SQL: {sql}")
    
    if sql:
        success, cols, rows, error = system.execute_sql(sql)
        if success:
            print(f"执行成功，列: {cols}")
            print(f"结果行数: {len(rows) if rows else 0}")
        else:
            print(f"执行失败: {error}") 