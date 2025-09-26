"""
Text2SQL系统封装类
用于Salila评估框架调用现有的Text2SQL系统
"""

import os
import sys
import logging
import re
import json
from pathlib import Path

# 添加Text2SQL系统路径
sys.path.append(str(Path(__file__).parent.parent.parent / "Text2SQL" / "Sakila"))

try:
    from pymilvus import MilvusClient
    import openai
    from sqlalchemy import create_engine, text
    from dotenv import load_dotenv
    import yaml
    
    # 尝试导入pymilvus的model模块，如果失败则使用替代方案
    try:
        from pymilvus import model
    except ImportError:
        # 如果pymilvus版本不支持model模块，使用openai直接调用
        model = None
        logging.warning("pymilvus.model模块不可用，将使用openai直接调用嵌入")
        
except ImportError as e:
    logging.error(f"导入依赖失败: {e}")
    raise

class Text2SQLSystem:
    """Text2SQL系统封装类"""
    
    def __init__(self, config=None):
        """初始化Text2SQL系统"""
        self.config = config or {}
        self.setup_environment()
        self.setup_clients()
        
    def setup_environment(self):
        """设置环境变量和配置"""
        load_dotenv()
        
        # OpenAI配置
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.model_name = os.getenv("OPENAI_MODEL", "o4-mini")
        
        # 数据库配置
        self.db_url = os.getenv(
            "SAKILA_DB_URL", 
            "mysql+pymysql://root:password@localhost:3306/sakila"
        )
        
        # Milvus配置
        self.milvus_db = os.getenv("MILVUS_DB_PATH", "text2sql_milvus_sakila.db")
        
    def setup_clients(self):
        """初始化客户端连接"""
        try:
            # 初始化嵌入函数
            if model is not None:
                # 使用pymilvus的model模块
                self.embedding_fn = model.dense.OpenAIEmbeddingFunction(
                    model_name='text-embedding-3-large',
                )
            else:
                # 使用openai直接调用作为备选方案
                self.embedding_fn = None
                logging.info("使用OpenAI直接调用嵌入")
            
            # 初始化Milvus客户端
            self.milvus_client = MilvusClient(self.milvus_db)
            
            # 初始化数据库连接
            self.db_engine = create_engine(self.db_url)
            
            logging.info("Text2SQL系统初始化成功")
            
        except Exception as e:
            logging.error(f"Text2SQL系统初始化失败: {e}")
            raise
    
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
            # 1. 问题嵌入
            q_emb = self._get_embedding(question)
            
            # 2. RAG检索
            context = self._retrieve_context(q_emb)
            
            # 3. 构建prompt
            prompt = self._build_prompt(question, context)
            
            # 4. 生成SQL
            sql = self._generate_sql_with_llm(prompt)
            
            if return_context:
                return sql, context
            return sql
            
        except Exception as e:
            logging.error(f"SQL生成失败: {e}")
            return None
    
    def _get_embedding(self, text):
        """获取文本嵌入"""
        try:
            if self.embedding_fn is not None:
                # 使用pymilvus的嵌入函数
                return self.embedding_fn([text])[0].tolist()
            else:
                # 使用OpenAI直接调用
                response = openai.embeddings.create(
                    model="text-embedding-3-large",
                    input=text
                )
                return response.data[0].embedding
        except Exception as e:
            logging.error(f"嵌入生成失败: {e}")
            return []
    
    def _retrieve_context(self, query_embedding):
        """检索相关上下文"""
        context = {}
        
        try:
            # 检索DDL
            ddl_hits = self._retrieve("ddl_knowledge", query_embedding, 
                                    top_k=3, fields=["ddl_text"])
            context['ddl'] = [hit.get("ddl_text", "") for hit in ddl_hits]
            
            # 检索示例
            q2sql_hits = self._retrieve("q2sql_knowledge", query_embedding,
                                      top_k=3, fields=["question", "sql_text"])
            context['examples'] = [(hit.get('question', ''), hit.get('sql_text', '')) 
                                 for hit in q2sql_hits]
            
            # 检索字段描述
            desc_hits = self._retrieve("dbdesc_knowledge", query_embedding,
                                     top_k=5, fields=["table_name", "column_name", "description"])
            context['descriptions'] = [(hit.get('table_name', ''), 
                                      hit.get('column_name', ''), 
                                      hit.get('description', '')) 
                                     for hit in desc_hits]
            
        except Exception as e:
            logging.error(f"上下文检索失败: {e}")
            context = {'ddl': [], 'examples': [], 'descriptions': []}
            
        return context
    
    def _retrieve(self, collection, query_emb, top_k=3, fields=None):
        """从指定集合检索"""
        try:
            results = self.milvus_client.search(
                collection_name=collection,
                data=[query_emb],
                limit=top_k,
                output_fields=fields
            )
            return results[0] if results else []
        except Exception as e:
            logging.error(f"检索失败 {collection}: {e}")
            return []
    
    def _build_prompt(self, question, context):
        """构建LLM提示"""
        # DDL上下文
        ddl_context = "\n".join(context.get('ddl', []))
        
        # 示例上下文
        example_context = "\n".join([
            f"NL: \"{q}\"\nSQL: \"{sql}\"" 
            for q, sql in context.get('examples', [])
        ])
        
        # 字段描述上下文
        desc_context = "\n".join([
            f"{table}.{column}: {desc}" 
            for table, column, desc in context.get('descriptions', [])
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
            q_emb = self._get_embedding(question)
            context = self._retrieve_context(q_emb)
            
            # 提取表名和列名
            retrieved_tables = set()
            retrieved_columns = set()
            
            # 从DDL中提取表名
            for ddl in context.get('ddl', []):
                tables = re.findall(r'CREATE TABLE\s+(\w+)', ddl, re.IGNORECASE)
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
    system = Text2SQLSystem()
    
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