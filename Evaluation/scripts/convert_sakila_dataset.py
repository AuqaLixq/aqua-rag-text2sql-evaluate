#!/usr/bin/env python3
"""
Sakila数据集转换脚本
将原始的q2sql_pairs.json转换为评估系统需要的格式，并自动推断表和列信息
"""

import json
import re
import yaml
from typing import Dict, List, Set, Any
from pathlib import Path

class SakilaDatasetConverter:
    def __init__(self, ddl_file: str, db_description_file: str):
        """初始化转换器"""
        self.ddl_file = ddl_file
        self.db_description_file = db_description_file
        self.table_info = {}
        self.column_info = {}
        self.load_schema_info()
    
    def load_schema_info(self):
        """加载数据库模式信息"""
        # 加载DDL信息
        with open(self.ddl_file, 'r', encoding='utf-8') as f:
            ddl_data = yaml.safe_load(f)
        
        # 加载描述信息
        with open(self.db_description_file, 'r', encoding='utf-8') as f:
            desc_data = yaml.safe_load(f)
        
        # 解析表和列信息
        for table_name, ddl in ddl_data.items():
            if 'CREATE TABLE' in ddl:  # 只处理表，不处理视图
                self.table_info[table_name] = {
                    'columns': self.extract_columns_from_ddl(ddl),
                    'primary_keys': self.extract_primary_keys_from_ddl(ddl),
                    'description': desc_data.get(table_name, {})
                }
    
    def extract_columns_from_ddl(self, ddl: str) -> List[str]:
        """从DDL中提取列名"""
        columns = []
        lines = ddl.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('`') and not line.startswith('PRIMARY KEY') and not line.startswith('KEY') and not line.startswith('CONSTRAINT'):
                # 提取列名
                match = re.match(r'`([^`]+)`', line)
                if match:
                    columns.append(match.group(1))
        return columns
    
    def extract_primary_keys_from_ddl(self, ddl: str) -> List[str]:
        """从DDL中提取主键"""
        primary_keys = []
        # 查找PRIMARY KEY定义
        pk_match = re.search(r'PRIMARY KEY \(`([^`]+)`\)', ddl)
        if pk_match:
            primary_keys.append(pk_match.group(1))
        return primary_keys
    
    def extract_tables_from_sql(self, sql: str) -> Set[str]:
        """从SQL语句中提取表名"""
        tables = set()
        sql_upper = sql.upper()
        
        # 常见的表名提取模式
        patterns = [
            r'\bFROM\s+`?(\w+)`?',
            r'\bJOIN\s+`?(\w+)`?',
            r'\bINTO\s+`?(\w+)`?',
            r'\bUPDATE\s+`?(\w+)`?',
            r'\bDELETE\s+FROM\s+`?(\w+)`?'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, sql_upper)
            for match in matches:
                table_name = match.lower()
                if table_name in self.table_info:
                    tables.add(table_name)
        
        return tables
    
    def extract_columns_from_sql(self, sql: str, tables: Set[str]) -> Set[str]:
        """从SQL语句中提取列名"""
        columns = set()
        sql_upper = sql.upper()
        
        # 提取SELECT子句中的列
        select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql_upper)
        if select_match:
            select_clause = select_match.group(1)
            if select_clause.strip() != '*':
                # 分割列名
                col_parts = select_clause.split(',')
                for part in col_parts:
                    part = part.strip()
                    # 移除别名
                    if ' AS ' in part:
                        part = part.split(' AS ')[0].strip()
                    # 移除表前缀
                    if '.' in part:
                        part = part.split('.')[-1]
                    # 移除反引号
                    part = part.replace('`', '').lower()
                    
                    # 检查是否是有效列名
                    for table in tables:
                        if part in self.table_info[table]['columns']:
                            columns.add(part)
        
        # 提取WHERE子句中的列
        where_patterns = [
            r'WHERE\s+`?(\w+)`?\s*[=<>!]',
            r'AND\s+`?(\w+)`?\s*[=<>!]',
            r'OR\s+`?(\w+)`?\s*[=<>!]'
        ]
        
        for pattern in where_patterns:
            matches = re.findall(pattern, sql_upper)
            for match in matches:
                col_name = match.lower()
                for table in tables:
                    if col_name in self.table_info[table]['columns']:
                        columns.add(col_name)
        
        return columns
    
    def identify_key_columns(self, tables: Set[str], columns: Set[str]) -> Set[str]:
        """识别关键列（主键、外键等）"""
        key_columns = set()
        
        for table in tables:
            table_info = self.table_info[table]
            # 添加主键
            for pk in table_info['primary_keys']:
                if pk in columns:
                    key_columns.add(pk)
            
            # 添加ID列（通常是关键列）
            for col in columns:
                if col.endswith('_id') or col == 'id':
                    key_columns.add(col)
        
        return key_columns
    
    def convert_dataset(self, input_file: str, output_file: str):
        """转换数据集"""
        # 读取原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
        
        converted_data = []
        
        for i, item in enumerate(original_data, 1):
            question = item['question']
            sql = item['sql']
            
            # 提取表和列信息
            tables = self.extract_tables_from_sql(sql)
            columns = self.extract_columns_from_sql(sql, tables)
            key_columns = self.identify_key_columns(tables, columns)
            
            # 构建转换后的数据项
            converted_item = {
                "id": i,
                "question": question,
                "expected_sql": sql,
                "expected_tables": list(tables),
                "expected_columns": list(columns),
                "key_columns": list(key_columns),
                "difficulty": self.assess_difficulty(sql),
                "sql_type": self.identify_sql_type(sql),
                "metadata": {
                    "source": "sakila",
                    "domain": "video_rental",
                    "complexity": len(tables) + len(columns)
                }
            }
            
            converted_data.append(converted_item)
        
        # 保存转换后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 数据集转换完成！")
        print(f"   原始数据: {len(original_data)} 条")
        print(f"   转换后数据: {len(converted_data)} 条")
        print(f"   输出文件: {output_file}")
        
        # 打印统计信息
        self.print_statistics(converted_data)
    
    def assess_difficulty(self, sql: str) -> str:
        """评估SQL难度"""
        sql_upper = sql.upper()
        
        # 简单：单表查询
        if 'JOIN' not in sql_upper and 'SUBQUERY' not in sql_upper:
            return "easy"
        
        # 困难：包含JOIN或子查询
        if 'JOIN' in sql_upper or 'SELECT' in sql_upper.count('SELECT') > 1:
            return "hard"
        
        # 中等：其他情况
        return "medium"
    
    def identify_sql_type(self, sql: str) -> str:
        """识别SQL类型"""
        sql_upper = sql.upper().strip()
        
        if sql_upper.startswith('SELECT'):
            return "SELECT"
        elif sql_upper.startswith('INSERT'):
            return "INSERT"
        elif sql_upper.startswith('UPDATE'):
            return "UPDATE"
        elif sql_upper.startswith('DELETE'):
            return "DELETE"
        else:
            return "OTHER"
    
    def print_statistics(self, data: List[Dict[str, Any]]):
        """打印数据集统计信息"""
        print("\n📊 数据集统计信息:")
        
        # SQL类型统计
        sql_types = {}
        difficulties = {}
        
        for item in data:
            sql_type = item['sql_type']
            difficulty = item['difficulty']
            
            sql_types[sql_type] = sql_types.get(sql_type, 0) + 1
            difficulties[difficulty] = difficulties.get(difficulty, 0) + 1
        
        print(f"   SQL类型分布: {sql_types}")
        print(f"   难度分布: {difficulties}")
        
        # 表使用统计
        table_usage = {}
        for item in data:
            for table in item['expected_tables']:
                table_usage[table] = table_usage.get(table, 0) + 1
        
        print(f"   最常用的表: {sorted(table_usage.items(), key=lambda x: x[1], reverse=True)[:5]}")

def main():
    """主函数"""
    # 文件路径
    base_dir = Path(__file__).parent.parent.parent  # 回到项目根目录
    ddl_file = base_dir / "Data" / "sakila" / "ddl_statements.yaml"
    db_desc_file = base_dir / "Data" / "sakila" / "db_description.yaml"
    input_file = base_dir / "Data" / "sakila" / "q2sql_pairs.json"
    output_file = base_dir / "Evaluation" / "datasets" / "sakila_real.json"
    
    # 确保输出目录存在
    output_file.parent.mkdir(exist_ok=True)
    
    # 创建转换器并执行转换
    converter = SakilaDatasetConverter(str(ddl_file), str(db_desc_file))
    converter.convert_dataset(str(input_file), str(output_file))

if __name__ == "__main__":
    main() 