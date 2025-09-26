"""
Text2SQL评估指标计算模块
实现执行准确率、语法正确率、语义相似度、检索召回率等指标
"""

import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

class EvaluationMetrics:
    """评估指标计算类"""
    
    def __init__(self, db_connection_string=None, mock_mode=False):
        """
        初始化评估指标计算器
        
        Args:
            db_connection_string (str): 数据库连接字符串
            mock_mode (bool): 是否启用模拟模式
        """
        self.db_connection_string = db_connection_string
        self.mock_mode = mock_mode
        
        if db_connection_string and not mock_mode:
            try:
                self.db_engine = create_engine(db_connection_string)
            except Exception as e:
                logging.warning(f"数据库连接失败，启用模拟模式: {e}")
                self.mock_mode = True
                self.db_engine = None
        else:
            self.db_engine = None
        
        # 初始化本地嵌入模型用于语义相似度计算（替代OpenAI）
        try:
            self.embedding_model = SentenceTransformer(
                os.getenv("SEM_SIM_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            )
        except Exception as e:
            logging.warning(f"本地嵌入模型加载失败，将使用退化相似度: {e}")
            self.embedding_model = None
        
    def calculate_execution_accuracy(self, generated_sql: str, expected_sql: str) -> Dict[str, Any]:
        """
        计算执行准确率
        
        Args:
            generated_sql (str): 生成的SQL
            expected_sql (str): 期望的SQL
            
        Returns:
            dict: 执行准确率结果
        """
        result = {
            'execution_accuracy': 0.0,
            'generated_executable': False,
            'expected_executable': False,
            'results_match': False,
            'error_message': None
        }
        
        # 模拟模式下跳过实际执行
        if self.mock_mode:
            result['execution_accuracy'] = 0.0
            result['error_message'] = "Mock mode - execution skipped"
            return result
        
        try:
            # 执行生成的SQL
            gen_success, gen_result = self._execute_sql_safe(generated_sql)
            result['generated_executable'] = gen_success
            
            # 执行期望的SQL
            exp_success, exp_result = self._execute_sql_safe(expected_sql)
            result['expected_executable'] = exp_success
            
            if gen_success and exp_success:
                # 比较结果
                results_match = self._compare_sql_results(gen_result, exp_result)
                result['results_match'] = results_match
                result['execution_accuracy'] = 1.0 if results_match else 0.0
            elif gen_success and not exp_success:
                # 生成的SQL可执行但期望的不可执行，可能是测试数据问题
                result['execution_accuracy'] = 0.5
                result['error_message'] = "Expected SQL not executable"
            else:
                result['execution_accuracy'] = 0.0
                
        except Exception as e:
            result['error_message'] = str(e)
            logging.error(f"执行准确率计算失败: {e}")
            
        return result
    
    def calculate_syntax_accuracy(self, generated_sql: str) -> Dict[str, Any]:
        """
        计算语法正确率
        
        Args:
            generated_sql (str): 生成的SQL
            
        Returns:
            dict: 语法正确率结果
        """
        result = {
            'syntax_accuracy': 0.0,
            'is_valid': False,
            'parse_error': None
        }
        
        try:
            # 简单的SQL语法检查
            sql_upper = generated_sql.upper().strip()
            
            # 检查基本SQL关键字
            valid_starts = ['SELECT', 'INSERT', 'UPDATE', 'DELETE']
            has_valid_start = any(sql_upper.startswith(keyword) for keyword in valid_starts)
            
            if has_valid_start:
                result['is_valid'] = True
                result['syntax_accuracy'] = 1.0
                
                # 进一步检查SQL结构
                if self._validate_basic_sql_structure(generated_sql):
                    result['syntax_accuracy'] = 1.0
                else:
                    result['syntax_accuracy'] = 0.5
                    result['parse_error'] = "SQL structure validation failed"
            else:
                result['syntax_accuracy'] = 0.0
                result['parse_error'] = "Invalid SQL start keyword"
                
        except Exception as e:
            result['parse_error'] = str(e)
            logging.error(f"语法正确率计算失败: {e}")
            
        return result
    
    def calculate_semantic_similarity(self, generated_sql: str, expected_sql: str, 
                                    question: str = None) -> Dict[str, Any]:
        """
        计算语义相似度
        
        Args:
            generated_sql (str): 生成的SQL
            expected_sql (str): 期望的SQL
            question (str): 原始问题（可选）
            
        Returns:
            dict: 语义相似度结果
        """
        result = {
            'semantic_similarity': 0.0,
            'embedding_similarity': 0.0,
            'structural_similarity': 0.0,
            'error_message': None
        }
        
        try:
            # 1. 基于嵌入的相似度（本地模型）
            embedding_sim = self._calculate_embedding_similarity_local(generated_sql, expected_sql)
            result['embedding_similarity'] = embedding_sim
            
            # 2. 结构相似度
            structural_sim = self._calculate_structural_similarity(generated_sql, expected_sql)
            result['structural_similarity'] = structural_sim
            
            # 3. 综合语义相似度（加权平均）
            result['semantic_similarity'] = 0.7 * embedding_sim + 0.3 * structural_sim
            
        except Exception as e:
            result['error_message'] = str(e)
            logging.error(f"语义相似度计算失败: {e}")
            
        return result

    def _calculate_embedding_similarity_local(self, text_a: str, text_b: str) -> float:
        """使用本地Sentence-Transformers计算余弦相似度"""
        try:
            if not text_a or not text_b:
                return 0.0
            if self.embedding_model is None:
                # 退化：简单字符串重叠率
                set_a = set(text_a.lower().split())
                set_b = set(text_b.lower().split())
                inter = len(set_a & set_b)
                union = len(set_a | set_b) or 1
                return inter / union
            emb = self.embedding_model.encode([text_a, text_b])
            sim = cosine_similarity([emb[0]], [emb[1]])[0][0]
            # 归一化到[0,1]
            return float(max(0.0, min(1.0, (sim + 1.0) / 2.0))) if sim < 0 else float(min(1.0, sim))
        except Exception as e:
            logging.error(f"嵌入相似度计算失败: {e}")
            return 0.0
    
    def calculate_retrieval_recall(self, retrieved_info: Dict, expected_info: Dict) -> Dict[str, Any]:
        """
        计算检索召回率 - 细分为多个维度
        
        Args:
            retrieved_info (dict): 检索到的信息 {'tables': [], 'columns': []}
            expected_info (dict): 期望的信息 {'tables': [], 'columns': [], 'key_columns': []}
            
        Returns:
            dict: 详细的检索召回率结果
        """
        result = {
            # 基础召回率
            'table_recall': 0.0,
            'column_recall': 0.0,
            'key_column_recall': 0.0,
            'overall_recall': 0.0,
            
            # 详细分析
            'table_analysis': {
                'expected_count': 0,
                'retrieved_count': 0,
                'matched_count': 0,
                'missing_tables': [],
                'extra_tables': []
            },
            'column_analysis': {
                'expected_count': 0,
                'retrieved_count': 0,
                'matched_count': 0,
                'missing_columns': [],
                'extra_columns': []
            },
            'key_column_analysis': {
                'expected_count': 0,
                'retrieved_count': 0,
                'matched_count': 0,
                'missing_key_columns': [],
                'key_column_coverage': 0.0
            },
            
            # SQL类型特定分析
            'sql_type_analysis': self._analyze_sql_type_specific_recall(retrieved_info, expected_info),
            
            # 权重召回率（根据重要性加权）
            'weighted_recall': 0.0
        }
        
        try:
            retrieved_tables = set(retrieved_info.get('tables', []))
            retrieved_columns = set(retrieved_info.get('columns', []))
            
            expected_tables = set(expected_info.get('expected_tables', []))
            expected_columns = set(expected_info.get('expected_columns', []))
            expected_key_columns = set(expected_info.get('key_columns', []))
            
            # 1. 表级召回率分析
            result['table_analysis'] = self._analyze_table_recall(retrieved_tables, expected_tables)
            if expected_tables:
                result['table_recall'] = result['table_analysis']['matched_count'] / len(expected_tables)
            
            # 2. 列级召回率分析
            result['column_analysis'] = self._analyze_column_recall(retrieved_columns, expected_columns)
            if expected_columns:
                result['column_recall'] = result['column_analysis']['matched_count'] / len(expected_columns)
            
            # 3. 关键列召回率分析
            result['key_column_analysis'] = self._analyze_key_column_recall(
                retrieved_columns, expected_key_columns, expected_columns
            )
            if expected_key_columns:
                result['key_column_recall'] = result['key_column_analysis']['matched_count'] / len(expected_key_columns)
            
            # 4. 计算权重召回率
            result['weighted_recall'] = self._calculate_weighted_recall(result)
            
            # 5. 综合召回率（使用权重）
            result['overall_recall'] = result['weighted_recall']
            
        except Exception as e:
            logging.error(f"检索召回率计算失败: {e}")
            
        return result
    
    def _execute_sql_safe(self, sql: str) -> Tuple[bool, Any]:
        """
        安全执行SQL
        
        Args:
            sql (str): SQL语句
            
        Returns:
            tuple: (是否成功, 结果)
        """
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text(sql))
                
                if sql.strip().upper().startswith('SELECT'):
                    rows = result.fetchall()
                    columns = list(result.keys())
                    return True, {'columns': columns, 'rows': rows}
                else:
                    return True, {'affected_rows': result.rowcount}
                    
        except Exception as e:
            logging.error(f"SQL执行失败: {e}")
            return False, str(e)
    
    def _compare_sql_results(self, result1: Dict, result2: Dict) -> bool:
        """
        比较两个SQL执行结果
        
        Args:
            result1 (dict): 第一个结果
            result2 (dict): 第二个结果
            
        Returns:
            bool: 结果是否相同
        """
        try:
            # 如果都是SELECT结果
            if 'rows' in result1 and 'rows' in result2:
                rows1 = [tuple(row) for row in result1['rows']]
                rows2 = [tuple(row) for row in result2['rows']]
                
                # 排序后比较（处理顺序不同的情况）
                return sorted(rows1) == sorted(rows2)
            
            # 如果都是修改操作结果
            elif 'affected_rows' in result1 and 'affected_rows' in result2:
                return result1['affected_rows'] == result2['affected_rows']
            
            return False
            
        except Exception as e:
            logging.error(f"结果比较失败: {e}")
            return False
    
    def _validate_basic_sql_structure(self, sql: str) -> bool:
        """
        验证基本SQL结构
        
        Args:
            sql (str): SQL语句
            
        Returns:
            bool: 结构是否有效
        """
        try:
            sql_upper = sql.upper().strip()
            
            # 检查SELECT语句结构
            if sql_upper.startswith('SELECT'):
                return 'FROM' in sql_upper and ';' in sql
            
            # 检查INSERT语句结构
            elif sql_upper.startswith('INSERT'):
                return 'INTO' in sql_upper and 'VALUES' in sql_upper and ';' in sql
            
            # 检查UPDATE语句结构
            elif sql_upper.startswith('UPDATE'):
                return 'SET' in sql_upper and ';' in sql
            
            # 检查DELETE语句结构
            elif sql_upper.startswith('DELETE'):
                return 'FROM' in sql_upper and ';' in sql
            
            return False
                   
        except Exception:
            return False
    
    def _calculate_embedding_similarity(self, sql1: str, sql2: str) -> float:
        """
        计算SQL语句的嵌入相似度
        
        Args:
            sql1 (str): 第一个SQL
            sql2 (str): 第二个SQL
            
        Returns:
            float: 相似度分数
        """
        try:
            # 使用OpenAI嵌入API
            response = openai.embeddings.create(
                model="text-embedding-3-large",
                input=[sql1, sql2]
            )
            
            embeddings = [data.embedding for data in response.data]
            
            # 计算余弦相似度
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                return max(0.0, similarity)  # 确保非负
            except ImportError:
                # 如果sklearn不可用，使用简单的余弦相似度计算
                import numpy as np
                vec1 = np.array(embeddings[0])
                vec2 = np.array(embeddings[1])
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                return max(0.0, cos_sim)
            
        except Exception as e:
            logging.error(f"嵌入相似度计算失败: {e}")
            # 如果API调用失败，使用简单的字符串相似度
            return self._simple_string_similarity(sql1, sql2)
    
    def _simple_string_similarity(self, str1: str, str2: str) -> float:
        """简单的字符串相似度计算"""
        try:
            # 使用Jaccard相似度
            set1 = set(str1.lower().split())
            set2 = set(str2.lower().split())
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_structural_similarity(self, sql1: str, sql2: str) -> float:
        """
        计算SQL结构相似度
        
        Args:
            sql1 (str): 第一个SQL
            sql2 (str): 第二个SQL
            
        Returns:
            float: 结构相似度分数
        """
        try:
            # 解析SQL结构
            struct1 = self._extract_sql_structure(sql1)
            struct2 = self._extract_sql_structure(sql2)
            
            # 计算结构相似度
            similarity_scores = []
            
            # 比较操作类型
            if struct1.get('operation') == struct2.get('operation'):
                similarity_scores.append(1.0)
            else:
                similarity_scores.append(0.0)
            
            # 比较表名
            tables1 = set(struct1.get('tables', []))
            tables2 = set(struct2.get('tables', []))
            if tables1 and tables2:
                table_sim = len(tables1.intersection(tables2)) / len(tables1.union(tables2))
                similarity_scores.append(table_sim)
            
            # 比较列名
            columns1 = set(struct1.get('columns', []))
            columns2 = set(struct2.get('columns', []))
            if columns1 and columns2:
                column_sim = len(columns1.intersection(columns2)) / len(columns1.union(columns2))
                similarity_scores.append(column_sim)
            
            return np.mean(similarity_scores) if similarity_scores else 0.0
            
        except Exception as e:
            logging.error(f"结构相似度计算失败: {e}")
            return 0.0
    
    def _extract_sql_structure(self, sql: str) -> Dict[str, Any]:
        """
        提取SQL结构信息
        
        Args:
            sql (str): SQL语句
            
        Returns:
            dict: 结构信息
        """
        structure = {
            'operation': None,
            'tables': [],
            'columns': []
        }
        
        try:
            # 确定操作类型
            sql_upper = sql.upper().strip()
            if sql_upper.startswith('SELECT'):
                structure['operation'] = 'SELECT'
            elif sql_upper.startswith('INSERT'):
                structure['operation'] = 'INSERT'
            elif sql_upper.startswith('UPDATE'):
                structure['operation'] = 'UPDATE'
            elif sql_upper.startswith('DELETE'):
                structure['operation'] = 'DELETE'
            
            # 提取表名
            table_patterns = [
                r'FROM\s+(\w+)',
                r'INTO\s+(\w+)',
                r'UPDATE\s+(\w+)',
                r'DELETE\s+FROM\s+(\w+)'
            ]
            
            for pattern in table_patterns:
                matches = re.findall(pattern, sql, re.IGNORECASE)
                structure['tables'].extend(matches)
            
            # 提取列名（简单实现）
            if structure['operation'] == 'SELECT':
                # 提取SELECT后的列名
                select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
                if select_match:
                    columns_str = select_match.group(1)
                    # 简单分割（实际应该更复杂的解析）
                    columns = [col.strip() for col in columns_str.split(',')]
                    structure['columns'] = [col for col in columns if col != '*']
            
        except Exception as e:
            logging.error(f"SQL结构提取失败: {e}")
            
        return structure
    
    def _analyze_table_recall(self, retrieved_tables: set, expected_tables: set) -> Dict[str, Any]:
        """分析表级召回率"""
        matched_tables = retrieved_tables.intersection(expected_tables)
        missing_tables = expected_tables - retrieved_tables
        extra_tables = retrieved_tables - expected_tables
        
        return {
            'expected_count': len(expected_tables),
            'retrieved_count': len(retrieved_tables),
            'matched_count': len(matched_tables),
            'missing_tables': list(missing_tables),
            'extra_tables': list(extra_tables),
            'precision': len(matched_tables) / len(retrieved_tables) if retrieved_tables else 0.0,
            'recall': len(matched_tables) / len(expected_tables) if expected_tables else 0.0
        }
    
    def _analyze_column_recall(self, retrieved_columns: set, expected_columns: set) -> Dict[str, Any]:
        """分析列级召回率"""
        matched_columns = retrieved_columns.intersection(expected_columns)
        missing_columns = expected_columns - retrieved_columns
        extra_columns = retrieved_columns - expected_columns
        
        return {
            'expected_count': len(expected_columns),
            'retrieved_count': len(retrieved_columns),
            'matched_count': len(matched_columns),
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns),
            'precision': len(matched_columns) / len(retrieved_columns) if retrieved_columns else 0.0,
            'recall': len(matched_columns) / len(expected_columns) if expected_columns else 0.0
        }
    
    def _analyze_key_column_recall(self, retrieved_columns: set, expected_key_columns: set, 
                                 expected_all_columns: set) -> Dict[str, Any]:
        """分析关键列召回率"""
        matched_key_columns = retrieved_columns.intersection(expected_key_columns)
        missing_key_columns = expected_key_columns - retrieved_columns
        
        # 关键列覆盖率：检索到的关键列占所有期望列的比例
        key_column_coverage = 0.0
        if expected_all_columns:
            key_column_coverage = len(matched_key_columns) / len(expected_all_columns)
        
        return {
            'expected_count': len(expected_key_columns),
            'retrieved_count': len(retrieved_columns.intersection(expected_key_columns)),
            'matched_count': len(matched_key_columns),
            'missing_key_columns': list(missing_key_columns),
            'key_column_coverage': key_column_coverage,
            'precision': len(matched_key_columns) / len(retrieved_columns) if retrieved_columns else 0.0,
            'recall': len(matched_key_columns) / len(expected_key_columns) if expected_key_columns else 0.0
        }
    
    def _analyze_sql_type_specific_recall(self, retrieved_info: Dict, expected_info: Dict) -> Dict[str, Any]:
        """根据SQL类型进行特定分析"""
        sql_type = expected_info.get('sql_type', 'UNKNOWN')
        
        analysis = {
            'sql_type': sql_type,
            'type_specific_score': 0.0,
            'critical_elements_found': [],
            'critical_elements_missing': []
        }
        
        retrieved_tables = set(retrieved_info.get('tables', []))
        retrieved_columns = set(retrieved_info.get('columns', []))
        expected_tables = set(expected_info.get('expected_tables', []))
        expected_columns = set(expected_info.get('expected_columns', []))
        expected_key_columns = set(expected_info.get('key_columns', []))
        
        if sql_type == 'SELECT':
            # SELECT查询：重点关注查询的表和列
            critical_elements = expected_tables.union(expected_columns)
            found_elements = retrieved_tables.union(retrieved_columns)
            
            analysis['critical_elements_found'] = list(critical_elements.intersection(found_elements))
            analysis['critical_elements_missing'] = list(critical_elements - found_elements)
            
            if critical_elements:
                analysis['type_specific_score'] = len(analysis['critical_elements_found']) / len(critical_elements)
        
        elif sql_type in ['INSERT', 'UPDATE', 'DELETE']:
            # 修改操作：重点关注目标表和关键列（如主键、外键）
            critical_elements = expected_tables.union(expected_key_columns)
            found_elements = retrieved_tables.union(retrieved_columns)
            
            analysis['critical_elements_found'] = list(critical_elements.intersection(found_elements))
            analysis['critical_elements_missing'] = list(critical_elements - found_elements)
            
            if critical_elements:
                # 对于修改操作，表的权重更高
                table_score = len(expected_tables.intersection(retrieved_tables)) / len(expected_tables) if expected_tables else 1.0
                key_col_score = len(expected_key_columns.intersection(retrieved_columns)) / len(expected_key_columns) if expected_key_columns else 1.0
                
                analysis['type_specific_score'] = 0.7 * table_score + 0.3 * key_col_score
        
        return analysis
    
    def _calculate_weighted_recall(self, result: Dict[str, Any]) -> float:
        """计算权重召回率"""
        # 权重配置：表 > 关键列 > 普通列
        weights = {
            'table': 0.5,      # 表是最重要的
            'key_column': 0.3,  # 关键列次之
            'column': 0.2       # 普通列权重最低
        }
        
        table_recall = result.get('table_recall', 0.0)
        key_column_recall = result.get('key_column_recall', 0.0)
        column_recall = result.get('column_recall', 0.0)
        
        # 如果某个维度没有期望值，则不参与权重计算
        active_weights = {}
        total_weight = 0.0
        
        if result['table_analysis']['expected_count'] > 0:
            active_weights['table'] = weights['table']
            total_weight += weights['table']
        
        if result['key_column_analysis']['expected_count'] > 0:
            active_weights['key_column'] = weights['key_column']
            total_weight += weights['key_column']
        
        if result['column_analysis']['expected_count'] > 0:
            active_weights['column'] = weights['column']
            total_weight += weights['column']
        
        # 重新归一化权重
        if total_weight > 0:
            weighted_score = 0.0
            if 'table' in active_weights:
                weighted_score += (active_weights['table'] / total_weight) * table_recall
            if 'key_column' in active_weights:
                weighted_score += (active_weights['key_column'] / total_weight) * key_column_recall
            if 'column' in active_weights:
                weighted_score += (active_weights['column'] / total_weight) * column_recall
            
            return weighted_score
        
        return 0.0


class MetricsAggregator:
    """指标聚合器"""
    
    def __init__(self, config=None):
        self.results = []
        self.config = config or {}
    
    def add_result(self, result: Dict[str, Any]):
        """添加单个评估结果"""
        self.results.append(result)
    
    def calculate_overall_metrics(self) -> Dict[str, Any]:
        """计算总体指标"""
        if not self.results:
            return {}
        
        overall = {
            'total_queries': len(self.results),
            'syntax_accuracy': {
                'mean': np.mean([r.get('syntax_accuracy', {}).get('syntax_accuracy', 0) for r in self.results]),
                'valid_rate': sum(1 for r in self.results if r.get('syntax_accuracy', {}).get('is_valid', False)) / len(self.results)
            },
            'semantic_similarity': {
                'mean': np.mean([r.get('semantic_similarity', {}).get('semantic_similarity', 0) for r in self.results])
            },
            'retrieval_recall': {
                'table_recall_mean': np.mean([r.get('retrieval_recall', {}).get('table_recall', 0) for r in self.results]),
                'column_recall_mean': np.mean([r.get('retrieval_recall', {}).get('column_recall', 0) for r in self.results]),
                'key_column_recall_mean': np.mean([r.get('retrieval_recall', {}).get('key_column_recall', 0) for r in self.results]),
                'weighted_recall_mean': np.mean([r.get('retrieval_recall', {}).get('weighted_recall', 0) for r in self.results]),
                'overall_recall_mean': np.mean([r.get('retrieval_recall', {}).get('overall_recall', 0) for r in self.results]),
                
                # 详细统计
                'table_analysis': self._aggregate_table_analysis(),
                'column_analysis': self._aggregate_column_analysis(),
                'key_column_analysis': self._aggregate_key_column_analysis(),
                'sql_type_analysis': self._aggregate_sql_type_analysis()
            }
        }
        
        # 只在启用执行准确率时添加该指标
        metrics_config = self.config.get('metrics', {})
        if metrics_config.get('execution_accuracy', {}).get('enabled', False):
            overall['execution_accuracy'] = {
                'mean': np.mean([r.get('execution_accuracy', {}).get('execution_accuracy', 0) for r in self.results]),
                'success_rate': sum(1 for r in self.results if r.get('execution_accuracy', {}).get('generated_executable', False)) / len(self.results)
            }
        
        # 计算加权综合分数 - 从配置文件读取权重
        metrics_config = self.config.get('metrics', {})
        weights = {
            'execution_accuracy': metrics_config.get('execution_accuracy', {}).get('weight', 0.4),
            'syntax_accuracy': metrics_config.get('syntax_accuracy', {}).get('weight', 0.2),
            'semantic_similarity': metrics_config.get('semantic_similarity', {}).get('weight', 0.2),
            'retrieval_recall': metrics_config.get('retrieval_recall', {}).get('weight', 0.2)
        }
        
        # 处理NaN值
        syntax_score = overall['syntax_accuracy']['mean']
        semantic_score = overall['semantic_similarity']['mean']
        recall_score = overall['retrieval_recall']['overall_recall_mean']
        
        # 替换NaN为0
        syntax_score = 0.0 if np.isnan(syntax_score) else syntax_score
        semantic_score = 0.0 if np.isnan(semantic_score) else semantic_score
        recall_score = 0.0 if np.isnan(recall_score) else recall_score
        
        # 计算加权分数，只包含启用的指标
        weighted_score = (
            weights['syntax_accuracy'] * syntax_score +
            weights['semantic_similarity'] * semantic_score +
            weights['retrieval_recall'] * recall_score
        )
        
        # 如果启用了执行准确率，则添加到计算中
        if 'execution_accuracy' in overall:
            exec_score = overall['execution_accuracy']['mean']
            exec_score = 0.0 if np.isnan(exec_score) else exec_score
            weighted_score += weights['execution_accuracy'] * exec_score
        
        overall['weighted_overall_score'] = weighted_score
        
        return overall
    
    def get_detailed_results(self) -> List[Dict[str, Any]]:
        """获取详细结果"""
        return self.results
    
    def _aggregate_table_analysis(self) -> Dict[str, Any]:
        """聚合表级分析"""
        table_analyses = [r.get('retrieval_recall', {}).get('table_analysis', {}) for r in self.results]
        
        total_expected = sum(analysis.get('expected_count', 0) for analysis in table_analyses)
        total_retrieved = sum(analysis.get('retrieved_count', 0) for analysis in table_analyses)
        total_matched = sum(analysis.get('matched_count', 0) for analysis in table_analyses)
        
        # 统计最常缺失的表
        missing_tables = []
        for analysis in table_analyses:
            missing_tables.extend(analysis.get('missing_tables', []))
        
        missing_table_counts = {}
        for table in missing_tables:
            missing_table_counts[table] = missing_table_counts.get(table, 0) + 1
        
        return {
            'total_expected': total_expected,
            'total_retrieved': total_retrieved,
            'total_matched': total_matched,
            'overall_precision': total_matched / total_retrieved if total_retrieved > 0 else 0.0,
            'overall_recall': total_matched / total_expected if total_expected > 0 else 0.0,
            'most_missed_tables': sorted(missing_table_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _aggregate_column_analysis(self) -> Dict[str, Any]:
        """聚合列级分析"""
        column_analyses = [r.get('retrieval_recall', {}).get('column_analysis', {}) for r in self.results]
        
        total_expected = sum(analysis.get('expected_count', 0) for analysis in column_analyses)
        total_retrieved = sum(analysis.get('retrieved_count', 0) for analysis in column_analyses)
        total_matched = sum(analysis.get('matched_count', 0) for analysis in column_analyses)
        
        # 统计最常缺失的列
        missing_columns = []
        for analysis in column_analyses:
            missing_columns.extend(analysis.get('missing_columns', []))
        
        missing_column_counts = {}
        for column in missing_columns:
            missing_column_counts[column] = missing_column_counts.get(column, 0) + 1
        
        return {
            'total_expected': total_expected,
            'total_retrieved': total_retrieved,
            'total_matched': total_matched,
            'overall_precision': total_matched / total_retrieved if total_retrieved > 0 else 0.0,
            'overall_recall': total_matched / total_expected if total_expected > 0 else 0.0,
            'most_missed_columns': sorted(missing_column_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _aggregate_key_column_analysis(self) -> Dict[str, Any]:
        """聚合关键列分析"""
        key_column_analyses = [r.get('retrieval_recall', {}).get('key_column_analysis', {}) for r in self.results]
        
        total_expected = sum(analysis.get('expected_count', 0) for analysis in key_column_analyses)
        total_matched = sum(analysis.get('matched_count', 0) for analysis in key_column_analyses)
        
        # 统计最常缺失的关键列
        missing_key_columns = []
        for analysis in key_column_analyses:
            missing_key_columns.extend(analysis.get('missing_key_columns', []))
        
        missing_key_column_counts = {}
        for column in missing_key_columns:
            missing_key_column_counts[column] = missing_key_column_counts.get(column, 0) + 1
        
        # 平均关键列覆盖率
        avg_coverage = np.mean([analysis.get('key_column_coverage', 0) for analysis in key_column_analyses])
        
        return {
            'total_expected': total_expected,
            'total_matched': total_matched,
            'overall_recall': total_matched / total_expected if total_expected > 0 else 0.0,
            'average_coverage': avg_coverage,
            'most_missed_key_columns': sorted(missing_key_column_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _aggregate_sql_type_analysis(self) -> Dict[str, Any]:
        """聚合SQL类型分析"""
        sql_type_analyses = [r.get('retrieval_recall', {}).get('sql_type_analysis', {}) for r in self.results]
        
        # 按SQL类型分组
        type_groups = {}
        for analysis in sql_type_analyses:
            sql_type = analysis.get('sql_type', 'UNKNOWN')
            if sql_type not in type_groups:
                type_groups[sql_type] = []
            type_groups[sql_type].append(analysis)
        
        # 计算每种类型的平均分数
        type_scores = {}
        for sql_type, analyses in type_groups.items():
            scores = [analysis.get('type_specific_score', 0) for analysis in analyses]
            type_scores[sql_type] = {
                'count': len(analyses),
                'average_score': np.mean(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores)
            }
        
        return {
            'type_distribution': {sql_type: info['count'] for sql_type, info in type_scores.items()},
            'type_performance': type_scores
        }


if __name__ == "__main__":
    # 测试代码
    metrics = EvaluationMetrics("mysql+pymysql://root:password@localhost:3306/sakila")
    
    # 测试执行准确率
    gen_sql = "SELECT actor_id, first_name, last_name FROM actor;"
    exp_sql = "SELECT actor_id, first_name, last_name FROM actor;"
    
    exec_result = metrics.calculate_execution_accuracy(gen_sql, exp_sql)
    print("执行准确率:", exec_result)
    
    # 测试语法正确率
    syntax_result = metrics.calculate_syntax_accuracy(gen_sql)
    print("语法正确率:", syntax_result)
