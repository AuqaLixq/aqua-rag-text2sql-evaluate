"""
Salila Text2SQL评估主程序
执行完整的Text2SQL系统评估流程
"""

import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

# 延迟导入Text2SQL系统，避免在模拟模式下产生不必要的警告
from evaluation_metrics import EvaluationMetrics, MetricsAggregator

def convert_numpy_types(obj):
    """转换numpy类型为Python原生类型，解决JSON序列化问题"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class SalilaEvaluator:
    """Salila Text2SQL评估器"""
    
    def __init__(self, config_path: str):
        """
        初始化评估器
        
        Args:
            config_path (str): 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # 创建输出目录（在setup_logging之前）
        self.output_dir = Path(self.config['evaluation']['output_path'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.setup_logging()
        
        # 初始化组件
        # 检查是否启用模拟模式，如果是则直接使用简化版系统
        mock_mode = self.config.get('evaluation', {}).get('mock_mode', False)
        
        if mock_mode:
            self.logger.info("模拟模式：直接使用简化版Text2SQL系统")
            from text2sql_system_simple import SimpleText2SQLSystem
            self.text2sql_system = SimpleText2SQLSystem(self.config.get('model', {}))
        else:
            # 检查是否使用Hugging Face模型
            use_huggingface = self.config.get('model', {}).get('use_huggingface', False)
            
            if use_huggingface:
                try:
                    from text2sql_system_huggingface import HuggingFaceText2SQLSystem
                    self.text2sql_system = HuggingFaceText2SQLSystem(self.config.get('model', {}))
                    self.logger.info("使用Hugging Face Text2SQL系统")
                except Exception as e:
                    self.logger.warning(f"Hugging Face Text2SQL系统初始化失败: {e}")
                    self.logger.info("回退到简化版Text2SQL系统")
                    from text2sql_system_simple import SimpleText2SQLSystem
                    self.text2sql_system = SimpleText2SQLSystem(self.config.get('model', {}))
            else:
                try:
                    # 只有在非模拟模式下才导入完整版系统
                    from text2sql_system import Text2SQLSystem
                    self.text2sql_system = Text2SQLSystem(self.config.get('model', {}))
                    self.logger.info("使用完整版Text2SQL系统")
                except Exception as e:
                    self.logger.warning(f"完整版Text2SQL系统初始化失败: {e}")
                    self.logger.info("回退到简化版Text2SQL系统")
                    from text2sql_system_simple import SimpleText2SQLSystem
                    self.text2sql_system = SimpleText2SQLSystem(self.config.get('model', {}))
        
        # 检查是否启用模拟模式
        mock_mode = self.config.get('evaluation', {}).get('mock_mode', False)
        self.metrics_calculator = EvaluationMetrics(
            self.config['database']['connection_string'],
            mock_mode=mock_mode
        )
        self.metrics_aggregator = MetricsAggregator(self.config)
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            raise Exception(f"配置文件加载失败: {e}")
    
    def setup_logging(self):
        """Configure logging system"""
        log_level = getattr(logging, self.config['evaluation'].get('log_level', 'INFO'))
        
        # Minimalist log format
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Console output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        # File logging
        log_file = self.output_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Setup logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        root_logger.handlers.clear()
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting evaluation with config: {self.config_path}")
    
    def load_test_dataset(self) -> List[Dict[str, Any]]:
        """加载测试数据集"""
        dataset_path = Path(self.config['evaluation']['dataset_path'])
        
        # 处理相对路径
        if not dataset_path.is_absolute():
            dataset_path = Path(self.config_path).parent / dataset_path
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            self.logger.info(f"成功加载测试数据集，共 {len(dataset)} 条记录")
            return dataset
            
        except Exception as e:
            self.logger.error(f"测试数据集加载失败: {e}")
            raise
    
    def evaluate_single_query(self, test_item: Dict[str, Any]) -> Dict[str, Any]:
        """
        评估单个查询
        
        Args:
            test_item (dict): 测试项目
            
        Returns:
            dict: 评估结果
        """
        query_id = test_item.get('id', 'unknown')
        question = test_item['question']
        expected_sql = test_item.get('expected_sql', test_item.get('sql', ''))
        
        self.logger.info(f"评估查询 {query_id}: {question}")
        
        result = {
            'id': query_id,
            'question': question,
            'expected_sql': expected_sql,
            'generated_sql': None,
            'execution_accuracy': {},
            'syntax_accuracy': {},
            'semantic_similarity': {},
            'retrieval_recall': {},
            'error_message': None,
            'evaluation_time': None
        }
        
        start_time = datetime.now()
        
        try:
            # 1. 生成SQL
            generated_sql, retrieval_context = self.text2sql_system.generate_sql(
                question, return_context=True
            )
            result['generated_sql'] = generated_sql
            
            if not generated_sql:
                result['error_message'] = "SQL生成失败"
                return result
            
            # 2. 计算执行准确率
            if self.config['metrics']['execution_accuracy']['enabled']:
                result['execution_accuracy'] = self.metrics_calculator.calculate_execution_accuracy(
                    generated_sql, expected_sql
                )
            
            # 3. 计算语法正确率
            if self.config['metrics']['syntax_accuracy']['enabled']:
                result['syntax_accuracy'] = self.metrics_calculator.calculate_syntax_accuracy(
                    generated_sql
                )
            
            # 4. 计算语义相似度
            if self.config['metrics']['semantic_similarity']['enabled']:
                result['semantic_similarity'] = self.metrics_calculator.calculate_semantic_similarity(
                    generated_sql, expected_sql, question
                )
            
            # 5. 计算检索召回率
            if self.config['metrics']['retrieval_recall']['enabled']:
                retrieved_info = self.text2sql_system.get_retrieval_info(question)
                expected_info = self._build_expected_info_from_item(test_item)
                result['retrieval_recall'] = self.metrics_calculator.calculate_retrieval_recall(
                    retrieved_info, expected_info
                )
            
        except Exception as e:
            self.logger.error(f"查询 {query_id} 评估失败: {e}")
            result['error_message'] = str(e)
        
        finally:
            result['evaluation_time'] = (datetime.now() - start_time).total_seconds()
        
        return result

    def _build_expected_info_from_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """从数据项构造期望的表/列信息，若不存在则从SQL粗略提取"""
        expected_tables = set(item.get('expected_tables', []))
        expected_columns = set(item.get('expected_columns', []))
        key_columns = set(item.get('key_columns', []))
        sql_text = item.get('expected_sql') or item.get('sql') or ''
        try:
            import re
            if not expected_tables and sql_text:
                tables = re.findall(r'\bFROM\s+([\w\.]+)|\bJOIN\s+([\w\.]+)', sql_text, flags=re.IGNORECASE)
                for a, b in tables:
                    tbl = a or b
                    if tbl:
                        expected_tables.add(tbl.split('.')[-1])
            if sql_text:
                # 提取SELECT列表
                m = re.search(r'\bSELECT\s+(.*?)\bFROM\b', sql_text, flags=re.IGNORECASE | re.DOTALL)
                if m:
                    select_part = m.group(1)
                    # 拆分列，去除函数/别名
                    for raw in [seg.strip() for seg in select_part.split(',') if seg.strip()]:
                        # 去除 AS 别名
                        raw = re.split(r'\bAS\b', raw, flags=re.IGNORECASE)[0].strip()
                        # 去除函数括号，只取最内层标识符
                        ident = re.findall(r'([\w]+\.[\w]+|[\w]+)', raw)
                        if not ident:
                            continue
                        token = ident[-1]
                        if '.' in token:
                            t, c = token.split('.', 1)
                            expected_columns.add(f"{t}.{c}")
                            if c.lower().endswith('id'):
                                key_columns.add(f"{t}.{c}")
                        else:
                            # 无前缀列，使用主表名（FROM的第一个表）
                            main_table = next(iter(expected_tables)) if expected_tables else ''
                            if main_table:
                                expected_columns.add(f"{main_table}.{token}")
                                if token.lower().endswith('id'):
                                    key_columns.add(f"{main_table}.{token}")
        except Exception:
            pass
        return {
            'expected_tables': list(expected_tables),
            'expected_columns': list(expected_columns),
            'key_columns': list(key_columns)
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """
        运行完整评估
        
        Returns:
            dict: 评估结果
        """
        self.logger.info("开始Text2SQL系统评估")
        
        # 加载测试数据
        test_dataset = self.load_test_dataset()
        
        # Debug mode check
        debug_mode = self.config.get('evaluation', {}).get('debug_mode', False)
        if debug_mode:
            original_count = len(test_dataset)
            test_dataset = test_dataset[:1]
            self.logger.info(f"DEBUG: Processing 1/{original_count} samples")
            self.logger.info(f"Sample: {test_dataset[0].get('question', 'N/A')}")
        else:
            self.logger.info(f"Processing {len(test_dataset)} samples")
        
        # Process queries
        detailed_results = []
        for i, test_item in enumerate(test_dataset, 1):
            self.logger.info(f"[{i:2d}/{len(test_dataset):2d}] {test_item.get('question', 'N/A')[:50]}")
            
            result = self.evaluate_single_query(test_item)
            detailed_results.append(result)
            self.metrics_aggregator.add_result(result)
            
            # Checkpoint
            if i % 5 == 0:
                self._save_intermediate_results(detailed_results, i)
                self.logger.info(f"Checkpoint: {i} queries")
        
        # 计算总体指标
        overall_metrics = self.metrics_aggregator.calculate_overall_metrics()
        
        # 生成最终报告
        final_report = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'config_file': str(self.config_path),
                'total_queries': len(test_dataset),
                'evaluation_duration': sum(r.get('evaluation_time', 0) for r in detailed_results)
            },
            'overall_metrics': overall_metrics,
            'detailed_results': detailed_results
        }
        
        # 保存结果
        self._save_results(final_report)
        
        # 生成报告
        if self.config['reporting']['generate_html']:
            self._generate_html_report(final_report)
        
        self.logger.info("Evaluation complete")
        return final_report
    
    def _save_intermediate_results(self, results: List[Dict], count: int):
        """保存中间结果"""
        try:
            # 转换numpy类型为Python原生类型
            results_clean = convert_numpy_types(results)
            
            intermediate_file = self.output_dir / f"intermediate_results_{count}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(results_clean, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"中间结果保存失败: {e}")
    
    def _save_results(self, report: Dict[str, Any]):
        """保存评估结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 转换numpy类型为Python原生类型
        report_clean = convert_numpy_types(report)
        
        # 保存JSON格式
        if self.config['reporting']['generate_json']:
            json_file = self.output_dir / f"evaluation_results_{timestamp}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(report_clean, f, ensure_ascii=False, indent=2)
            self.logger.info(f"结果已保存到: {json_file}")
        
        # 保存简化的指标摘要
        summary_file = self.output_dir / f"metrics_summary_{timestamp}.json"
        summary = {
            'timestamp': report_clean['evaluation_info']['timestamp'],
            'total_queries': report_clean['evaluation_info']['total_queries'],
            'overall_metrics': report_clean['overall_metrics']
        }
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    
    def _generate_html_report(self, report: Dict[str, Any]):
        """生成HTML报告"""
        try:
            html_content = self._create_html_content(report)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = self.output_dir / f"evaluation_report_{timestamp}.html"
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML报告已生成: {html_file}")
            
        except Exception as e:
            self.logger.error(f"HTML报告生成失败: {e}")
    
    def _create_html_content(self, report: Dict[str, Any]) -> str:
        """Generate terminal-style HTML report"""
        overall = report['overall_metrics']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Text2SQL Evaluation</title>
            <style>
                body {{ font-family: 'Courier New', monospace; margin: 0; padding: 20px; background: #0d1117; color: #c9d1d9; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ border: 1px solid #30363d; padding: 20px; margin-bottom: 20px; background: #161b22; }}
                .header h1 {{ margin: 0; color: #58a6ff; font-size: 24px; }}
                .header .info {{ margin-top: 10px; color: #8b949e; font-size: 14px; }}
                .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric {{ border: 1px solid #30363d; padding: 15px; background: #161b22; }}
                .metric .label {{ color: #f0f6fc; font-weight: bold; font-size: 14px; }}
                .metric .value {{ color: #58a6ff; font-size: 28px; margin: 10px 0; font-weight: bold; }}
                .metric .detail {{ color: #8b949e; font-size: 12px; margin: 3px 0; }}
                .section {{ border: 1px solid #30363d; padding: 20px; margin: 20px 0; background: #161b22; }}
                .section h2 {{ margin: 0 0 15px 0; color: #f0f6fc; border-bottom: 1px solid #30363d; padding-bottom: 10px; font-size: 18px; }}
                .table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 13px; }}
                .table th {{ background: #21262d; padding: 12px; text-align: left; border: 1px solid #30363d; color: #f0f6fc; font-weight: bold; }}
                .table td {{ padding: 12px; border: 1px solid #30363d; }}
                .table tr:nth-child(even) {{ background: #0d1117; }}
                .success {{ color: #3fb950; font-weight: bold; }}
                .error {{ color: #f85149; font-weight: bold; }}
                .bar {{ background: #21262d; height: 6px; margin: 8px 0; border: 1px solid #30363d; }}
                .bar-fill {{ background: #58a6ff; height: 100%; }}
                .footer {{ text-align: center; margin-top: 30px; color: #8b949e; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Text2SQL Evaluation Report</h1>
                    <div class="info">
                        {report['evaluation_info']['timestamp']} | 
                        {report['evaluation_info']['total_queries']} queries | 
                        {report['evaluation_info']['evaluation_duration']:.2f}s
                    </div>
                </div>
            
                <div class="metrics">
                    {"" if "mock" in report['evaluation_info']['config_file'] else f'''
                    <div class="metric">
                        <div class="label">EXECUTION ACCURACY</div>
                        <div class="value">{overall['execution_accuracy']['mean']:.1%}</div>
                        <div class="detail">Success: {overall['execution_accuracy']['success_rate']:.1%}</div>
                        <div class="bar">
                            <div class="bar-fill" style="width: {overall['execution_accuracy']['mean']*100:.1f}%"></div>
                        </div>
                    </div>
                    '''}
                    <div class="metric">
                        <div class="label">SYNTAX ACCURACY</div>
                        <div class="value">{overall['syntax_accuracy']['mean']:.1%}</div>
                        <div class="detail">Valid: {overall['syntax_accuracy']['valid_rate']:.1%}</div>
                        <div class="bar">
                            <div class="bar-fill" style="width: {overall['syntax_accuracy']['mean']*100:.1f}%"></div>
                        </div>
                    </div>
                    
                    <div class="metric">
                        <div class="label">SEMANTIC SIMILARITY</div>
                        <div class="value">{overall['semantic_similarity']['mean']:.1%}</div>
                        <div class="bar">
                            <div class="bar-fill" style="width: {overall['semantic_similarity']['mean']*100:.1f}%"></div>
                        </div>
                    </div>
                    
                    <div class="metric">
                        <div class="label">RETRIEVAL RECALL</div>
                        <div class="value">{overall['retrieval_recall']['overall_recall_mean']:.1%}</div>
                        <div class="detail">Table: {overall['retrieval_recall']['table_recall_mean']:.1%}</div>
                        <div class="detail">Column: {overall['retrieval_recall']['column_recall_mean']:.1%}</div>
                        <div class="detail">Key: {overall['retrieval_recall']['key_column_recall_mean']:.1%}</div>
                        <div class="detail">Weighted: {overall['retrieval_recall']['weighted_recall_mean']:.1%}</div>
                        <div class="bar">
                            <div class="bar-fill" style="width: {overall['retrieval_recall']['overall_recall_mean']*100:.1f}%"></div>
                        </div>
                    </div>
                    
                    <div class="metric">
                        <div class="label">OVERALL SCORE</div>
                        <div class="value">{overall['weighted_overall_score']:.1%}</div>
                        <div class="bar">
                            <div class="bar-fill" style="width: {overall['weighted_overall_score']*100:.1f}%"></div>
                        </div>
                    </div>
                </div>
            
                <div class="section">
                    <h2>ANALYSIS</h2>
                    <div class="metrics">
                        <div class="metric">
                            <div class="label">TABLE ANALYSIS</div>
                            <div class="detail">Expected: {overall['retrieval_recall']['table_analysis']['total_expected']}</div>
                            <div class="detail">Retrieved: {overall['retrieval_recall']['table_analysis']['total_retrieved']}</div>
                            <div class="detail">Matched: {overall['retrieval_recall']['table_analysis']['total_matched']}</div>
                            <div class="detail">Precision: {overall['retrieval_recall']['table_analysis']['overall_precision']:.1%}</div>
                            <div class="detail">Recall: {overall['retrieval_recall']['table_analysis']['overall_recall']:.1%}</div>
                        </div>
                        
                        <div class="metric">
                            <div class="label">COLUMN ANALYSIS</div>
                            <div class="detail">Expected: {overall['retrieval_recall']['column_analysis']['total_expected']}</div>
                            <div class="detail">Retrieved: {overall['retrieval_recall']['column_analysis']['total_retrieved']}</div>
                            <div class="detail">Matched: {overall['retrieval_recall']['column_analysis']['total_matched']}</div>
                            <div class="detail">Precision: {overall['retrieval_recall']['column_analysis']['overall_precision']:.1%}</div>
                            <div class="detail">Recall: {overall['retrieval_recall']['column_analysis']['overall_recall']:.1%}</div>
                        </div>
                        
                        <div class="metric">
                            <div class="label">KEY COLUMN ANALYSIS</div>
                            <div class="detail">Expected: {overall['retrieval_recall']['key_column_analysis']['total_expected']}</div>
                            <div class="detail">Matched: {overall['retrieval_recall']['key_column_analysis']['total_matched']}</div>
                            <div class="detail">Recall: {overall['retrieval_recall']['key_column_analysis']['overall_recall']:.1%}</div>
                            <div class="detail">Coverage: {overall['retrieval_recall']['key_column_analysis']['average_coverage']:.1%}</div>
                        </div>
                        
                        <div class="metric">
                            <div class="label">SQL TYPES</div>
                            {"".join([f"<div class='detail'>{sql_type}: {count}</div>" for sql_type, count in overall['retrieval_recall']['sql_type_analysis']['type_distribution'].items()])}
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>QUERY RESULTS</h2>
                    <table class="table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Query</th>
                                {"" if "mock" in report['evaluation_info']['config_file'] else "<th>Exec</th>"}
                                <th>Syntax</th>
                                <th>Semantic</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
        """
        
        for result in report['detailed_results']:
            exec_acc = result.get('execution_accuracy', {}).get('execution_accuracy', 0)
            syntax_acc = result.get('syntax_accuracy', {}).get('syntax_accuracy', 0)
            semantic_sim = result.get('semantic_similarity', {}).get('semantic_similarity', 0)
            
            status = "OK" if not result.get('error_message') else "FAIL"
            status_class = "success" if status == "OK" else "error"
            
            exec_acc_cell = "" if "mock" in report['evaluation_info']['config_file'] else f"<td>{exec_acc:.0%}</td>"
            html += f"""
                            <tr>
                                <td>{result.get('id', 'N/A')}</td>
                                <td>{result.get('question', 'N/A')[:40]}...</td>
                                {exec_acc_cell}
                                <td>{syntax_acc:.0%}</td>
                                <td>{semantic_sim:.0%}</td>
                                <td class="{status_class}">{status}</td>
                            </tr>
            """
        
        html += """
                        </tbody>
                    </table>
                </div>
                
                <div class="footer">
                    <p>Text2SQL Evaluation System v1.0 | Terminal Interface</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Text2SQL系统Salila评估')
    parser.add_argument(
        '--config', 
        type=str, 
        default='../configs/salila_config.yaml',
        help='配置文件路径'
    )
    
    args = parser.parse_args()
    
    try:
        # 创建评估器
        evaluator = SalilaEvaluator(args.config)
        
        # 运行评估
        results = evaluator.run_evaluation()
        
        # Print summary
        overall = results['overall_metrics']
        print("\n" + "="*50)
        print("EVALUATION COMPLETE")
        print("="*50)
        print(f"Queries: {overall['total_queries']}")
        
        # Check if execution accuracy is enabled
        if results.get('evaluation_info', {}).get('config_file', '').find('mock') == -1:
            print(f"Execution: {overall['execution_accuracy']['mean']:.1%}")
        
        print(f"Syntax:    {overall['syntax_accuracy']['mean']:.1%}")
        print(f"Semantic:  {overall['semantic_similarity']['mean']:.1%}")
        print(f"Retrieval: {overall['retrieval_recall']['overall_recall_mean']:.1%}")
        print(f"  Table:   {overall['retrieval_recall']['table_recall_mean']:.1%}")
        print(f"  Column:  {overall['retrieval_recall']['column_recall_mean']:.1%}")
        print(f"  Key:     {overall['retrieval_recall']['key_column_recall_mean']:.1%}")
        print(f"  Weight:  {overall['retrieval_recall']['weighted_recall_mean']:.1%}")
        print(f"Overall:   {overall['weighted_overall_score']:.1%}")
        print("="*50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 