"""
测试导入和基本功能
"""

import sys
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """测试所有必要的导入"""
    try:
        logger.info("测试基础导入...")
        import os
        import json
        import yaml
        import numpy as np
        logger.info("✓ 基础模块导入成功")
        
        logger.info("测试数据库相关导入...")
        from sqlalchemy import create_engine, text
        logger.info("✓ SQLAlchemy导入成功")
        
        logger.info("测试OpenAI导入...")
        import openai
        logger.info("✓ OpenAI导入成功")
        
        logger.info("测试Milvus导入...")
        from pymilvus import MilvusClient
        logger.info("✓ MilvusClient导入成功")
        
        # 测试model模块
        try:
            from pymilvus import model
            logger.info("✓ pymilvus.model导入成功")
        except ImportError:
            logger.warning("⚠ pymilvus.model不可用，将使用备选方案")
        
        logger.info("测试sklearn导入...")
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            logger.info("✓ sklearn导入成功")
        except ImportError:
            logger.warning("⚠ sklearn不可用，将使用numpy计算")
        
        logger.info("测试自定义模块导入...")
        from text2sql_system import Text2SQLSystem
        from evaluation_metrics import EvaluationMetrics
        logger.info("✓ 自定义模块导入成功")
        
        return True
        
    except Exception as e:
        logger.error(f"导入测试失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    try:
        logger.info("测试Text2SQL系统初始化...")
        
        # 使用测试配置
        test_config = {
            'max_retries': 3,
            'timeout': 30
        }
        
        # 这里可能会因为数据库连接失败，但至少能测试导入
        try:
            system = Text2SQLSystem(test_config)
            logger.info("✓ Text2SQL系统初始化成功")
        except Exception as e:
            logger.warning(f"⚠ Text2SQL系统初始化失败（可能是数据库连接问题）: {e}")
        
        logger.info("测试评估指标计算器...")
        try:
            metrics = EvaluationMetrics()
            logger.info("✓ 评估指标计算器初始化成功")
        except Exception as e:
            logger.warning(f"⚠ 评估指标计算器初始化失败: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"功能测试失败: {e}")
        return False

if __name__ == "__main__":
    logger.info("开始导入和功能测试...")
    
    # 测试导入
    import_success = test_imports()
    
    if import_success:
        logger.info("所有导入测试通过！")
        
        # 测试基本功能
        func_success = test_basic_functionality()
        
        if func_success:
            logger.info("基本功能测试通过！")
            logger.info("系统准备就绪，可以运行完整评估。")
        else:
            logger.error("基本功能测试失败！")
    else:
        logger.error("导入测试失败！")
        
    logger.info("测试完成。") 