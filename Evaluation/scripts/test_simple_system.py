#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版Text2SQL系统测试脚本
"""

import sys
import os
import json
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from text2sql_system_simple import SimpleText2SQLSystem

def test_simple_system():
    """测试简化版Text2SQL系统"""
    print("🚀 开始测试简化版Text2SQL系统...")
    
    # 配置
    config = {
        'embedding_model': 'text-embedding-3-large',
        'llm_model': 'gpt-4o-mini',  # 使用更快的模型
        'max_retries': 2,
        'timeout': 15  # 减少超时时间
    }
    
    try:
        # 初始化系统
        print("📝 初始化系统...")
        system = SimpleText2SQLSystem(config)
        
        # 测试问题
        test_questions = [
            "显示所有演员的姓名",
            "有多少部电影？",
            "找出租金最高的电影"
        ]
        
        print(f"🔍 开始测试 {len(test_questions)} 个问题...")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n--- 测试 {i}/{len(test_questions)} ---")
            print(f"问题: {question}")
            
            try:
                # 生成SQL
                sql, context = system.generate_sql(question, return_context=True)
                
                if sql:
                    print(f"✅ 生成SQL: {sql}")
                    print(f"📚 检索到的表: {context.get('retrieved_tables', [])}")
                else:
                    print("❌ SQL生成失败")
                    
            except Exception as e:
                print(f"❌ 错误: {e}")
        
        print("\n🎉 测试完成！")
        return True
        
    except Exception as e:
        print(f"❌ 系统初始化失败: {e}")
        return False

if __name__ == "__main__":
    success = test_simple_system()
    sys.exit(0 if success else 1) 