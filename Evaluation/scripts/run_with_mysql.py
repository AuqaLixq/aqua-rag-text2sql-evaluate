#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查MySQL可用性并运行相应的评估模式
"""

import subprocess
import sys
import os
from pathlib import Path

def check_mysql_connection():
    """检查MySQL连接是否可用"""
    try:
        import pymysql
        connection = pymysql.connect(
            host='localhost',
            user='root',
            password='password',
            database='sakila',
            port=3306
        )
        connection.close()
        return True
    except Exception as e:
        print(f"MySQL连接失败: {e}")
        return False

def main():
    """主函数"""
    print("🔍 检查MySQL数据库连接...")
    
    if check_mysql_connection():
        print("✅ MySQL连接成功！使用完整模式评估")
        config_file = "../configs/salila_config.yaml"
    else:
        print("⚠️  MySQL连接失败，使用模拟模式评估")
        config_file = "../configs/salila_config_mock.yaml"
    
    print(f"📊 运行评估: {config_file}")
    
    # 运行评估
    cmd = [sys.executable, "run_evaluation.py", "--config", config_file]
    subprocess.run(cmd)

if __name__ == "__main__":
    main() 