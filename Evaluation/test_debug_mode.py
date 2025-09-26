#!/usr/bin/env python3
"""
快速测试调试模式是否正常工作
"""

import sys
import json
from pathlib import Path

# 添加脚本目录到路径
sys.path.append(str(Path(__file__).parent / "scripts"))

def test_debug_mode():
    """测试调试模式配置"""
    print("🧪 测试调试模式配置...")
    
    # 1. 测试配置文件加载
    try:
        import yaml
        config_path = Path(__file__).parent / "configs" / "salila_config_debug.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        debug_mode = config.get('evaluation', {}).get('debug_mode', False)
        print(f"✅ 调试模式配置: {debug_mode}")
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        return False
    
    # 2. 测试数据集加载
    try:
        dataset_path = Path(__file__).parent / "datasets" / "sakila_test.json"
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"✅ 数据集加载成功，共 {len(dataset)} 条数据")
        print(f"   第一条数据: {dataset[0].get('question', 'N/A')}")
        
        # 测试调试模式切片
        debug_dataset = dataset[:1]
        print(f"✅ 调试模式数据集: {len(debug_dataset)} 条数据")
        
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        return False
    
    # 3. 测试评估器初始化
    try:
        from run_evaluation import SalilaEvaluator
        
        evaluator = SalilaEvaluator(str(config_path))
        print("✅ 评估器初始化成功")
        
    except Exception as e:
        print(f"❌ 评估器初始化失败: {e}")
        return False
    
    print("\n🎉 调试模式测试通过！")
    print("现在可以运行: python run_evaluation.py --config ../configs/salila_config_debug.yaml")
    return True

if __name__ == "__main__":
    test_debug_mode()
