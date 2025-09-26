#!/usr/bin/env python3
"""
测试修复后的评估系统
"""

import sys
import json
import numpy as np
from pathlib import Path

# 添加脚本目录到路径
sys.path.append(str(Path(__file__).parent / "scripts"))

def test_json_serialization():
    """测试JSON序列化修复"""
    print("🧪 测试JSON序列化修复...")
    
    # 模拟包含numpy类型的数据
    test_data = {
        'int32_value': np.int32(42),
        'float64_value': np.float64(3.14),
        'array_value': np.array([1, 2, 3]),
        'nested_dict': {
            'numpy_int': np.int32(100),
            'normal_string': 'test'
        },
        'normal_value': 'hello'
    }
    
    try:
        # 导入修复函数
        from run_evaluation import convert_numpy_types
        
        # 转换数据
        cleaned_data = convert_numpy_types(test_data)
        
        # 尝试JSON序列化
        json_str = json.dumps(cleaned_data, ensure_ascii=False, indent=2)
        
        print("✅ JSON序列化测试通过")
        print(f"   原始类型: {type(test_data['int32_value'])}")
        print(f"   转换后类型: {type(cleaned_data['int32_value'])}")
        return True
        
    except Exception as e:
        print(f"❌ JSON序列化测试失败: {e}")
        return False

def test_debug_config():
    """测试调试配置"""
    print("\n🧪 测试调试配置...")
    
    try:
        import yaml
        
        # 测试模拟模式配置
        config_path = Path(__file__).parent / "configs" / "salila_config_debug.yaml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        mock_mode = config.get('evaluation', {}).get('mock_mode', False)
        debug_mode = config.get('evaluation', {}).get('debug_mode', False)
        
        print(f"✅ 调试配置加载成功")
        print(f"   模拟模式: {mock_mode}")
        print(f"   调试模式: {debug_mode}")
        return True
        
    except Exception as e:
        print(f"❌ 调试配置测试失败: {e}")
        return False

def test_evaluator_init():
    """测试评估器初始化"""
    print("\n🧪 测试评估器初始化...")
    
    try:
        from run_evaluation import SalilaEvaluator
        
        # 使用模拟模式配置
        config_path = Path(__file__).parent / "configs" / "salila_config_debug.yaml"
        evaluator = SalilaEvaluator(str(config_path))
        
        print("✅ 评估器初始化成功")
        return True
        
    except Exception as e:
        print(f"❌ 评估器初始化失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试修复...")
    
    tests = [
        test_json_serialization,
        test_debug_config,
        test_evaluator_init
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！现在可以运行调试模式了。")
        print("\n运行命令:")
        print("cd Evaluation/scripts")
        print("python run_evaluation.py --config ../configs/salila_config_debug.yaml")
    else:
        print("❌ 部分测试失败，请检查错误信息。")

if __name__ == "__main__":
    main()
