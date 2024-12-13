import torch
from text_encoder import TextEncoder, TransformerLayer

def main():
    print("开始测试模型...")
    
    # 创建模型实例
    try:
        model = TextEncoder()
        print("模型实例创建成功")
    except Exception as e:
        print(f"创建模型时发生错误: {e}")
        return
    
    print(f"模型层数: {model.num_layers}")
    print(f"隐藏层大小: {model.hidden_size}")
    
    # 测试前向传播
    try:
        test_input = torch.randn(1, 10, 5120)
        output = model(test_input)
        print(f"输出形状: {output.shape}")
    except Exception as e:
        print(f"前向传播时发生错误: {e}")
    
    layer = TransformerLayer(hidden_size=5120)
    test_input = torch.randn(1, 10, 5120)
    output = layer(test_input)
    print(f"单层输出形状: {output.shape}")

if __name__ == "__main__":
    main()
