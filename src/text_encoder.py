import torch
import torch.nn as nn

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        print("初始化 SelfAttentionLayer...")
        self.num_heads = 8
        self.head_dim = hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.q_norm = nn.LayerNorm(hidden_size)
        self.k_norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        print("开始 TextEncoder 前向传播...")
        for i, layer in enumerate(self.layers):
            print(f"正在处理第 {i + 1} 层...")
            x = layer(x)
            print(f"第 {i + 1} 层输出形状: {x.shape}")  # 输出每层的形状
        return x

class MLPLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        print("初始化 MLPLayer...")
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4)

    def forward(self, x):
        print("开始 MLPLayer 前向传播...")
        return x  # 示例返回

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        print("初始化 TransformerLayer...")
        self.self_attn = SelfAttentionLayer(hidden_size)
        self.mlp = MLPLayer(hidden_size)
        
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.post_feedforward_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        print("开始 TransformerLayer 前向传播...")
        attn_output = self.self_attn(x)
        x = self.post_attention_layernorm(x + attn_output)
        
        ff_output = self.mlp(x)
        x = self.post_feedforward_layernorm(x + ff_output)
        
        return x

class TextEncoder(nn.Module):
    def __init__(self, hidden_size=5120, num_layers=29):
        super().__init__()
        self.layers = nn.ModuleList([TransformerLayer(hidden_size) for _ in range(num_layers)])

    @classmethod
    def from_pretrained(cls, pretrained_path):
        model = cls()
        print("开始加载权重...")
        
        try:
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)  # 设置 weights_only=True
            print("权重加载完成，开始映射...")
        except Exception as e:
            print(f"加载权重时发生错误: {e}")
            return model

        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('model.layers.', 'layers.')
            new_state_dict[new_key] = value
        
        print("映射完成，开始加载状态字典...")
        try:
            model.load_state_dict(new_state_dict, strict=False)
            print(f"成功从 {pretrained_path} 加载预训练权重")
        except Exception as e:
            print(f"加载状态字典时发生错误: {e}")
        
        return model

    def forward(self, x):
        print("开始 TextEncoder 前向传播...")
        for i, layer in enumerate(self.layers):
            print(f"正在处理第 {i + 1} 层...")
            x = layer(x)  # 确保每一层都能处理输入
            print(f"第 {i + 1} 层输出形状: {x.shape}")  # 输出每层的形状
        return x
