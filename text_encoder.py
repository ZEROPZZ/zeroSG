import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.num_heads = 8
        self.head_dim = hidden_size // self.num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.q_norm = nn.LayerNorm(hidden_size)
        self.k_norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        q = self.q_norm(self.q_proj(x))
        k = self.k_norm(self.k_proj(x))
        v = self.v_proj(x)
        
        B, T, C = x.size()
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        attn_output = (attn_probs @ v).transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, T, C)
        
        return self.out_proj(attn_output)

class MLPLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4)
    
    def forward(self, x):
        gate = F.sigmoid(self.gate_proj(x))
        up = F.silu(self.up_proj(x))
        
        hidden = gate * up
        output = self.down_proj(hidden)
        
        return output

class TransformerLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = SelfAttentionLayer(hidden_size)
        self.mlp = MLPLayer(hidden_size)
        
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)
        self.post_feedforward_layernorm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        attn_output = self.self_attn(x)
        x = self.post_attention_layernorm(x + attn_output)
        
        ff_output = self.mlp(x)
        x = self.post_feedforward_layernorm(x + ff_output)
        
        return x

class TextEncoder(nn.Module):
    def __init__(self, hidden_size=5120, num_layers=29):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_size) 
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# 可选：添加模型初始化和测试函数
def test_model():
    model = TextEncoder()
    test_input = torch.randn(1, 10, 5120)
    output = model(test_input)
    print(f"输出形状: {output.shape}")

if __name__ == "__main__":
    test_model() 