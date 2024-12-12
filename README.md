# TextEncoder Model

## Overview
这是一个基于Transformer的文本编码器模型。

## Features
- 29层Transformer结构
- 隐藏层大小：5120
- 多头自注意力机制

## Usage
python
from text_encoder import TextEncoder
初始化模型
model = TextEncoder()
准备输入
input_tensor = torch.randn(1, 10, 5120)
前向传播
output = model(input_tensor)


