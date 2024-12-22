import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelConfig:
    def __init__(self):
        # 基础配置
        self.model_name = "MultiModalFramework"
        self.seed = 42
        
        # 模型架构
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.num_hidden_layers = 12
        self.intermediate_size = 3072
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        
        # 图像相关
        self.image_size = 224
        self.image_channels = 3
        self.patch_size = 16
        
        # 文本相关
        self.max_position_embeddings = 512
        self.vocab_size = 30522
        self.type_vocab_size = 2
        
        # 语音相关
        self.audio_sample_rate = 16000
        self.audio_frame_size = 1024
        self.audio_hop_size = 512
        
        # 任务相关
        self.enable_vqa = True
        self.enable_caption = True
        self.enable_retrieval = True
        self.enable_asr = True  # 语音识别
        self.enable_realtime_asr = True  # 实时语音识别
        
        # 训练相关
        self.batch_size = 32
        self.learning_rate = 1e-4
        self.weight_decay = 0.01
        self.warmup_steps = 10000
        self.max_steps = 100000

class ImageEncoder(nn.Module):
    def __init__(self, config):
        super(ImageEncoder, self).__init__()
        self.config = config
        
        # 图像编码层
        self.encoder_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 111 * 111, 768)  # 修改为 64 * 111 * 111
        )
        
    def forward(self, image):
        print("Image shape:", image.shape)
        # 图像编码
        image_features = self.encoder_layer(image)
        print("Image features shape:", image_features.shape)
        
        return image_features

class TextEncoder(nn.Module):
    def __init__(self, config):
        super(TextEncoder, self).__init__()
        self.config = config
        
        # 使用 Transformer 层，设置 batch_first=True
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size, 
            nhead=config.num_attention_heads,
            batch_first=True  # 设置 batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, 
            num_layers=config.num_hidden_layers
        )
        
    def forward(self, text):
        print("Text shape:", text.shape)
        # 文本编码
        text_features = self.transformer_encoder(text)
        # 将文本特征从三维变为二维
        text_features = text_features.mean(dim=1)
        print("Text features shape:", text_features.shape)
        
        return text_features

class AudioEncoder(nn.Module):
    def __init__(self, config):
        super(AudioEncoder, self).__init__()
        self.config = config
        
        # 语音编码层
        self.encoder_layer = nn.Sequential(
            nn.Linear(config.audio_sample_rate, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
    def forward(self, audio):
        print("Audio shape:", audio.shape)
        # 语音编码
        audio_features = self.encoder_layer(audio)
        print("Audio features shape:", audio_features.shape)
        
        return audio_features

class FusionLayer(nn.Module):
    def __init__(self, config):
        super(FusionLayer, self).__init__()
        self.config = config
        
        # 多模态融合层
        self.fusion_layer = nn.Linear(config.hidden_size * 3, config.hidden_size)
        
    def forward(self, image_features, text_features, audio_features):
        print("Image features shape:", image_features.shape)
        print("Text features shape:", text_features.shape)
        print("Audio features shape:", audio_features.shape)
        # 多模态融合
        fused_features = torch.cat((image_features, text_features, audio_features), dim=1)
        fused_features = self.fusion_layer(fused_features)
        print("Fused features shape:", fused_features.shape)
        
        return fused_features

class VQALayer(nn.Module):
    def __init__(self, config):
        super(VQALayer, self).__init__()
        self.config = config
        
        # VQA层
        self.vqa_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, fused_features):
        print("Fused features shape:", fused_features.shape)
        # VQA输出
        vqa_output = self.vqa_layer(fused_features)
        print("VQA output shape:", vqa_output.shape)
        
        return vqa_output

class CaptionLayer(nn.Module):
    def __init__(self, config):
        super(CaptionLayer, self).__init__()
        self.config = config
        
        # Caption层
        self.caption_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, fused_features):
        print("Fused features shape:", fused_features.shape)
        # Caption输出
        caption_output = self.caption_layer(fused_features)
        print("Caption output shape:", caption_output.shape)
        
        return caption_output

class RetrievalLayer(nn.Module):
    def __init__(self, config):
        super(RetrievalLayer, self).__init__()
        self.config = config
        
        # Retrieval层
        self.retrieval_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, fused_features):
        print("Fused features shape:", fused_features.shape)
        # Retrieval输出
        retrieval_output = self.retrieval_layer(fused_features)
        print("Retrieval output shape:", retrieval_output.shape)
        
        return retrieval_output

class ASRLayer(nn.Module):
    def __init__(self, config):
        super(ASRLayer, self).__init__()
        self.config = config
        
        # 语音识别层
        self.asr_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, fused_features):
        print("Fused features shape:", fused_features.shape)
        # 语音识别输出
        asr_output = self.asr_layer(fused_features)
        print("ASR output shape:", asr_output.shape)
        
        return asr_output

class RealtimeASRLayer(nn.Module):
    def __init__(self, config):
        super(RealtimeASRLayer, self).__init__()
        self.config = config
        
        # 实时语音识别层
        self.realtime_asr_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, fused_features):
        print("Fused features shape:", fused_features.shape)
        # 实时语音识别输出
        realtime_asr_output = self.realtime_asr_layer(fused_features)
        print("Realtime ASR output shape:", realtime_asr_output.shape)
        
        return realtime_asr_output

class MultiModalFramework(nn.Module):
    def __init__(self, config):
        super(MultiModalFramework, self).__init__()
        self.config = config
        
        # 图像编码器
        self.image_encoder = ImageEncoder(config)
        
        # 文本编码器
        self.text_encoder = TextEncoder(config)
        
        # 语音编码器
        self.audio_encoder = AudioEncoder(config)
        
        # 多模态融合层
        self.fusion_layer = FusionLayer(config)
        
        # 任务相关层
        self.vqa_layer = VQALayer(config)
        self.caption_layer = CaptionLayer(config)
        self.retrieval_layer = RetrievalLayer(config)
        self.asr_layer = ASRLayer(config)
        self.realtime_asr_layer = RealtimeASRLayer(config)
        
    def forward(self, image, text, audio):
        # 图像编码
        image_features = self.image_encoder(image)
        
        # 文本编码
        text_features = self.text_encoder(text)
        
        # 语音编码
        audio_features = self.audio_encoder(audio)
        
        # 多模态融合
        fused_features = self.fusion_layer(image_features, text_features, audio_features)
        
        # 任务相关输出
        vqa_output = self.vqa_layer(fused_features)
        caption_output = self.caption_layer(fused_features)
        retrieval_output = self.retrieval_layer(fused_features)
        asr_output = self.asr_layer(fused_features)
        realtime_asr_output = self.realtime_asr_layer(fused_features)
        
        return vqa_output, caption_output, retrieval_output, asr_output, realtime_asr_output

    def load_model_parameters(self, model_state_dict):
        # 加载模型参数
        self.load_state_dict(model_state_dict, strict=False)

# 初始化模型
config = ModelConfig()
model = MultiModalFramework(config)

# 输入数据
image = torch.randn(1, 3, 224, 224)
text = torch.randn(1, 512, 768)  # max_position_embeddings
audio = torch.randn(1, 16000)  # audio_sample_rate

# 前向传播
vqa_output, caption_output, retrieval_output, asr_output, realtime_asr_output = model(image, text, audio)

# 打印输出
print("VQA output shape:", vqa_output.shape)
print("Caption output shape:", caption_output.shape)
print("Retrieval output shape:", retrieval_output.shape)
print("ASR output shape:", asr_output.shape)
print("Realtime ASR output shape:", realtime_asr_output.shape)

# 加载模型参数
model_state_dict = {
    'layers.10.mlp.up_proj.bias': torch.Size([20480]),
    'layers.10.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.10.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.10.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.10.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.11.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.11.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.11.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.11.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.11.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.11.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.11.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.11.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.11.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.11.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.11.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.11.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.11.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.11.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.11.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.11.mlp.down_proj.bias': torch.Size([5120]),
    'layers.11.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.8.mlp.up_proj.bias': torch.Size([20480]),
    'layers.8.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.8.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.8.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.8.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.9.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.9.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.9.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.9.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.9.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.9.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.9.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.9.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.9.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.9.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.9.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.9.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.9.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.9.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.9.mlp.down_proj.bias': torch.Size([5120]),
    'layers.9.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.3.mlp.up_proj.bias': torch.Size([20480]),
    'layers.3.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.3.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.3.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.3.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.4.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.4.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.4.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.4.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.4.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.4.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.4.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.4.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.4.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.4.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.4.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.4.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.4.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.4.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.4.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.4.mlp.down_proj.bias': torch.Size([5120]),
    'layers.4.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.6.mlp.up_proj.bias': torch.Size([20480]),
    'layers.6.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.6.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.6.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.6.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.7.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.7.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.7.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.7.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.7.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.7.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.7.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.7.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.7.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.7.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.7.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.7.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.7.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.7.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.7.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.7.mlp.down_proj.bias': torch.Size([5120]),
    'layers.7.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.11.mlp.up_proj.bias': torch.Size([20480]),
    'layers.11.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.11.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.11.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.11.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.12.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.12.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.12.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.12.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.12.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.12.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.12.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.12.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.12.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.12.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.12.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.12.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.12.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.12.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.12.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.12.mlp.down_proj.bias': torch.Size([5120]),
    'layers.12.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.9.mlp.up_proj.bias': torch.Size([20480]),
    'layers.9.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.9.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.9.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.9.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.10.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.10.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.10.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.10.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.10.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.10.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.10.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.10.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.10.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.10.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.10.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.10.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.10.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.10.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.10.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.10.mlp.down_proj.bias': torch.Size([5120]),
    'layers.10.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.12.mlp.up_proj.bias': torch.Size([20480]),
    'layers.12.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.12.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.12.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.12.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.13.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.13.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.13.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.13.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.13.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.13.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.13.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.13.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.13.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.13.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.13.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.13.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.13.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.13.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.13.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.13.mlp.down_proj.bias': torch.Size([5120]),
    'layers.13.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.14.mlp.up_proj.bias': torch.Size([20480]),
    'layers.14.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.14.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.14.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.14.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.15.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.15.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.15.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.15.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.15.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.15.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.15.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.15.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.15.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.15.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.15.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.15.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.15.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.15.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.15.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.15.mlp.down_proj.bias': torch.Size([5120]),
    'layers.15.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.21.mlp.up_proj.bias': torch.Size([20480]),
    'layers.21.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.21.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.21.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.21.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.22.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.22.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.22.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.22.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.22.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.22.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.22.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.22.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.22.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.22.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.22.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.22.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.22.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.22.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.22.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.22.mlp.down_proj.bias': torch.Size([5120]),
    'layers.22.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.24.mlp.up_proj.bias': torch.Size([20480]),
    'layers.24.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.24.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.24.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.24.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.25.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.25.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.25.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.25.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.25.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.25.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.25.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.25.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.25.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.25.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.25.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.25.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.25.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.25.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.25.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.25.mlp.down_proj.bias': torch.Size([5120]),
    'layers.25.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.17.mlp.up_proj.bias': torch.Size([20480]),
    'layers.17.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.17.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.17.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.17.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.18.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.18.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.18.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.18.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.18.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.18.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.18.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.18.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.18.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.18.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.18.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.18.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.18.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.18.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.18.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.18.mlp.down_proj.bias': torch.Size([5120]),
    'layers.18.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.25.mlp.up_proj.bias': torch.Size([20480]),
    'layers.25.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.25.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.25.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.25.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.26.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.26.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.26.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.26.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.26.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.26.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.26.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.26.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.26.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.26.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.26.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.26.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.26.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.26.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.26.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.26.mlp.down_proj.bias': torch.Size([5120]),
    'layers.26.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.27.mlp.up_proj.bias': torch.Size([20480]),
    'layers.27.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.27.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.27.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.27.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.28.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.28.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.28.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.28.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.28.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.28.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.28.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.28.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.28.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.28.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.28.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.28.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.28.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.28.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.28.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.28.mlp.down_proj.bias': torch.Size([5120]),
    'layers.28.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.4.mlp.up_proj.bias': torch.Size([20480]),
    'layers.4.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.4.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.4.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.4.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.5.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.5.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.5.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.5.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.5.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.5.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.5.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.5.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.5.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.5.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.5.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.5.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.5.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.5.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.5.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.5.mlp.down_proj.bias': torch.Size([5120]),
    'layers.5.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.20.mlp.up_proj.bias': torch.Size([20480]),
    'layers.20.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.20.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.20.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.20.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.21.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.21.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.21.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.21.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.21.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.21.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.21.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.21.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.21.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.21.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.21.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.21.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.21.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.21.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.21.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.21.mlp.down_proj.bias': torch.Size([5120]),
    'layers.21.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.7.mlp.up_proj.bias': torch.Size([20480]),
    'layers.7.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.7.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.7.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.7.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.8.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.8.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.8.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.8.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.8.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.8.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.8.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.8.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.8.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.8.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.8.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.8.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.8.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.8.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.8.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.8.mlp.down_proj.bias': torch.Size([5120]),
    'layers.8.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.15.mlp.up_proj.bias': torch.Size([20480]),
    'layers.15.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.15.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.15.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.15.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.16.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.16.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.16.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.16.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.16.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.16.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.16.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.16.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.16.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.16.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.16.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.16.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.16.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.16.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.16.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.16.mlp.down_proj.bias': torch.Size([5120]),
    'layers.16.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.23.mlp.up_proj.bias': torch.Size([20480]),
    'layers.23.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.23.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.23.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.23.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.24.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.24.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.24.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.24.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.24.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.24.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.24.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.24.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.24.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.24.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.24.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.24.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.24.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.24.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.24.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.24.mlp.down_proj.bias': torch.Size([5120]),
    'layers.24.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.18.mlp.up_proj.bias': torch.Size([20480]),
    'layers.18.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.18.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.18.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.18.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.19.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.19.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.19.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.19.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.19.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.19.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.19.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.19.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.19.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.19.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.19.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.19.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.19.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.19.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.19.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.19.mlp.down_proj.bias': torch.Size([5120]),
    'layers.19.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.26.mlp.up_proj.bias': torch.Size([20480]),
    'layers.26.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.26.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.26.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.26.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.27.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.27.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.27.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.27.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.27.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.27.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.27.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.27.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.27.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.27.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.27.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.27.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.27.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.27.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.27.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.27.mlp.down_proj.bias': torch.Size([5120]),
    'layers.27.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.22.mlp.up_proj.bias': torch.Size([20480]),
    'layers.22.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.22.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.22.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.22.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.23.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.23.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.23.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.23.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.23.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.23.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.23.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.23.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.23.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.23.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.23.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.23.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.23.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.23.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.23.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.23.mlp.down_proj.bias': torch.Size([5120]),
    'layers.23.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.2.mlp.up_proj.bias': torch.Size([20480]),
    'layers.2.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.2.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.2.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.2.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.3.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.3.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.3.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.3.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.3.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.3.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.3.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.3.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.3.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.3.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.3.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.3.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.3.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.3.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.3.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.3.mlp.down_proj.bias': torch.Size([5120]),
    'layers.3.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.28.mlp.up_proj.bias': torch.Size([20480]),
    'layers.28.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.28.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.28.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.28.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.19.mlp.up_proj.bias': torch.Size([20480]),
    'layers.19.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.19.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.19.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.19.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.20.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.20.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.20.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.20.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.20.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.20.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.20.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.20.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.20.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.20.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.20.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.20.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.20.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.20.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.20.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.20.mlp.down_proj.bias': torch.Size([5120]),
    'layers.20.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.5.mlp.up_proj.bias': torch.Size([20480]),
    'layers.5.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.5.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.5.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.5.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.6.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.6.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.6.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.6.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.6.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.6.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.6.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.6.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.6.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.6.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.6.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.6.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.6.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.6.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.6.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.6.mlp.down_proj.bias': torch.Size([5120]),
    'layers.6.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.16.mlp.up_proj.bias': torch.Size([20480]),
    'layers.16.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.16.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.16.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.16.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.17.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.17.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.17.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.17.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.17.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.17.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.17.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.17.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.17.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.17.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.17.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.17.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.17.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.17.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.17.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.17.mlp.down_proj.bias': torch.Size([5120]),
    'layers.17.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.13.mlp.up_proj.bias': torch.Size([20480]),
    'layers.13.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.13.post_attention_layernorm.bias': torch.Size([5120]),
    'layers.13.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.13.post_feedforward_layernorm.bias': torch.Size([5120]),
    'layers.14.self_attn.q_proj.weight': torch.Size([5120, 5120]),
    'layers.14.self_attn.q_proj.bias': torch.Size([5120]),
    'layers.14.self_attn.k_proj.weight': torch.Size([5120, 5120]),
    'layers.14.self_attn.k_proj.bias': torch.Size([5120]),
    'layers.14.self_attn.v_proj.weight': torch.Size([5120, 5120]),
    'layers.14.self_attn.v_proj.bias': torch.Size([5120]),
    'layers.14.self_attn.out_proj.weight': torch.Size([5120, 5120]),
    'layers.14.self_attn.out_proj.bias': torch.Size([5120]),
    'layers.14.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.14.self_attn.q_norm.bias': torch.Size([5120]),
    'layers.14.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.14.self_attn.k_norm.bias': torch.Size([5120]),
    'layers.14.mlp.gate_proj.weight': torch.Size([20480, 5120]),
    'layers.14.mlp.gate_proj.bias': torch.Size([20480]),
    'layers.14.mlp.down_proj.weight': torch.Size([5120, 20480]),
    'layers.14.mlp.down_proj.bias': torch.Size([5120]),
    'layers.14.mlp.up_proj.weight': torch.Size([20480, 5120]),
    'layers.0.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.0.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.0.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.0.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.0.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.0.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.0.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.1.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.1.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.1.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.1.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.1.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.1.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.1.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.2.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.2.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.2.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.2.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.2.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.2.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.2.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.3.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.3.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.3.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.3.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.3.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.3.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.3.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.4.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.4.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.4.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.4.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.4.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.4.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.4.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.5.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.5.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.5.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.5.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.5.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.5.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.5.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.6.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.6.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.6.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.6.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.6.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.6.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.6.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.7.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.7.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.7.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.7.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.7.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.7.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.7.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.8.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.8.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.8.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.8.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.8.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.8.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.8.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.9.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.9.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.9.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.9.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.9.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.9.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.9.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.10.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.10.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.10.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.10.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.10.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.10.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.10.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.11.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.11.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.11.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.11.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.11.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.11.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.11.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.12.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.12.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.12.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.12.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.12.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.12.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.12.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.13.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.13.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.13.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.13.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.13.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.13.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.13.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.14.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.14.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.14.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.14.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.14.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.14.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.14.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.15.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.15.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.15.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.15.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.15.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.15.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.15.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.16.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.16.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.16.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.16.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.16.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.16.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.16.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.17.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.17.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.17.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.17.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.17.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.17.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.17.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.18.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.18.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.18.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.18.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.18.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.18.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.18.self_attn.q_norm.weight': torch.Size([5120]),
    'layers.19.mlp.down_proj.weight': torch.Size([5120, 13824]),
    'layers.19.mlp.gate_proj.weight': torch.Size([13824, 5120]),
    'layers.19.mlp.up_proj.weight': torch.Size([13824, 5120]),
    'layers.19.post_attention_layernorm.weight': torch.Size([5120]),
    'layers.19.post_feedforward_layernorm.weight': torch.Size([5120]),
    'layers.19.self_attn.k_norm.weight': torch.Size([5120]),
    'layers.19.self_attn.q_norm.weight': torch.Size([5120]),
}

print("\n模型参数:")
for name, param in model.named_parameters():
    print(f"参数名称: {name}, 参数形状: {param.shape}")

# 计算并打印总参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数数量: {total_params}")


torch.save(model, 'zero_multimodal_model.pth')
print("整个模型已保存！")
