import copy

import numpy as np
import torch.nn as nn
import torch
import math

from torch.nn import LayerNorm
from torch.nn.modules.utils import _pair
import configs


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.num_attention_heads = config.transformer['num_attention_heads']
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.out = nn.Linear(config.hidden_size, self.all_head_size)
        self.attention_dropout = nn.Dropout(config.transformer['attention_dropout_rate'])
        self.proj_dropout = nn.Dropout(config.transformer['attention_dropout_rate'])

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads,
                                       self.attention_head_size)  #(batch_size,seq_len,num_attention_head,attenrion_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  #(batch_size,num_attention_head,seq_len,attention_head_size)

    def forward(self, hidden_states):
        mixed_query = self.query(hidden_states)
        mixed_key = self.key(hidden_states)
        mixed_value = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query)
        key_layer = self.transpose_for_scores(mixed_key)
        value_layer = self.transpose_for_scores(mixed_value)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output


ActiveFunc = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer['mlp_dim'])
        self.fc2 = nn.Linear(config.transformer['mlp_dim'], config.hidden_size)
        self.activate_func = ActiveFunc["gelu"]
        self.dropout = nn.Dropout(config.transformer['dropout_rate'])

    def forward(self, x):
        x = self.fc1(x)
        x = self.activate_func(x)
        x = self.dropout
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels):
        super(Embeddings, self).__init__()
        self.config = config
        img_size = _pair(img_size)
        patch_size = _pair(config.patches['size'])
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embedding = nn.Conv2d(in_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer['dropout_rate'])

    def forward(self, x):
        x = self.patch_embedding(x)  # (B, hidden n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)
        embedding = x + self.position_embedding
        embedding = self.dropout(embedding)
        return embedding


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = MLP(config)
        self.attention = Attention(config)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attention(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer['num_layers']):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embedding = Embeddings(config, img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_out, features = self.embedding(input_ids)
        encoded = self.encoder(embedding_out)
        return encoded, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         bias=not (use_batchnorm))

        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, relu, bn)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, kernel_size=3, padding=1,
                                use_batchnorm=use_batchnorm)
        self.conv2 = Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, config.head_channels, kernel_size=3, padding=1,
                                    use_batchnorm=True)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        skip_channels = [0, 0, 0, 0]

        blocks = [DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in
                  zip(in_channels, out_channels, skip_channels)]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            skip = None
            x = decoder_block(x, skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(in_channels=config['decoder_channels'][-1],
                                                  out_channels=config['n_classes'], kernel_size=3)
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, attn_weights, features = self.transformer(x)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    CONFIGS = {configs.get_config()}
