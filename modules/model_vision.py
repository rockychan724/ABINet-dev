import logging

from fastai.vision import *
import numpy as np

from modules.attention import *
from modules.backbone import ResTransformer
from modules.embedding_head import Embedding
from modules.model import Model
from modules.resnet import resnet45


class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.embedding_loss_weight = ifnone(config.model_vision_embedding_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTransformer(config)
        else:
            self.backbone = resnet45()
        
        self.embedding = Embedding(8*32, 512)

        if config.model_vision_attention == 'position':
            mode = ifnone(config.model_vision_attention_mode, 'nearest')
            self.attention = PositionAttention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                mode=mode,
            )
        elif config.model_vision_attention == 'attention':
            self.attention = Attention(
                max_length=config.dataset_max_length + 1,  # additional stop token
                n_feature=8 * 32,
            )
        else:
            raise Exception(f'{config.model_vision_attention} is not valid.')
        self.cls = nn.Linear(self.out_channels, self.charset.num_classes)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint)

    def forward(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)  # [n, 512, 8, 32]
        # TODO: backbone 后面接一个 embedding，添加一个 loss

        embedding_vector = self.embedding(features)
        # embedding_vector = np.zeros((features.shape[0], 300))

        # TODO: 添加
        attn_vecs, attn_scores = self.attention(features)  # (N, T, E), (N, T, H, W)  # [n, 26, 512], [n, 26, 8, 32]
        logits = self.cls(attn_vecs)  # (N, T, C)  # [n, 26, 37]
        pt_lengths = self._get_length(logits)

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision',
                'embedding_vector': embedding_vector, 'embedding_loss_weight': self.embedding_loss_weight}
