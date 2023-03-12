import logging

from fastai.vision import *

from modules.attention import *
from modules.backbone import ResTransformer
from modules.embedding_head import Embedding
from modules.model import Model
from modules.resnet import resnet45
from modules.token_learner import TokenLearner
from utils import TokenLabelConverter


class BaseVision(Model):
    def __init__(self, config):
        super().__init__(config)
        self.loss_weight = ifnone(config.model_vision_loss_weight, 1.0)
        self.out_channels = ifnone(config.model_vision_d_model, 512)

        if config.model_vision_backbone == 'transformer':
            self.backbone = ResTransformer(config)
        else:
            self.backbone = resnet45()

        self.char_tokenLearner = TokenLearner(self.out_channels, config.dataset_max_length + 2)

        self.bpe_tokenLearner = TokenLearner(self.out_channels, config.dataset_max_length + 2)
        self.wp_tokenLearner = TokenLearner(self.out_channels, config.dataset_max_length + 2)

        self.converter = TokenLabelConverter(max_length=config.dataset_max_length)
        self.char_cls = nn.Linear(self.out_channels, len(self.converter.character))  # 36 + start token + end token
        self.bpe_cls = nn.Linear(self.out_channels, 50257)
        self.wp_cls = nn.Linear(self.out_channels, 30522)

        if config.model_vision_checkpoint is not None:
            logging.info(f'Read vision model from {config.model_vision_checkpoint}.')
            self.load(config.model_vision_checkpoint)

    def forward(self, images, *args):
        features = self.backbone(images)  # (N, E, H, W)  # [n, 512, 8, 32]

        # TODO: 预测 char, subword, word
        # 使用三个 TokenLearner 或者 PositionAttention:
        # 1. 使用三个 PositionAttention: attention 后面接 Linear 直接预测
        # 2. 使用三个 TokenLearner: 参考 MGP-STR

        b, c = features.shape[:2]
        features = features.reshape((b, c, -1)).transpose(1,2)
        char_attn_vecs, char_attn_scores = self.char_tokenLearner(features)  # [n, 27, 512], [n, 27, 8, 32]
        char_logits = self.char_cls(char_attn_vecs)  # [n, 27, 37]
        char_pt_lengths = self._get_length(char_logits)  # TODO: 可以去掉

        bpe_attn_vecs, bpe_attn_scores = self.bpe_tokenLearner(features)
        bpe_logits = self.bpe_cls(bpe_attn_vecs)  # [n, 27, 50257]
        bpe_pt_lengths = self._get_length(bpe_logits)

        wp_attn_vecs, wp_attn_scores = self.wp_tokenLearner(features)
        wp_logits = self.wp_cls(wp_attn_vecs)  # [n, 27, 30522]
        wp_pt_lengths = self._get_length(wp_logits)

        attn_vecs = [char_attn_vecs, bpe_attn_vecs, wp_attn_vecs]
        attn_scores = [char_attn_scores, bpe_attn_scores, wp_attn_scores]
        logits = [char_logits, bpe_logits, wp_logits]
        pt_lengths = [char_pt_lengths, bpe_pt_lengths, wp_pt_lengths]

        return {'feature': attn_vecs, 'logits': logits, 'pt_lengths': pt_lengths,
                'attn_scores': attn_scores, 'loss_weight': self.loss_weight, 'name': 'vision'}
