# from time import time
from fastai.vision import *

from .model_alignment import BaseAlignment
from .model_language import BCNLanguage
from .model_vision_embedding import BaseVision


class ABINetIterModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.iter_size = ifnone(config.model_iter_size, 1)
        self.max_length = config.dataset_max_length + 1  # additional stop token
        self.vision = BaseVision(config)
        self.language = BCNLanguage(config)
        self.alignment = BaseAlignment(config)
        self.total_time = 0.0
        self.total_num = 0

    def forward(self, images, *args):
        # torch.cuda.synchronize()
        # start = time()
        res = self._forward(images)
        # end = time()
        # self.total_time += (end - start)
        # self.total_num += images.size(0)
        return res

    def _forward(self, images):
        v_res = self.vision(images)
        a_res = v_res
        all_l_res, all_a_res = [], []
        for _ in range(self.iter_size):
            # TODO: 弄清这里几个变量的维度，猜测 token 是字符的表示，
            # ABINet 的视觉模型和语言模型之间没有梯度传递，因此输入
            # 语言模型的不是 feature，而是预测的字符。
            tokens = torch.softmax(a_res['logits'], dim=-1)
            lengths = a_res['pt_lengths']
            lengths.clamp_(2, self.max_length)  # TODO:move to language model
            l_res = self.language(tokens, lengths)
            all_l_res.append(l_res)
            a_res = self.alignment(l_res['feature'], v_res['feature'])
            all_a_res.append(a_res)
        if self.training:
            return all_a_res, all_l_res, v_res
        else:
            return a_res, all_l_res[-1], v_res
