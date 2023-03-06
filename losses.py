from fastai.vision import *


class MultiLosses(nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
        self.ce = SoftCrossEntropyLoss() if one_hot else torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()

    @property
    def last_losses(self):
        return self.losses

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res

        def merge(items):
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)
            else:
                return items[0]

        res = dict()
        for key in all_res[0].keys():
            items = [r[key] for r in all_res]
            res[key] = merge(items)
        return res

    def _ce_loss(self, output, gt_labels, gt_lengths, gt_embedding, idx=None, record=True):
        loss_name = output.get('name')
        pt_logits, weight = output['logits'], output['loss_weight']

        assert pt_logits.shape[0] % gt_labels.shape[0] == 0
        iter_size = pt_logits.shape[0] // gt_labels.shape[0]
        if iter_size > 1:
            gt_labels = gt_labels.repeat(3, 1, 1)
            gt_lengths = gt_lengths.repeat(3)
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)

        nll = output.get('nll')
        if nll is not None:
            loss = self.ce(flat_pt_logits, flat_gt_labels, softmax=False) * weight
        else:
            loss = self.ce(flat_pt_logits, flat_gt_labels) * weight
        if record and loss_name is not None: self.losses[f'{loss_name}_loss'] = loss

        return loss

    def forward(self, outputs, *args):
        self.losses = {}
        if isinstance(outputs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outputs]
            return sum([self._ce_loss(o, *args) for o in outputs if o['loss_weight'] > 0.])
        else:
            return self._ce_loss(outputs, *args, record=False)


class MultiLossesWithEmbedding(nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
        self.ce = SoftCrossEntropyLoss() if one_hot else torch.nn.CrossEntropyLoss()
        self.bce = torch.nn.BCELoss()
        self.embedding_ce = EmbeddingRegressionLoss(loss_func="cosin")

    @property
    def last_losses(self):
        return self.losses

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res

        def merge(items):
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)
            else:
                return items[0]

        res = dict()
        for key in all_res[0].keys():
            items = [r[key] for r in all_res]
            res[key] = merge(items)
        return res

    def _ce_loss(self, output, gt_labels, gt_lengths, gt_embedding, idx=None, record=True):
        loss_name = output.get('name')
        pt_logits, weight = output['logits'], output['loss_weight']

        assert pt_logits.shape[0] % gt_labels.shape[0] == 0
        iter_size = pt_logits.shape[0] // gt_labels.shape[0]
        if iter_size > 1:
            gt_labels = gt_labels.repeat(3, 1, 1)
            gt_lengths = gt_lengths.repeat(3)
        flat_gt_labels = self._flatten(gt_labels, gt_lengths)
        flat_pt_logits = self._flatten(pt_logits, gt_lengths)

        nll = output.get('nll')
        if nll is not None:
            loss = self.ce(flat_pt_logits, flat_gt_labels, softmax=False) * weight
        else:
            loss = self.ce(flat_pt_logits, flat_gt_labels) * weight
        if record and loss_name is not None: self.losses[f'{loss_name}_loss'] = loss

        # calculate embedding loss
        if loss_name == "vision":
            embedding_weight = output['embedding_loss_weight']
            loss_embedding = self.embedding_ce(output['embedding_vector'], gt_embedding) * embedding_weight
            loss += loss_embedding

        return loss

    def forward(self, outputs, *args):
        self.losses = {}
        if isinstance(outputs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outputs]
            return sum([self._ce_loss(o, *args) for o in outputs if o['loss_weight'] > 0.])
        else:
            return self._ce_loss(outputs, *args, record=False)


class MultiLossesWithMGP(nn.Module):
    def __init__(self, one_hot=True):
        super().__init__()
        if one_hot:
            raise Exception("The value of one_hot must be False!")
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=0)
        # self.ce = SoftCrossEntropyLoss() if one_hot else torch.nn.CrossEntropyLoss()
        # self.bce = torch.nn.BCELoss()
        # self.embedding_ce = EmbeddingRegressionLoss(loss_func="cosin")

    @property
    def last_losses(self):
        return self.losses

    def _flatten(self, sources, lengths):
        return torch.cat([t[:l] for t, l in zip(sources, lengths)])

    def _merge_list(self, all_res):
        if not isinstance(all_res, (list, tuple)):
            return all_res

        def merge(items):
            if isinstance(items[0], torch.Tensor):
                return torch.cat(items, dim=0)
            else:
                return items[0]

        res = dict()
        for key in all_res[0].keys():
            items = [r[key] for r in all_res]
            res[key] = merge(items)
        return res

    # def _ce_loss(self, output, gt_labels, gt_lengths, gt_embedding, idx=None, record=True):
    def _ce_loss(self, output, gt_char_labels, gt_bpe_labels, gt_wp_labels,
                 gt_lengths, idx=None, record=True):
        loss_name = output.get('name')
        pt_logits, weight = output['logits'], output['loss_weight']
        char_logits, bpe_logits, wp_logits = pt_logits

        assert char_logits.shape[0] % gt_char_labels.shape[0] == 0
        iter_size = char_logits.shape[0] // gt_char_labels.shape[0]
        if iter_size > 1:
            gt_char_labels = gt_char_labels.repeat(3, 1, 1)
            gt_lengths = gt_lengths.repeat(3)

        flat_gt_char_labels = self._flatten(gt_char_labels, gt_lengths)
        flat_gt_bpe_labels = self._flatten(gt_bpe_labels, gt_lengths)
        flat_gt_wp_labels = self._flatten(gt_wp_labels, gt_lengths)
        flat_char_logits = self._flatten(char_logits, gt_lengths)
        flat_bpe_logits = self._flatten(bpe_logits, gt_lengths)
        flat_wp_logits = self._flatten(wp_logits, gt_lengths)

        # 参考 MGP-STR 的 loss 计算
        nll = output.get('nll')
        if nll is not None:
            char_loss = self.ce(flat_char_logits, flat_gt_char_labels, softmax=False) * weight
            bpe_loss = self.ce(flat_bpe_logits, flat_gt_bpe_labels, softmax=False) * weight
            wp_loss = self.ce(flat_wp_logits, flat_gt_wp_labels, softmax=False) * weight
        else:
            char_loss = self.ce(flat_char_logits, flat_gt_char_labels) * weight
            bpe_loss = self.ce(flat_bpe_logits, flat_gt_bpe_labels) * weight
            wp_loss = self.ce(flat_wp_logits, flat_gt_wp_labels) * weight
        loss = char_loss + bpe_loss + wp_loss
        if record and loss_name is not None:
            self.losses[f'{loss_name}_loss'] = loss
            self.losses[f'{loss_name}_char_loss'] = char_loss
            self.losses[f'{loss_name}_bpe_loss'] = bpe_loss
            self.losses[f'{loss_name}_wp_loss'] = wp_loss

        return loss

    def forward(self, outputs, *args):
        self.losses = {}
        if isinstance(outputs, (tuple, list)):
            outputs = [self._merge_list(o) for o in outputs]
            return sum([self._ce_loss(o, *args) for o in outputs if o['loss_weight'] > 0.])
        else:
            return self._ce_loss(outputs, *args, record=False)


class SoftCrossEntropyLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, softmax=True):
        if softmax:
            log_prob = F.log_softmax(input, dim=-1)
        else:
            log_prob = torch.log(input)
        loss = -(target * log_prob).sum(dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


def _assert_no_grad(variable):
  assert not variable.requires_grad, \
    "nn criterions don't compute the gradient w.r.t. targets - please " \
    "mark these variables as not requiring gradients"

class EmbeddingRegressionLoss(nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 ignore_index=-100,
                 sequence_normalize=False,
                 sample_normalize=True,
                 loss_func='cosin'):
        super(EmbeddingRegressionLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize
        # self.loss_func = torch.nn.MSELoss()
        self.is_cosin_loss = False
        if loss_func == 'smooth_l1':
            self.loss_func = torch.nn.SmoothL1Loss()
        elif loss_func == 'cosin':
            self.loss_func = torch.nn.CosineEmbeddingLoss()
            self.is_cosin_loss = True

    def forward(self, input, target):
        _assert_no_grad(target)
        if not self.is_cosin_loss:
            Loss = self.loss_func(input, target)
        else:
            label_target = torch.ones(input.size(0)).cuda()
            Loss = self.loss_func(input, target, label_target)
        return Loss

    def logistic_dot_loss(self, input, target):
        dot_result = torch.mm(input, target.t())
        _diagaonal = dot_result.diagonal()
        logistic_loss = torch.log(1 + torch.exp(-1 * _diagaonal))

        # logistic_loss = torch.mean(logistic_loss, dim=0)

        return logistic_loss
        # _trace = torch.trace(dot_result)
        # loss = _trace / input.size(0)
        #
        # logistic_loss = nn.sigmoid(loss)
