"""
We build our architecture on top of the ANs proposed in

@InProceedings{huang2018and,
  title={Unsupervised Deep Learning by Neighbourhood Discovery},
  author={Jiabo Huang, Qi Dong, Shaogang Gong and Xiatian Zhu},
  booktitle={Proceedings of the International Conference on machine learning (ICML)},
  year={2019},
}

The code is available online under https://github.com/Raymond-sci/AND

"""


from torch.autograd import Function

import os
import torchvision.models as models
import math
import torch
import torch.nn.functional as F
import torch.nn as nn


def resnet18(pretrained=False):
    model = models.resnet18(pretrained=pretrained)
    model.fc = Identity()
    model.avgpool = Identity()
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Backbone(nn.Module):

    def __init__(self, name='vgg16', pretrained=True, freeze_all=False):
        super(Backbone, self).__init__()
        self.name = name
        self.freeze_all = freeze_all
        self.pretrained = pretrained
        self.backbone = resnet18(pretrained=self.pretrained)

        if self.freeze_all:
            # List all layers (even inside sequential module)
            layers = [module for module in self.backbone.modules() if type(module) != nn.Sequential]
            for layer in layers:
                if hasattr(layer, 'requires_grad_'):
                    layer.requires_grad_(False)

    def forward(self, x):
        return self.backbone(x)


class SimpleDecoder(nn.Module):

    def __init__(self, hidden_dimension=512):
        super(SimpleDecoder, self).__init__()

        self.conv_up_5 = nn.Conv2d(hidden_dimension, hidden_dimension//2, 3, padding=1)
        self.conv_up_4 = nn.Conv2d(hidden_dimension//2, hidden_dimension//4, 3, padding=1)
        self.conv_up_3 = nn.Conv2d(hidden_dimension//4, hidden_dimension//8, 3, padding=1)
        self.conv_up_2 = nn.Conv2d(hidden_dimension//8, hidden_dimension//16, 3, padding=1)
        self.conv_up_1 = nn.Conv2d(hidden_dimension//16, hidden_dimension//32, 5, padding=2)
        self.decoder = nn.Conv2d(hidden_dimension//32, 3, 5, padding=2)

    def forward(self, z):

        h = nn.ReLU()(self.conv_up_5(z))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        h = nn.ReLU()(self.conv_up_4(h))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        h = nn.ReLU()(self.conv_up_3(h))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        h = nn.ReLU()(self.conv_up_2(h))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        h = nn.ReLU()(self.conv_up_1(h))
        h = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=False)(h)
        x_hat = nn.Sigmoid()(self.decoder(h))

        return x_hat


class CAE_DNR(nn.Module):

    def __init__(self, pretrained=True, n_channels=3, hidden_dimension=512):
        super(CAE_DNR, self).__init__()

        self.n_channels = n_channels
        self.encoder = Backbone(name='resnet18', pretrained=pretrained, freeze_all=False)

        if self.n_channels != self.encoder.backbone.conv1.in_channels:
            conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            data = self.encoder.backbone.conv1.weight.data[:, :2, :, :]  # Better than nothing ... ?
            self.encoder.backbone.conv1 = conv1
            self.encoder.backbone.conv1.weight.data = data

        self.decoder = SimpleDecoder(hidden_dimension=512)

        self.hidden_dimension = hidden_dimension

    def restore_model(self, paths):
        for attr, path in paths.items():
            self._load(attr=attr, path=path)
        return self

    def _load(self, attr, path):
        if not os.path.exists(path):
            print('Unknown path: {}'.format(path))
        if not hasattr(self, attr):
            print('No attribute: {}'.format(attr))

        self.__getattr__(attr).load_state_dict(torch.load(path), strict=True)

        return self

    def forward(self, x, decode=False):

        z = self.encode(x, pool=False)
        zp = nn.AvgPool2d(7, 7)(z).squeeze(dim=3).squeeze(dim=2)

        if decode:
            x_hat = self.decoder(z)
        else:
            x_hat = None

        return x_hat, zp

    def encode(self, x, pool=False):
        h = self.encoder(x)
        h = h.view((-1, 512, 7, 7))
        if pool:
            return nn.AvgPool2d(7, 7)(h).squeeze(dim=3).squeeze(dim=2)
        else:
            return h

    def calculate_objective(self, x_in, x_out, index, npc, ANs_discovery, criterion, round):

        x_hat, z = self.forward(x_in, decode=True)

        z_n = torch.div(z, torch.norm(z, p=2, dim=1, keepdim=True))
        outputs = npc(z_n, index)  # For each image get similarity with neighbour
        loss_inst, loss_ans = criterion(outputs, index, ANs_discovery)
        loss = loss_inst + loss_ans
        l_loss = {'loss': loss, 'loss_inst': loss_inst, 'loss_ans': loss_ans}

        if x_hat is not None:
            loss_mse = nn.MSELoss()(x_hat, x_out)
            loss = loss + loss_mse
            l_loss['loss_mse'] = loss_mse
            l_loss['loss'] = loss

        return l_loss


class NonParametricClassifierOP(Function):

    @staticmethod
    def forward(self, x, y, memory, params):
        T = params[0].item()

        # inner product
        out = torch.mm(x.data, memory.t())
        out.div_(T)  # batchSize * N

        self.save_for_backward(x, memory, y, params)

        return out

    @staticmethod
    def backward(self, gradOutput):
        x, memory, y, params = self.saved_tensors
        T = params[0].item()
        momentum = params[1].item()

        # add temperature
        gradOutput.data.div_(T)

        # gradient of linear
        gradInput = torch.mm(gradOutput.data, memory)
        gradInput.resize_as_(x)

        # update the memory
        weight_pos = memory.index_select(0, y.data.view(-1)).resize_as_(x)
        weight_pos.mul_(momentum)
        weight_pos.add_(torch.mul(x.data, 1 - momentum))
        w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
        updated_weight = weight_pos.div(w_norm)
        memory.index_copy_(0, y, updated_weight)

        return gradInput, None, None, None, None


class NonParametricClassifier(nn.Module):
    """Non-parametric Classifier

    Non-parametric Classifier from
    "Unsupervised Feature Learning via Non-Parametric Instance Discrimination"

    Extends:
        nn.Module
    """

    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.5):
        """Non-parametric Classifier initial functin

        Initial function for non-parametric classifier

        Arguments:
            inputSize {int} -- in-channels dims
            outputSize {int} -- out-channels dims

        Keyword Arguments:
            T {int} -- distribution temperate (default: {0.05})
            momentum {int} -- memory update momentum (default: {0.5})
        """
        super(NonParametricClassifier, self).__init__()
        self.nLem = outputSize
        self.register_buffer('params', torch.tensor([T, momentum]))
        stdv = 1. / math.sqrt(inputSize / 3)
        self.register_buffer('memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv))

    def forward(self, x, y):
        out = NonParametricClassifierOP.apply(x, y, self.memory, self.params)
        return out


class ANsDiscovery(nn.Module):
    """Discovery ANs

    Discovery ANs according to current round, select_rate and most importantly,
    all sample's corresponding entropy
    """

    def __init__(self, nsamples):
        """Object used to discovery ANs

        Discovery ANs according to the total amount of samples, ANs selection
        rate, ANs size

        Arguments:
            nsamples {int} -- total number of sampels
            select_rate {float} -- ANs selection rate
            ans_size {int} -- ANs size

        Keyword Arguments:
            device {str} -- [description] (default: {'cpu'})
        """
        super(ANsDiscovery, self).__init__()

        # not going to use ``register_buffer'' as
        # they are determined by configs
        self.select_rate = 0.25
        self.ANs_size = 1
        # number of samples
        self.register_buffer('samples_num', torch.tensor(nsamples))
        # indexes list of anchor samples
        self.register_buffer('anchor_indexes', torch.LongTensor(nsamples//2))
        # indexes list of instance samples
        self.register_buffer('instance_indexes', torch.arange(nsamples//2).long())
        # anchor samples' and instance samples' position
        self.register_buffer('position', -1 * torch.arange(nsamples).long() - 1)
        # anchor samples' neighbours
        self.register_buffer('neighbours', torch.LongTensor(nsamples//2, 1))
        # each sample's entropy
        self.register_buffer('entropy', torch.FloatTensor(nsamples))
        # consistency
        self.register_buffer('consistency', torch.tensor(0.))


    def get_ANs_num(self, round):
        """Get number of ANs

        Get number of ANs at target round according to the select rate

        Arguments:
            round {int} -- target round

        Returns:
            int -- number of ANs
        """
        return int(self.samples_num.float() * self.select_rate * round)

    def update(self, round, npc, cheat_labels=None):
        """Update ANs

        Discovery new ANs and update `anchor_indexes`, `instance_indexes` and
        `neighbours`

        Arguments:
            round {int} -- target round
            npc {Module} -- non-parametric classifier
            cheat_labels {list} -- used to compute consistency of chosen ANs only

        Returns:
            number -- [updated consistency]
        """
        with torch.no_grad():
            batch_size = 100
            ANs_num = self.get_ANs_num(round)
            features = npc.memory

            for start in range(0, self.samples_num, batch_size):
                end = start + batch_size
                end = min(end, self.samples_num)

                preds = F.softmax(npc(features[start:end], None), 1)
                self.entropy[start:end] = -(preds * preds.log()).sum(1)

            # get the anchor list and instance list according to the computed
            # entropy
            self.anchor_indexes = self.entropy.topk(ANs_num, largest=False)[1]
            self.instance_indexes = (torch.ones_like(self.position)
                                     .scatter_(0, self.anchor_indexes, 0)
                                     .nonzero().view(-1))
            anchor_entropy = self.entropy.index_select(0, self.anchor_indexes)
            instance_entropy = self.entropy.index_select(0, self.instance_indexes)

            # get position
            # if the anchor sample x whose index is i while position is j, then
            # sample x_i is the j-th anchor sample at current round
            # if the instance sample x whose index is i while position is j, then
            # sample x_i is the (-j-1)-th instance sample at current round

            instance_cnt = 0
            for i in range(self.samples_num):

                # for anchor samples
                if (i == self.anchor_indexes).any():
                    self.position[i] = (self.anchor_indexes == i).max(0)[1]
                    continue
                # for instance samples
                instance_cnt -= 1
                self.position[i] = instance_cnt

            anchor_features = features.index_select(0, self.anchor_indexes)
            self.neighbours = (torch.LongTensor(ANs_num, self.ANs_size)
                               .to('cuda'))
            for start in range(0, ANs_num, batch_size):

                end = start + batch_size
                end = min(end, ANs_num)

                sims = torch.mm(anchor_features[start:end], features.t())
                sims.scatter_(1, self.anchor_indexes[start:end].view(-1, 1), -1.)
                _, self.neighbours[start:end] = (
                    sims.topk(self.ANs_size, largest=True, dim=1))

            # if cheat labels is provided, then compute consistency
            if cheat_labels is None:
                return 0.
            anchor_label = cheat_labels.index_select(0, self.anchor_indexes)
            neighbour_label = cheat_labels.index_select(0,
                                                        self.neighbours.view(-1)).view_as(self.neighbours)
            self.consistency = ((anchor_label.view(-1, 1) == neighbour_label)
                                .float().mean())

            return self.consistency


class Criterion(nn.Module):

    def __init__(self):
        super(Criterion, self).__init__()

    def forward(self, x, y, ANs):
        batch_size, _ = x.shape

        # split anchor and instance list
        anchor_indexes, instance_indexes = self._split(y[:batch_size//2], ANs)
        preds = F.softmax(x, 1)

        l_ans = torch.tensor(0).cuda()
        if anchor_indexes.size(0) > 0:
            # compute loss for anchor samples
            y_ans = y.index_select(0, anchor_indexes)
            y_ans_p = y.index_select(0, anchor_indexes + batch_size//2)
            y_ans_neighbour = ANs.position.index_select(0, y_ans)
            neighbours = ANs.neighbours.index_select(0, y_ans_neighbour)
            # p_i = \sum_{j \in \Omega_i} p_{i,j}
            x_ans = preds.index_select(0, anchor_indexes)
            x_ans_p = preds.index_select(0, anchor_indexes + batch_size//2)

            x_ans_neighbour = x_ans.gather(1, neighbours).sum(1)
            x_ans_p = x_ans_p.gather(1, y_ans_p.view(-1, 1)).view(-1)
            x_ans = x_ans.gather(1, y_ans.view(-1, 1)).view(-1)
            # sum all terms : self + sim + neighbors
            # NLL: l = -log(p_i)
            l_ans = -1 * torch.log(x_ans + x_ans_p + x_ans_neighbour).sum(0)

        l_inst = torch.tensor(0).cuda()
        if instance_indexes.size(0) > 0:
            # compute loss for instance samples
            y_inst = y.index_select(0, instance_indexes)
            y_inst_p = y.index_select(0, instance_indexes + batch_size//2)
            x_inst = preds.index_select(0, instance_indexes)
            x_inst_p = preds.index_select(0, instance_indexes + batch_size//2)
            # p_i = p_{i, i}
            x_inst = x_inst.gather(1, y_inst.view(-1, 1))
            x_inst_p = x_inst_p.gather(1, y_inst_p.view(-1, 1))
            # NLL: l = -log(p_i)
            l_inst = -1 * torch.log(x_inst + x_inst_p).sum(0)

        return l_inst / batch_size, l_ans / batch_size

    def _split(self, y, ANs):
        pos = ANs.position.index_select(0, y.view(-1))
        return (pos >= 0).nonzero().view(-1), (pos < 0).nonzero().view(-1)
