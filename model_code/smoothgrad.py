import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad, Variable
import numpy as np

class SmoothGrad(object):
    def __init__(self, model, device='cpu', train=False, 
                 x_stddev=0.015, t_stddev=0.015, nsamples=20, magnitude=2):
        self.model = model
        self.device = device
        self.train = train
        self.x_stddev = x_stddev
        self.t_stddev = t_stddev
        self.nsamples = nsamples
        self.magnitude = magnitude
        self.features = model

    def get_gradients(self, z, RBP_emb, pred_label=None):
        self.model.eval()
        self.model.zero_grad()
        z = z.to(self.device)
        z.requires_grad = True
        output = self.model(z, RBP_emb)
        output = torch.sigmoid(output)
        output.backward()
        return z.grad

    def get_smooth_gradients(self, z, RBP_emb, y=None):
        return self.__call__(z, RBP_emb, y)

    def __call__(self, z, RBP_emb, y=None):
        """[summary]
        
        Args:
            z ([type]): [description]
            y ([type]): [description]
            x_stddev (float, optional): [description]. Defaults to 0.15.
            t_stddev (float, optional): [description]. Defaults to 0.15.
            nsamples (int, optional):   [description]. Defaults to 20.
            magnitude (int, optional):  magnitude:0,1,2; 0: original gradient, 1: absolute value of the gradient,
                                        2: square value of the gradient. Defaults to 2.
        
        Returns:
            [type]: [description]
        """

        # z 形状: [batch_size, channels, sequence_length]
        # print(f"Shape of z: {z.shape}")
        
        # 选择前4个通道作为序列部分（x），后2个通道作为结构部分（t）#系统检查这些中文翻译为英文
        x = z[:, :4, :]  # 前4个通道
        t = z[:, 4:, :]  # 后2个通道
        
        # 计算标准差
        x_stddev = (self.x_stddev * (x.max() - x.min())).to(self.device).item()
        t_stddev = (self.t_stddev * (t.max() - t.min())).to(self.device).item()

        total_grad = torch.zeros(z.shape).to(self.device)
        x_noise = torch.zeros(x.shape).to(self.device)
        t_noise = torch.zeros(t.shape).to(self.device)

        # 在多次采样中计算梯度
        for i in range(self.nsamples):
            # 给序列和结构部分分别添加噪声
            x_plus_noise = x + x_noise.zero_().normal_(0, x_stddev)
            t_plus_noise = t + t_noise.zero_().normal_(0, t_stddev)
            z_plus_noise = torch.cat((x_plus_noise, t_plus_noise), dim=1)  # 合并序列和结构部分

            grad = self.get_gradients(z_plus_noise, RBP_emb, y)
            if self.magnitude == 1:
                total_grad += torch.abs(grad)
            elif self.magnitude == 2:
                total_grad += grad * grad

        total_grad /= self.nsamples
        return total_grad

    def get_batch_gradients(self, X, RBP_emb, Y=None):
        if Y is not None:
            assert len(X) == len(Y), "The size of input {} and target {} are not matched.".format(len(X), len(Y))
        g = torch.zeros_like(X)
        for i in range(X.shape[0]):
            x = X[i:i+1]
            if Y is not None:
                y = Y[i:i+1]
            else:
                y = None
            g[i:i+1] = self.get_smooth_gradients(x, RBP_emb, y)
        return g


def generate_saliency(model, x, y=None, smooth=False, nsamples=2, stddev=0.15, \
                      train=False):
    saliency = SmoothGrad(model, train=train)
    x_grad = saliency.get_smooth_gradients(x, y, nsamples=nsamples, x_stddev=stddev, t_stddev=stddev)
    return x_grad


class GuidedBackpropReLU(torch.autograd.Function):

    def __init__(self, inplace=False):
        super(GuidedBackpropReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        pos_mask = (input > 0).type_as(input)
        output = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            input,
            pos_mask)
        self.save_for_backward(input, output)
        return output

    def backward(self, grad_output):
        input, output = self.saved_tensors

        pos_mask_1 = (input > 0).type_as(grad_output)
        pos_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(
            torch.zeros(input.size()).type_as(input),
            torch.addcmul(
                torch.zeros(input.size()).type_as(input), grad_output, pos_mask_1),
            pos_mask_2)

        return grad_input

    def __repr__(self):
        inplace_str = ', inplace' if self.inplace else ''
        return self.__class__.__name__ + ' (' + inplace_str + ')'


class GuidedBackpropSmoothGrad(SmoothGrad):

    def __init__(self, model, device='cpu', train=False, 
                 x_stddev=0.15, t_stddev=0.15, nsamples=20, magnitude=2):
        super(GuidedBackpropSmoothGrad, self).__init__(
            model, device, train, x_stddev, t_stddev, nsamples, magnitude)
        for idx, module in self.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.features._modules[idx] = GuidedBackpropReLU()
