import torch
import torchvision
import math
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable

class VGG19(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, device='cuda'):
        super(VGGLoss, self).__init__()
        # self.vgg = VGG19().to(device)
        index = 31
        vgg_model = torchvision.models.vgg16(pretrained=True).to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        self.vgg = nn.Sequential(*list(vgg_model.features.children())[:index])
        self.instancenorm = nn.InstanceNorm2d(512, affine=False)
        # self.criterion = nn.MSELoss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(self.vgg_preprocess(x)), self.vgg(self.vgg_preprocess(y))
        # loss = 0
        # for i in range(len(x_vgg)):
        #     loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        loss = torch.mean((self.instancenorm(x_vgg) - self.instancenorm(y_vgg)) ** 2)
        return loss
    
    def vgg_preprocess(self, batch): 
        tensor_type = type(batch.data) 
        (r, g, b) = torch.chunk(batch, 3, dim=1) 
        batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR 
        batch = batch * 255  # * 0.5  [-1, 1] -> [0, 255] 
        mean = tensor_type(batch.data.size()).cuda() 
        mean[:, 0, :, :] = 103.939 
        mean[:, 1, :, :] = 116.779 
        mean[:, 2, :, :] = 123.680 
        batch = batch.sub(Variable(mean))  # subtract mean 
        return batch 
    
def range_compressor(hdr_img, mu=5000):
    return (torch.log(1 + mu * hdr_img)) / math.log(1 + mu)

class HDRloss(nn.Module):
    # https://github.com/megvii-research/HDR-Transformer/blob/main/loss/losses.py
    def __init__(self, mu=5000):
        super(HDRloss, self).__init__()
        self.mu = mu
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, label):
        mu_pred = range_compressor(pred, self.mu)
        mu_label = range_compressor(label, self.mu)
        return self.l1_loss(mu_pred, mu_label)
    
class HistogramMatchingLoss(nn.Module):
    # Aamir Mustafa, Hongjie You, Rafal K. Mantiuk, "A Comparative Study on the Loss Functions for Image Enhancement Networks"  
    # in London Imaging Meeting,  2022,  pp 11 - 15,  https://doi.org/10.2352/lim.2022.1.1.04
    def __init__(self, num_bins=256, lambda_weight=1.0):
        super(HistogramMatchingLoss, self).__init__()
        self.num_bins = num_bins
        self.lambda_weight = lambda_weight

    def forward(self, img1, img2):
        assert img1.shape == img2.shape, "Input tensors must have the same shape"
        batch_size, channels, height, width = img1.shape
        loss = 0.0
        
        for c in range(channels):  # Iterate over channels
            # Compute histograms
            hist1 = torch.histc(img1[:, c, :, :], bins=self.num_bins, min=0.0, max=1.0)
            hist2 = torch.histc(img2[:, c, :, :], bins=self.num_bins, min=0.0, max=1.0)

            # Normalize histograms
            hist1 /= hist1.sum()
            hist2 /= hist2.sum()

            # Compute CDFs
            cdf1 = torch.cumsum(hist1, dim=0)
            cdf2 = torch.cumsum(hist2, dim=0)

            # Calculate the loss as the sum of absolute differences between the CDFs
            channel_loss = torch.sum(torch.abs(cdf1 - cdf2))
            loss += channel_loss
        
        return loss/3.
    