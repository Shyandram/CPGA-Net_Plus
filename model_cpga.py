import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from guided_filter_pytorch.guided_filter import FastGuidedFilter
from utils import weight_init

# from kornia.color import rgb_to_y
def _rgb_to_y(r: Tensor, g: Tensor, b: Tensor) -> Tensor:
    y: Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    return y

def rgb_to_y(image: Tensor) -> Tensor:
    r"""Convert an RGB image to Y.

    Args:
        image: RGB Image to be converted to Y with shape :math:`(*, 3, H, W)`.

    Returns:
        Y version of the image with shape :math:`(*, 1, H, W)`.

    Examples:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_y(input)  # 2x1x4x5
    """
    if not isinstance(image, Tensor):
        raise TypeError(f"Input type is not a Tensor. Got {type(image)}")

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {image.shape}")

    r: Tensor = image[..., 0:1, :, :]
    g: Tensor = image[..., 1:2, :, :]
    b: Tensor = image[..., 2:3, :, :]

    y: Tensor = _rgb_to_y(r, g, b)
    return y

class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1,
            stride=[1, 1]
            ):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias, stride=stride[i]))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale
        
    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += F.interpolate(x, size=res.size()[2:], mode='bilinear', align_corners=True)

        return res
    
def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), 
        stride=stride,
        bias=bias)
  
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
    
class ResCBAM(nn.Module):
    def __init__(self, inplanes, planes, strides=[1, 1], act=nn.ReLU(True), downsample=None):
        super(ResCBAM, self).__init__()
        self.conv1 = default_conv(inplanes, planes, kernel_size=3, stride=strides[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = act
        self.conv2 = default_conv(inplanes, planes, kernel_size=3, stride=strides[1])
        self.bn2 = nn.BatchNorm2d(planes)

        if planes < 16:
            ratio = planes
        else:
            ratio = 16
        self.ca = ChannelAttention(planes, ratio=ratio)
        self.sa = SpatialAttention()

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class t_est(nn.Module):
    def __init__(self, n_channels=3):
        super(t_est, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=n_channels*2, out_channels=n_channels, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=n_channels*2, out_channels=n_channels, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=n_channels*4, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def __call__(self, x):
        dbc = self.get_dbc(x)
        x1 = F.relu(self.conv1(dbc))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), 1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), 1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), 1)
        t = F.relu(self.conv5(cat3))
        return t
    
    def get_dbc(self, rgb):
        img_max, _ = torch.max(rgb, 1, keepdim=True)
        img_min, _ = torch.min(rgb, 1, keepdim=True)
        y = rgb_to_y(rgb)
        return torch.cat((img_max, img_min, y), dim=1)

class At_est(nn.Module):
    def __init__(self, n_channels=8):
        super(At_est, self).__init__()
        self.conv1_A = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_A = ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3)
        self.conv3_A = ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3)
        self.conv4_A = nn.Conv2d(in_channels=n_channels, out_channels=3, kernel_size=1, stride=1, padding=0)
           
    def __call__(self, x):
        y = self.conv1_A(x)
        y = self.conv2_A(y)
        y = F.relu(y)
        y = self.conv3_A(y)
        y = self.conv4_A(y)
        return y
    
class Gamma_est(nn.Module):
    def __init__(self, n_channels=16, act=nn.ReLU(True), *args, **kwargs) -> None:
        super(Gamma_est, self).__init__(*args, **kwargs)
        self.conv1_g = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_g = ResCBAM(n_channels, n_channels, act=act)
        self.conv3_g = ResCBAM(n_channels, n_channels, act=act)
        self.conv4_g = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.gap_g = nn.AdaptiveAvgPool2d(1)

        # self.resize = 224

    def __call__(self, x): 
        # x = F.interpolate(x, [self.resize, self.resize], mode='bicubic', align_corners=True)

        y = self.conv1_g(x)
        y = self.conv2_g(y)
        # y = F.relu(y)
        y = self.conv3_g(y)
        g = self.conv4_g(y)
        g = self.gap_g(g)
        g = torch.clamp(input=g, min=1e-8, max=1e8)
        return g

class CPnet(nn.Module):
    def __init__(self, n_channels=16, isplus=False):
        super(CPnet, self).__init__()
        self.t_module = t_est() 
        self.a_module = At_est(n_channels=n_channels)
        self.apply(weight_init)
        
    def forward(self, x):
        t_ = self.t_module(x)
        a_ = self.a_module(x)
        out = ((x-a_)*t_ + a_)
        out = torch.clamp(out, min=1e-9, max=1)
        return out

class IAAF(nn.Module):
    def __init__(self, in_channel = 6, out_channel = 3, n_channels=16, act=nn.ReLU(True),*args, **kwargs) -> None:
        super(IAAF, self).__init__(*args, **kwargs)
        self.conv1_post_g = nn.Conv2d(in_channels=in_channel, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_post_g = ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3, act=act)
        self.conv3_post_g = nn.Conv2d(in_channels=n_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0)

    def __call__(self, gamma, llie, get_all=False): 
        
        out_g = torch.pow(llie, gamma)
        out = self.conv1_post_g(torch.cat((out_g, llie), dim=1))
        out = self.conv2_post_g(out)
        intersection = self.conv3_post_g(out)
        output = out_g + llie -intersection 
        
        output = torch.clamp(output, min=1e-9, max=1)
        if get_all:
            return output, out_g, intersection
        return output, out_g

class CPGAnet(nn.Module):
    def __init__(self, n_channels=16, isdgf=False, *args, **kwargs) -> None:
        super(CPGAnet, self).__init__(*args, **kwargs)
        self.n_CPA = n_channels
        self.n_Ga = 16
        self.cpnet = CPnet(n_channels=self.n_CPA)
        self.ga_est = Gamma_est()
        self.iaaf = IAAF()

        if isdgf:
            self.gf = FastGuidedFilter(r = 1)
        else:
            self.gf = isdgf

        self.apply(weight_init)
            
    def __call__(self, x, get_all=False):
        if self.gf:
            xx = x
            x = F.interpolate(x, [x.shape[2]//2, x.shape[3]//2], mode='bicubic', align_corners=True)
        out = self.cpnet(x=x)

        gamma = self.ga_est(x=x)

        if get_all:
            output, out_g, intersection,  = self.iaaf(gamma=gamma, llie=out, get_all=True)
        else:
            output, out_g = self.iaaf(gamma=gamma, llie=out)

        if self.gf:
            output = self.gf(x, output, xx)            
            output = torch.clamp(output, min=1e-9, max=1)
        if get_all:
            return output, gamma, intersection, out
        return output

class Gamma_est_p(nn.Module):
    def __init__(self, n_channels=16, act=nn.ReLU(True) ,*args, **kwargs) -> None:
        super(Gamma_est_p, self).__init__(*args, **kwargs)
        self.conv1_g = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv2_g = nn.Sequential(
            ResCBAM(n_channels, n_channels, strides=[2, 1], 
                    downsample=nn.Conv2d(n_channels, n_channels, kernel_size=1, stride=2),
                    act=act
                ),
        )
        self.conv3_g = nn.Sequential(
            ResCBAM(n_channels, n_channels, act=act),
            ResCBAM(n_channels, n_channels, act=act),
        )
        self.conv4_g = nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.gap_g = nn.AdaptiveAvgPool2d(1)

    def __call__(self, x): 
        y_ = self.conv1_g(x)
        y = self.conv2_g(y_)
        # y = F.relu(y)
        y = self.conv3_g(y) + F.interpolate(y_, size=y.size()[2:], mode='bilinear', align_corners=True)
        g = self.conv4_g(y)
        g = self.gap_g(g)
        g = torch.clamp(input=g, min=1e-8, max=1e8)
        return g

class t_module(nn.Module):
    def __init__(self, in_channel, out_channel, n_channels=8):
        super(t_module, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.conv = ResBlock(conv=default_conv, n_feats=n_channels+in_channel, kernel_size=3, act=nn.GELU(), res_scale=1)
        self.postp = nn.Conv2d(in_channels=n_channels+in_channel*2, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        self.gelu = nn.GELU()
           
    def __call__(self, x):
        x_dbc = self.get_dbc(x)
        y = self.prep(x_dbc)
        y = torch.cat((y, x), 1)
        y = self.conv(y)
        y = torch.cat((y, x), 1)
        y = self.gelu(y)
        y = self.postp(y)
        return y
    
    def get_dbc(self, rgb):
        img_max, _ = torch.max(rgb, 1, keepdim=True)
        img_min, _ = torch.min(rgb, 1, keepdim=True)
        img_mean = torch.mean(rgb, 1, keepdim=True)
        return torch.cat((img_max, img_min, img_mean), dim=1)
class At_module(nn.Module):
    def __init__(self, in_channel, out_channel, n_channels=8):
        super(At_module, self).__init__()
        self.conv1_A = nn.Conv2d(in_channels=in_channel, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_A = nn.Sequential(
            ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3, 
                     stride=[2, 1], act=nn.GELU(),
            ),
        )
        self.gelu = nn.GELU()
        self.conv3_A = ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3, act=nn.GELU())
        self.conv4_A = nn.Sequential(
            nn.Conv2d(in_channels=n_channels*2, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3, act=nn.GELU()),
        )

        self.conv5_A = nn.Conv2d(in_channels=n_channels*2, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
           
    def __call__(self, x):
        y_ = self.conv1_A(x)
        y__ = self.conv2_A(y_)
        y = self.conv3_A(y__)
        y = torch.cat((y, y__), 1)
        y = self.gelu(y)
        y = self.conv4_A(y) 
        y = torch.cat((y, y_), 1)
        y = self.conv5_A(y) 
        return y
class CPblock(nn.Module):
    def __init__(self, in_channel, out_channel, n_channels=8):
        super(CPblock, self).__init__()
        self.prep_module = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.t_module = t_module(in_channel=in_channel, out_channel=1, n_channels=n_channels//2)
        self.a_module = At_module(in_channel=in_channel, out_channel=out_channel, n_channels=n_channels)
        self.apply(weight_init)
        
    def forward(self, x):
        t_ = self.t_module(x)
        a_ = self.a_module(x)
        x_ = self.prep_module(x)
        out = ((x_-a_)*t_ + a_)
        out = torch.clamp(out, min=1e-9, max=1)
        return out
class CPGAnet_blk(nn.Module):
    def __init__(self, n_channels=8, gamma_n_channel=16, n_cpblks=2, n_IAAFch=16, isdgf=False, iscpgablks=False, *args, **kwargs) -> None:
        super(CPGAnet_blk, self).__init__(*args, **kwargs)
        self.cpblks_pre = CPblock(in_channel=3, out_channel=n_channels, n_channels=n_channels)
        cpblks = []
        for _ in range(n_cpblks):
            cpblks.append(CPblock(in_channel=n_channels, out_channel=n_channels, n_channels=n_channels))
        self.cpblks = nn.ModuleList(cpblks)
        self.cpblk_post = CPblock(in_channel=n_channels, out_channel=3, n_channels=n_channels)
        self.ga_est = Gamma_est_p(n_channels=gamma_n_channel, act=nn.GELU())
        self.iaaf = IAAF(n_channels=n_IAAFch,)
        
        if iscpgablks:
            iaafblks = []
            for _ in range(n_cpblks):
                iaafblks.append(IAAF(n_channels=n_IAAFch, in_channel=n_channels*2, out_channel=n_channels, act=nn.GELU()))
            self.iaafblks = nn.ModuleList(iaafblks)
            
        if isdgf:
            self.gf = FastGuidedFilter(r = 1)
        else:
            self.gf = isdgf

        self.apply(weight_init)
    def __call__(self, x):
        if self.gf:
            xx = x
            x = F.interpolate(x, [x.shape[2]//2, x.shape[3]//2], mode='bicubic', align_corners=True)
            
        gamma = self.ga_est(x=x)

        y = self.cpblks_pre(x)
        for index, cpblk in enumerate(self.cpblks):
            if self.iaafblks:
                y = self.cpga_blk(y, gamma, cpblk, self.iaafblks[index])
            else:
                y = cpblk(y) + y
        out = self.cpblk_post(y) + x

        output, out_g = self.iaaf(gamma=gamma, llie=out)

        if self.gf:
            output = self.gf(x, output, xx)            
            output = torch.clamp(output, min=1e-9, max=1)
        return output
    
    def cpga_blk(self, y, gamma, cpblk, iaaf):
        
        y = cpblk(y) + y

        output, out_g = iaaf(gamma=gamma, llie=y)

        return output + y

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(1, 3, 400, 600).to(device)
    model = CPGAnet_blk(
        n_channels=8, gamma_n_channel=32, n_cpblks=2, n_IAAFch=16,
        isdgf=False, iscpgablks=True
    ).to(device)
    out = model(x)

    from demo import measure_network_efficiency
    measure_network_efficiency(device, model)