import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
try:
    from guided_filter_pytorch.guided_filter import FastGuidedFilter
except:
    print("FastGuidedFilter not found, please install guided-filter-pytorch package.")
    FastGuidedFilter = None
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
        if x.size() != res.size():
            res += F.interpolate(x, size=res.size()[2:], mode='bilinear', align_corners=False)
        else:
            res += x

        return res
        
class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio=1, act=nn.ReLU6(inplace=True), bias=False):
        super(InvertedResidualBlock, self).__init__()
        
        assert stride in [1, 2]

        hidden_channels = int(in_channels * expand_ratio)
        
        self.use_res_connect = (stride == 1 and in_channels == out_channels)

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=bias),
                nn.BatchNorm2d(hidden_channels),
                act
            ])
        
        layers.extend([
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, 
                      stride=stride, padding=kernel_size // 2, groups=hidden_channels, bias=bias),
            nn.BatchNorm2d(hidden_channels),
            act
        ])
        
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.body(x)
        else:
            return self.body(x)
   
def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), 
        stride=stride,
        bias=bias)

def default_ds_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    # Depthwise separable convolution: depthwise + pointwise
    depthwise = nn.Conv2d(
        in_channels, in_channels, kernel_size,
        padding=(kernel_size // 2),
        stride=stride,
        bias=bias, groups=in_channels
    )
    pointwise = nn.Conv2d(
        in_channels, out_channels, kernel_size=1,
        stride=1, bias=bias
    )
    return nn.Sequential(depthwise, pointwise)

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
    def __init__(self, inplanes, planes, strides=[1, 1], kernel_size=3, act=nn.ReLU(True), conv=default_conv, downsample=None):
        super(ResCBAM, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=kernel_size, stride=strides[0])
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = act
        self.conv2 = conv(planes, planes, kernel_size=kernel_size, stride=strides[1])
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

class LayerNorm(nn.Module):
    """
    一個對 Channels-First (NCHW) 格式的張量進行 Layer Normalization 的輔助類。
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)
        return x
    
class ConvNeXtBlock(nn.Module):

    def __init__(self, in_channels, out_channels=None, kernel_size=7, stride=1, 
                 mlp_expand_ratio=4., layer_scale_init_value=1e-6):
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels

        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, 
                                padding=kernel_size // 2, groups=in_channels, stride=stride)
        
        self.norm = LayerNorm(in_channels, eps=1e-6)
        
        hidden_channels = int(mlp_expand_ratio * in_channels)
        self.pwconv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)
        
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels, 1, 1)), requires_grad=True)
        else:
            self.gamma = None

        self.downsample = None
        if stride > 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
            )

    def forward(self, x):
        residual = x
        
        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x

        x = residual + x
        return x

class t_est(nn.Module):
    def __init__(self, n_channels=3):
        super(t_est, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=n_channels*2, out_channels=n_channels, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(in_channels=n_channels*2, out_channels=n_channels, kernel_size=7, stride=1, padding=3)
        self.conv5 = nn.Conv2d(in_channels=n_channels*4, out_channels=1, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
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
           
    def forward(self, x):
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

    def forward(self, x): 
        # x = F.interpolate(x, [self.resize, self.resize], mode='bicubic', align_corners=True)

        y = self.conv1_g(x)
        y = self.conv2_g(y)
        # y = F.relu(y)
        y = self.conv3_g(y)
        g = self.conv4_g(y)
        g = self.gap_g(g)
        g = torch.clamp(input=g, min=.2, max=4.)
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
        out = torch.clamp(out, min=1e-4, max=1)
        return out

class IAAF(nn.Module):
    def __init__(self, in_channel = 6, out_channel = 3, n_channels=16, act=nn.ReLU(True),*args, **kwargs) -> None:
        super(IAAF, self).__init__()
        self.conv1_post_g = nn.Conv2d(in_channels=in_channel, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv2_post_g = ResBlock(conv=default_conv, n_feats=n_channels, kernel_size=3, act=act)
        self.conv3_post_g = nn.Conv2d(in_channels=n_channels, out_channels=out_channel, kernel_size=1, stride=1, padding=0)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, gamma, llie, x=None, get_all=False,*args, **kwarg): 
        
        base_img = x if x is not None else llie
        out_g = torch.pow(base_img, gamma)

        out = self.conv1_post_g(torch.cat((out_g, llie), dim=1))
        out = self.conv2_post_g(out)
        intersection = self.conv3_post_g(out)
        output = out_g + llie - intersection
        
        output = torch.clamp(output, min=1e-4, max=1)
        if get_all:
            return output, out_g, intersection
        return output, out_g
        
class IAAF_masking(nn.Module):
    def __init__(self, in_channel=6, out_channel=3, n_channels=16, act=nn.GELU(),
                 n_ConvScoring=2, masking_range=[0.1, 1.], ablation=False, block='ResBlock',
                 conv=default_conv, bn=False, global_proc=None,
                 *args, **kwargs):
        super(IAAF_masking, self).__init__()
        self.conv1_post_g = conv(in_channel, n_channels, 1, 1)
        # Use a single block instance for scoring if possible to save memory
        block_map = {
            'ResBlock': lambda: ResBlock(conv=conv, n_feats=n_channels, kernel_size=3, act=act, bn=bn),
            'InvertedResBlock': lambda: InvertedResidualBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1, act=act, bias=True),
            'ConvNeXtBlock': lambda: ConvNeXtBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=1)
        }
        if block not in block_map:
            raise ValueError(f"Unsupported block type: {block}")
        self.conv2_post_g = block_map[block]()
        self.conv3_post_g = conv(n_channels, out_channel, 1, 1)

        if ablation != 'no_scoring' or ablation == 'no':
            scoring_layers = [conv(in_channel // 2, n_channels, 1, 1)]
            for _ in range(n_ConvScoring):
                scoring_layers.append(block_map[block]())
            if n_ConvScoring == 0:
                scoring_layers.append(conv(n_channels, n_channels, 3, 1))
                scoring_layers.append(act)
            scoring_layers.append(conv(n_channels, out_channel, 1, 1))
            scoring_layers.append(nn.Sigmoid())
            self.conv_scoring = nn.Sequential(*scoring_layers)
        self.masking_range = masking_range
        self.ablation = ablation

        if global_proc is not None:
            self.global_proc = global_proc
        else:
            self.global_proc = torch.pow

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, gamma, llie, x=None, get_all=False, masking_rate=.5, SST=False, annealing=False):
        try:
            llie = llie.float()
            gamma = gamma.float()
        except:
            pass
        
        if x is not None:
            x = x.float()
            out_g = self.global_proc(x, gamma)
        else:
            out_g = self.global_proc(llie, gamma)

        if self.ablation and (self.ablation == 'no_masking' or self.ablation == 'no'):
            masking_rate = .5 * torch.ones_like(llie[:, 0:1, :, :], dtype=torch.float32)
        
        if not annealing:
            msk_rate = torch.clamp(masking_rate, min=self.masking_range[0], max=self.masking_range[1]) if not SST else 0.
        else:
            msk_rate = masking_rate

        if self.ablation and (self.ablation == 'no_scoring' or self.ablation == 'no'):
            out_g_msk = torch.ones_like(out_g)
            llie_msk = torch.ones_like(llie)
        else:
            out_g_msk = self.conv_scoring(out_g) if not SST else torch.ones_like(out_g)
            llie_msk = self.conv_scoring(llie) if not SST else torch.ones_like(llie)

        out = self.conv1_post_g(torch.cat((out_g * out_g_msk, llie * llie_msk), dim=1))
        out = self.conv2_post_g(out)

        intersection = self.conv3_post_g(out)

        output = (1- msk_rate) * out_g + (msk_rate) * llie - intersection
        output = torch.clamp(output, min=1e-4, max=1)
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
            
    def forward(self, x, get_all=False):
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
            output = torch.clamp(output, min=1e-4, max=1)
        if get_all:
            return output, gamma, intersection, out
        return output

class Gamma_est_p(nn.Module):
    def __init__(self, n_channels=16, act=nn.ReLU(True) , conv=default_conv, *args, **kwargs) -> None:
        super(Gamma_est_p, self).__init__()
        self.conv1_g = conv(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1)
        self.conv2_g = nn.Sequential(
            ResCBAM(n_channels, n_channels, strides=[2, 1], 
                    downsample=conv(n_channels, n_channels, kernel_size=1, stride=2),
                    act=act, conv=conv
                ),
        )
        self.conv3_g = nn.Sequential(
            ResCBAM(n_channels, n_channels, act=act, conv=conv),
            ResCBAM(n_channels, n_channels, act=act, conv=conv),
        )
        self.conv4_g = conv(n_channels, 1, kernel_size=1, stride=1)
        self.gap_g = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, keep_feat=False): 
        y_ = self.conv1_g(x)
        y = self.conv2_g(y_)
        # y = F.relu(y)
        y = self.conv3_g(y) + F.interpolate(y_, size=y.size()[2:], mode='bilinear', align_corners=False)
        g = self.conv4_g(y)
        g = self.gap_g(g)
        g = torch.clamp(input=g, min=1e-2, max=4.)
        if keep_feat:
            return g, y
        return g

class t_module(nn.Module):
    def __init__(self, in_channel, out_channel, n_channels=8, conv=default_conv):
        super(t_module, self).__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=n_channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
        self.conv = ResBlock(conv=conv, n_feats=n_channels+in_channel, kernel_size=3, act=nn.GELU(), res_scale=1)
        # self.postp = nn.Conv2d(in_channels=n_channels+in_channel*2, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
        self.postp = conv(in_channels=n_channels+in_channel*2, out_channels=out_channel, kernel_size=1, stride=1)
        self.gelu = nn.GELU()
           
    def forward(self, x):
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
    def __init__(self, in_channel, out_channel, n_channels=8, conv=default_conv, block='ResBlock'):
        super(At_module, self).__init__()
        # self.conv1_A = nn.Conv2d(in_channels=in_channel, out_channels=n_channels, kernel_size=1, stride=1, padding=0)
        self.conv1_A = conv(in_channels=in_channel, out_channels=n_channels, kernel_size=1, stride=1)
        # Efficient block selection for conv2_A and conv3_A
        block_map = {
            'ResBlock': lambda stride: ResBlock(conv=conv, n_feats=n_channels, kernel_size=3, stride=[stride, 1], act=nn.GELU()),
            'InvertedResBlock': lambda stride: InvertedResidualBlock(
            in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=stride, expand_ratio=1, act=nn.GELU(), bias=True),
            'ConvNeXtBlock': lambda stride: ConvNeXtBlock(in_channels=n_channels, out_channels=n_channels, kernel_size=3, stride=stride)
        }
        if block not in block_map:
            raise ValueError(f"Unsupported block type: {block}. Use 'ResBlock', 'InvertedResBlock', or 'ConvNeXtBlock'.")
        self.conv2_A = nn.Sequential(block_map[block](2))
        self.gelu = nn.GELU()
        self.conv3_A = block_map[block](1)
        # Efficient conv4_A
        conv4_layers = [
            conv(n_channels*2, n_channels, kernel_size=3, stride=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            block_map[block](1)
        ]
        self.conv4_A = nn.Sequential(*conv4_layers)

        self.conv5_A = nn.Conv2d(in_channels=n_channels*2, out_channels=out_channel, kernel_size=1, stride=1, padding=0)
           
    def forward(self, x):
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
    def __init__(self, in_channel, out_channel, n_channels=8, conv=default_conv, bn=False, block='ResBlock'):
        super(CPblock, self).__init__()
        self.prep_module = nn.Sequential(
            # nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            conv(in_channel, out_channel, kernel_size=3, stride=1),
            nn.GELU(),
        )
        self.t_module = t_module(in_channel=in_channel, out_channel=1, n_channels=n_channels//2, conv=conv)
        self.a_module = At_module(in_channel=in_channel, out_channel=out_channel, n_channels=n_channels, conv=conv, block=block)
        if bn:
            self.bn = nn.BatchNorm2d(out_channel) 
        self.apply(weight_init)
        
    def forward(self, x):
        t_ = self.t_module(x)
        a_ = self.a_module(x)
        x_ = self.prep_module(x)
        out = ((x_-a_)*t_ + a_)
        if hasattr(self, 'bn'):
            out = self.bn(out)
        out = torch.clamp(out, min=1e-4, max=1)
        return out
class CPGAnet_blk(nn.Module):
    def __init__(self, n_channels=8, gamma_n_channel=16, n_cpblks=2, n_IAAFch=16, isdgf=False, iscpgablks=False, iaaf_type='IAAF', 
                 iaaf_ablation=False, iaaf_scoring=[2, 2], conv='conv', block='ResBlock', bn=False, efficient=False,
                 global_proc=None,
                 *args, **kwargs) -> None:
        super(CPGAnet_blk, self).__init__()

        IAAF_type = IAAF_masking if iaaf_type == 'IAAF_masking' else IAAF
        if conv == 'conv':
            conv = default_conv
        elif conv == 'ds_conv':
            conv = default_ds_conv
        else:
            raise ValueError(f"Unsupported conv type: {conv}. Use 'conv' or 'ds_conv'.")
                
        self.global_proc = global_proc

        self.cpblks_pre = CPblock(in_channel=3, out_channel=n_channels, n_channels=n_channels, conv=conv, block=block)
        cpblks = []
        for _ in range(n_cpblks):
            cpblks.append(CPblock(in_channel=n_channels, out_channel=n_channels, n_channels=n_channels, conv=conv, 
                                  bn=bn, block=block))
        self.cpblks = nn.ModuleList(cpblks)
        self.cpblk_post = CPblock(in_channel=n_channels, out_channel=3, n_channels=n_channels, conv=conv, block=block)
        self.ga_est = Gamma_est_p(n_channels=gamma_n_channel, act=nn.GELU(), conv=conv)
        self.iaaf = IAAF_type(n_channels=n_IAAFch, ablation = iaaf_ablation, n_ConvScoring=iaaf_scoring[0], 
                              conv=conv, block=block, global_proc=None)
        if isinstance(self.iaaf, IAAF_masking):
            self.iaaf_masking_rate = nn.Parameter(torch.tensor([0.5], dtype=torch.float32, requires_grad=True),)
        
        if iscpgablks:
            iaafblks = []
            for _ in range(n_cpblks):
                iaafblks.append(
                    IAAF_type(n_channels=n_IAAFch, in_channel=n_channels*2, out_channel=n_channels, 
                              act=nn.GELU(), ablation = iaaf_ablation, n_ConvScoring=iaaf_scoring[1], conv=conv,
                               block=block, global_proc=self.global_proc))
            self.iaafblks = nn.ModuleList(iaafblks)
            if isinstance(self.iaafblks[0], IAAF_masking):
                self.iaafblks_masking_rate = nn.Parameter(
                    torch.tensor([.5] * n_cpblks, dtype=torch.float32, requires_grad=True)
                )
        if isdgf:
            self.gf = FastGuidedFilter(r = 1)
        else:
            self.gf = isdgf

        self.efficient = efficient

        
        self.apply(weight_init)

    def forward(self, x, get_all=False, SST=False, bdsf=False, masking_rate=None):
        if self.gf:
            xx = x
            x = F.interpolate(x, [x.shape[2]//2, x.shape[3]//2], mode='bicubic', align_corners=True)
            
        gamma_output = self.ga_est(x=x, keep_feat=self.global_proc)
        if self.global_proc:
            gamma, gamma_features = gamma_output
            self.global_proc.generate_dynamic_lut((gamma, gamma_features))
        else:
            gamma = gamma_output

        if bdsf or self.efficient:
           out = x
        else: 
            y = self.cpblks_pre(x)
            for index, cpblk in enumerate(self.cpblks):
                if hasattr(self, 'iaafblks') and self.iaafblks:
                    y = self.cpga_blk(y, gamma, cpblk, self.iaafblks[index], index, SST=SST)
                else:
                    y = cpblk(y) + y
            out = self.cpblk_post(y) + x

        if isinstance(self.iaaf, IAAF_masking):
            if not get_all:
                output, out_g = self.iaaf(gamma, out, x=x, masking_rate=self.iaaf_masking_rate, 
                                          SST=SST)
            else:
                output, out_g, intersection = self.iaaf(gamma, out, x=x, get_all=True, masking_rate=self.iaaf_masking_rate, 
                                                        SST=SST)
        else:
            if not get_all:
                output, out_g = self.iaaf(gamma, out, get_all=False)
            else:
                output, out_g, intersection = self.iaaf(gamma, out, get_all=True)

        if self.gf:
            output = self.gf(x, output, xx)            
            output = torch.clamp(output, min=1e-4, max=1)

        if get_all:
            _ = out_g
            return output, gamma, out_g, intersection, out        
    
        return output
    
    def cpga_blk(self, y, gamma, cpblk, iaaf, index, SST=False):
        
        y_ = cpblk(y) + y

        if isinstance(iaaf, IAAF_masking):
            output, out_g = iaaf(gamma, y_, masking_rate=
                self.iaafblks_masking_rate[index], SST=SST
            )
        else:
            output, out_g = iaaf(gamma, y_, get_all=False)

        return output + y_

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    try:
        device = 'mps' if torch.backends.mps.is_available() else device
    except:
        print("MPS backend is not available, using CPU or CUDA.")
    x = torch.randn(1, 3, 400, 600).to(device)
    model = CPGAnet_blk(
        n_channels=8, gamma_n_channel=16, n_cpblks=2, n_IAAFch=16,
        isdgf=False, iscpgablks=True,
        iaaf_type='IAAF_masking', iaaf_ablation=None, iaaf_scoring=[2, 2],
        # conv='ds_conv',  # 'conv' or 'ds_conv'
        block = 'ConvNeXtBlock',  # 'ResBlock', 'InvertedResBlock', or 'ConvNeXtBlock'
        # efficient=True,  
    ).to(device)
    out = model(x)

    from demo import measure_network_efficiency
    measure_network_efficiency(device, model)