# YOLOv5 common modules

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, p=None, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, p=p, g=math.gcd(c1, c2), act=act)


def DWConvWoBN(c1, c2, k=1, s=1, p=None, act=True):
    # Depthwise convolution
    return ConvWoBN(c1, c2, k, s, p=p, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, d=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConvWoBN(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, d=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvWoBN, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), dilation=d, groups=g, bias=True)
        self.act = nn.ReLU(inplace=True) if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.conv(x))


class Resizer(nn.Module):
    # Image resizer
    def __init__(self, size, shape, mode='nearest'):
        super(Resizer, self).__init__()
        self.size = size
        self.mode = mode
        assert shape == 'square'  # or rectangular
        self.shape = shape

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            size = (self.size, self.size)
        else:
            x, template = x
            size = template.shape[-2:]

        return F.interpolate(x, mode=self.mode, size=size,
                             align_corners=False if self.mode == "bilinear" else None)


class ResBlock(nn.Module):
    # Residual block
    def __init__(self, c1, c2, bn=False):  # ch_in, ch_out
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            ConvWoBN(c1, c1, k=3, s=1),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=True)
        )
        self.shortcut = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False) if c1 != c2 else nn.Identity()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.shortcut(x) + self.body(x))

        return x


class Add(nn.Module):
    # Add a list of tensors
    def __init__(self, act=False):
        super(Add, self).__init__()
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(sum(x))


class FuseBlock(nn.Module):
    # Prediction head
    def __init__(self, c1, c2, bn=False):  # ch_in, ch_out
        super(FuseBlock, self).__init__()
        self.conv1 = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(c1) if bn else nn.Identity()
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))  # + x
        x = self.conv2(x)
        
        return x


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads, norm=False):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)
        if norm:
            self.norm1 = nn.LayerNorm(c)
            self.norm2 = nn.LayerNorm(c)
        else:
            self.norm1 = None
            self.norm2 = None

    def forward(self, x, pos_embed=None):
        if pos_embed is not None:
            q, k, v = self.q(x), self.k(x + pos_embed), self.v(x + pos_embed)
        else:
            q, k, v = self.q(x), self.k(x), self.v(x)
        x = self.ma(q, k, v)[0] + x
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.fc2(self.fc1(x)) + x
        if self.norm2 is not None:
            x = self.norm2(x)
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)         # [b, c, w*h]
        p = p.unsqueeze(0)       # [1, b, c, w*h]
        p = p.transpose(0, 3)    # [w*h, b, c, 1]
        p = p.squeeze(3)         # [w*h, b, c]
        e = self.linear(p)       # [w*h, b, c]
        x = p + e                # [w*h, b, c]

        x = self.tr(x)           # [w*h, b, c]
        x = x.unsqueeze(3)       # [w*h, b, c, 1]
        x = x.transpose(0, 3)    # [1, b, c, w*h]
        x = x.reshape(b, self.c2, w, h)
        return x


class ASFF(nn.Module):
    # Adaptive Spatial Feature Fusion https://arxiv.org/abs/1911.09516
    def __init__(self, ch1, c2, e=0.5):
        super().__init__()
        assert tuple(ch1) == tuple(sorted(ch1))
        self.num_layers = len(ch1)
        self.c = int(c2 * e)
        self.c_reducer = nn.ModuleList(Conv(x, self.c, 3, 1) for x in ch1)
        self.adapter = nn.Conv2d(self.c * self.num_layers, self.num_layers,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.c_expander = Conv(self.c, c2, 1)
    
    def forward(self, x):
        assert len(x) == self.num_layers
        bs, c0, h, w = x[0].shape
        for i in range(self.num_layers):
            x[i] = self.c_reducer[i](x[i])
            scale_factor = w // x[i].shape[-1]
            if scale_factor > 1:
                # mode='bilinear', align_corners=False
                # x[i] = F.interpolate(x[i], scale_factor=scale_factor, mode='nearest')
                x[i] = nn.Upsample(size=None, scale_factor=scale_factor, mode='nearest')(x[i])
                
        layer_weights = self.adapter(torch.cat(x, dim=1)).chunk(self.num_layers, dim=1)
        
        fused = sum([x[i] * layer_weights[i] for i in range(self.num_layers)])
        out = self.c_expander(fused)
        
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, c, num_heads, num_layers):
        super().__init__()
        # self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.num_layers = num_layers
        self.tr = nn.ModuleList([TransformerLayer(c, num_heads, norm=False) for _ in range(num_layers)])
        self.c = c

    def forward(self, x):
        x, pos, pos_embed = x
        for i in range(self.num_layers):
            x = self.tr[i](x, pos_embed)
            
        # for conv ops
        return self._organize_shape(x), self._organize_shape(pos)

    @staticmethod
    def _organize_shape(x):  # [w*h, b, c, *]
        p = x.transpose(0, 1)  # [b, w*h, c, *]
        p = p.transpose(1, 2)  # [b, c, w*h, *]
        p = p.unsqueeze(2)  # [b, c, 1, w*h, *]
    
        return p


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckWoBN(Bottleneck):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckWoBN, self).__init__(c1, c2, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvWoBN(c1, c_, 1, 1)
        self.cv2 = ConvWoBN(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3WoBN(C3):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3WoBN, self).__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvWoBN(c1, c_, 1, 1)
        self.cv2 = ConvWoBN(c1, c_, 1, 1)
        self.cv3 = ConvWoBN(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[BottleneckWoBN(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        

class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class ASPP(nn.Module):
    # Atrous spatial pyramid pooling layer
    def __init__(self, c1, c2, d=(1, 2, 4, 6)):
        super(ASPP, self).__init__()
        assert c1 == c2 and c2 % len(d) == 0
        c_ = c2 // len(d)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.m = nn.ModuleList([Conv(c_, c_, k=3, s=1, p=x, d=x) for x in d])

    def forward(self, x):
        x = self.cv1(x)
        return torch.cat([m(x) for m in self.m], 1)


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(self.contract(x))
    
    @staticmethod
    def contract(x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)


class FocusWoBN(Focus):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(FocusWoBN, self).__init__(c1, c2, k, s, p, g, act)
        self.conv = ConvWoBN(c1 * 4, c2, k, s, p, g, act)


class Focus2(Focus):
    # Focus module with 2 input
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act)
    
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        # return self.conv(self.contract(x1)) + self.conv(self.contract(x2))  # share conv weight
        return self.conv(torch.cat((self.contract(x1), self.contract(x2)), dim=1))


class Blur(nn.Module):
    # Blur c information into wh-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Blur, self).__init__()
        self.conv = Conv(c1 // 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,4c,w,h) -> y(b,c,2w,2h)
        return self.conv(F.pixel_shuffle(x, 2))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Identity(nn.Module):
    # identity map for feature-pyramid
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x):
        return x


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)
