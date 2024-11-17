
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
# from convMAE import CBlock
# from einops import rearrange

#########################################################################
from timm.models.layers import DropPath

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)#保持图像尺寸不变

    def forward(self, x):
        x = self.proj(x)

        return x

class Downsample(nn.Module):
    def __init__(self, in_c=24,out_c = 12):
        super().__init__()
        self.proj =nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)#特征图尺寸保持不变,chanel减半

    def forward(self, x):
        x = self.proj(x)
        return x

class CMlp(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CBlock(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.conv2 = nn.Conv2d(dim, dim, 1)#pw
        self.attn = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)#保持图像尺寸不变
#        self.attn = nn.Conv2d(dim, dim, 13, padding=6, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x + self.drop_path(self.conv2(self.attn(mask * self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))
        else:
            x = x + self.drop_path(self.conv2(self.attn(self.conv1(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))))###self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)保持bchw
        x = x + self.drop_path(self.mlp(self.norm2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)))
        return x

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True) # 计算均值
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias # 添加偏置

class LayerNorm(nn.Module): # 层归一化
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)



    # (b,c,h,w)->(b,h*w,c)
    def to_3d(self,x):
        return rearrange(x, 'b c h w -> b (h w) c')

    # (b,h*w,c)->(b,c,h,w)
    def to_4d(self,x, h, w):
        return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

    def forward(self, x): # (b,c,h,w)
        h, w = x.shape[-2:]
        return self.to_4d(self.body(self.to_3d(x)), h, w)
        # to_3d后：(b,h*w,c)
        # body后：(b,h*w,c)
        # to_4d后：(b,c,h,w)

class DownsamplePs(nn.Module):
    def __init__(self, n_feat):
        super(DownsamplePs, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        #x: (b,n_feat,h,w)
        # Conv2d后：(b,n_feat/4,h,w)
        # PixelUnshuffle: (b,n_feat,h/2,w/2)
        return self.body(x)

class UpsamplePs(nn.Module):
    def __init__(self, n_feat):
        super(UpsamplePs, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        # x: (b,n_feat,h,w)
        #Conv2d后：(b,n_feat*2,h,w)
        #PixelShuffle后：(b,n_feat/2,h*2,w*2)
        return self.body(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias = False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1)) # 初始化是(num_heads,1,1)

        # point-wise 1x1的卷积
        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        # depth-wise groups=in_channels
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x): # x: (b,dim,h,w)
        b,c,h,w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        # qkv后：(b,3*dim,h,w)
        # qkv_dwconv后: (b,3*dim,h,w)
        q,k,v = qkv.chunk(3, dim=1)
        # chunk后：q、k、v的大学均为(b,dim,h,w)

        # (b,dim,h,w)->(b,num_head,c,h*w)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # 在最后一维进行归一化
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # (b,num_head,c,h*w) @ (b,num_head,h*w,c) -> (b,num_head,c,c)
        # 然后乘以temperature这个可学习的参数(指的是注意力机制中的sqrt(d),d表示特征的维度)
        attn = (q @ k.transpose(-2, -1)) * self.temperature # @ 表示数学中的矩阵乘法
        # softmax 函数归一化，得到注意力得分
        attn = attn.softmax(dim=-1) #  (b,num_head,c,c)
        # attn和v做矩阵乘法：(b,num_head,c,c) @ (b,num_head,c,h*w)->(b,num_head,c,h*w)
        out = (attn @ v)
        # reshape: (b,num_head,c,h*w)->(b,num_head*c,h,w)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        # 1x1conv: (b,dim,h,w)
        out = self.project_out(out) # dim=c*num_head
        return out # (b,c,h,w)

class RGBformerBlock(nn.Module):
    def __init__(self, dim, num_heads, bias=False):
        super(RGBformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = Attention(dim, num_heads, bias)


    def forward(self, x): # (b,c,h,w)
        x = x + self.attn(self.norm1(x))
        # LN->残差连接
        return x

class RGBformer(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 24,
        num_blocks = 4,
        num_refinement_blocks = 4,
        num_heads = 2,
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(RGBformer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.RGBformerR = nn.Sequential(*[RGBformerBlock(dim=dim, num_heads=3) for i in range(num_blocks)])
        self.RGBformerB = nn.Sequential(*[RGBformerBlock(dim=dim, num_heads=3) for i in range(num_blocks)])
        self.RGBformerG = nn.Sequential(*[RGBformerBlock(dim=dim, num_heads=3) for i in range(num_blocks)])

        self.downsampleR = DownsamplePs(dim)
        self.downsampleB = DownsamplePs(dim)

        self.upsampleR = UpsamplePs(dim)
        self.upsampleB = UpsamplePs(dim)#特征图尺寸保持不变,chanel减半
        self.out3channel = OverlapPatchEmbed(dim, out_channels)


    def forward(self, img):
        extend_img = self.patch_embed(img)#B,dim,h*w

        img_r = self.downsampleR(extend_img)#B,dim,h/2*w/2
        img_b = self.downsampleB(extend_img)#B,dim,h/2*w/2

        img_r = self.RGBformerR(img_r)#B,dim,h/2*w/2
        img_b = self.RGBformerB(img_b)#B,dim,h/2*w/2
        img_g = self.RGBformerB(extend_img)#B,dim,h*w

        img_r = self.upsampleR(img_r)#B,dim/2,h*w
        img_b = self.upsampleB(img_b)#B,dim/2,h*w

        img_rb = torch.cat((img_r, img_b), dim=1)  # 拼接回B,dim,h*w
        img_rgb = self.out3channel(img_rb+img_g)+img

        return img_rgb

class RGBformerCNN(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        dim = 24,
        num_blocks = 4,
        num_refinement_blocks = 4,
        num_heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(RGBformerCNN, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        # self.norm1 = LayerNorm(dim)
        # self.attn = Attention(dim, num_heads, bias)
        self.restormerR = nn.Sequential(*[CBlock(dim=dim//2, num_heads=3) for i in range(num_blocks)])
        self.restormerB = nn.Sequential(*[CBlock(dim=dim//2, num_heads=3) for i in range(num_blocks)])
        self.restormer3out = nn.Sequential(*[CBlock(dim=dim, num_heads=3) for i in range(num_blocks)])
        self.downsampleR = Downsample(dim,dim//2)
        self.downsampleB = Downsample(dim,dim//2)#特征图尺寸保持不变,chanel减半
        self.out3channel = OverlapPatchEmbed(dim, out_channels)


    def forward(self, inp_img):
        inp_img = self.patch_embed(inp_img)#B,dim,h*w

        img_r = self.downsampleR(inp_img)#B,dim/2,h*w
        img_b = self.downsampleB(inp_img)#B,dim/2,h*w

        img_r = self.restormerR(img_r)#B,dim/2,h*w
        img_b = self.restormerB(img_b)#B,dim/2,h*w

        img_rb = torch.cat((img_r, img_b), dim=1)  # 拼接回B,dim,h*w
        img_rgb = self.out3channel(img_rb+inp_img)

        return img_rgb

if __name__ == '__main__':
    model = RGBformer()
    img = torch.Tensor(1, 3, 400, 592)
    out = model.forward(img)
    print(out.shape)
