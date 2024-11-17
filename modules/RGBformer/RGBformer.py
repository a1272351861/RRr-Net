from torch import nn
# from .layers import *
from patch_embed import *
from blocks import *

class RGBformer(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, block_num=2, num_heads=3,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 alpha=0.5, local_ws=2, **kwargs):
        super().__init__()

        # self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.local_ws = local_ws

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        patches_resolution = self.patch_embed.patches_resolution
        self.reshape = DTM(patches_resolution,self.embed_dim)

        # build blocks block_num个block
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, qk_scale=qk_scale,
                  drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=drop_path_rate,
                  norm_layer=norm_layer, local_ws=local_ws, alpha=alpha)
            for i in range(block_num)])


    def forward(self, x):
        x = self.patch_embed(x)# b,3,224,224->b,96,56,56 (b,3136,96)
        for blk in self.blocks:
            x = blk(x)# 不改变形状
        x = self.reshape(x)###b,96,56,56->b,4,224,224
        return x

if __name__ == '__main__':
    model = RGBformer()
    img = torch.Tensor(1, 3, 224, 224)
    # img = torch.Tensor(1, 3, 400, 592)
    out = model.forward(img)
    print(out.shape)
