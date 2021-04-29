# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MIN_NUM_PATCHES = 16
defaultcfg = {
    # 6 : [512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512]
    # 6 : [[510, 375, 512, 443], [512, 399, 479, 286], [511, 367, 370, 196], [512, 404, 111, 95], [512, 425, 60, 66], [512, 365, 356, 223]]
    6 : [[360, 512], [408, 479], [360, 370], [408, 111], [432, 60], [360, 356]]
}

class channel_selection(nn.Module):
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (B, num_patches + 1, dim). 
        """
        output = input_tensor.mul(self.indexes)
        return output

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, dim),
        #     nn.Dropout(dropout)
        # )
        self.net1 = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        # self.select1 = channel_selection(dim)
        # self.select2 = channel_selection(dim)
    def forward(self, x):
        # pruning   torch.Size([4, 65, 512])
        # x = self.select1(x)
        x = self.net1(x)
        # pruning   torch.Size([4, 65, 512])
        # x = self.select2(x)
        x = self.net2(x)
        return x
        # return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, dim1, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim1 ** -0.5

        # self.to_qkv = nn.Linear(dim, dim1 * 3, bias = False)
        self.to_q = nn.Linear(dim, dim1 , bias = False)
        self.to_k = nn.Linear(dim, dim1, bias = False)
        self.to_v = nn.Linear(dim, dim1, bias = False)
        self.to_out = nn.Sequential(
            nn.Linear(dim1, dim),
            nn.Dropout(dropout)
        )
        # self.select1 = channel_selection(dim1)
        # self.select2 = channel_selection(dim2)

    def forward(self, x, mask = None):
        b, n, _, h = *x.shape, self.heads
        # pruning   torch.Size([4, 65, 512])
        # x = self.select1(x)

        q = self.to_q(x)
        # q = self.select1(q)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = self.to_k(x)
        # k = self.select1(k)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = self.to_v(x)
        # v = self.select1(v)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        # pruning   torch.Size([4, 65, 512])
        # out = self.select2(out)
        out =  self.to_out(out)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, cfg):
        super().__init__()
        self.layers = nn.ModuleList([])
        if cfg is not None:
            for num in cfg:
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, num[0], heads = heads, dropout = dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, num[1], dropout = dropout)))
                ]))
        else:
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    Residual(PreNorm(dim, Attention(dim, dim, heads = heads, dropout = dropout))),
                    Residual(PreNorm(dim, FeedForward(dim, dim, dropout = dropout)))
                ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers:
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class ViT_slim(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, cfg=None, channels = 3, dropout = 0., emb_dropout = 0.):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, cfg)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask = None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    # setup_seed(200)
    b,c,h,w = 4, 3, 32, 32
    x = torch.randn(b, c, h, w)
    net = ViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    y = net(x)
    # print(y)
    print(y.size())