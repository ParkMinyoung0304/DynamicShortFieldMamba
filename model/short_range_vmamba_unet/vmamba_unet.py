import torch
import torch.nn as nn
from einops import rearrange
from vmamba import VSSM
from color_map_converted import COLOR_MAP

class VMambaUnetConf:
    def __init__(self, num_classes):
        self.depths = [2, 2, 2]
        self.dims = [96, 192, 384]
        self.in_chans = 3
        self.mid_depths = [2]
        self.mid_dims = [self.dims[-1] * 2]
        self.patch_size = 4 #patch_size是指将输入图像分成多少个patch，这里是4*4=16个patch
        self.img_size = 224 #img_size是指
        self.patch_norm = True
        self.num_classes = num_classes


class VMambaUnet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        num_classes = max(COLOR_MAP.values()) + 1
        conf = VMambaUnetConf(num_classes=num_classes)
        self.patch_embed = nn.Sequential(
            nn.Conv2d(conf.in_chans, conf.in_chans * (conf.patch_size ** 2), kernel_size=conf.patch_size,
                      stride=conf.patch_size, bias=True),
            Permute(0, 2, 3, 1),
            nn.LayerNorm(conf.in_chans * (conf.patch_size ** 2)) if conf.patch_norm else nn.Identity(),
        )
        self.down = VSSM(depths=conf.depths, dims=conf.dims, in_chans=conf.in_chans)
        self.mid = VSSM(depths=conf.mid_depths, dims=conf.mid_dims, in_chans=conf.dims[-1], if_down=False)
        self.up = VSSM(depths=conf.depths, dims=conf.dims[::-1], in_chans=conf.dims[-1], if_down=False)
        self.patch_expanding = FinalPatchExpand_X4(
            input_resolution=(conf.img_size // conf.patch_size, conf.img_size // conf.patch_size), dim_scale=4,
            dim=conf.dims[0]) 
        self.output = nn.Conv2d(in_channels=conf.dims[0], out_channels=conf.num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.patch_embed(x)
        x, all_feat = self.down(x) #all_feat: [B, H*W, C]
        x = self.mid(x)
        x = self.up(x, all_feat)
        x = self.patch_expanding(x)
        x = x.permute(0, 3, 1, 2)
        x = self.output(x)
        # print(1)
        return x


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = self.norm(x)
        return x


if __name__ == '__main__':
    model = VMambaUnet().cuda()
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of params: {n_parameters}")
    for i in range(100):
        x = torch.randn((12, 3, 224, 224), dtype=torch.float32, device="cuda") #这里的randn的参数是batch_size, channel, height, width
        res = model(x)
    print("over")