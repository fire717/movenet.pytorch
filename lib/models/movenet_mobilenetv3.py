"""
@Fire
https://github.com/fire717
"""
import torch
import torch.nn as nn
import math




"""
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, 
        padding=0, dilation=1, groups=1, bias=True))
"""

def conv_3x3_act(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_act(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def conv_1x1_act2(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )



def dw_conv(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup)
    )

def dw_conv2(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )

def dw_conv3(inp, oup, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )



def upsample(inp, oup, scale=2):
    return nn.Sequential(
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp),
                nn.ReLU(inplace=True),
                conv_1x1_act2(inp,oup),
                nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False))



def channel_shuffle(x, groups: int):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int
    ) -> None:
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(
        i: int,
        o: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
    ) -> nn.Conv2d:
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

#0.5:[4, 8, 4], [24, 48, 96, 192, 1024]
#1.0:[4, 8, 4], [24, 116, 232, 464, 1024]
class Backbone(nn.Module):
    def __init__(
        self,
        stages_repeats=[4, 8, 4], 
        stages_out_channels=[24, 64, 128, 192, 256],
        num_classes = 1000,
        inverted_residual = InvertedResidual
        ) -> None:
        super(Backbone, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        # output_channels = self._stage_out_channels[-1]
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
        #     nn.BatchNorm2d(output_channels),
        #     nn.ReLU(inplace=True),
        # )



        self.upsample2 = upsample(stages_out_channels[3], stages_out_channels[2])
        self.upsample1 = upsample(stages_out_channels[2], stages_out_channels[1])

        self.conv3 = nn.Conv2d(stages_out_channels[2], stages_out_channels[2], 1, 1, 0)
        self.conv2 = nn.Conv2d(stages_out_channels[1], stages_out_channels[1], 1, 1, 0)

        self.conv4 = dw_conv3(stages_out_channels[1], 24, 1)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = x/127.5-1
        
        x = self.conv1(x)
        # x = self.maxpool(x)
        f1 = self.stage2(x)
        #print(f1.shape)#2, 116, 24, 24]
        f2 = self.stage3(f1)
        #print(f2.shape)#2, 232, 12, 12]
        x = self.stage4(f2)
        #print(x.shape)#2, 464, 6, 6]

        x = self.upsample2(x)
        f2 = self.conv3(f2)
        x += f2

        x = self.upsample1(x)
        f1 = self.conv2(f1)
        x += f1

        x = self.conv4(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)




class HardSigmoid(nn.Module):
    """Implements the Had Mish activation module from `"H-Mish" <https://github.com/digantamisra98/H-Mish>`_
    This activation is computed as follows:
    .. math::
        f(x) = \\frac{x}{2} \\cdot \\min(2, \\max(0, x + 2))
    """

    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        return 0.5 * (x/(1+torch.abs(x)))+0.5

class Header(nn.Module):
    def __init__(self, num_classes, mode='train'):
        super(Header, self).__init__()

        self.mode = mode

        #heatmaps, centers, regs, offsets
        #Person keypoint heatmap
        self.header_heatmaps = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2d(96, num_classes, 1, 1, 0, bias=True),
                        # nn.Sigmoid(),
                        HardSigmoid(),
                    ])

        #Person center heatmap
        self.header_centers = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2d(96, 1, 1, 1, 0, bias=True),
                        # nn.Sigmoid(),
                        HardSigmoid(),
                        # MulReshapeArgMax()
                    ])

        #Keypoint regression field:
        self.header_regs = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2d(96, num_classes*2, 1, 1, 0, bias=True),
                    ])

        #2D per-keypoint offset field
        self.header_offsets = nn.Sequential(*[
                        dw_conv3(24, 96),
                        nn.Conv2d(96, num_classes*2, 1, 1, 0, bias=True),
                    ])

    def argmax2loc(self, x, h=48, w=48):
        ## n,1
        y0 = torch.div(x,w).long()
        x0 = torch.sub(x, y0*w).long()
        return x0,y0


    def forward(self, x):

        res = []
        if self.mode=='train':
            h1 = self.header_heatmaps(x)
            h2 = self.header_centers(x)
            h3 = self.header_regs(x)
            h4 = self.header_offsets(x)
            res = [h1,h2,h3,h4]


        elif self.mode=='test':
            pass


        else:
            print("[ERROR] wrong mode.")

        

        return res


class MoveNet(nn.Module):
    def __init__(self, num_classes=7, width_mult=1.,mode='train'):
        super(MoveNet, self).__init__()

        self.backbone = Backbone()

        self.header = Header(num_classes, mode)
        

        self._initialize_weights()


    def forward(self, x):
        x = self.backbone(x) # n,24,48,48
        # print(x.shape)

        x = self.header(x)
        # print([x0.shape for x0 in x])

        return x


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.01)
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.weight.data.normal_(0, 0.01)
            #     m.bias.data.zero_()




if __name__ == "__main__":
    from torchsummary import summary
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    model = MoveNet().cuda()
    print(summary(model, (3, 192, 192)))


    dummy_input1 = torch.randn(1, 3, 192, 192).cuda()
    input_names = [ "input1"] #自己命名
    output_names = [ "output1" ]
    
    torch.onnx.export(model, dummy_input1, "pose.onnx", 
        verbose=True, input_names=input_names, output_names=output_names,
        do_constant_folding=True, opset_version=11)