import torch.nn as nn
import torch

def passthrough(x, **kwargs):
    return x

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),  # in_ch、out_ch是通道数
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1+x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
		# 返回加权的 x
        return x*psi

class InputTransition(nn.Module):
    def __init__(self, inChans):
        super(InputTransition, self).__init__()
        self.conv = DoubleConv(inChans, 64)
    def forward(self, x):
        out = self.conv(x)
        return out

class DownTransition(nn.Module):
    def __init__(self, inChans):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.pool = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        self.conv = DoubleConv(inChans, outChans)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        return out

class UpTransition(nn.Module):
    def __init__(self, inChans):
        super(UpTransition, self).__init__()
        outChans = inChans // 2
        self.up_conv = nn.ConvTranspose2d(inChans, outChans, 2, stride=2)
        self.atten = Attention_block(outChans,outChans,outChans//2)
        self.conv = DoubleConv(inChans, outChans)

    def forward(self, x, xskip):
        up = self.up_conv(x)
        att= self.atten(up,xskip)
        out = torch.cat((att,up),1)
        out = self.conv(out)
        return out

class OutputTransition(nn.Module):
    def __init__(self, out_ch):
        super(OutputTransition, self).__init__()
        self.up_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        out = self.up_conv(x)
        return out

class UNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UNet, self).__init__()
        self.down1= InputTransition(in_ch)
        self.down2= DownTransition(64)
        self.down3= DownTransition(128)
        self.down4= DownTransition(256)
        self.down5= DownTransition(512)
        # self.conv1 = DoubleConv(in_ch, 64)
        # self.pool1 = nn.MaxPool2d(2)  # 每次把图像尺寸缩小一半
        # self.conv2 = DoubleConv(64, 128)
        # self.pool2 = nn.MaxPool2d(2)
        # self.conv3 = DoubleConv(128, 256)
        # self.pool3 = nn.MaxPool2d(2)
        # self.conv4 = DoubleConv(256, 512)
        # self.pool4 = nn.MaxPool2d(2)
        # self.conv5 = DoubleConv(512, 1024)
        # 逆卷积
        self.up1= UpTransition(1024)
        self.up2= UpTransition(512)
        self.up3= UpTransition(256)
        self.up4= UpTransition(128)
        self.up5= OutputTransition(out_ch)
        # self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        # self.conv6 = DoubleConv(1024, 512)
        # self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        # self.conv7 = DoubleConv(512, 256)
        # self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        # self.conv8 = DoubleConv(256, 128)
        # self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        # self.conv9 = DoubleConv(128, 64)
        # self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        d1=self.down1(x)
        d2=self.down2(d1)
        d3=self.down3(d2)
        d4=self.down4(d3)
        d5=self.down5(d4)
        # c1 = self.conv1(x)
        # p1 = self.pool1(c1)
        # c2 = self.conv2(p1)
        # p2 = self.pool2(c2)
        # c3 = self.conv3(p2)
        # p3 = self.pool3(c3)
        # c4 = self.conv4(p3)
        # p4 = self.pool4(c4)
        # c5 = self.conv5(p4)

        u1=self.up1(d5,d4)
        u2=self.up2(u1,d3)
        u3=self.up3(u2,d2)
        u4=self.up4(u3,d1)
        u5=self.up5(u4)
        out = nn.Sigmoid()(u5)  # 化成(0~1)区间
        # up_6 = self.up6(c5)
        # merge6 = torch.cat([up_6, c4], dim=1)  # 按维数1（列）拼接,列增加
        # c6 = self.conv6(merge6)
        #
        # up_7 = self.up7(c6)
        # merge7 = torch.cat([up_7, c3], dim=1)
        # c7 = self.conv7(merge7)
        #
        # up_8 = self.up8(c7)
        # merge8 = torch.cat([up_8, c2], dim=1)
        # c8 = self.conv8(merge8)
        #
        # up_9 = self.up9(c8)
        # merge9 = torch.cat([up_9, c1], dim=1)
        # c9 = self.conv9(merge9)

        # c10 = self.conv10(c9)

        # out = nn.Sigmoid()(c10)  # 化成(0~1)区间
        return out
