from module import * 

class CategoryKeypointNet(nn.Module):
    def __init__(self, n_channels, n_classes=3, n_heatmap=2, bilinear=False):
        """
        Initialize the CategoryKeypointNet model.
        """
        super(CategoryKeypointNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
        self.embedding = nn.Embedding(n_classes, 64)
        self.hclass = OutConv(128, n_heatmap)
        self.hscore = OutConv(128, 1)

    def forward(self, x):
        """
        Define the forward pass of the CategoryKeypointNet model.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = F.softmax(logits, dim=1)
        argmax_indices = torch.argmax(logits, dim=1) 
        embedded_argmax = self.embedding(argmax_indices).permute(0, 3, 1, 2) 
        heatmap = torch.cat([x, embedded_argmax], dim=1)
        heatmap_score = torch.sigmoid(self.hscore(heatmap))
        heatmap_class = F.softmax(self.hclass(heatmap), dim=1)
        heatmap = heatmap_score * heatmap_class
        return logits, heatmap

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        """
        Initialize the UNet model.
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Define the forward pass of the UNet model.
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits