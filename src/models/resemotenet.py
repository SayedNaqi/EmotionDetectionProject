
import torch
import torch.nn as nn


# -----------------------------------------
# 1. Convolution Block
# -----------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------
# 2. Squeeze-and-Excitation Block
# -----------------------------------------
class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        weights = self.pool(x).view(batch_size, channels)
        weights = self.fc(weights)
        weights = weights.view(batch_size, channels, 1, 1)

        return x * weights


# -----------------------------------------
# 3. Residual Block
# -----------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out


# -----------------------------------------
# 4. ResEmoteNet
# -----------------------------------------
class ResEmoteNet(nn.Module):

    def __init__(
        self,
        num_classes=7,
        in_channels=1,
        channels=(64, 128, 256),
        se_reduction=16,
        num_residual_blocks=3,
        dropout_rate=0.5
    ):
        super().__init__()

        # Feature Extraction
        self.cnet = nn.Sequential(
            ConvBlock(in_channels, channels[0]),
            ConvBlock(channels[0], channels[1]),
            ConvBlock(channels[1], channels[2])
        )

        # Squeeze-and-Excitation
        self.se = SqueezeExcitation(
            channels=channels[2],
            reduction=se_reduction
        )

        # Residual Blocks
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(channels[2])
                for _ in range(num_residual_blocks)
            ]
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classifier with Dropout
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(channels[2], num_classes)
        )

    def forward(self, x):

        # Feature Extraction
        x = self.cnet(x)

        # Channel Attention
        x = self.se(x)

        # Residual Learning
        x = self.residual_blocks(x)

        # Global Average Pooling
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Classification
        x = self.classifier(x)

        return x


# -----------------------------------------
# 5. Model Test
# -----------------------------------------
if __name__ == "__main__":

    model = ResEmoteNet()

    # Test input: Batch of 4 grayscale images
    test_input = torch.randn(4, 1, 48, 48)

    # Forward pass
    model.eval()

    with torch.no_grad():
        output = model(test_input)

    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Total Parameters:", f"{total_params:,}")

    assert output.shape == (4, 7), "Output shape is incorrect!"

    print("Model test passed successfully!")
