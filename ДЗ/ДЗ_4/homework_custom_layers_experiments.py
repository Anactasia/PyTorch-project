import torch
import torch.nn as nn
import time
from models.custom_layers import Swish, L2Pooling, CustomConv, ChannelAttention, BasicResidualBlock, BottleneckBlock, WideResidualBlock
from utils.comparison_utils import count_parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_custom_layers():
    print("Тестируем кастомные слои")

    x = torch.randn(4, 3, 32, 32, requires_grad=True).to(device)

    # Swish vs ReLU
    swish = Swish().to(device)
    relu = nn.ReLU().to(device)

    # Forward
    out_swish = swish(x)
    out_relu = relu(x)

    print("Swish output mean:", out_swish.mean().item())
    print("ReLU output mean:", out_relu.mean().item())
    print("Swish output shape:", out_swish.shape)
    print("ReLU output shape:", out_relu.shape)

    # Backward
    out_swish.mean().backward(retain_graph=True)
    grad_swish = x.grad.clone()
    x.grad.zero_()

    out_relu.mean().backward()
    grad_relu = x.grad.clone()
    x.grad.zero_()

    print("Swish grad mean:", grad_swish.mean().item())
    print("ReLU grad mean:", grad_relu.mean().item())

    # Time measurement
    start = time.time()
    for _ in range(100):
        _ = swish(x)
    print("Swish forward time (100 runs):", time.time() - start)

    start = time.time()
    for _ in range(100):
        _ = relu(x)
    print("ReLU forward time (100 runs):", time.time() - start)

    # L2Pooling vs MaxPool2d
    l2pool = L2Pooling(kernel_size=2, stride=2).to(device)
    maxpool = nn.MaxPool2d(kernel_size=2, stride=2).to(device)

    out_l2pool = l2pool(x)
    out_maxpool = maxpool(x)

    print("L2Pooling output shape:", out_l2pool.shape)
    print("MaxPool2d output shape:", out_maxpool.shape)

    # Backward check for L2Pooling
    x2 = x.clone().detach().requires_grad_(True)
    out_l2pool = l2pool(x2)
    out_l2pool.mean().backward()
    print("L2Pooling grad mean:", x2.grad.mean().item())

    # CustomConv тест
    custom_conv = CustomConv(3, 16, kernel_size=3, padding=1).to(device)
    out_custom_conv = custom_conv(x)
    print("CustomConv output shape:", out_custom_conv.shape)

    x3 = x.clone().detach().requires_grad_(True)
    out_custom_conv = custom_conv(x3)
    out_custom_conv.mean().backward()
    print("CustomConv input grad mean:", x3.grad.mean().item())
    print("CustomConv weight grad mean:", custom_conv.conv.weight.grad.mean().item())

    # ChannelAttention тест с новым входом
    x4 = x.clone().detach().requires_grad_(True)
    out_custom_conv = custom_conv(x4)
    ca = ChannelAttention(16).to(device)
    out_ca = ca(out_custom_conv)
    print("ChannelAttention output shape:", out_ca.shape)

    out_ca.mean().backward()

    print("ChannelAttention input grad mean:", x4.grad.mean().item())
    print("ChannelAttention CustomConv weight grad mean:", custom_conv.conv.weight.grad.mean().item())

def benchmark_residual_blocks():
    print("\nBenchmark Residual блоков")
    input_tensor = torch.randn(16, 64, 32, 32).to(device)

    blocks = {
        "BasicResidualBlock": BasicResidualBlock(64, 64),
        "BottleneckBlock": BottleneckBlock(64, 16),  # 64 -> 64
        "WideResidualBlock": WideResidualBlock(64, widen_factor=2)  # 64 -> 128
    }

    for name, block in blocks.items():
        block = block.to(device)
        params = count_parameters(block)

        # Measure forward pass time
        start = time.time()
        for _ in range(100):
            _ = block(input_tensor)
        elapsed = time.time() - start

        # Check gradient stability
        input_tensor.requires_grad = True
        output = block(input_tensor)
        loss = output.mean()
        loss.backward()
        grad_mean = input_tensor.grad.mean().item()

        print(f"{name:<20} | Params: {params:<8} | Time (100 runs): {elapsed:.4f}s | Grad mean: {grad_mean:.2e}")

def test_residual_blocks():
    print("\n Тест Residual блоков ")
    x = torch.randn(4, 64, 32, 32).to(device)

    basic = BasicResidualBlock(64, 64).to(device)
    print("BasicResidualBlock output:", basic(x).shape)

    bottle = BottleneckBlock(64, 16).to(device)
    print("BottleneckBlock output:", bottle(x).shape)

    wide = WideResidualBlock(64, widen_factor=2).to(device)
    print("WideResidualBlock output:", wide(x).shape)


if __name__ == "__main__":
    test_custom_layers()
    test_residual_blocks()
    benchmark_residual_blocks()
