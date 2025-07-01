import torch

# Задание 2: Автоматическое дифференцирование

def simple_gradients():
    # 2.1 Простые вычисления с градиентами

    print("2.1 Простые вычисления с градиентами")
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(4.0, requires_grad=True)
    z = torch.tensor(6.0, requires_grad=True)

    #  функция f(x,y,z) = x² + y² + z² + 2xyz
    f = x**2 + y**2 + z**2 + 2*x*y*z
    print("f =", f.item())

    f.backward()

    print(f"df/dx = {x.grad.item()} (должно быть 2x + 2yz = {2*x.item() + 2*y.item()*z.item()}), {x.grad.item() == 2*x.item() + 2*y.item()*z.item()}")
    print(f"df/dy = {y.grad.item()} (должно быть 2y + 2xz = {2*y.item() + 2*x.item()*z.item()}), {y.grad.item() == 2*y.item() + 2*x.item()*z.item()}")
    print(f"df/dz = {z.grad.item()} (должно быть 2z + 2xy = {2*z.item() + 2*x.item()*y.item()}), {z.grad.item() == 2*z.item() + 2*x.item()*y.item()}")


def mse_gradients():
    # 2.2 Градиент функции потерь

    print("\n2.2 Градиент функции потерь")
    
    x = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([2.0, 4.0, 6.0])

    w = torch.rand(1, requires_grad=True)
    b = torch.rand(1, requires_grad=True)

    y_pred = w * x + b

    mse = ((y_pred - y_true) ** 2).mean()
    print(f"MSE = {mse.item():.4f}")

    mse.backward()

    print(f"df/dw = {w.grad.item()}")
    print(f"df/db = {b.grad.item()}")


def chain_rule():
    # 2.3 Цепное правило

    print("\n2.3 Цепное правило")

    x = torch.tensor(2.0, requires_grad=True)
    f = torch.sin(x**2 + 1)

    f.backward(retain_graph=True)
    grad_autograd = x.grad.item()

    grad_analytic = torch.cos(x**2 + 1) * 2 * x

    grad_torch_grad = torch.autograd.grad(f, x)[0].item()

    print(f"df/dx (autograd backward) = {grad_autograd}")
    print(f"df/dx (аналитически) = {grad_analytic.item()}")
    print(f"df/dx (torch.autograd.grad) = {grad_torch_grad}")



if __name__ == "__main__":
    simple_gradients()
    mse_gradients()
    chain_rule()
