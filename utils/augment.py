import torch
import torchvision.transforms as T
import torchvision.transforms.autoaugment as A

def build_transforms():
    return T.Compose([
        A.AutoAugment(A.AutoAugmentPolicy.CIFAR10),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010))
    ])

def cutmix(x, y, alpha=1.0, device="cpu"):
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    B, _, H, W = x.size()
    index = torch.randperm(B, device=device)

    cx = torch.randint(W, (1,), device=device)
    cy = torch.randint(H, (1,), device=device)

    w = int(W * torch.sqrt(torch.tensor(1 - lam)))
    h = int(H * torch.sqrt(torch.tensor(1 - lam)))

    x1, y1 = torch.clamp(cx - w // 2, 0, W), torch.clamp(cy - h // 2, 0, H)
    x2, y2 = torch.clamp(cx + w // 2, 0, W), torch.clamp(cy + h // 2, 0, H)

    x_cut = x.clone()
    x_cut[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    return x_cut.to(device), y.to(device), y[index].to(device), lam

def mixup(x, y, alpha=1.0, device="cpu"):
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    B = x.size(0)
    index = torch.randperm(B, device=device)
    x_mixed = lam * x + (1 - lam) * x[index]
    return x_mixed.to(device), y.to(device), y[index].to(device), lam
