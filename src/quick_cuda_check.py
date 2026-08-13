import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        print("matmul ok:", float(y.mean().detach().cpu()))


if __name__ == "__main__":
    main()

