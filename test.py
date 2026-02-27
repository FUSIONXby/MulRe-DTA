import torch

def inspect_checkpoint(pth_path):
    # 加载 checkpoint
    checkpoint = torch.load(pth_path, map_location='cpu')

    print("=== checkpoint 里的 key ===")
    for key in checkpoint.keys():
        print(key)
    print()

    # 查看训练轮数
    if 'epoch' in checkpoint:
        print(f"训练轮数 epoch: {checkpoint['epoch']}")
    print()

    # 查看模型参数
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        print("模型参数层级键名及形状:")
        for k, v in model_state_dict.items():
            print(f"{k}: {tuple(v.shape)}")
        print()

        # 查看第一个参数的值示例
        first_key = list(model_state_dict.keys())[0]
        print(f"第一个参数名: {first_key}")
        print(f"参数形状: {model_state_dict[first_key].shape}")
        print(f"参数值 (前10个元素): {model_state_dict[first_key].view(-1)[:10]}")
        print()

    # 查看优化器状态
    if 'optimizer' in checkpoint:
        optimizer_state = checkpoint['optimizer']
        print("优化器状态 key:")
        for key in optimizer_state.keys():
            print(key)
        print()

if __name__ == "__main__":
    path_to_pth = 'checkpoints/ressl-graphormer-200.pth'  # 修改为你的.pth路径
    inspect_checkpoint(path_to_pth)
