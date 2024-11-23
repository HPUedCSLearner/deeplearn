import torch
import os

def save_checkpoint(model, epoch, optimizer, filename):
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = output_dir + '/' + filename
    
    torch.save({
        "model_dict": model.state_dict(),
        "optimizer_dict": optimizer.state_dict(),
        "epoch": epoch
    }, filename)


def read_checkpoint(path, model, optimizer):
    # 首先判断 模型路径是否存在
    # 然后读取
    checkpoint = torch.load(path, weights_only=True, map_location='cuda')
    model.load_state_dict(checkpoint['model_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_dict'])
    epoch = checkpoint['epoch']
    return epoch