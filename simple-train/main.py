import argparse
import torch
from torch import optim, nn

from dataset import get_cirfar100_dataloader
from modules import SimpleModel, ComplexModel
from train_eval import train_model, ressume_train_model
from utils import read_checkpoint

def default_args():
    parser = argparse.ArgumentParser(description="This is a default args")
    parser.add_argument('-c', '--checkpoint', default='latest_model_checkpoint.pth')
    parser.add_argument('--assume', action='store_true')
    parser.add_argument('--dataset', default='/home/feng/train/train-cifar-100/dataset')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('-e', '--epochs', type=int, default=300)
    args = parser.parse_args()
    return args



def main():
    # 参数解析
    args = default_args()

    trainloader, testloader = get_cirfar100_dataloader(args.dataset, batch_size=50, num_workers=2)
    criterion = nn.CrossEntropyLoss()
    # 这里，使用全局变量设置device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # model = SimpleModel().to(device)
    model = ComplexModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    


    if args.assume:
        checkpoint_path = args.checkpoint
        epoch = read_checkpoint(checkpoint_path, model, optimizer) + 1
        ressume_train_model(model, trainloader, testloader, criterion, optimizer, start_epoch=epoch, end_epochs=args.epochs)

    else:
        train_model(model, trainloader, testloader, criterion, optimizer, args.epochs)



# 简单训练： python main.py
# 续训练：   python main.py --assume --checkpoint output/epoch-50-model.pth --lr 0.0001 --epochs 110

if __name__ == '__main__':
    main()