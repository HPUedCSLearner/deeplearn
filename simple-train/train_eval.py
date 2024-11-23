import torch
from utils import save_checkpoint

# 默认设备是cuda
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def eval_model(model, datasetLoader, trainsetOrTestset):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, lables in datasetLoader:
            inputs, lables = inputs.to(device), lables.to(device)
            outputs = model(inputs)
            _, predict = torch.max(outputs, dim=1)
            correct += (predict == lables).sum().item()
            total += lables.size(0)
        print(f'in {trainsetOrTestset}, \ttotal:{total}, \tcorret:{correct}, \tAcc:{correct/total*100:.3f}%')


def train_model(model, trainloader, testloader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for iter, (inputs, lables) in enumerate(trainloader):
            inputs, lables = inputs.to(device), lables.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predict = torch.max(outputs, dim=1)
            total += lables.size(0)
            correct += (predict == predict).sum().item()

            # 设置iter输出训练过程
            if iter % 200 == 0:
                print(f'Iter:{iter + epoch*len(trainloader)}, \tEpoch:{epoch+1}/{epochs}, \tLoss:{running_loss/total:.6f}')
            
        # 训练完一个epoch，做评测
        eval_model(model, testloader,  'testloader')
        # eval_model(model, trainloader, 'trainloader')

        # 模型checkpoint保存
        if (epoch+1) % 50 == 0:
            save_checkpoint(model, epoch, optimizer, f'epoch-{epoch+1}-model.pth')



def ressume_train_model(model, trainloader, testloader, criterion, optimizer, start_epoch, end_epochs):
    epoch = start_epoch
    epochs = end_epochs
    while epoch < end_epochs:
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for iter, (inputs, lables) in enumerate(trainloader):
            inputs, lables = inputs.to(device), lables.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predict = torch.max(outputs, dim=1)
            total += lables.size(0)
            correct += (predict == predict).sum().item()

            # 设置iter输出训练过程
            if iter % 200 == 0:
                print(f'Iter:{iter + epoch*len(trainloader)}, \tEpoch:{epoch+1}/{epochs}, \tLoss:{running_loss/total:.6f}')
            
        # 训练完一个epoch，做评测
        eval_model(model, testloader,  'testloader')
        eval_model(model, trainloader, 'trainloader')

        # 模型checkpoint保存
        if (epoch+1) % 30 == 0:
            save_checkpoint(model, epoch, optimizer, f'epoch-{epoch+1}-model.pth')
        
        epoch += 1
