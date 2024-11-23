from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision



def get_cirfar100_dataloader(dataset_path, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5074, 0.4867, 0.4417), (0.2675, 0.2565, 0.2761)) # CIFAR-100的均值和标准差
    ])

    trainset = torchvision.datasets.CIFAR100(root=dataset_path, train=True,  download=False, transform=transform)
    testset  = torchvision.datasets.CIFAR100(root=dataset_path, train=False, download=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader  = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f'Your dateset batch: {batch_size}, len(train):{len(trainset)}, len(test):{len(testset)}')

    return trainloader, testloader