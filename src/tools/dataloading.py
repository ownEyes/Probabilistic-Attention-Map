import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.backends import cudnn


def CIFAR_data_loaders(args):
    # Configure device and DataLoader parameters
    if len(args.gpu_ids) > 0 and torch.cuda.is_available():
        cudnn.benchmark = True  # Enable cudnn optimization
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        args.device = torch.device(f"cuda:{args.gpu_ids[0]}")
    else:
        kwargs = {}
        args.device = torch.device("cpu")

    # Normalization transforms
    normlizer = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    print(f"Building dataset: {args.dataset}")

    train_transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normlizer
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normlizer
    ])

    # Data loading logic based on dataset
    if args.dataset == "cifar10":
        args.num_class = 10

        full_train_set = datasets.CIFAR10(
            args.dataset_dir, train=True, download=True, transform=train_transform)

        test_set = datasets.CIFAR10(
            args.dataset_dir, train=False, download=True, transform=test_transform)

    elif args.dataset == "cifar100":
        args.num_class = 100

        full_train_set = datasets.CIFAR100(
            args.dataset_dir, train=True, download=True, transform=train_transform)

        test_set = datasets.CIFAR100(
            args.dataset_dir, train=False, download=True, transform=test_transform)

    if args.validation:

        train_size = len(full_train_set) - args.val_size

        # Split the training set into train and validation sets
        train_set, val_set = torch.utils.data.random_split(
            full_train_set, [train_size, args.val_size])

        # Create DataLoaders for the train and validation sets
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=100, shuffle=False, **kwargs)

        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=100, shuffle=False, **kwargs)

    else:
        train_loader = torch.utils.data.DataLoader(
            full_train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            test_set, batch_size=100, shuffle=False, **kwargs)

        test_loader = None

    return train_loader, val_loader, test_loader
