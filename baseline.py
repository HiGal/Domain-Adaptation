import torch
from torch import nn
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Function
from torch import optim




class GradientReverseLayer(Function):
    """
    Gradient Reversal Layer implementation.
    Taken from https://discuss.pytorch.org/t/solved-reverse-gradients-in-backward-pass/3589/3
    """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        output = grad.neg() * ctx.alpha
        return output, None


class SVHNModel(nn.Module):
    """
    Architecture of the Neural Network.
    Taken from http://sites.skoltech.ru/compvision/projects/grl/files/suppmat.pdf
    """

    def __init__(self):
        super(SVHNModel, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5),  # 28
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 13
            nn.Dropout2d(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5),  # 9
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4),  # 1
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(128, 3072),
            nn.ReLU(),
            nn.BatchNorm1d(3072),
            nn.Linear(3072, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 10)
        )

        self.domain_classifier = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 2)
        )

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        feature = feature.view(-1, 128)
        reverse_feature = GradientReverseLayer.apply(feature, alpha)
        classifier = self.classifier(feature)
        domain = self.domain_classifier(reverse_feature)

        return classifier, domain


def train(model, device, src_dataloader, target_dataloader, criterion, optimizer, epoch, scheduler=None):
    """
    Implementation of training process.
    See more high-level details on http://sites.skoltech.ru/compvision/projects/grl/files/paper.pdf

    :param model: model which we will train
    :param device: cpu or gpu
    :param src_dataloader: dataloader with source images
    :param target_dataloader:  dataloader with target images
    :param criterion: loss function
    :param optimizer: method for update weights in NN
    :param epoch: current epoch
    :param scheduler: algorithm for changing learning rate
    """
    model.train()
    len_dataloader = min(len(src_dataloader), len(target_dataloader))
    data_zip = enumerate(zip(src_dataloader, target_dataloader))

    for batch_idx, ((imgs_src, src_class), (imgs_target, _)) in data_zip:

        """
        Calculating alpha (lambda in the paper)
        """
        p = float(batch_idx + epoch * len_dataloader) / (EPOCHS * len_dataloader)
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        """
        Set up labels for domain classifier
        The images which belong to source class will have zeros, to target class will have ones as labels
        """
        labels_src = torch.zeros(len(imgs_src)).long().to(device)
        labels_target = torch.ones(len(imgs_target)).long().to(device)

        """
        Putting everything to device (cpu or gpu)
        """
        src_class = src_class.to(device)
        imgs_src = imgs_src.to(device)
        imgs_target = imgs_target.to(device)

        optimizer.zero_grad()

        # train on target domain
        _, t_domain_predict = model(imgs_target, alpha)
        t_domain_loss = criterion(t_domain_predict, labels_target)

        # train on source domain
        class_predict, src_domain_predict = model(imgs_src, alpha)
        src_domain_loss = criterion(src_domain_predict, labels_src)
        src_class_loss = criterion(class_predict, src_class)

        # calculating loss
        loss = src_class_loss + src_domain_loss + t_domain_loss

        if scheduler is not None:
            scheduler.step()

        """
        Calculating gradients and update weights
        """
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, max):
    """
    Provide accuracy on test dataset
    :param model: Model of the NN
    :param device: cpu or gpu
    :param test_loader: loader of the test dataset
    :param max: the current max accuracy of the model
    :return: max accuracy for overall observations
    """
    model.eval()

    accuracy = 0
    accuracy_domain = 0

    for (imgs, labels) in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        prediction, domain_prediction = model(imgs, alpha=0)
        domain_labels = torch.ones(len(labels)).long().to(device)
        pred_cls = prediction.data.max(1)[1]
        pred_domain = domain_prediction.data.max(1)[1]
        accuracy += pred_cls.eq(labels.data).sum().item()
        accuracy_domain += pred_domain.eq(domain_labels.data).sum().item()

    accuracy /= len(test_loader.dataset)
    accuracy_domain /= len(test_loader.dataset)
    print(f'Accuracy on MNIST-test: {100 * accuracy:.2f}%')

    """
    If accuracy is bigger then current max and domain accuracy is lower then threshold,
    then we should update and save our best model
    """
    if accuracy > max and accuracy_domain < DOMAIN_THRSH:
        max = accuracy
        torch.save(model.state_dict(), 'best_model.pth')

    return max


if __name__ == '__main__':
    """
    Set up random seed for reproducibility
    """
    np.random.seed(1337)
    torch.manual_seed(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """
    Set up number of epochs, domain threshold, loss function, device
    """
    EPOCHS = 100
    DOMAIN_THRSH = 0.2
    BATCH_SIZE = 1024
    criterion = nn.CrossEntropyLoss(reduction='sum')
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    """
    Set up transforms for source and target domains
    """
    source_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    target_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    """
    Downloading datasets and with applying transforms on them
    """
    dataset_source = datasets.SVHN(root='./', download=True, transform=source_transform)
    dataset_source_val = datasets.SVHN(root='./', split='test', download=True, transform=source_transform)
    dataset_target = datasets.MNIST(root='./', download=True, transform=target_transform)
    dataset_target_val = datasets.MNIST(root='./', train=False, download=True, transform=target_transform)

    """
    Creating PyTorch dataloaders
    """
    source_dataloader = DataLoader(dataset=dataset_source, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    source_dataloader_val = DataLoader(dataset=dataset_source_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    target_dataloader = DataLoader(dataset=dataset_target, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    target_dataloader_val = DataLoader(dataset=dataset_target_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    """
    Create a model and send it to device. After create an optimizer and scheduler
    """
    model = SVHNModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=EPOCHS // 100, eta_min=0.00005)

    """
    Training loop
    """
    max = 0
    for epoch in range(EPOCHS):
        train(model, device, source_dataloader, target_dataloader, criterion, optimizer, epoch)
        max = test(model, device, target_dataloader_val, max)

    """
    Evaluating of the model by loading the best weights after training.
    """
    model = SVHNModel()
    model.load_state_dict(torch.load('best_model.pth'))
    model = model.to(device)
    test(model, device, target_dataloader_val, 100)
