import torch
from torch import nn, optim
from torchvision.transforms import transforms
import cv2

from datasets.dataset import get_imsize
from train_model import train
from test_model import test

from datasets import ImplementedDatasets, get_dataloader

from suas_transforms import RollingShutter


class ShapeClassifier(nn.Module):
    def __init__(self, input_size = 200, in_channels = 1, num_classes=9):
        super().__init__()
        self.in_channels = in_channels

        hidden_dims = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]]
        #hidden_dims = [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512]]
        layers = []

        for conv_group in hidden_dims:
            for num_filters in conv_group:
                layers.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = num_filters,
                            kernel_size = (3,3),
                            stride = 1,
                            padding = 1,
                        ),
                        nn.BatchNorm2d(num_filters),
                        nn.ReLU(),
                    )
                )
                in_channels = num_filters

            layers.append(nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.backbone = nn.Sequential(*layers)

        linear_in = (input_size >> len(hidden_dims))**2 * hidden_dims[-1][-1]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(linear_in, 4096),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(4096, num_classes)
        )


    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

if __name__ == "__main__":
    epochs = 300
    model = ShapeClassifier()
    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=1e-3,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    imsz = get_imsize(ImplementedDatasets.SUAS)
    tf = [
        transforms.ToTensor(),
        transforms.Resize((imsz, imsz)),
        transforms.Grayscale(),
        RollingShutter(1.3)
    ]
    train_set = get_dataloader(dataset=ImplementedDatasets.SUAS, batch_size=32, train=True, transform=tf)
    val_set   = get_dataloader(dataset=ImplementedDatasets.SUAS, batch_size=32, train=False, transform=tf)

    #img = next(iter(train_set))[0][0]
    #img = img.detach().numpy().transpose(1, 2, 0)
    #cv2.imshow("jie", img)
    #cv2.waitKey(0)
    #exit()

    print_interval = 10
    model_save_path="VGGshape-normalgrayscale.pt"
    device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")

    def test_fn():
        return test(model, val_set, device, criterion)

    model.to(device)
    train(
        model=model,
        optim=optim,
        scheduler=scheduler,
        criterion=criterion,
        train_set=train_set,
        print_interval=print_interval,
        model_save_path=model_save_path,
        epochs=epochs,
        device=device,
        test_fn=test_fn
    )
