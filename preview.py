import torch
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import time
import model

data_dir = "./"
data_transform = {
    x: transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor()
    ])
    for x in ["train", "valid"]
}

image_datasets = {
    x: datasets.ImageFolder(root=os.path.join(data_dir, x),
                            transform=data_transform[x])
    for x in ["train", "valid"]
}

dataloader = {
    x: torch.utils.data.DataLoader(dataset=image_datasets[x],
                                   batch_size=16,
                                   shuffle=True)
    for x in ["train", "valid"]
}
# 查看图片样本
# x_example, y_example = next(iter(dataloader["train"]))  # x_example = 16*3*64*64
#
# index_classes = image_datasets["train"].class_to_idx
# print(index_classes) # 装载时候会根据独热编码进行数字化
#
# example_clasees = image_datasets["train"].classes
#
# img = torchvision.utils.make_grid(x_example)
# img = img.numpy().transpose([1, 2, 0])
# print([example_clasees[i] for i in y_example])
# plt.imshow(img)
# plt.show()
#
# exit(0)
# 加载model
model = model.Models()
model.cuda()

# 定义损失函数为 交叉熵损失函数 为什么用交叉熵不用等差 等差多用于回归
loss_f = torch.nn.CrossEntropyLoss()

# 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 训练10个回合
epoch_n = 10
time_open = time.time()

for epoch in range(epoch_n):
    print("Epoch{}/{}".format(epoch, epoch_n - 1))
    print("-" * 10)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training...")
            model.train(True)
        else:
            print("Validing...")
            model.train(False)

        running_loss = 0.0
        running_corrects = 0
        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            X, y = Variable(X.cuda()), Variable(y.cuda())
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)

            optimizer.zero_grad()
            loss = loss_f(y_pred, y)

            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.data.item()
            running_corrects += torch.sum(pred == y.data)

            if batch % 500 == 0 and phase == "train":
                print("Batch {},Train Loss:{:.4f},Train ACC:{:.4f}".format(batch, running_loss / batch,
                                                                           100 * running_corrects / (16 * batch)))

                epoch_loss = running_loss * 16 / len(image_datasets[phase])
                epoch_acc = 100 * running_corrects / len(image_datasets[phase])
                print("{} Loss:{:.4f} Acc:{:.4f}%".format(phase, epoch_loss, epoch_acc))

        time_end = time.time()-time_open
        print(time_end)
