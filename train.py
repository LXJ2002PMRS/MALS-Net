import os
from prefetch_generator import BackgroundGenerator
import torch
from datasets.datasets import MoonDetection3
from torch.utils.tensorboard import SummaryWriter
import time
import network
from collections import OrderedDict
from tqdm import tqdm
import torch.optim as optim
import sys
import torch.nn as nn
import torch.nn.functional as F
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 确保内容及时写入文件

    def flush(self):
        pass


def get_time_hhmmss(start_time, end_time):
    m, s = divmod(end_time - start_time, 60)
    h, m = divmod(m, 60)
    time_cout = "%02d:%02d:%02d" % (h, m, s)
    return time_cout

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt 是预测的概率
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()        
        else:
            return focal_loss


def train_net(train_data_path, val_data_path, epochs=100, batch_size=16, lr=0.0001):
    # 创建日志目录
    log_dir = "training_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # 设置日志文件名（包含时间戳）
    log_filename = os.path.join(log_dir, f"training_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    # 重定向标准输出到文件和终端
    sys.stdout = Logger(log_filename)
    
    # 加载数据集
    train_dataset = MoonDetection3(train_data_path)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_dataset = MoonDetection3(val_data_path)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # MALS-Net Models
    model = network.deeplabv3plus_ECAResNet50(
        num_classes=1,
        output_stride=16,        
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device=device)
    optimizer = optim.NAdam(model.parameters(), lr=lr,eps=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.96)
    criterion = FocalLoss(alpha=0.25, gamma=2, reduction='mean')

    # best loss统计， 初始化为正无穷
    best_accuracy = 0.0
    epoch_losses = []
    val_losses = []

    # 初始化准确率列表
    train_accuracies = []
    val_accuracies = []

    # TensorBoard记录器
    writer = SummaryWriter('log')
    for epoch in range(epochs):
        # train model
        epoch_loss = 0
        correct_train = 0
        total_train = 0
        model.train()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (image, dem, label) in BackgroundGenerator(pbar):

            feature = OrderedDict()
            optimizer.zero_grad()

            feature['image'] = image.to(device=device, dtype=torch.float32)
            feature['dem'] = dem.to(device=device, dtype=torch.float32)

            pred = model(feature)
            loss = criterion(pred, label.to(device=device, dtype=torch.float32))
            loss.backward()
            optimizer.step()

            # 计算准确率
            pred = torch.sigmoid(pred)  # 将预测结果通过sigmoid函数转换为概率
            predicted = (pred > 0.5).float().squeeze()  # 应用阈值并去除单维度
            correct_train += (predicted == label.to(device).float().squeeze()).sum().item()
            total_train += label.numel()

            if i % 10 == 0:
                pbar.set_description("loss:{}".format(loss))
            epoch_loss += loss.item()
        scheduler.step()
    
        # 打印信息
        current_lr = scheduler.get_last_lr()[0]        
        epoch_loss = epoch_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss:.10f}, LR: {current_lr:.10f}')
        print('Epoch:{} train Loss：{} train Accuracy：{:.4f}'.format(epoch, epoch_loss, train_accuracy))
        epoch_losses.append(epoch_loss)
        train_accuracies.append(train_accuracy)
        writer.add_scalar("train_epoch_loss", epoch_loss, epoch)
        writer.add_scalar("train_accuracy", train_accuracy, epoch)

        with torch.no_grad():
            model.eval()
            val_loss = 0
            correct_val = 0
            total_val = 0
            for i, (image_val, dem_val, label_val) in BackgroundGenerator(
                    tqdm(enumerate(val_loader), total=len(val_loader))):
                feature_val = OrderedDict()
                feature_val['image'] = image_val.to(device=device, dtype=torch.float32)
                feature_val['dem'] = dem_val.to(device=device, dtype=torch.float32)
                pred_val = model(feature_val)
                loss_val = criterion(pred_val, label_val.to(device=device, dtype=torch.float32))
                # 计算准确率
                pred_val = torch.sigmoid(pred_val)  # 将预测结果通过sigmoid函数转换为概率
                predicted_val = (pred_val > 0.5).float().squeeze()
                correct_val += (predicted_val == label_val.to(device).float().squeeze()).sum().item()
                total_val += label_val.numel()

                val_loss += loss_val.item()
            val_loss = val_loss / len(val_loader)

            val_accuracy = correct_val / total_val
            print('Epoch:{} val Loss：{} val Accuracy：{:.4f}'.format(epoch, val_loss, val_accuracy))
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            writer.add_scalar("val_epoch_loss", val_loss, epoch)
            writer.add_scalar("val_accuracy", val_accuracy, epoch)
        
        if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'utils/check.pth')
                print(f"Best model saved with accuracy: {best_accuracy:.4f}")
        
        import matplotlib.pyplot as plt
        plt.switch_backend('Agg')
        plt.figure()
        plt.plot(epoch_losses)
        plt.plot(val_losses)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'])
        plt.savefig(os.path.join("utils/loss.jpg"))

        plt.figure()
        plt.plot(train_accuracies)
        plt.plot(val_accuracies)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train accuracy', 'val accuracy'])
        plt.savefig(os.path.join("utils/acc.jpg"))
    writer.close()


if __name__ == '__main__':
    start_time = time.time()
    train_data_path = '../data/train'
    val_data_path = '../data/val'
   
    train_net(train_data_path, val_data_path)
    end_time = time.time()
    time_cout = get_time_hhmmss(start_time, end_time)
    print('time cost:', time_cout)
