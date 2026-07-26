import os
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal
import cv2
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.utils import cvtColor, preprocess_input, resize_image
from utils.utils_metrics_get25_2 import compute_mIoU
import time


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True, patience=5):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag
        self.patience = patience  
        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []
        self.best_val_loss = float('inf')          
        self.epochs_without_improvement = 0  
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            device = next(model.parameters()).device
            dummy_input = torch.randn(2, 3, input_shape[0], input_shape[1]).to(device)
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss=None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(f"Epoch: {epoch}, loss: {loss}\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(f"Epoch: {epoch}, val_loss: {val_loss}\n")

        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)

        self.loss_plot()

        if self.val_loss_flag and val_loss is not None:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0              
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping at epoch {epoch}. Best validation loss: {self.best_val_loss}")
                return True
        return False

    def loss_plot(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth=2, label='val loss')

        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15

            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle='--', linewidth=2,
                     label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle='--',
                         linewidth=2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")


class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
                 miou_out_path=".temp_miou_out", eval_flag=True, period=1, eval_batch_size=4,
                 eval_metrics=["miou","accuracy", "pa", "precision", "f1", "dice", "hd95"]):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.miou_out_path      = miou_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        self.eval_batch_size    = eval_batch_size
        self.eval_metrics       = [m.lower() for m in eval_metrics]         
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        
        self.epoches    = [0]
        self.mious      = [0]
        self.PA_recalls = [0]
        self.precisions = [0]
        self.accuracies = [0]
        self.F1_scores  = [0]
        self.Dice       = [0]
        self.HD95       = [0]

    def on_epoch_end(self, epoch, UnFreeze_Epoch, model_eval):
        if epoch >= 0 and (epoch % self.period == 0 or epoch == 1 or epoch == UnFreeze_Epoch-1) and self.eval_flag:
            self.net = model_eval
            gt_dir      = os.path.join(self.dataset_path, r"masks")
            pred_dir    = os.path.join(self.miou_out_path, r'detection-results')
            
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            else:
                shutil.rmtree(pred_dir)
                os.makedirs(pred_dir)
                
            print(f"开始生成验证集预测结果 (Batch Size: {self.eval_batch_size})...")
            
            num_imgs = len(self.image_ids)
            with tqdm(total=num_imgs, desc="Inference", unit="img") as pbar:
                for i in range(0, num_imgs, self.eval_batch_size):
                    batch_ids = self.image_ids[i : min(i + self.eval_batch_size, num_imgs)]
                    
                    imgs_tensor = []
                    meta_infos  = []                     
                    for image_id in batch_ids:
                        image_path  = os.path.join(self.dataset_path, "train/" + image_id + ".jpg")
                        image       = Image.open(image_path)
                        image       = cvtColor(image)
                        
                        ow, oh = image.size
                        
                        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
                        
                        data = preprocess_input(np.array(image_data, np.float32))
                        data = np.transpose(data, (2, 0, 1))
                        
                        imgs_tensor.append(data)
                        meta_infos.append((nw, nh, ow, oh))
                        
                    imgs_tensor = np.array(imgs_tensor)
                    with torch.no_grad():
                        images = torch.from_numpy(imgs_tensor)
                        if self.cuda:
                            images = images.cuda()
                            
                        outputs = self.net(images)
                        if isinstance(outputs, (list, tuple)):
                            outputs = outputs[0]
                        
                        outputs = F.softmax(outputs, dim=1)
                        
                        for idx, pr in enumerate(outputs):
                            nw, nh, ow, oh = meta_infos[idx]
                            
                            y1 = int((self.input_shape[0] - nh) // 2)
                            y2 = int((self.input_shape[0] - nh) // 2 + nh)
                            x1 = int((self.input_shape[1] - nw) // 2)
                            x2 = int((self.input_shape[1] - nw) // 2 + nw)
                            
                            pr_cropped = pr[:, y1:y2, x1:x2]
                            
                            pr_resized = F.interpolate(
                                pr_cropped.unsqueeze(0), 
                                size=(oh, ow), 
                                mode='bilinear', 
                                align_corners=False
                            )
                            
                            mask = torch.argmax(pr_resized, dim=1).squeeze(0)
                            
                            mask_np = mask.cpu().numpy().astype(np.uint8)
                            pred_img = Image.fromarray(mask_np)
                            pred_img.save(os.path.join(pred_dir, batch_ids[idx] + ".png"))
                            
                    pbar.update(len(batch_ids))

            print("预测完成，开始计算评估指标...")

            temp_miou = temp_PA_Recall = temp_Precision = temp_accuracy = \
            temp_F1 = temp_dice = temp_hd95 = 0 
            
            results = compute_mIoU(
                gt_dir, 
                pred_dir, 
                self.image_ids, 
                self.num_classes, 
                None, 
                metrics_list=self.eval_metrics 
            )
            
            _, IoUs, PA_Recall, Precision, accuracy, _, dice, raw_hd95 = results
            
            if "miou" in self.eval_metrics:
                temp_miou = np.nanmean(IoUs) * 100
            
            if "pa" in self.eval_metrics:
                temp_PA_Recall = np.nanmean(PA_Recall) * 100
                temp_accuracy = np.nanmean(accuracy) * 100

            if "precision" in self.eval_metrics:
                temp_Precision = np.nanmean(Precision) * 100

            if "f1" in self.eval_metrics:
                p, r = np.nanmean(Precision), np.nanmean(PA_Recall)
                temp_F1 = (2 * p * r / (p + r) * 100) if (p + r) > 0 else 0

            if "dice" in self.eval_metrics:
                temp_dice = np.nanmean(dice) * 100

            if "hd95" in self.eval_metrics:
                temp_hd95 = raw_hd95

            self.mious.append(temp_miou if "miou" in self.eval_metrics else self.mious[-1])
            self.PA_recalls.append(temp_PA_Recall if "pa" in self.eval_metrics else self.PA_recalls[-1])
            self.precisions.append(temp_Precision if "precision" in self.eval_metrics else self.precisions[-1])
            self.accuracies.append(temp_accuracy if "pa" in self.eval_metrics else self.accuracies[-1])
            self.F1_scores.append(temp_F1 if "f1" in self.eval_metrics else self.F1_scores[-1])
            self.Dice.append(temp_dice if "dice" in self.eval_metrics else self.Dice[-1])
            self.HD95.append(temp_hd95 if "hd95" in self.eval_metrics else self.HD95[-1])
            self.epoches.append(epoch)

            print("指标计算完成，正在记录日志...")
            metrics_to_log = {
                "miou":      ("epoch_miou.txt",      temp_miou),
                "pa":        ("epoch_PA_Recall.txt", temp_PA_Recall),
                "precision": ("epoch_Precision.txt", temp_Precision),
                "accuracy":  ("epoch_accuracy.txt",  temp_accuracy),
                "f1":        ("epoch_F1.txt",        temp_F1),
                "dice":      ("dice.txt",            temp_dice),
                "hd95":      ("HD95.txt",            temp_hd95)
            }

            for m_name, (f_name, val) in metrics_to_log.items():
                if m_name in self.eval_metrics:
                    with open(os.path.join(self.log_dir, f_name), 'a') as f:
                        f.write(f"Epoch: {epoch}, {m_name.upper()}: {val}\n")

            print(f">>> Epoch {epoch} 记录完毕。")

            for attempt in range(5):
                try:
                    if os.path.exists(pred_dir):
                        shutil.rmtree(pred_dir)
                    break
                except Exception as e:
                    time.sleep(0.5)
    
