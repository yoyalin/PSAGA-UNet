
import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from medpy.metric.binary import hd95
# 计算f1值，用于评估模型的精确性和召回率
    # inputs：模型的输出（预测值）。
    # target：真实标签。
    # beta：控制精确率和召回率的权重。
    # smooth：避免除零错误的平滑项。
    # threshold：将预测值转化为二进制值的阈值（默认为 0.5）。
def f_score(inputs, target, beta=1, smooth = 1e-5, threhold = 0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)
        
    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c),-1)
    temp_target = target.view(n, -1, ct)

    #--------------------------------------------#
    #   计算dice系数
    #--------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[...,:-1] * temp_inputs, axis=[0,1])
    fp = torch.sum(temp_inputs                       , axis=[0,1]) - tp
    fn = torch.sum(temp_target[...,:-1]              , axis=[0,1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

# 计算混淆矩阵，a 是真实标签，b 是预测结果，n 是类别数。
# 设标签宽W，长H
def fast_hist(a, b, n):
    #--------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    #--------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    #--------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    #--------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)  

# 这些函数分别计算每个类别的 IoU、像素准确率、召回率和精确率，使用混淆矩阵 hist 作为输入。
def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1) 

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1) 

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1) 

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1) 

# 计算整个数据集的 mIoU 和其他指标。
    # 读取真实标签和预测结果的路径。
    # 逐张图片计算混淆矩阵，并累计。
    # 每 10 张图片输出当前的 mIoU、像素准确率和准确性。
def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, return_samples=True, metrics_list=None):  
    print('Num classes', num_classes)  
    #-----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    #-----------------------------------------#
    hist = np.zeros((num_classes, num_classes))
    sample_iou_dict = {} 
    hd95_scores = []
    
    # 确保 metrics_list 是列表，防止 None 报错
    if metrics_list is None:
        metrics_list = ["miou", "pa", "precision", "f1", "dice", "hd95"] # 默认全算

    #------------------------------------------------#
    #   获得验证集标签路径列表
    #------------------------------------------------#
    gt_imgs     = [join(gt_dir, x + ".png") for x in png_name_list]  
    pred_imgs   = [join(pred_dir, x + ".png") for x in png_name_list]  

    #------------------------------------------------#
    #   读取每一个（图片-标签）对
    #------------------------------------------------#
    for ind in range(len(gt_imgs)): 
        image_id = png_name_list[ind]
        #------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        #------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))  
        #------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        #------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))  

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):  
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            sample_iou_dict[image_id] = np.nan
            if 'hd95' in metrics_list:
                hd95_scores.append(np.nan)
            continue
            
        #------------------------------------------------#
        #   计算当前样本的混淆矩阵 (mIoU/Dice/PA的基础)
        #------------------------------------------------#
        sample_hist = fast_hist(label.flatten(), pred.flatten(), num_classes)
        
        #------------------------------------------------#
        #   计算当前样本的IoU
        #------------------------------------------------#
        iou_per_class = per_class_iu(sample_hist)
        sample_iou = np.nanmean(iou_per_class)
        sample_iou_dict[image_id] = sample_iou 

        #------------------------------------------------#
        #   累加全局混淆矩阵
        #------------------------------------------------#
        hist += sample_hist

        # ==========================================================
        #   [关键修改] HD95 计算控制开关
        #   只有在 metrics_list 包含 'hd95' 时才执行耗时计算
        # ==========================================================
        if 'hd95' in metrics_list:
            sample_hd95_per_class = []
            for c in range(1, num_classes): # 假设类别0是背景
                pred_c = (pred == c)
                label_c = (label == c)
                
                if np.sum(pred_c) > 0 and np.sum(label_c) > 0:
                    try:
                        # 调用 medpy 或自定义函数计算 hd95
                        sample_hd95_per_class.append(hd95(pred_c, label_c))
                    except Exception:
                        sample_hd95_per_class.append(np.nan)
                else:
                    sample_hd95_per_class.append(np.nan)

            if sample_hd95_per_class: 
                mean_sample_hd95 = np.nanmean(sample_hd95_per_class)
            else:
                mean_sample_hd95 = 0
            hd95_scores.append(mean_sample_hd95)

        # 每计算10张输出进度
        if name_classes is not None and ind > 0 and ind % 10 == 0: 
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind, len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            ))

    def per_class_dice(hist):
        dice = np.zeros(num_classes)
        for i in range(num_classes):
            gt_count = hist[i].sum()
            pred_count = hist[:, i].sum()
            intersection = hist[i, i]
            if gt_count + pred_count == 0:
                dice[i] = 0.0 
            else:
                dice[i] = 2.0 * intersection / (gt_count + pred_count)
        return dice

    #------------------------------------------------#
    #   计算全局指标
    #------------------------------------------------#
    IoUs        = per_class_iu(hist)
    PA_Recall   = per_class_PA_Recall(hist)
    Precision   = per_class_Precision(hist)
    accuracy    = per_Accuracy(hist)
    Dices       = per_class_dice(hist)
    
    # [关键修改] 如果没算 HD95，直接给 0
    if 'hd95' in metrics_list and len(hd95_scores) > 0:
        mean_hd95 = np.nanmean(hd95_scores)
    else:
        mean_hd95 = 0

    #------------------------------------------------#
    #   输出结果
    #------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                + '; Recall-' + str(round(PA_Recall[ind_class] * 100, 2))+ '; Precision-' 
                + str(round(Precision[ind_class] * 100, 2))
                + ':\tDice-' + str(round(Dices[ind_class] * 100, 2)))

    print(f'===> mIoU: {np.nanmean(IoUs) * 100:.2f}; mPA: {np.nanmean(PA_Recall) * 100:.2f}; Accuracy: {per_Accuracy(hist) * 100:.2f}; mDice: {np.nanmean(Dices) * 100:.2f}; HD95: {mean_hd95:.2f}')  
    
    # 保持返回值格式一致
    if return_samples:
        return np.array(hist, int), IoUs, PA_Recall, Precision, accuracy, sample_iou_dict, Dices, mean_hd95
    else:
        return np.array(hist, int), IoUs, PA_Recall, Precision, accuracy, Dices, mean_hd95
    
def adjust_axes(r, t, fig, axes):
    # 获取文本的窗口范围
    bb                  = t.get_window_extent(renderer=r)
    # 计算文本的宽度（以英寸为单位）
    text_width_inches   = bb.width / fig.dpi
    # 获取当前图形的宽度
    current_fig_width   = fig.get_figwidth()
    # 计算新的图形宽度
    new_fig_width       = current_fig_width + text_width_inches
    # 计算比例
    propotion           = new_fig_width / current_fig_width
    # 获取x轴的范围
    x_lim               = axes.get_xlim()
    # 设置新的x轴范围
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

# 绘制水平条形图，显示每个类别的指标（如 mIoU、召回率等）
def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size = 12, plt_show = True):
    # 获取当前图形
    fig     = plt.gcf() 
    # 获取当前坐标轴
    axes    = plt.gca()
    # 绘制水平条形图
    plt.barh(range(len(values)), values, color='royalblue')
    # 设置标题
    plt.title(plot_title, fontsize=tick_font_size + 2)
    # 设置x轴标签
    plt.xlabel(x_label, fontsize=tick_font_size)
    # 设置y轴刻度标签，标签内容为name_classes，字体大小为tick_font_size
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    # 获取画布的渲染器
    r = fig.canvas.get_renderer()
    
    # 遍历values列表
    for i, val in enumerate(values):
        # 将val转换为字符串，并在前面加一个空格
        str_val = " " + str(val) 
        # 如果val小于1.0，则将val格式化为两位小数，并在前面加一个空格
        if val < 1.0:
            str_val = " {0:.2f}".format(val)
        # 在val的位置绘制文本，文本内容为str_val，颜色为royalblue，垂直对齐方式为center，字体加粗
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        # 如果i等于values列表的最后一个元素的索引，则调整坐标轴
        if i == (len(values)-1):
            adjust_axes(r, t, fig, axes)

    # 调整布局
    fig.tight_layout()
    # 保存图像到output_path
    fig.savefig(output_path)
    # 如果plt_show为True，则显示图像
    if plt_show:
        plt.show()
    # 关闭图像
    plt.close()

# 展示和保存各类指标的结果和混淆矩阵
def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, tick_font_size = 12):
    # 绘制mIoU图
    draw_plot_func(IoUs, name_classes, "mIoU = {0:.2f}%".format(np.nanmean(IoUs)*100), "Intersection over Union", \
        os.path.join(miou_out_path, "mIoU.png"), tick_font_size = tick_font_size, plt_show = True)
    # 打印保存mIoU图的路径
    print("Save mIoU out to " + os.path.join(miou_out_path, "mIoU.png"))

    # 绘制mPA图
    draw_plot_func(PA_Recall, name_classes, "mPA = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Pixel Accuracy", \
        os.path.join(miou_out_path, "mPA.png"), tick_font_size = tick_font_size, plt_show = False)
    # 打印保存mPA图的路径
    print("Save mPA out to " + os.path.join(miou_out_path, "mPA.png"))
    
    # 绘制mRecall图
    draw_plot_func(PA_Recall, name_classes, "mRecall = {0:.2f}%".format(np.nanmean(PA_Recall)*100), "Recall", \
        os.path.join(miou_out_path, "Recall.png"), tick_font_size = tick_font_size, plt_show = False)
    # 打印保存mRecall图的路径
    print("Save Recall out to " + os.path.join(miou_out_path, "Recall.png"))

    # 绘制mPrecision图
    draw_plot_func(Precision, name_classes, "mPrecision = {0:.2f}%".format(np.nanmean(Precision)*100), "Precision", \
        os.path.join(miou_out_path, "Precision.png"), tick_font_size = tick_font_size, plt_show = False)
    # 打印保存mPrecision图的路径
    print("Save Precision out to " + os.path.join(miou_out_path, "Precision.png"))

    # 保存混淆矩阵
    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer          = csv.writer(f)
        writer_list     = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
            