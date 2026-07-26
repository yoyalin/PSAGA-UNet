import os
import matplotlib.pyplot as plt

def plot_and_save_lr(log_dir: str, dpi: int = 150) -> str:
    """
    绘制学习率变化曲线并保存到指定目录
    
    Args:
        log_dir: 日志保存目录路径
        lr_path: 学习率历史文件路径
        dpi: 图片分辨率，默认150
    
    Returns:
        保存生成的图片完整路径
    
    Raises:
        FileNotFoundError: 当学习率文件不存在时抛出
    """
    lr_path = os.path.join(log_dir, "lr_history.txt") 
    if not os.path.isfile(lr_path):
        return (f"学习率文件 {lr_path} 不存在")

    with open(lr_path, 'r') as f:
        lr_history = [float(line.strip()) for line in f]

    if log_dir != '':
        os.makedirs(log_dir, exist_ok=True)
        
        save_path = os.path.join(log_dir, "lr_curve.png")
    
    plt.figure(figsize=(10, 5), dpi=dpi)
    plt.plot(lr_history, 
             marker='.', 
             linestyle='-', 
             linewidth=2,
             color='#2B5B84',               alpha=0.8)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.title("Learning Rate Schedule", fontsize=14, pad=20)
    plt.xlabel("Training Epochs", fontsize=12)
    plt.ylabel("Learning Rate", fontsize=12)
    plt.xticks(range(0, len(lr_history), max(1, len(lr_history)//10)))      
    if log_dir != '':
        plt.savefig(save_path, bbox_inches='tight', facecolor='white')
        print('学习率图像保存完成')
    plt.close()      
    return save_path

if __name__ == "__main__":
    log_dir = r'logs\loss_2025_03_08_23_01_16'
    lr_path = os.path.join(log_dir, "lr_history.txt") 
    plot_and_save_lr(log_dir,lr_path)