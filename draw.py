import re
import matplotlib.pyplot as plt
import pandas as pd

# 读取文件
file_path = 'practice record.txt'  # 你的日志文件路径
with open(file_path, 'r') as f:
    data = f.read()

# 分段：按'///'划分
sections = data.split('///')

# 定义提取函数
def extract_times(section_text, pattern, label_name):
    times = re.findall(pattern, section_text)
    return [float(t) for t in times], label_name

# 配置每个部分提取的正则表达式
extract_config = {
    'YOLOv8': [
        (r'([\d\.]+)ms preprocess', 'Preprocess'),
        (r'([\d\.]+)ms inference', 'Inference'),
        (r'([\d\.]+)ms postprocess per image', 'Postprocess')
    ],
    'Depth Segmentation': [
        (r'depth_image_callback took ([\d\.]+) seconds', 'Depth Callback'),
        (r'apply_cumulative_hist_depth_filter took ([\d\.]+) seconds', 'Depth Filter'),
        (r'inference_callback took ([\d\.]+) seconds', 'Inference Callback')
    ],
    'EKF': [
        (r'depth_image_callback took ([\d\.]+) seconds', 'Depth Callback'),
        (r'apply_cumulative_hist_depth_filter took ([\d\.]+) seconds', 'Depth Filter'),
        (r'inference_callback took ([\d\.]+) seconds', 'Inference Callback')
    ],
    'MFE Pose': [
        (r'depth_image_callback took ([\d\.]+) seconds', 'Depth Callback'),
        (r'apply_cumulative_hist_depth_filter took ([\d\.]+) seconds', 'Depth Filter'),
        (r'estimate_manhattan_frame took ([\d\.]+) seconds', 'Estimate Manhattan'),
        (r'inference_callback took ([\d\.]+) seconds', 'Inference Callback')
    ]
}

# 提取所有部分时间
results = {}

for name, config in zip(['YOLOv8', 'Depth Segmentation', 'EKF', 'MFE Pose'], sections):
    section_result = {}
    for pattern, label in extract_config[name]:
        times, label_name = extract_times(config, pattern, label)
        section_result[label_name] = times
    results[name] = section_result

# 绘制函数
def plot_section_times(section_name, time_dict):
    plt.figure(figsize=(10, 6))
    for label, times in time_dict.items():
        plt.plot(times, label=label)
    plt.title(f'{section_name} Timing per Frame')
    plt.xlabel('Frame Index')
    plt.ylabel('Time (s)' if 'seconds' in file_path else 'Time (ms)')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制所有部分的曲线
for section, time_data in results.items():
    plot_section_times(section, time_data)

# 计算统计数据
for section, time_data in results.items():
    print(f'===== {section} =====')
    for label, times in time_data.items():
        if times:
            series = pd.Series(times)
            print(f'[{label}] Mean: {series.mean():.4f}, Max: {series.max():.4f}, Min: {series.min():.4f}, Std: {series.std():.4f}')
    print('\n')
