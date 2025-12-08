import glob
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import cv2
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import math
import datetime
gdal.DontUseExceptions()

# read tif dataset
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "The file unable to open")

    tif_XSize = dataset.RasterXSize
    tif_YSize = dataset.RasterYSize
    band = dataset.GetRasterBand(1)
    tif_data = band.ReadAsArray(0, 0, tif_XSize, tif_YSize)
    tif_data_np = np.array(tif_data)
    del dataset
    return tif_data_np

def extract_connected_components(mask, min_component_size=1, connectivity=8):
    # 确保是二值图像
    binary_mask = (mask > 0).astype(np.uint8)
    
    # 使用OpenCV的连通对象分析
    num_components, labeled_mask = cv2.connectedComponents(binary_mask, connectivity=connectivity)
    
    components = []
    centroids = []
    bboxes = []
    
    for i in range(1, num_components):  # 跳过背景(0)
        component_mask = (labeled_mask == i)
        component_pixels = np.argwhere(component_mask)
        
        # 过滤小区域
        if len(component_pixels) >= min_component_size:
            components.append(component_pixels)
            
            # 计算质心
            centroid = np.mean(component_pixels, axis=0)
            centroids.append(centroid)
            
            # 计算边界框
            y_min, x_min = np.min(component_pixels, axis=0)
            y_max, x_max = np.max(component_pixels, axis=0)
            bboxes.append((y_min, x_min, y_max, x_max))
    
    return components, centroids, bboxes, labeled_mask

def calculate_component_iou(component1, component2):
    # 创建最小边界框
    all_coords = np.vstack([component1, component2])
    y_min, x_min = np.min(all_coords, axis=0)
    y_max, x_max = np.max(all_coords, axis=0)
    
    # 创建局部掩码
    height = int(y_max - y_min + 1)
    width = int(x_max - x_min + 1)
    
    mask1 = np.zeros((height, width), dtype=np.uint8)
    mask2 = np.zeros((height, width), dtype=np.uint8)
    
    # 填充掩码
    for y, x in component1:
        mask1[int(y - y_min), int(x - x_min)] = 1
    
    for y, x in component2:
        mask2[int(y - y_min), int(x - x_min)] = 1
    
    # 计算IoU
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    return intersection / union if union > 0 else 0

def hungarian_component_matching(label_components, predict_components, iou_threshold=0.5):
    if not label_components or not predict_components:
        return [], list(range(len(label_components))), list(range(len(predict_components)))
    
    # 构建成本矩阵（1 - IoU）
    cost_matrix = np.ones((len(label_components), len(predict_components)))
    
    for i, label_comp in enumerate(label_components):
        for j, predict_comp in enumerate(predict_components):
            iou = calculate_component_iou(label_comp, predict_comp)
            if iou >= iou_threshold:
                cost_matrix[i, j] = 1 - iou  # 成本为1-IoU
    
    # 匈牙利算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    matched_labels = set()
    matched_predicts = set()
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < (1 - iou_threshold):  # IoU > 阈值
            matches.append((i, j, 1 - cost_matrix[i, j]))  # 返回匹配对和IoU
            matched_labels.add(i)
            matched_predicts.add(j)
    
    # 找出未匹配的对象
    unmatched_labels = [i for i in range(len(label_components)) if i not in matched_labels]
    unmatched_predicts = [j for j in range(len(predict_components)) if j not in matched_predicts]
    
    return matches, unmatched_labels, unmatched_predicts

def calculate_spatial_metrics(label_components, predict_components, matches, image_shape):
    spatial_metrics = {}
    
    # 计算对象质心
    label_centroids = [np.mean(comp, axis=0) for comp in label_components]
    predict_centroids = [np.mean(comp, axis=0) for comp in predict_components]
    
    # 计算图像对角线长度（用于归一化）
    image_diagonal = math.sqrt(image_shape[0]**2 + image_shape[1]**2)
    
    if matches:
        # 计算匹配对象的质心距离
        centroid_distances = []
        for label_idx, predict_idx, iou in matches:
            dist = np.linalg.norm(label_centroids[label_idx] - predict_centroids[predict_idx])
            centroid_distances.append(dist)
        spatial_metrics['mean_normalized_centroid_distance'] = np.mean(centroid_distances) / image_diagonal
    
    return spatial_metrics

def calculate_compactness(component):
    if len(component) == 0:
        return 0
    
    # 创建对象掩码
    y_min, x_min = np.min(component, axis=0)
    y_max, x_max = np.max(component, axis=0)
    
    height = int(y_max - y_min + 1)
    width = int(x_max - x_min + 1)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    for y, x in component:
        mask[int(y - y_min), int(x - x_min)] = 1
    
    # 计算面积
    area = len(component)
    
    # 计算周长（使用轮廓检测）
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        perimeter = cv2.arcLength(contours[0], True)
        return perimeter**2 / area if area > 0 else 0
    else:
        return 0

def calculate_compactness_similarity(label_components, predict_components, matches):

    compactness_metrics = {}
    
    # 计算所有匹配对象对的紧凑度相似性
    if matches:
        compactness_similarities = []
        
        for label_idx, predict_idx in matches:
            label_comp = label_components[label_idx]
            predict_comp = predict_components[predict_idx]
            
            # 计算紧凑度（周长²/面积）
            label_compactness = calculate_compactness(label_comp)
            predict_compactness = calculate_compactness(predict_comp)
            
            # 计算紧凑度相似性
            if label_compactness > 0 and predict_compactness > 0:
                # 使用比值作为相似性度量，值越接近1表示形状越相似
                similarity = min(label_compactness, predict_compactness) / max(label_compactness, predict_compactness)
                compactness_similarities.append(similarity)
        
        if compactness_similarities:
            compactness_metrics['mean_compactness_similarity'] = np.mean(compactness_similarities) 
        else:
            compactness_metrics['mean_compactness_similarity'] = 0            
    else:
        compactness_metrics['mean_compactness_similarity'] = 0               
    
    return compactness_metrics

def comprehensive_component_assessment(label_mask, predict_mask, min_component_size=10):
    # 提取连通对象
    label_components = extract_connected_components(
        label_mask, min_component_size
    )
    predict_components = extract_connected_components(
        predict_mask, min_component_size
    )
    
    # 1. 对象匹配评估
    matches, unmatched_labels, unmatched_predicts = hungarian_component_matching(
        label_components, predict_components
    )
    
    # 对象级混淆矩阵
    component_tp = len(matches)
    component_fn = len(unmatched_labels)
    component_fp = len(unmatched_predicts)
    
    # 对象级指标
    component_precision = component_tp / (component_tp + component_fp) if (component_tp + component_fp) > 0 else 0
    component_recall = component_tp / (component_tp + component_fn) if (component_tp + component_fn) > 0 else 0
    component_f1 = 2 * component_precision * component_recall / (component_precision + component_recall) if (component_precision + component_recall) > 0 else 0
    
    # 平均匹配质量
    mean_match_iou = np.mean([iou for _, _, iou in matches]) if matches else 0
    
    matching_metrics = {
        'num_label_components': len(label_components),
        'num_predict_components': len(predict_components),
        'component_tp': component_tp,
        'component_fp': component_fp,
        'component_fn': component_fn,
        'component_precision': component_precision,
        'component_recall': component_recall,
        'component_f1': component_f1,
        'mean_match_iou': mean_match_iou,        
    }
    
    # 2. 空间分布评估
    spatial_metrics = calculate_spatial_metrics(
        label_components, predict_components, matches, label_mask.shape
    )
    
    # 3. 紧凑度相似性评估
    compactness_metrics = calculate_compactness_similarity(
        label_components, predict_components, matches
    )
    
    return {
        'matching_metrics': matching_metrics,
        'spatial_metrics': spatial_metrics,
        'compactness_metrics': compactness_metrics,
        'label_components': label_components,
        'predict_components': predict_components,
        'matches': matches
    }

def estimate(predict, label):
    predict_data = readTif(predict)
    label_data = readTif(label)
    
    # 像素级的混淆矩阵
    # TP(True Positive)
    TP = ((predict_data == 255) & (label_data == 1)).sum()
    # TN(True Negative)
    TN = ((predict_data == 0) & (label_data == 0)).sum()
    # FN(False Negative)
    FN = ((predict_data == 0) & (label_data == 1)).sum()
    # FP(False Positive)
    FP = ((predict_data == 255) & (label_data == 0)).sum()
    
    # 计算当前图片的像素级Precision
    current_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    
    # 综合对象评估
    component_assessment = comprehensive_component_assessment(label_data, predict_data)
    
    return TP, TN, FN, FP, component_assessment, current_precision

def write_to_file(file, content):
    """将内容同时写入文件和打印到控制台"""
    print(content)
    file.write(content + '\n')



if  __name__=='__main__':
    
    # 创建输出文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"evaluation_results_{timestamp}.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        # 写入文件头
        write_to_file(f, "=" * 80)
        write_to_file(f, "陡坎检测模型评估结果")
        write_to_file(f, f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        write_to_file(f, "=" * 80)
        write_to_file(f, "")
        
        TP_sum, TN_sum, FN_sum, FP_sum = 0, 0, 0, 0
        
        # 对象级指标累加
        total_matching_metrics = {
            'num_label_components': 0,
            'num_predict_components': 0,
            'component_tp': 0,
            'component_fp': 0,
            'component_fn': 0,
            'sum_match_iou': 0,
            'match_count': 0
        }
        
        # 空间和紧凑度指标
        spatial_distances = []
        compactness_similarities = []
        processed_images = 0
        
        test_predict_Path = glob.glob('utils/predict/*.tif')
        write_to_file(f, f"开始处理 {len(test_predict_Path)} 张预测图片...")
        write_to_file(f, "")
        
        for predict_Path in tqdm(test_predict_Path):
            label_Path = "../data/test/label/" + predict_Path.split('\\')[-1]
            TP, TN, FN, FP, component_assessment, current_precision = estimate(predict_Path, label_Path)
                       
            # 累加统计指标
            TP_sum += TP
            TN_sum += TN
            FN_sum += FN
            FP_sum += FP
            
            # 累加对象级指标
            matching_metrics = component_assessment['matching_metrics']
            total_matching_metrics['num_label_components'] += matching_metrics['num_label_components']
            total_matching_metrics['num_predict_components'] += matching_metrics['num_predict_components']
            total_matching_metrics['component_tp'] += matching_metrics['component_tp']
            total_matching_metrics['component_fp'] += matching_metrics['component_fp']
            total_matching_metrics['component_fn'] += matching_metrics['component_fn']
            total_matching_metrics['sum_match_iou'] += matching_metrics['mean_match_iou'] * matching_metrics['component_tp']
            total_matching_metrics['match_count'] += matching_metrics['component_tp']
            
            # 收集空间和紧凑度指标
            spatial_metrics = component_assessment['spatial_metrics']
            if 'mean_normalized_centroid_distance' in spatial_metrics:
                spatial_distances.append(spatial_metrics['mean_normalized_centroid_distance'])
            
            compactness_metrics = component_assessment['compactness_metrics']
            if 'mean_compactness_similarity' in compactness_metrics:
                compactness_similarities.append(compactness_metrics['mean_compactness_similarity'])
            
            processed_images += 1

        
        # 像素级指标
        Precision = TP_sum / (TP_sum + FP_sum) if (TP_sum + FP_sum) > 0 else 0
        Recall = TP_sum / (TP_sum + FN_sum) if (TP_sum + FN_sum) > 0 else 0
        F1 = (2 * Recall * Precision) / (Recall + Precision) if (Recall + Precision) > 0 else 0
        OA = (TP_sum + TN_sum) / (TP_sum + TN_sum + FP_sum + FN_sum) if (TP_sum + TN_sum + FP_sum + FN_sum) > 0 else 0
        MIoU = TP_sum / (TP_sum + FP_sum + FN_sum) if (TP_sum + FP_sum + FN_sum) > 0 else 0
        
        # 对象级指标
        component_precision = total_matching_metrics['component_tp'] / (total_matching_metrics['component_tp'] + total_matching_metrics['component_fp']) if (total_matching_metrics['component_tp'] + total_matching_metrics['component_fp']) > 0 else 0
        component_recall = total_matching_metrics['component_tp'] / (total_matching_metrics['component_tp'] + total_matching_metrics['component_fn']) if (total_matching_metrics['component_tp'] + total_matching_metrics['component_fn']) > 0 else 0
        component_f1 = 2 * component_precision * component_recall / (component_precision + component_recall) if (component_precision + component_recall) > 0 else 0
        mean_match_iou = total_matching_metrics['sum_match_iou'] / total_matching_metrics['match_count'] if total_matching_metrics['match_count'] > 0 else 0
        
        
        
        # 空间和紧凑度指标统计
        mean_spatial_distance = np.mean(spatial_distances) if spatial_distances else 0
        mean_compactness_similarity = np.mean(compactness_similarities) if compactness_similarities else 0
        
        write_to_file(f, "=" * 60)
        write_to_file(f, "像素级精度评估：")
        write_to_file(f, "=" * 60)
        write_to_file(f, " TP: %.3f" % (TP_sum / (TP_sum + TN_sum + FP_sum + FN_sum) * 100) + "%")
        write_to_file(f, " TN: %.3f" % (TN_sum / (TP_sum + TN_sum + FP_sum + FN_sum) * 100) + "%")
        write_to_file(f, " FN: %.3f" % (FN_sum / (TP_sum + TN_sum + FP_sum + FN_sum) * 100) + "%")
        write_to_file(f, " FP: %.3f" % (FP_sum / (TP_sum + TN_sum + FP_sum + FN_sum) * 100) + "%")
        write_to_file(f, " Precision: %.3f" % (Precision * 100) + "%")
        write_to_file(f, " Recall: %.3f" % (Recall * 100) + "%")
        write_to_file(f, " F1: %.3f" % (F1 * 100) + "%")
        write_to_file(f, " Overall Accuracy: %.3f" % (OA * 100) + "%")
        write_to_file(f, " IoU: %.3f" % (MIoU * 100) + "%")
        
        write_to_file(f, "")
        write_to_file(f, "=" * 60)
        write_to_file(f, "对象级精度评估：")
        write_to_file(f, "=" * 60)
        write_to_file(f, " 标签中陡坎总数: %d" % total_matching_metrics['num_label_components'])
        write_to_file(f, " 预测中陡坎总数: %d" % total_matching_metrics['num_predict_components'])
        write_to_file(f, " 对象级TP: %d" % total_matching_metrics['component_tp'])
        write_to_file(f, " 对象级FP: %d" % total_matching_metrics['component_fp'])
        write_to_file(f, " 对象级FN: %d" % total_matching_metrics['component_fn'])
        write_to_file(f, " 对象级Precision: %.3f" % (component_precision * 100) + "%")
        write_to_file(f, " 对象级Recall: %.3f" % (component_recall * 100) + "%")
        write_to_file(f, " 对象级F1: %.3f" % (component_f1 * 100) + "%")
        write_to_file(f, " 平均匹配IoU: %.3f" % (mean_match_iou * 100) + "%")
                
        write_to_file(f, "")
        write_to_file(f, "=" * 60)
        write_to_file(f, "空间与形状评估：")
        write_to_file(f, "=" * 60)
        write_to_file(f, " 平均归一化质心距离: %.3f" % (mean_spatial_distance * 100) + "%")
        write_to_file(f, " 平均紧凑度相似性: %.3f" % (mean_compactness_similarity * 100) + "%")
        
        write_to_file(f, "")
        write_to_file(f, "=" * 80)
        write_to_file(f, "评估完成！")
        write_to_file(f, f"结果已保存至: {output_filename}")
        write_to_file(f, "=" * 80)
    
    print(f"\n评估结果已保存至: {output_filename}")