"""
Codi utilitzat per analitzar els resultats del millor model.

"""


import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score

# --- RUTES ---
labels_folder      = "runs/detect/val4/labels/"
images_folder      = "dataset/images/test/"
gt_labels_folder   = "dataset/labels/test/"
output_dir         = "D:/fotos/plots/"
os.makedirs(output_dir, exist_ok=True)

iou_threshold = 0.5
heatmap_size  = 50

# --- FUNCIONS ---
def compute_iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def save_plot(fig, name):
    fig.savefig(os.path.join(output_dir, f"{name}.png"), dpi=150)
    plt.close(fig)


def plot_and_save(data, title, name, kind='line', **kwargs):
    fig, ax = plt.subplots(figsize=(8, 5))
    if kind == 'line':
        data.plot(ax=ax, marker=kwargs.get('marker', 'o'))
    elif kind == 'hist':
        sns.histplot(data=data, ax=ax, **kwargs)
    elif kind == 'heatmap':
        sns.heatmap(data, ax=ax, cmap=kwargs.get('cmap', 'Reds'))
    elif kind == 'box':
        sns.boxplot(data=data, ax=ax, **kwargs)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    save_plot(fig, name)


def compute_ap_for_bin(df_bin):
    if df_bin.empty:
        return np.nan
    y_true  = df_bin['is_tp'].astype(int)
    y_score = df_bin['conf']
    if y_true.sum() == 0:
        return 0.0
    return average_precision_score(y_true, y_score)

records = []

# --- PREDICCIONS i TP/FP ---
for fname in os.listdir(labels_folder):
    if not fname.endswith('.txt'):
        print("no trobat")
        continue
    image_id  = os.path.splitext(fname)[0]
    pred_path = os.path.join(labels_folder, fname)
    gt_path   = os.path.join(gt_labels_folder, image_id + '.txt')
    img       = cv2.imread(os.path.join(images_folder, image_id + '.jpg'))
    if img is None:
        print("no trobat image")

        continue
    h, w = img.shape[:2]

    # GT boxes
    gt_boxes = []
    if os.path.exists(gt_path):
        with open(gt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h
                gt_boxes.append([x1, y1, x2, y2])
    matched_gt = set()
    with open(pred_path) as f:
        ii=0
        for line in f:
            ii+=1
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            _, cx, cy, bw, bh, conf = map(float, parts)
            # coords
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            bbox = [x1, y1, x2, y2]

            # contrast i area
            crop = img[int(y1):int(y2), int(x1):int(x2)]
            lesion_mean     = crop.mean() if crop.size else 0
            margin = img[max(0, int(y1-5)):min(h, int(y2+5)), max(0, int(x1-5)):min(w, int(x2+5))]
            background_mean = margin.mean() if margin.size else 0
            contrast = abs(lesion_mean - background_mean)
            area     = bw * bh

            # Troba millor IOU
            best_iou, best_idx = 0, -1
            for i, gt in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = compute_iou(bbox, gt)
                if iou > best_iou:
                    best_iou, best_idx = iou, i

            is_tp = best_iou >= iou_threshold
            if is_tp:
                matched_gt.add(best_idx)
            records.append({
                'image': image_id,
                'conf': conf,
                'area': area,
                'contrast': contrast,
                'cx': cx,
                'cy': cy,
                'iou': best_iou,
                'is_tp': is_tp,
                'is_fn': False
            })


# --- FALS NEGATIUS (FN) ---
for fname in os.listdir(gt_labels_folder):
    image_id = os.path.splitext(fname)[0]
    gt_path  = os.path.join(gt_labels_folder, fname)
    img      = cv2.imread(os.path.join(images_folder, image_id + '.jpg'))
    if img is None:
        continue
    h, w = img.shape[:2]

    gt_boxes = []
    with open(gt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, cx, cy, bw, bh = map(float, parts)
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            gt_boxes.append((cx, cy, bw, bh, x1, y1, x2, y2))

    preds = []
    pfile = os.path.join(labels_folder, image_id + '.txt')
    if os.path.exists(pfile):
        with open(pfile) as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                if len(parts) != 6:
                    continue
                _, cx_p, cy_p, bw_p, bh_p, _ = parts
                x1 = (cx_p - bw_p/2) * w
                y1 = (cy_p - bh_p/2) * h
                x2 = (cx_p + bw_p/2) * w
                y2 = (cy_p + bh_p/2) * h
                preds.append([x1, y1, x2, y2])

    for cx, cy, bw, bh, x1_gt, y1_gt, x2_gt, y2_gt in gt_boxes:
        if not any(compute_iou([x1_gt, y1_gt, x2_gt, y2_gt], p) >= iou_threshold for p in preds):
            crop = img[int(y1_gt):int(y2_gt), int(x1_gt):int(x2_gt)]
            lesion_mean = crop.mean() if crop.size else 0
            margin = img[max(0, int(y1_gt-5)):min(h, int(y2_gt+5)), max(0, int(x1_gt-5)):min(w, int(x2_gt+5))]
            background_mean = margin.mean() if margin.size else 0
            contrast = abs(lesion_mean - background_mean)
            area     = bw * bh
            records.append({
                'image': image_id,
                'conf': 0.0,
                'area': area,
                'contrast': contrast,
                'cx': cx,
                'cy': cy,
                'iou': 0.0,
                'is_tp': False,
                'is_fn': True
            })

# --- CREACIÓ DataFrames i Etiquetatge ---
df = pd.DataFrame(records).dropna(subset=['cx','cy','area','contrast'])
df['type'] = df.apply(lambda r: 'TP' if r['is_tp'] else ('FN' if r['is_fn'] else 'FP'), axis=1)

# --- BINNING en tot df ---
df['area_bin']     = pd.qcut(df['area'],     q=10, duplicates='drop')
df['contrast_bin'] = pd.qcut(df['contrast'], q=10, duplicates='drop')
# Nomes prediccions amb confianca 0.5
df_pred = df[df['conf'] > 0.5].copy()


# --- mAP ---
map_area = df_pred.groupby(df_pred['area_bin']).apply(compute_ap_for_bin)
plot_and_save(map_area, 'mAP@0.5 segons mida', 'map_area')
map_contrast = df_pred.groupby(df_pred['contrast_bin']).apply(compute_ap_for_bin)
plot_and_save(map_contrast, 'mAP@0.5 segons contrast', 'map_contrast', marker='s')

# --- Precision-Recall global ---
y_true = df_pred['is_tp'].astype(int)
y_score = df_pred['conf']
precision, recall, _ = precision_recall_curve(y_true, y_score)
fig, ax = plt.subplots(figsize=(7,5))
ax.plot(recall, precision, marker='.')
ax.set_title('Corba Precision-Recall')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.grid(True)
plt.tight_layout()
save_plot(fig, 'precision_recall')

# --- Histogrames TP/FP i FN ---
df_hist = df[(df['type'] != 'FP') | (df['conf'] >= 0.5)].copy()


plot_and_save(df_hist,      'Distribució mida (TP/FP/FN)', 'hist_area_all',    kind='hist',
              x='area', hue='type', bins=20, multiple='stack')
plot_and_save(df_pred, 'Distribució mida (TP/FP)',     'hist_area_preds',  kind='hist',
              x='area', hue='type', bins=20, multiple='stack')
plot_and_save(df_hist,      'Distribució contrast (TP/FP/FN)', 'hist_contrast_all',   kind='hist',
              x='contrast', hue='type', bins=20, multiple='stack')
plot_and_save(df_pred, 'Distribució contrast (TP/FP)',     'hist_contrast_preds', kind='hist',
              x='contrast', hue='type', bins=20, multiple='stack')

# --- Heatmaps per tipus ---
for key, cmap in [('FP','Reds'), ('FN','Blues')]:
    heat = np.zeros((heatmap_size, heatmap_size))
    for _, row in df[df['type']==key].iterrows():
        xi = int(row['cx'] * (heatmap_size-1))
        yi = int(row['cy'] * (heatmap_size-1))
        heat[yi, xi] += 1
    plot_and_save(heat, f'Heatmap {key}', f'heatmap_{key.lower()}', kind='heatmap', cmap=cmap)

# --- Confiança ---
plot_and_save(df, 'Confiança per tipus', 'box_conf', kind='box', x='type', y='conf')
plot_and_save(df, 'Distribució confiança', 'hist_conf', kind='hist', x='conf', bins=20, kde=True)
# --- Llindar de confiança ---
conf_threshold = 0.5

# --- Recall per bins (només TP amb conf≥thr, ) ---
rec_df = df[((df['type'] == 'TP') & (df['conf'] >= conf_threshold))
            | (df['type'] == 'FN')].copy()

recall_by_area = rec_df.groupby('area_bin') \
    .apply(lambda g: g['is_tp'].sum() / len(g) if len(g) > 0 else np.nan)
plot_and_save(recall_by_area, 'Recall segons mida (conf ≥ 0.5)', 'recall_area')

recall_by_contrast = rec_df.groupby('contrast_bin') \
    .apply(lambda g: g['is_tp'].sum() / len(g) if len(g) > 0 else np.nan)
plot_and_save(recall_by_contrast, 'Recall segons contrast (conf ≥ 0.5)', 'recall_contrast', marker='s')


# --- Precisió per bins (només TP/FP amb conf≥0.5) ---
pr_df = df[((df['type'] == 'TP') | (df['type'] == 'FP'))
           & (df['conf'] >= conf_threshold)].copy()

precision_by_area = pr_df.groupby('area_bin') \
    .apply(lambda g: g['is_tp'].sum() / len(g) if len(g) > 0 else np.nan)
plot_and_save(precision_by_area, 'Precisió segons mida (conf ≥ 0.5)', 'precision_area')

precision_by_contrast = pr_df.groupby('contrast_bin') \
    .apply(lambda g: g['is_tp'].sum() / len(g) if len(g) > 0 else np.nan)
plot_and_save(precision_by_contrast, 'Precisió segons contrast (conf ≥ 0.5)', 'precision_contrast', marker='s')
