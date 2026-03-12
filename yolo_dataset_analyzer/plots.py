import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def class_distribution_filtered(df, class_names=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(df) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    class_id = df['class'].iloc[0]
    class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"

    ax.bar([class_name], [len(df)], color='skyblue', edgecolor='black')
    ax.set_ylabel('Count')
    ax.set_title(f'Class Distribution - {class_name}')
    plt.tight_layout()
    return fig

def class_distribution(df, class_names=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    counts = df["class"].value_counts().sort_index()

    if class_names:
        labels = [class_names.get(i, f"Class {i}") for i in counts.index]
        ax.bar(range(len(counts)), counts.values)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
    else:
        ax.bar(counts.index, counts.values)
        ax.set_xlabel("Class")

    ax.set_ylabel("Count")
    ax.set_title("Class Distribution")
    plt.tight_layout()
    return fig

def objects_per_image(counts):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(counts, bins=30)
    ax.set_title("Objects per Image")
    plt.tight_layout()
    return fig

def bbox_size(df):
    fig, ax = plt.subplots(figsize=(10, 6))

    if len(df) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center')
        return fig

    IMG_WIDTH = 1280
    IMG_HEIGHT = 720

    width_pixels = df["bw"] * IMG_WIDTH
    height_pixels = df["bh"] * IMG_HEIGHT

    categories = ['0-15', '15-30', '30-50', '50-150', '150+']

    width_counts = [
        sum((width_pixels >= 0) & (width_pixels < 15)),
        sum((width_pixels >= 15) & (width_pixels < 30)),
        sum((width_pixels >= 30) & (width_pixels < 50)),
        sum((width_pixels >= 50) & (width_pixels < 150)),
        sum(width_pixels >= 150)
    ]

    height_counts = [
        sum((height_pixels >= 0) & (height_pixels < 15)),
        sum((height_pixels >= 15) & (height_pixels < 30)),
        sum((height_pixels >= 30) & (height_pixels < 50)),
        sum((height_pixels >= 50) & (height_pixels < 150)),
        sum(height_pixels >= 150)
    ]

    x = range(len(categories))
    width = 0.35

    ax.bar([i - width/2 for i in x], width_counts, width, label='Width', color='skyblue', edgecolor='black')
    ax.bar([i + width/2 for i in x], height_counts, width, label='Height', color='lightcoral', edgecolor='black')

    ax.set_xlabel('Size Categories (pixels)')
    ax.set_ylabel('Count')
    ax.set_title('BBox Size Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    plt.tight_layout()
    return fig

def ratio_analysis(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(df["aspect"], bins=40)
    ax.set_title("Aspect Ratio")
    plt.tight_layout()
    return fig

def spatial_distribution(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    heatmap, _, _ = np.histogram2d(df["xc"], df["yc"], bins=40)
    ax.imshow(heatmap)
    ax.set_title("Spatial Distribution")
    plt.tight_layout()
    return fig

def resolution_analysis(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["img_w"], df["img_h"], s=5)
    ax.set_title("Resolution Distribution")
    plt.tight_layout()
    return fig