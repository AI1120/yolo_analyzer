import numpy as np
from sklearn.cluster import KMeans


def class_imbalance(df, class_names=None):

    counts = df["class"].value_counts().sort_index()
    
    # Create readable output with class names
    if class_names:
        readable_counts = "\n".join([f"{class_names.get(i, f'Class {i}')}: {count}" 
                                    for i, count in counts.items()])
    else:
        readable_counts = "\n".join([f"Class {i}: {count}" for i, count in counts.items()])

    ratio = counts.max() / counts.min() if len(counts) > 1 else 1.0

    return readable_counts, ratio


def object_size_stats(df):

    area=df["area"]

    small=sum(area<0.01)
    medium=sum((area>=0.01)&(area<0.05))
    large=sum(area>=0.05)

    return small,medium,large


def anchor_boxes(df, sample_size=50000):

    boxes=df[["bw","bh"]].values

    if len(boxes) > sample_size:
        # Random sampling without replacement
        indices = np.random.choice(len(boxes), sample_size, replace=False)
        boxes = boxes[indices]

    km=KMeans(n_clusters=9).fit(boxes)

    return km.cluster_centers_


def health_score(summary, ratio):

    score = 100

    if ratio > 10:
        score -= 20

    if summary["classes"] < 2:
        score -= 20

    return max(score, 0)