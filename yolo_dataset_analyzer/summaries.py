def generate_class_summary(train_df, test_df, class_names):
    train_counts = train_df["class"].value_counts().sort_index()
    test_counts = test_df["class"].value_counts().sort_index()
    total_train = len(train_df)
    total_test = len(test_df)

    summary = "CLASS DISTRIBUTION ANALYSIS\n" + "="*50 + "\n\n"
    summary += f"Total Training Objects: {total_train}\n"
    summary += f"Total Test Objects: {total_test}\n"
    summary += f"Number of Classes: {len(train_counts)}\n\n"
    summary += "Per-Class Breakdown:\n" + "-"*30 + "\n"

    for class_id in train_counts.index:
        class_name = class_names.get(class_id, f"Class {class_id}") if class_names else f"Class {class_id}"
        train_count = train_counts.get(class_id, 0)
        test_count = test_counts.get(class_id, 0)
        train_pct = (train_count / total_train * 100) if total_train > 0 else 0
        test_pct = (test_count / total_test * 100) if total_test > 0 else 0
        
        summary += f"{class_name}:\n"
        summary += f"  Train: {train_count} ({train_pct:.1f}%)\n"
        summary += f"  Test:  {test_count} ({test_pct:.1f}%)\n\n"

    return summary

def generate_objects_summary(train_counts, test_counts):
    import numpy as np
    train_avg = np.mean(train_counts)
    test_avg = np.mean(test_counts)
    train_std = np.std(train_counts)
    test_std = np.std(test_counts)

    summary = "OBJECTS PER IMAGE ANALYSIS\n" + "="*50 + "\n\n"
    summary += "Training Set:\n"
    summary += f"  Average objects/image: {train_avg:.2f}\n"
    summary += f"  Standard deviation: {train_std:.2f}\n"
    summary += f"  Min objects/image: {min(train_counts)}\n"
    summary += f"  Max objects/image: {max(train_counts)}\n\n"
    summary += "Test Set:\n"
    summary += f"  Average objects/image: {test_avg:.2f}\n"
    summary += f"  Standard deviation: {test_std:.2f}\n"
    summary += f"  Min objects/image: {min(test_counts)}\n"
    summary += f"  Max objects/image: {max(test_counts)}\n\n"

    if train_avg < 1.5:
        summary += "⚠️  Low object density - consider data augmentation\n"
    elif train_avg > 10:
        summary += "⚠️  High object density - may impact detection performance\n"
    else:
        summary += "✅ Good object density for training\n"

    return summary

def generate_bbox_summary(train_df, test_df):
    import numpy as np
    IMG_WIDTH = 1280
    IMG_HEIGHT = 720

    train_w_px = train_df["bw"] * IMG_WIDTH
    train_h_px = train_df["bh"] * IMG_HEIGHT
    test_w_px = test_df["bw"] * IMG_WIDTH
    test_h_px = test_df["bh"] * IMG_HEIGHT

    summary = "BOUNDING BOX SIZE ANALYSIS\n" + "="*50 + "\n\n"
    summary += "Resolution: 1280x720 pixels\n\n"
    summary += "Training Set:\n"
    summary += f"  Average width: {np.mean(train_w_px):.1f}px\n"
    summary += f"  Average height: {np.mean(train_h_px):.1f}px\n"
    summary += f"  Width range: {train_w_px.min():.1f} - {train_w_px.max():.1f}px\n"
    summary += f"  Height range: {train_h_px.min():.1f} - {train_h_px.max():.1f}px\n\n"
    summary += "Test Set:\n"
    summary += f"  Average width: {np.mean(test_w_px):.1f}px\n"
    summary += f"  Average height: {np.mean(test_h_px):.1f}px\n"
    summary += f"  Width range: {test_w_px.min():.1f} - {test_w_px.max():.1f}px\n"
    summary += f"  Height range: {test_h_px.min():.1f} - {test_h_px.max():.1f}px\n\n"

    def categorize_sizes(widths, heights):
        tiny = sum((widths < 15) | (heights < 15))
        small = sum(((widths >= 15) & (widths < 30)) | ((heights >= 15) & (heights < 30)))
        medium = sum(((widths >= 30) & (widths < 50)) | ((heights >= 30) & (heights < 50)))
        large = sum(((widths >= 50) & (widths < 150)) | ((heights >= 50) & (heights < 150)))
        xlarge = sum((widths >= 150) | (heights >= 150))
        return tiny, small, medium, large, xlarge

    tiny_train, small_train, medium_train, large_train, xlarge_train = categorize_sizes(train_w_px, train_h_px)

    summary += "Object Size Distribution (Training):\n"
    summary += f"  Tiny (0-15px): {tiny_train}\n"
    summary += f"  Small (15-30px): {small_train}\n"
    summary += f"  Medium (30-50px): {medium_train}\n"
    summary += f"  Large (50-150px): {large_train}\n"
    summary += f"  X-Large (150px+): {xlarge_train}\n\n"

    if tiny_train > len(train_df) * 0.3:
        summary += "⚠️  Many tiny objects - may be hard to detect\n"
    if xlarge_train > len(train_df) * 0.5:
        summary += "⚠️  Many large objects - consider data augmentation\n"
    if medium_train + large_train > len(train_df) * 0.6:
        summary += "✅ Good size distribution for detection\n"

    return summary

def generate_ratio_summary(train_df, test_df):
    import numpy as np
    train_ratios = train_df["aspect"]
    test_ratios = test_df["aspect"]

    summary = "ASPECT RATIO ANALYSIS\n" + "="*50 + "\n\n"
    summary += "Training Set:\n"
    summary += f"  Average aspect ratio: {np.mean(train_ratios):.2f}\n"
    summary += f"  Median aspect ratio: {np.median(train_ratios):.2f}\n"
    summary += f"  Range: {train_ratios.min():.2f} - {train_ratios.max():.2f}\n\n"
    summary += "Test Set:\n"
    summary += f"  Average aspect ratio: {np.mean(test_ratios):.2f}\n"
    summary += f"  Median aspect ratio: {np.median(test_ratios):.2f}\n"
    summary += f"  Range: {test_ratios.min():.2f} - {test_ratios.max():.2f}\n\n"

    square = sum((train_ratios >= 0.9) & (train_ratios <= 1.1))
    wide = sum(train_ratios > 1.5)
    tall = sum(train_ratios < 0.67)

    summary += "Shape Distribution (Training):\n"
    summary += f"  Square-ish (0.9-1.1): {square}\n"
    summary += f"  Wide (>1.5): {wide}\n"
    summary += f"  Tall (<0.67): {tall}\n"

    return summary

def generate_spatial_summary(train_df, test_df):
    import numpy as np
    summary = "SPATIAL DISTRIBUTION ANALYSIS\n" + "="*50 + "\n\n"

    train_x_center = np.mean(train_df["xc"])
    train_y_center = np.mean(train_df["yc"])
    test_x_center = np.mean(test_df["xc"])
    test_y_center = np.mean(test_df["yc"])

    summary += "Training Set Centers:\n"
    summary += f"  Average X position: {train_x_center:.3f}\n"
    summary += f"  Average Y position: {train_y_center:.3f}\n\n"
    summary += "Test Set Centers:\n"
    summary += f"  Average X position: {test_x_center:.3f}\n"
    summary += f"  Average Y position: {test_y_center:.3f}\n\n"

    if abs(train_x_center - 0.5) > 0.1:
        summary += "⚠️  Horizontal bias detected in object positions\n"
    if abs(train_y_center - 0.5) > 0.1:
        summary += "⚠️  Vertical bias detected in object positions\n"
    if abs(train_x_center - 0.5) <= 0.1 and abs(train_y_center - 0.5) <= 0.1:
        summary += "✅ Objects are well-distributed spatially\n"

    return summary

def generate_resolution_summary(train_df, test_df):
    import numpy as np
    train_resolutions = list(zip(train_df["img_w"], train_df["img_h"]))
    test_resolutions = list(zip(test_df["img_w"], test_df["img_h"]))

    train_unique = len(set(train_resolutions))
    test_unique = len(set(test_resolutions))

    summary = "RESOLUTION ANALYSIS\n" + "="*50 + "\n\n"
    summary += "Training Set:\n"
    summary += f"  Unique resolutions: {train_unique}\n"
    summary += f"  Average width: {np.mean(train_df['img_w']):.0f}px\n"
    summary += f"  Average height: {np.mean(train_df['img_h']):.0f}px\n"
    summary += f"  Width range: {train_df['img_w'].min():.0f} - {train_df['img_w'].max():.0f}px\n"
    summary += f"  Height range: {train_df['img_h'].min():.0f} - {train_df['img_h'].max():.0f}px\n\n"
    summary += "Test Set:\n"
    summary += f"  Unique resolutions: {test_unique}\n"
    summary += f"  Average width: {np.mean(test_df['img_w']):.0f}px\n"
    summary += f"  Average height: {np.mean(test_df['img_h']):.0f}px\n"
    summary += f"  Width range: {test_df['img_w'].min():.0f} - {test_df['img_w'].max():.0f}px\n"
    summary += f"  Height range: {test_df['img_h'].min():.0f} - {test_df['img_h'].max():.0f}px\n\n"

    if train_unique == 1:
        summary += "✅ Consistent resolution across all images\n"
    elif train_unique < 5:
        summary += "✅ Low resolution variance - good for training\n"
    else:
        summary += "⚠️  High resolution variance - consider resizing\n"

    return summary