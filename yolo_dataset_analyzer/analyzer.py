import enum
import os
import yaml
import cv2
import numpy as np
import pandas as pd
from PIL import Image


class DatasetAnalyzer:

    def __init__(self, yaml_path, progress_callback=None):
        self.progress_callback = progress_callback

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        base = os.path.dirname(yaml_path)

        self.train = self._resolve(base, data.get("train"))
        self.val = self._resolve(base, data.get("val"))
        self.test = self._resolve(base, data.get("test"))
        
        # Extract class names
        self.class_names = data.get("names", {})
        if isinstance(self.class_names, list):
            self.class_names = {i: name for i, name in enumerate(self.class_names)}

    def _resolve(self, base, value):
        """
        Resolve path(s) from YAML. Returns LIST of paths to support multiple directories.
        """
        if value is None:
            return []
        
        if isinstance(value, list):
            # Return ALL paths in the list
            return [os.path.join(base, v) for v in value]
        else:
            # Single path - return as list for consistency
            return [os.path.join(base, value)]

    def analyze_split(self, img_dirs):

        # label_dir = img_dir.replace("images", "labels")

        rows = []
        counts = []
        images = []

        for img_dir in img_dirs:
            if img_dir is None or not os.path.exists(img_dir):
                continue
            for root, _, files in os.walk(img_dir):
                for f in files:
                    if f.lower().endswith((".jpg", ".png", ".jpeg")):
                        images.append(os.path.join(root, f))

        total_images = len(images)

        for i, img_path in enumerate(images):

            if self.progress_callback and i % 1000 == 0:
                self.progress_callback(f"Processing {i}/{total_images} images")

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
            except Exception:
                continue

            label_path = img_path.replace("images","labels").rsplit(".",1)[0]+".txt"

            c=0

            if os.path.exists(label_path):

                with open(label_path) as f:

                    for line in f:

                        cl,xc,yc,bw,bh = map(float,line.split())

                        rows.append({
                            "class":int(cl),
                            "xc":xc,
                            "yc":yc,
                            "bw":bw,
                            "bh":bh,
                            "aspect":bw/bh if bh > 0 else 0,
                            "area":bw*bh,
                            "img_w":w,
                            "img_h":h
                        })

                        c+=1

            counts.append(c)

        return pd.DataFrame(rows), counts, images

    def run(self):
        if self.progress_callback:
            self.progress_callback("Analyzing training data...")

        train_df, train_counts, train_imgs = self.analyze_split(self.train)

        test_dir = self.test if self.test else self.val
        test_df, test_counts, test_imgs = self.analyze_split(test_dir)

        summary = {
            "train_images":len(train_imgs),
            "test_images":len(test_imgs),
            "train_objects":len(train_df),
            "test_objects":len(test_df),
            "classes":train_df["class"].nunique()
        }

        return train_df, test_df, train_counts, test_counts, train_imgs, summary, self.class_names