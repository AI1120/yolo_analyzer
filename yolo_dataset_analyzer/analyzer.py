import os
import yaml
import cv2
import numpy as np
import pandas as pd


class DatasetAnalyzer:

    def __init__(self, yaml_path):

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

        if value is None:
            return None

        if isinstance(value, list):
            value = value[0]

        return os.path.join(base, value)

    def analyze_split(self, img_dir):

        label_dir = img_dir.replace("images", "labels")

        rows = []
        counts = []
        images = []

        for root, _, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith((".jpg",".png",".jpeg")):
                    images.append(os.path.join(root,f))

        for img_path in images:

            img = cv2.imread(img_path)
            if img is None:
                continue

            h,w = img.shape[:2]

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
                            "aspect":bw/bh,
                            "area":bw*bh,
                            "img_w":w,
                            "img_h":h
                        })

                        c+=1

            counts.append(c)

        return pd.DataFrame(rows), counts, images

    def run(self):

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