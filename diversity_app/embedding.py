import os
import numpy as np
import torch
import sklearn
import yaml
import open_clip
import random
import multiprocessing as mp
from multiprocessing import Pool
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from ultralytics import YOLO
import pickle
from datetime import datetime
try:
    from safetensors.torch import load_file as load_safetensors
except:
    load_safetensors = None

try:
    import umap
except ImportError:
    umap = None

try:
    from openTSNE import TSNE as OpenTSNE
except ImportError:
    OpenTSNE = None


# Check sklearn version for correct parameter
SKLEARN_VERSION = tuple(map(int, sklearn.__version__.split('.')[:2]))
TSNE_ITER_PARAM = 'max_iter' if SKLEARN_VERSION >= (1, 2) else 'n_iter'

# ============================================================
# WORKER THREAD
# ============================================================

class EmbeddingWorker(QThread):
    progress_signal = pyqtSignal(str, int)
    finished_signal = pyqtSignal(object, object, object, object, object)  # coords, paths, class_ids, image_dir, class_names
    error_signal = pyqtSignal(str)

    def __init__(self, mode, yolo_weights, yolo_model_name, yaml_path, clip_model_name, sampling_enabled, sample_size, clip_weights_path=None, viz_method='tsne', use_multiprocessing=False, num_processes=None, load_from_file=None):
        super().__init__()
        self.mode = mode
        self.yolo_weights = yolo_weights
        self.yolo_model_name = yolo_model_name
        self.yaml_path = yaml_path
        self.clip_model_name = clip_model_name
        self.sampling_enabled = sampling_enabled
        self.sample_size = sample_size
        self.clip_weights_path = clip_weights_path
        self.viz_method = viz_method.lower()
        self.class_names = {}  # Store class names from YAML
        self.use_multiprocessing = use_multiprocessing
        self.num_processes = num_processes or mp.cpu_count()
        self.load_from_file = load_from_file  # Path to load embeddings/results from
        
        # Create history directory if it doesn't exist
        os.makedirs('./history', exist_ok=True)

    def run(self):
        try:
            # Check if loading from file
            if self.load_from_file and os.path.exists(self.load_from_file):
                self.progress_signal.emit("Loading from history file...", 10)
                
                if 'final_' in os.path.basename(self.load_from_file):
                    # Load final result file
                    with open(self.load_from_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    coords = data['coords']
                    paths = data['paths']
                    class_ids = data['class_ids']
                    image_dir = data['image_dir']
                    class_names = data['class_names']
                    
                    self.progress_signal.emit("Loaded final result from history!", 100)
                    self.finished_signal.emit(coords, paths, class_ids, image_dir, class_names)
                    return
                    
                elif 'embedding_' in os.path.basename(self.load_from_file):
                    # Load embedding file and continue with dimensionality reduction
                    with open(self.load_from_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    embeddings = data['embeddings']
                    paths = data['paths']
                    class_ids = data['class_ids']
                    image_dir = data['image_dir']
                    class_names = data['class_names']
                    
                    self.progress_signal.emit("Loaded embeddings from history!", 50)
                else:
                    raise Exception("Unknown file format in history")
            else:
                # Normal processing - extract features
                if self.mode == 'yolo':
                    embeddings, paths, class_ids, image_dir, class_names = self.extract_yolo_features()
                else:
                    embeddings, paths, class_ids, image_dir, class_names = self.extract_clip_features()
                
                # Save embeddings to history
                self.save_embeddings(embeddings, paths, class_ids, image_dir, class_names)
            
            if len(embeddings) < 2:
                raise Exception("Not enough images processed to run visualization (Need >= 2).")

            # Run dimensionality reduction
            if self.viz_method == 'umap':
                self.progress_signal.emit("Running UMAP...", 90)
                coords = self.run_umap(embeddings)
            elif self.viz_method == 'opentsne':
                self.progress_signal.emit("Running OpenTSNE...", 90)
                coords = self.run_opentsne(embeddings)
            else:
                self.progress_signal.emit("Running t-SNE...", 90)
                coords = self.run_tsne(embeddings)
            
            # Ensure coords is a numpy array (for OpenTSNE compatibility)
            if not isinstance(coords, np.ndarray):
                coords = np.array(coords)
            
            # Save final results to history
            self.save_final_results(coords, paths, class_ids, image_dir, class_names)
            
            self.progress_signal.emit("Complete!", 100)
            self.finished_signal.emit(coords, paths, class_ids, image_dir, class_names)

        except Exception as e:
            import traceback
            self.error_signal.emit(f"{str(e)}\n{traceback.format_exc()}")

    def get_image_paths_from_yaml(self, yaml_path):
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Extract class names from YAML
        self.class_names = data.get('names', {})
        if isinstance(self.class_names, list):
            self.class_names = {i: name for i, name in enumerate(self.class_names)}

        # Dataset root
        root = data.get('path', '')
        if not os.path.isabs(root):
            root = os.path.join(os.path.dirname(yaml_path), root)

        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        files_with_splits = []

        split_keys = [
            ('train', 'train'),
            ('val', 'val'),
            ('valid', 'val'),
            ('test', 'test')
        ]

        # Helper to normalize path fields
        def normalize_paths(p):
            if not p:
                return []
            if isinstance(p, str):
                return [p]
            if isinstance(p, list):
                return p
            return []

        # Collect images
        for key, label in split_keys:
            subpaths = normalize_paths(data.get(key))

            for subpath in subpaths:
                dir_path = subpath if os.path.isabs(subpath) else os.path.join(root, subpath)

                if not os.path.exists(dir_path):
                    continue

                # Walk directories recursively
                for r, _, files in os.walk(dir_path):
                    for f in files:
                        if f.lower().endswith(extensions):
                            files_with_splits.append((os.path.join(r, f), label))

        # Determine primary directory
        primary_dir = root
        train_paths = normalize_paths(data.get("train"))

        if train_paths:
            first = train_paths[0]
            candidate = first if os.path.isabs(first) else os.path.join(root, first)
            if os.path.exists(candidate):
                primary_dir = candidate

        # Apply random sampling if enabled
        if self.sampling_enabled and self.sample_size and len(files_with_splits) > self.sample_size:
            random.seed(42)  # For reproducibility
            total_before = len(files_with_splits)
            
            # Shuffle the list in place, then slice
            random.shuffle(files_with_splits)
            files_with_splits = files_with_splits[:self.sample_size]
            self.progress_signal.emit(f"Sampled {self.sample_size} images from {total_before} total", 15)

        return files_with_splits, primary_dir, self.class_names

    @staticmethod
    def _get_class_id_from_label(image_path):
        """Extract first object's class_id from corresponding YOLO label txt file"""
        # Normalize path separators
        image_path = os.path.normpath(image_path)
        image_dir = os.path.dirname(image_path)
        image_name = os.path.basename(image_path)
        image_name_no_ext = os.path.splitext(image_name)[0]
        
        # Try multiple possible label locations
        possible_label_paths = []
        
        # Option 1: Replace 'images' with 'labels' in directory
        if 'images' in image_dir:
            label_dir = image_dir.replace('images', 'labels')
            possible_label_paths.append(os.path.join(label_dir, image_name_no_ext + '.txt'))
        
        # Option 2: Same directory as image (some datasets do this)
        possible_label_paths.append(os.path.join(image_dir, image_name_no_ext + '.txt'))
        
        # Option 3: Root labels directory (common YOLO structure)
        parts = image_dir.split(os.sep)
        if 'images' in parts:
            idx = parts.index('images')
            parts[idx] = 'labels'
            possible_label_paths.append(os.path.join(os.sep.join(parts), image_name_no_ext + '.txt'))
        
        # Try each possible path
        for label_path in possible_label_paths:
            if os.path.exists(label_path):
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                return int(parts[0])
                except:
                    continue
        
        return -1

    def get_class_id_from_label(self, image_path):
        """Instance method wrapper for class_id extraction"""
        return self._get_class_id_from_label(image_path)

    @staticmethod
    def _process_yolo_batch(args):
        """Process a batch of images with YOLO model on CPU"""
        batch_data, model_path = args
        
        device = "cpu"
        model = YOLO(model_path)
        model.to(device)
        model.eval()
        
        yolo_model = model.model if hasattr(model, 'model') else model
        
        results = []
        for img_path, split_label in batch_data:
            try:
                if hasattr(yolo_model, 'model') and len(yolo_model.model) > 0:
                    backbone_layers = yolo_model.model[:10]
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(640, 640), mode='bilinear', align_corners=False)
                    img_tensor = img_tensor.to(device)
                    
                    feat = img_tensor
                    for layer in backbone_layers:
                        feat = layer(feat)
                    
                    if len(feat.shape) == 4:
                        feat_pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
                    else:
                        feat_pooled = feat
                    
                    class_id = EmbeddingWorker._get_class_id_from_label(img_path)
                    
                    results.append({
                        'features': feat_pooled.cpu().numpy(),
                        'path': img_path,
                        'class_id': class_id
                    })
            except Exception as e:
                print(f"Skip {img_path}: {e}")
                continue
        
        return results

    @staticmethod
    def _process_clip_batch(args):
        """Process a batch of images with CLIP model on CPU"""
        batch_data, model_name, weights_path = args
        
        device = "cpu"
        model, preprocess, _ = EmbeddingWorker._load_openclip(model_name, weights_path, device)
        
        results = []
        for img_path, split_label in batch_data:
            try:
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    feat = model.encode_image(img_t)
                
                class_id = EmbeddingWorker._get_class_id_from_label(img_path)
                
                results.append({
                    'features': feat.cpu().numpy().squeeze(),
                    'path': img_path,
                    'class_id': class_id
                })
            except Exception as e:
                print(f"Skip {img_path}: {e}")
                continue
        
        return results

    @staticmethod
    def _load_openclip(model_name, weights_path=None, device="cpu"):
        """Static version of load_openclip for multiprocessing"""
        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None if weights_path else "openai",
            device=device
        )
        
        if weights_path and os.path.exists(weights_path):
            ext = os.path.splitext(weights_path)[1].lower()
            
            if ext == ".safetensors":
                if load_safetensors is None:
                    raise ImportError("safetensors not installed")
                checkpoint = load_safetensors(weights_path, device=device)
            elif ext in [".pt", ".bin"]:
                checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
            else:
                raise ValueError(f"Unsupported format: {ext}")
            
            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]
                elif "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]
            
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v
            
            model.load_state_dict(new_state_dict, strict=False)
        
        model.eval()
        model.to(device)
        return model, preprocess, None

    def extract_yolo_features(self):
        if self.use_multiprocessing:
            return self.extract_yolo_features_mp()
        else:
            return self.extract_yolo_features_single()
    
    def extract_yolo_features_single(self):
        """Original single-threaded YOLO feature extraction"""
        self.progress_signal.emit("Loading YOLO model...", 10)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        model = YOLO(self.yolo_weights) if self.yolo_weights and os.path.exists(self.yolo_weights) else YOLO(self.yolo_model_name)
        model.to(device)
        model.eval()
        
        yolo_model = model.model if hasattr(model, 'model') else model
        features_list, paths_list, class_ids_list = [], [], []
        
        image_data, image_dir, class_names = self.get_image_paths_from_yaml(self.yaml_path)
        total = len(image_data)
        self.progress_signal.emit(f"Found {total} images. Processing...", 20)

        for i, (img_path, split_label) in enumerate(tqdm(image_data)):
            try:
                if hasattr(yolo_model, 'model') and len(yolo_model.model) > 0:
                    backbone_layers = yolo_model.model[:10]
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(640, 640), mode='bilinear', align_corners=False)
                    img_tensor = img_tensor.to(device)
                    
                    feat = img_tensor
                    for layer in backbone_layers:
                        feat = layer(feat) 
                    
                    if len(feat.shape) == 4:
                        feat_pooled = torch.nn.functional.adaptive_avg_pool2d(feat, (1, 1)).squeeze(-1).squeeze(-1)
                    else:
                        feat_pooled = feat
                    
                    features_list.append(feat_pooled.cpu().numpy())
                    paths_list.append(img_path)
                    class_id = self.get_class_id_from_label(img_path)
                    class_ids_list.append(class_id)
                
                progress = int(20 + (i/total)*70)
                if i % 100 == 99:
                    self.progress_signal.emit(f"Processing {i}/{total}", progress)
            except Exception as e:
                print(f"Skip {img_path}: {e}")
                continue
        
        if len(features_list) == 0:
            raise Exception("No features extracted.")
        
        return np.vstack(features_list), paths_list, np.array(class_ids_list), image_dir, self.class_names
    
    def extract_yolo_features_mp(self):
        """Multiprocessing YOLO feature extraction"""
        self.progress_signal.emit(f"Preparing YOLO multiprocessing ({self.num_processes} processes)...", 10)
        
        image_data, image_dir, class_names = self.get_image_paths_from_yaml(self.yaml_path)
        total = len(image_data)
        self.progress_signal.emit(f"Found {total} images. Processing with {self.num_processes} processes...", 20)
        
        # Determine model path
        model_path = self.yolo_weights if self.yolo_weights and os.path.exists(self.yolo_weights) else self.yolo_model_name
        
        batch_size = max(1, total // (self.num_processes * 4))
        batches = []
        for i in range(0, total, batch_size):
            batch_data = image_data[i:i + batch_size]
            batches.append((batch_data, model_path))
        
        self.progress_signal.emit(f"Processing {len(batches)} batches with {self.num_processes} processes...", 30)
        
        # Process batches
        features_list, paths_list, class_ids_list = [], [], []
        
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self._process_yolo_batch, batches)
            
            processed = 0
            for batch_results in results:
                for result in batch_results:
                    features_list.append(result['features'])
                    paths_list.append(result['path'])
                    class_ids_list.append(result['class_id'])
                    processed += 1
                    
                    if processed % 100 == 0:
                        progress = int(30 + (processed / total) * 60)
                        self.progress_signal.emit(f"Processed {processed}/{total} images", progress)
        
        if len(features_list) == 0:
            raise Exception("No features extracted.")
        
        return np.vstack(features_list), paths_list, np.array(class_ids_list), image_dir, self.class_names

    def extract_clip_features(self):
        if self.use_multiprocessing:
            return self.extract_clip_features_mp()
        else:
            return self.extract_clip_features_single()
    
    def extract_clip_features_single(self):
        """Original single-threaded CLIP feature extraction"""
        self.progress_signal.emit("Loading OpenCLIP model...", 10)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.clip_weights_path and os.path.exists(self.clip_weights_path):
            model, preprocess, _ = self.load_openclip(self.clip_model_name, weights_path=self.clip_weights_path, device=device)
        else:
            model, preprocess, _ = self.load_openclip(self.clip_model_name, device=device)

        image_data, image_dir, class_names = self.get_image_paths_from_yaml(self.yaml_path)
        total = len(image_data)
        self.progress_signal.emit(f"Found {total} images. Processing...", 20)
        
        embeddings, paths_list, class_ids_list = [], [], []
        batch, batch_paths, batch_class_ids = [], [], []
        
        for i, (img_path, split_label) in enumerate(tqdm(image_data)):
            try:
                # Get class_id from label file
                class_id = self.get_class_id_from_label(img_path)
                
                img = Image.open(img_path).convert('RGB')
                img_t = preprocess(img).unsqueeze(0).to(device)
                batch.append(img_t)
                batch_paths.append(img_path)
                batch_class_ids.append(class_id)
                
                if len(batch) >= 16:
                    with torch.no_grad():
                        feat = model.encode_image(torch.cat(batch, dim=0))
                        embeddings.append(feat.cpu().numpy())
                    paths_list.extend(batch_paths)
                    class_ids_list.extend(batch_class_ids)
                    batch, batch_paths, batch_class_ids = [], [], []
                    
                progress = int(20 + (i/total)*70)
                if i % 100 == 99:
                    self.progress_signal.emit(f"Processing {i+1}/{total}", progress)
            except Exception as e:
                print(f"Skip {img_path}: {e}")
        
        if batch:
            with torch.no_grad():
                feat = model.encode_image(torch.cat(batch, dim=0))
                embeddings.append(feat.cpu().numpy())
            paths_list.extend(batch_paths)
            class_ids_list.extend(batch_class_ids)
                
        if len(embeddings) > 0:
            return np.vstack(embeddings), paths_list, np.array(class_ids_list), image_dir, self.class_names
        else:
            raise Exception("No CLIP embeddings extracted")
    
    def extract_clip_features_mp(self):
        """Multiprocessing CLIP feature extraction"""
        self.progress_signal.emit(f"Preparing CLIP multiprocessing ({self.num_processes} processes)...", 10)
        
        image_data, image_dir, class_names = self.get_image_paths_from_yaml(self.yaml_path)
        total = len(image_data)
        self.progress_signal.emit(f"Found {total} images. Processing with {self.num_processes} processes...", 20)
        
        batch_size = max(1, total // (self.num_processes * 4))
        batches = []
        for i in range(0, total, batch_size):
            batch_data = image_data[i:i + batch_size]
            batches.append((batch_data, self.clip_model_name, self.clip_weights_path))
        
        self.progress_signal.emit(f"Processing {len(batches)} batches with {self.num_processes} processes...", 30)
        
        embeddings_list, paths_list, class_ids_list = [], [], []
        
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self._process_clip_batch, batches)
            
            processed = 0
            for batch_results in results:
                for result in batch_results:
                    embeddings_list.append(result['features'])
                    paths_list.append(result['path'])
                    class_ids_list.append(result['class_id'])
                    processed += 1
                    
                    if processed % 100 == 0:
                        progress = int(30 + (processed / total) * 60)
                        self.progress_signal.emit(f"Processed {processed}/{total} images", progress)
        
        if len(embeddings_list) == 0:
            raise Exception("No CLIP embeddings extracted")
        
        return np.vstack(embeddings_list), paths_list, np.array(class_ids_list), image_dir, self.class_names

    def load_openclip(
        self, model_name: str = "ViT-B-32",
        pretrained: str | None = "openai",
        weights_path: str | None = None,
        device: str | None = None,
        precision: str = "fp32",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model, preprocess, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=None if weights_path else pretrained,
            device=device
        )

        tokenizer = open_clip.get_tokenizer(model_name)

        if weights_path:
            if not os.path.exists(weights_path):
                raise FileNotFoundError(weights_path)

            ext = os.path.splitext(weights_path)[1].lower()

            print(f"Loading OpenCLIP weights: {weights_path}")

            # -------- SAFETENSORS --------
            if ext == ".safetensors":
                if load_safetensors is None:
                    raise ImportError(
                        "safetensors not installed. Install with: pip install safetensors"
                    )

                checkpoint = load_safetensors(weights_path, device=device)

            # -------- PT / BIN --------
            elif ext in [".pt", ".bin"]:
                checkpoint = torch.load(
                    weights_path,
                    map_location=device,
                    weights_only=False  # PyTorch >=2.6 compatibility
                )

            else:
                raise ValueError(f"Unsupported checkpoint format: {ext}")

            if isinstance(checkpoint, dict):
                if "state_dict" in checkpoint:
                    checkpoint = checkpoint["state_dict"]

                elif "model_state_dict" in checkpoint:
                    checkpoint = checkpoint["model_state_dict"]

            new_state_dict = {}

            for k, v in checkpoint.items():
                if k.startswith("module."):
                    k = k[7:]
                new_state_dict[k] = v

            checkpoint = new_state_dict

            missing, unexpected = model.load_state_dict(checkpoint, strict=False)

            if missing:
                print(f"Missing keys: {len(missing)}")

            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")

        if precision == "fp16":
            model = model.half()

        elif precision == "bf16":
            model = model.to(dtype=torch.bfloat16)

        model.eval()
        model.to(device)

        print(f"OpenCLIP model loaded: {model_name} on {device}")

        return model, preprocess, tokenizer

    def run_tsne(self, embeddings):
        X_scaled = StandardScaler().fit_transform(embeddings)
        perplexity_val = min(30, len(embeddings) - 1)
        if perplexity_val < 1: perplexity_val = 1
        
        tsne = TSNE(n_components=2, perplexity=50, random_state=42, init='pca', **{TSNE_ITER_PARAM: 1500})
        return tsne.fit_transform(X_scaled)

    def run_umap(self, embeddings):
        if umap is None:
            raise ImportError("UMAP is not installed. Install it with: pip install umap-learn")

        X_scaled = StandardScaler().fit_transform(embeddings)
        n_neighbors = min(15, max(2, len(embeddings) - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.1,
            metric='euclidean',
            random_state=42
        )
        return reducer.fit_transform(X_scaled)
    
    def run_opentsne(self, embeddings):
        """Run openTSNE dimensionality reduction"""
        if OpenTSNE is None:
            raise ImportError("openTSNE is not installed. Install it with: pip install openTSNE")
        
        X_scaled = StandardScaler().fit_transform(embeddings)
        
        # openTSNE parameters
        perplexity_val = min(30, len(embeddings) - 1)
        if perplexity_val < 1: 
            perplexity_val = 1
        
        # Run openTSNE with optimized settings
        tsne = OpenTSNE(
            perplexity=50,
            learning_rate='auto',
            n_iter=1000,
            random_state=42,
            initialization='pca',
            n_jobs=-1,  # Use all available cores
            verbose=False
        )
        
        # Fit and transform, then extract coordinates as numpy array
        embedding_result = tsne.fit(X_scaled)
        
        # Convert to numpy array to avoid Windows temp file issues when pickling
        coords = np.array(embedding_result)
        
        return coords
    
    def save_embeddings(self, embeddings, paths, class_ids, image_dir, class_names):
        """Save embeddings to history directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./history/embedding_{timestamp}.pkl"
        
        data = {
            'embeddings': embeddings,
            'paths': paths,
            'class_ids': class_ids,
            'image_dir': image_dir,
            'class_names': class_names,
            'mode': self.mode,
            'viz_method': self.viz_method
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        self.progress_signal.emit(f"Saved embeddings to {filename}", 85)
    
    def save_final_results(self, coords, paths, class_ids, image_dir, class_names):
        """Save final results to history directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./history/final_{self.viz_method}_{timestamp}.pkl"
        
        data = {
            'coords': coords,
            'paths': paths,
            'class_ids': class_ids,
            'image_dir': image_dir,
            'class_names': class_names,
            'mode': self.mode,
            'viz_method': self.viz_method
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        self.progress_signal.emit(f"Saved final results to {filename}", 95)
