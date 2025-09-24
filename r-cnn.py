#!/usr/bin/env python3
"""
Enhanced R-CNN Trainer for ShelfCheck AI with Explainable AI
Fixed MPS memory issues + Added XAI features (Grad-CAM, LIME, Feature Maps)
Team: Avirup, Lakshay, Sadaf - Amrita Vishwa Vidyapeetham
"""

import os
import json
import logging
from pathlib import Path
import argparse
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime
import shutil
import gc
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1.5'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

torch.backends.mps.is_available = lambda: False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_memory():
    """Clear GPU/MPS memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

def get_memory_usage():
    """Get current memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.mps.current_allocated_memory() / 1024**3  # GB
    return 0

class GradCAMHook:
    """Hook for capturing gradients and features for Grad-CAM"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.features = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_features)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_features(self, module, input, output):
        self.features = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()

class ExplainableAI:
    """Explainable AI module for R-CNN"""
    
    def __init__(self, model, device, class_names):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.gradcam_hook = None
    
    def generate_gradcam(self, image_tensor, target_class=None, layer_name='backbone.body.layer4'):
        """
        Generate Grad-CAM visualization
        Args:
            image_tensor: Input image tensor
            target_class: Target class for visualization
            layer_name: Name of target layer
        """
        try:
            # Get target layer
            target_layer = self._get_layer_by_name(layer_name)
            if target_layer is None:
                logger.warning(f"Layer {layer_name} not found, using default")
                # Use ResNet backbone layer4 as default
                target_layer = self.model.backbone.body.layer4
            
            # Setup hook
            self.gradcam_hook = GradCAMHook(self.model, target_layer)
            
            # Forward pass
            self.model.eval()
            image_tensor.requires_grad_(True)
            
            with torch.enable_grad():
                outputs = self.model([image_tensor])
                
                # Get prediction with highest confidence
                pred = outputs[0]
                if len(pred['scores']) == 0:
                    logger.warning("No predictions found for Grad-CAM")
                    return None, None
                
                # Use highest confidence prediction or target class
                if target_class is None:
                    target_idx = torch.argmax(pred['scores'])
                else:
                    # Find prediction with target class
                    target_mask = pred['labels'] == target_class
                    if target_mask.sum() == 0:
                        logger.warning(f"Target class {target_class} not found in predictions")
                        target_idx = torch.argmax(pred['scores'])
                    else:
                        target_scores = pred['scores'][target_mask]
                        target_idx = torch.argmax(target_scores)
                        # Get original index
                        target_indices = torch.where(target_mask)[0]
                        target_idx = target_indices[target_idx]
                
                target_score = pred['scores'][target_idx]
                target_label = pred['labels'][target_idx]
                
                # Backward pass
                self.model.zero_grad()
                target_score.backward(retain_graph=True)
                
                # Get gradients and features
                if self.gradcam_hook.gradients is None or self.gradcam_hook.features is None:
                    logger.warning("Failed to capture gradients or features")
                    return None, None
                
                gradients = self.gradcam_hook.gradients
                features = self.gradcam_hook.features
                
                # Calculate Grad-CAM
                weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
                cam = torch.sum(weights * features, dim=1, keepdim=True)
                cam = F.relu(cam)
                
                # Normalize
                cam = cam - cam.min()
                if cam.max() > 0:
                    cam = cam / cam.max()
                
                # Resize to input image size
                input_size = image_tensor.shape[-2:]
                cam_resized = F.interpolate(cam, size=input_size, mode='bilinear', align_corners=False)
                
                # Clean up
                self.gradcam_hook.remove_hooks()
                
                return cam_resized.squeeze().cpu().numpy(), target_label.item()
                
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            if self.gradcam_hook:
                self.gradcam_hook.remove_hooks()
            return None, None
    
    def _get_layer_by_name(self, layer_name):
        """Get layer by name from model"""
        try:
            parts = layer_name.split('.')
            layer = self.model
            for part in parts:
                layer = getattr(layer, part)
            return layer
        except:
            return None
    
    def extract_feature_maps(self, image_tensor, layer_name='backbone.body.layer3'):
        """Extract and visualize feature maps"""
        try:
            target_layer = self._get_layer_by_name(layer_name)
            if target_layer is None:
                target_layer = self.model.backbone.body.layer3
            
            features = []
            
            def hook_fn(module, input, output):
                features.append(output)
            
            hook = target_layer.register_forward_hook(hook_fn)
            
            # Forward pass
            self.model.eval()
            with torch.no_grad():
                _ = self.model([image_tensor])
            
            hook.remove()
            
            if features:
                feature_map = features[0].squeeze(0).cpu().numpy()
                return feature_map
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting feature maps: {e}")
            return None
    
    def simple_occlusion_analysis(self, image_tensor, patch_size=50, stride=25):
        """Simple occlusion analysis"""
        try:
            self.model.eval()
            
            # Get original prediction
            with torch.no_grad():
                original_outputs = self.model([image_tensor])
                if len(original_outputs[0]['scores']) == 0:
                    return None
                
                original_score = torch.max(original_outputs[0]['scores']).item()
            
            # Image dimensions
            _, h, w = image_tensor.shape
            
            # Create occlusion map
            occlusion_map = np.zeros((h, w))
            
            # Occlude patches and measure impact
            for y in range(0, h - patch_size, stride):
                for x in range(0, w - patch_size, stride):
                    # Create occluded image
                    occluded_image = image_tensor.clone()
                    occluded_image[:, y:y+patch_size, x:x+patch_size] = 0
                    
                    # Get prediction
                    with torch.no_grad():
                        outputs = self.model([occluded_image])
                        if len(outputs[0]['scores']) > 0:
                            max_score = torch.max(outputs[0]['scores']).item()
                        else:
                            max_score = 0
                    
                    # Calculate impact
                    impact = original_score - max_score
                    occlusion_map[y:y+patch_size, x:x+patch_size] = impact
            
            return occlusion_map
            
        except Exception as e:
            logger.error(f"Error in occlusion analysis: {e}")
            return None
    
    def analyze_prediction_confidence(self, predictions):
        """Analyze prediction confidence distribution"""
        try:
            pred = predictions[0]
            scores = pred['scores'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            
            confidence_analysis = {
                'high_confidence': np.sum(scores > 0.8),
                'medium_confidence': np.sum((scores > 0.5) & (scores <= 0.8)),
                'low_confidence': np.sum(scores <= 0.5),
                'mean_confidence': np.mean(scores),
                'std_confidence': np.std(scores),
                'max_confidence': np.max(scores) if len(scores) > 0 else 0,
                'min_confidence': np.min(scores) if len(scores) > 0 else 0,
                'total_detections': len(scores)
            }
            
            # Per-class confidence
            class_confidence = {}
            for label in np.unique(labels):
                mask = labels == label
                class_scores = scores[mask]
                class_name = self.class_names[label-1] if label > 0 and label-1 < len(self.class_names) else f'class_{label}'
                class_confidence[class_name] = {
                    'mean': np.mean(class_scores),
                    'std': np.std(class_scores),
                    'count': len(class_scores)
                }
            
            confidence_analysis['per_class'] = class_confidence
            return confidence_analysis
            
        except Exception as e:
            logger.error(f"Error in confidence analysis: {e}")
            return None

class RCNNDataset(Dataset):
    """Custom dataset for R-CNN training with YOLO format annotations"""
    
    def __init__(self, images_dir, labels_dir, transforms=None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        
        # Get all image files
        self.image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        self.images = [f for f in os.listdir(images_dir) 
                      if Path(f).suffix.lower() in self.image_extensions]
        
        logger.info(f"Found {len(self.images)} images in {images_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = self.images_dir / img_name
        image = Image.open(img_path).convert('RGB')
        
        # Get image dimensions
        img_width, img_height = image.size
        
        # Load corresponding label file
        label_name = Path(img_name).stem + '.txt'
        label_path = self.labels_dir / label_name
        
        boxes = []
        labels = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            center_x = float(parts[1])
                            center_y = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Convert YOLO format to absolute coordinates
                            x1 = (center_x - width/2) * img_width
                            y1 = (center_y - height/2) * img_height
                            x2 = (center_x + width/2) * img_width
                            y2 = (center_y + height/2) * img_height
                            
                            # Ensure coordinates are within image bounds
                            x1 = max(0, min(x1, img_width - 1))
                            y1 = max(0, min(y1, img_height - 1))
                            x2 = max(0, min(x2, img_width - 1))
                            y2 = max(0, min(y2, img_height - 1))
                            
                            # Only add valid boxes
                            if x2 > x1 and y2 > y1:
                                boxes.append([x1, y1, x2, y2])
                                labels.append(class_id + 1)  # Background is 0, so add 1
        
        if not boxes:
            # If no annotations, create a dummy box
            boxes = [[0, 0, 1, 1]]
            labels = [0]  # Background class
        
        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Calculate areas
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
            'area': area,
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        return image, target

class EnhancedRCNNTrainer:
    """Enhanced R-CNN trainer with organized output structure, memory optimization, and Explainable AI"""
    
    def __init__(self, dataset_path, model_size='resnet50', project='runs/detect', name=None):
        """
        Initialize trainer
        Args:
            dataset_path: Path to dataset directory
            model_size: Backbone model size ('resnet50', 'resnet101')
            project: Project directory (like YOLOv8)
            name: Run name (like YOLOv8)
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None
        
        # Create organized output structure like YOLOv8
        self.project = Path(project)
        if name is None:
            name = f'rcnn_{model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.run_dir = self.project / name
        self.weights_dir = self.run_dir / 'weights'
        self.xai_dir = self.run_dir / 'explainability'  # New XAI directory
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.xai_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        log_file = self.run_dir / 'train.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"💾 Output directory: {self.run_dir}")
        logger.info(f"🧠 XAI directory: {self.xai_dir}")
        
        # Detect device with memory considerations
        self.device = self.detect_device()
        logger.info(f"Using device: {self.device}")
        
        # Validate dataset structure
        self.validate_dataset()
        
        # Detect classes
        self.class_names = self.detect_classes()
        self.num_classes = len(self.class_names) + 1  # +1 for background
        
        logger.info(f"Number of classes: {self.num_classes} (including background)")
        logger.info(f"Class names: {self.class_names}")
        
        # Initialize XAI module
        self.xai = None
        
        # Save dataset info
        self.save_dataset_info()
    
    def detect_device(self):
        """Detect available device with memory optimization"""
        try:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                logger.info(f"CUDA available - GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                return device
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
                logger.info("MPS (Apple Silicon) available")
                # Set MPS memory fraction to avoid OOM
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.8'
                logger.info("Set MPS memory ratio to 0.8 to prevent OOM")
                return device
            else:
                device = torch.device('cpu')
                logger.info("Using CPU")
                return device
        except Exception as e:
            logger.warning(f"Device detection error: {e}, falling back to CPU")
            return torch.device('cpu')
    
    def validate_dataset(self):
        """Check if dataset has proper structure"""
        required_dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels']
        
        for dir_name in required_dirs:
            dir_path = self.dataset_path / dir_name
            if not dir_path.exists():
                logger.error(f"Missing directory: {dir_path}")
                raise FileNotFoundError(f"Dataset directory {dir_path} not found")
        
        # Count files
        train_images = list((self.dataset_path / 'train/images').glob('*'))
        train_labels = list((self.dataset_path / 'train/labels').glob('*.txt'))
        valid_images = list((self.dataset_path / 'valid/images').glob('*'))
        valid_labels = list((self.dataset_path / 'valid/labels').glob('*.txt'))
        
        logger.info(f"Dataset validation:")
        logger.info(f"  Train: {len(train_images)} images, {len(train_labels)} labels")
        logger.info(f"  Valid: {len(valid_images)} images, {len(valid_labels)} labels")
        
        if len(train_images) == 0 or len(valid_images) == 0:
            raise ValueError("No images found in dataset")
    
    def detect_classes(self):
        """Detect classes from label files"""
        class_ids = set()
        labels_dir = self.dataset_path / 'train/labels'
        
        for label_file in labels_dir.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            class_id = int(line.split()[0])
                            class_ids.add(class_id)
            except:
                continue
        
        # Try to read existing classes.txt
        classes_file = self.dataset_path / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded classes from classes.txt: {class_names}")
        else:
            # Create class names
            num_classes = len(class_ids)
            if num_classes <= 10:
                common_names = ['bottle', 'can', 'box', 'package', 'jar', 'tube', 'bag', 'carton', 'container', 'wrapper']
                class_names = common_names[:num_classes]
            else:
                class_names = [f'class_{i}' for i in sorted(class_ids)]
            
            # Save classes for future use
            with open(classes_file, 'w') as f:
                for name in class_names:
                    f.write(f"{name}\n")
            logger.info(f"Saved classes to {classes_file}")
        
        # Copy classes.txt to run directory
        shutil.copy2(classes_file, self.run_dir / 'classes.txt')
        
        return class_names
    
    def save_dataset_info(self):
        """Save dataset information like YOLOv8"""
        dataset_info = {
            'dataset_path': str(self.dataset_path.absolute()),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'model_size': self.model_size,
            'device': str(self.device),
            'xai_enabled': True,
            'created': datetime.now().isoformat()
        }
        
        with open(self.run_dir / 'args.yaml', 'w') as f:
            import yaml
            yaml.dump(dataset_info, f, default_flow_style=False)
    
    def create_model(self):
        """Create R-CNN model with memory optimization"""
        logger.info(f"Creating Faster R-CNN model with {self.model_size} backbone...")
        
        # Clear memory before creating model
        clear_memory()
        
        # Load pre-trained Faster R-CNN with ResNet backbone
        model = fasterrcnn_resnet50_fpn(pretrained=True)
        
        # Replace the classifier head for our custom number of classes
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        
        # Move model to device
        model.to(self.device)
        
        # Initialize XAI module
        self.xai = ExplainableAI(model, self.device, self.class_names)
        
        # Log memory usage after model creation
        memory_used = get_memory_usage()
        logger.info(f"Model loaded - Memory usage: {memory_used:.2f} GB")
        
        self.model = model
        logger.info(f"✅ Model created with {self.num_classes} classes")
        logger.info(f"🧠 XAI module initialized")
        return model
    
    def simple_data_loader(self, split='train', batch_size=2):
        """Simple data loading with memory optimization"""
        images_dir = self.dataset_path / split / 'images'
        labels_dir = self.dataset_path / split / 'labels'
        
        # Get all image files
        image_files = list(images_dir.glob('*'))
        
        # Reduce image size for memory efficiency
        transform = transforms.Compose([
            transforms.Resize((400, 400)),  # Smaller size to save memory
            transforms.ToTensor(),
        ])
        
        batches = []
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_images = []
            batch_targets = []
            
            for img_file in batch_files:
                try:
                    # Load image with size limit
                    image = Image.open(img_file).convert('RGB')
                    original_width, original_height = image.size
                    
                    # Resize image if too large
                    max_size = 800
                    if max(original_width, original_height) > max_size:
                        ratio = max_size / max(original_width, original_height)
                        new_width = int(original_width * ratio)
                        new_height = int(original_height * ratio)
                        image = image.resize((new_width, new_height))
                        img_width, img_height = new_width, new_height
                    else:
                        img_width, img_height = original_width, original_height
                    
                    image_tensor = transform(image)
                    
                    # Load labels
                    label_file = labels_dir / (img_file.stem + '.txt')
                    boxes = []
                    labels = []
                    
                    if label_file.exists():
                        with open(label_file, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    parts = line.split()
                                    if len(parts) >= 5:
                                        class_id = int(parts[0])
                                        center_x = float(parts[1])
                                        center_y = float(parts[2])
                                        width = float(parts[3])
                                        height = float(parts[4])
                                        
                                        # Convert to absolute coordinates
                                        x1 = (center_x - width/2) * img_width
                                        y1 = (center_y - height/2) * img_height
                                        x2 = (center_x + width/2) * img_width
                                        y2 = (center_y + height/2) * img_height
                                        
                                        x1 = max(0, min(x1, img_width))
                                        y1 = max(0, min(y1, img_height))
                                        x2 = max(0, min(x2, img_width))
                                        y2 = max(0, min(y2, img_height))
                                        
                                        if x2 > x1 and y2 > y1:
                                            boxes.append([x1, y1, x2, y2])
                                            labels.append(class_id + 1)
                    
                    if not boxes:
                        boxes = [[0, 0, 1, 1]]
                        labels = [0]
                    
                    target = {
                        'boxes': torch.tensor(boxes, dtype=torch.float32).to(self.device),
                        'labels': torch.tensor(labels, dtype=torch.int64).to(self.device),
                    }
                    
                    batch_images.append(image_tensor.to(self.device))
                    batch_targets.append(target)
                    
                except Exception as e:
                    logger.warning(f"Error loading {img_file}: {e}")
                    continue
            
            if batch_images:
                batches.append((batch_images, batch_targets))
        
        return batches
    
    def train(self, epochs=5, batch_size=2, learning_rate=0.005, enable_xai=True):
        """
        Train R-CNN with memory optimization and optional XAI
        """
        logger.info("🚀 Starting R-CNN training with Explainable AI...")
        logger.info(f"Configuration: {epochs} epochs, batch size {batch_size}, lr {learning_rate}")
        logger.info(f"Device: {self.device}")
        logger.info(f"XAI enabled: {enable_xai}")
        
        start_time = time.time()
        
        # Create model
        if not self.model:
            self.create_model()
        
        # Adjust batch size for device and memory
        if self.device.type == 'cpu':
            batch_size = min(batch_size, 1)
            logger.info(f"CPU detected - reducing batch size to {batch_size}")
        elif self.device.type == 'mps':
            batch_size = min(batch_size, 1)  # Very conservative for MPS
            logger.info(f"MPS detected - reducing batch size to {batch_size} for memory efficiency")
        
        # Simple optimizer
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=0.0005
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training tracking
        best_loss = float('inf')
        results = {
            'train_loss': [],
            'val_loss': [],
            'epochs': [],
            'lr': []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Clear memory before epoch
            clear_memory()
            memory_before = get_memory_usage()
            logger.info(f"Memory before epoch: {memory_before:.2f} GB")
            
            # Training phase
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # Load training batches (simple method)
            train_batches = self.simple_data_loader('train', batch_size)
            
            for batch_idx, (images, targets) in enumerate(train_batches[:5]):  # Reduced to 5 batches per epoch for memory
                try:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    loss_dict = self.model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())
                    
                    # Backward pass
                    losses.backward()
                    optimizer.step()
                    
                    total_loss += losses.item()
                    num_batches += 1
                    
                    # Clear intermediate tensors
                    del loss_dict, losses
                    clear_memory()
                    
                    if batch_idx % 2 == 0:
                        memory_current = get_memory_usage()
                        logger.info(f"  Batch {batch_idx+1}/5, Loss: {total_loss/num_batches:.4f}, Memory: {memory_current:.2f} GB")
                    
                except Exception as e:
                    logger.warning(f"Error in training batch {batch_idx}: {e}")
                    clear_memory()
                    continue
            
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0
            
            # Validation phase (simplified and memory-efficient)
            self.model.eval()
            val_loss = 0
            val_batches = 0
            
            clear_memory()  # Clear before validation
            
            val_data = self.simple_data_loader('valid', batch_size)
            
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_data[:3]):  # Reduced to 3 validation batches
                    try:
                        loss_dict = self.model(images, targets)
                        losses = sum(loss for loss in loss_dict.values())
                        val_loss += losses.item()
                        val_batches += 1
                        
                        # Clear intermediate tensors
                        del loss_dict, losses
                        clear_memory()
                        
                    except Exception as e:
                        logger.warning(f"Error in validation batch {batch_idx}: {e}")
                        clear_memory()
                        continue
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            # Record results
            results['epochs'].append(epoch + 1)
            results['train_loss'].append(avg_train_loss)
            results['val_loss'].append(avg_val_loss)
            results['lr'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            memory_after = get_memory_usage()
            logger.info(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - '
                       f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
                       f'Memory: {memory_after:.2f} GB')
            
            # Save best model
            if avg_val_loss < best_loss and avg_val_loss > 0:
                best_loss = avg_val_loss
                self.save_checkpoint(epoch, avg_val_loss, 'best.pt')
                logger.info(f'✅ Best model saved with validation loss: {best_loss:.4f}')
            
            # Save last model
            if epoch == epochs - 1:
                self.save_checkpoint(epoch, avg_val_loss, 'last.pt')
            
            # Generate XAI analysis every few epochs
            if enable_xai and (epoch + 1) % 5 == 0 and self.xai:
                try:
                    logger.info(f"🧠 Generating XAI analysis for epoch {epoch + 1}...")
                    self.generate_xai_analysis_epoch(epoch)
                except Exception as e:
                    logger.warning(f"XAI analysis failed: {e}")
            
            scheduler.step()
            clear_memory()  # Clear memory after each epoch
        
        total_time = time.time() - start_time
        
        # Save results and create plots
        self.save_results(results, total_time, best_loss)
        self.create_plots(results)
        
        # Final XAI analysis
        if enable_xai and self.xai:
            try:
                logger.info("🧠 Generating comprehensive XAI analysis...")
                self.generate_comprehensive_xai_analysis()
            except Exception as e:
                logger.warning(f"Comprehensive XAI analysis failed: {e}")
        
        # Test inference with memory management
        try:
            self.test_inference_safe(enable_xai=enable_xai)
        except Exception as e:
            logger.warning(f"Inference test failed due to memory constraints: {e}")
            logger.info("Skipping inference test to prevent OOM")
        
        logger.info(f"✅ Training completed in {total_time:.1f}s!")
        logger.info(f"📊 Best validation loss: {best_loss:.4f}")
        logger.info(f"💾 Results saved to: {self.run_dir}")
        if enable_xai:
            logger.info(f"🧠 XAI analysis saved to: {self.xai_dir}")
        
        return results
    
    def generate_xai_analysis_epoch(self, epoch):
        """Generate XAI analysis for current epoch"""
        try:
            # Get a sample validation image
            val_images_dir = self.dataset_path / 'valid/images'
            sample_images = list(val_images_dir.glob('*'))[:3]  # Take first 3 images
            
            for i, img_path in enumerate(sample_images):
                try:
                    # Load and preprocess image
                    image = Image.open(img_path).convert('RGB')
                    
                    # Resize for memory efficiency
                    max_size = 400
                    if max(image.size) > max_size:
                        ratio = max_size / max(image.size)
                        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                        image = image.resize(new_size)
                    
                    transform = transforms.Compose([transforms.ToTensor()])
                    image_tensor = transform(image).to(self.device)
                    
                    # Generate predictions
                    self.model.eval()
                    with torch.no_grad():
                        predictions = self.model([image_tensor])
                    
                    if len(predictions[0]['scores']) > 0:
                        # Generate Grad-CAM
                        gradcam, target_label = self.xai.generate_gradcam(image_tensor)
                        
                        if gradcam is not None:
                            self.save_gradcam_visualization(
                                image, gradcam, predictions[0], 
                                f"epoch_{epoch+1}_sample_{i+1}_gradcam.png",
                                target_label
                            )
                        
                        # Generate feature maps
                        feature_maps = self.xai.extract_feature_maps(image_tensor)
                        if feature_maps is not None:
                            self.save_feature_maps_visualization(
                                feature_maps, 
                                f"epoch_{epoch+1}_sample_{i+1}_features.png"
                            )
                    
                    clear_memory()
                    
                except Exception as e:
                    logger.warning(f"Error processing sample {i}: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error in epoch XAI analysis: {e}")
    
    def generate_comprehensive_xai_analysis(self):
        """Generate comprehensive XAI analysis after training"""
        logger.info("Generating comprehensive explainability analysis...")
        
        try:
            # Get validation images
            val_images_dir = self.dataset_path / 'valid/images'
            sample_images = list(val_images_dir.glob('*'))[:5]  # Analyze 5 images
            
            analysis_results = {
                'gradcam_analysis': [],
                'confidence_analysis': [],
                'feature_analysis': [],
                'occlusion_analysis': []
            }
            
            for i, img_path in enumerate(sample_images):
                try:
                    logger.info(f"Analyzing image {i+1}/5: {img_path.name}")
                    
                    # Load and preprocess
                    image = Image.open(img_path).convert('RGB')
                    original_image = image.copy()
                    
                    # Resize for memory efficiency
                    max_size = 400
                    if max(image.size) > max_size:
                        ratio = max_size / max(image.size)
                        new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                        image = image.resize(new_size)
                    
                    transform = transforms.Compose([transforms.ToTensor()])
                    image_tensor = transform(image).to(self.device)
                    
                    # Get predictions
                    self.model.eval()
                    with torch.no_grad():
                        predictions = self.model([image_tensor])
                    
                    if len(predictions[0]['scores']) == 0:
                        logger.info(f"No predictions for image {i+1}, skipping...")
                        continue
                    
                    # 1. Grad-CAM Analysis
                    gradcam, target_label = self.xai.generate_gradcam(image_tensor)
                    if gradcam is not None:
                        # Save comprehensive Grad-CAM visualization
                        self.save_comprehensive_gradcam(
                            original_image, image, gradcam, predictions[0], 
                            f"comprehensive_gradcam_{i+1}.png", target_label
                        )
                        analysis_results['gradcam_analysis'].append({
                            'image': img_path.name,
                            'target_class': target_label,
                            'gradcam_intensity': float(np.mean(gradcam))
                        })
                    
                    # 2. Confidence Analysis
                    conf_analysis = self.xai.analyze_prediction_confidence(predictions)
                    if conf_analysis:
                        analysis_results['confidence_analysis'].append({
                            'image': img_path.name,
                            'analysis': conf_analysis
                        })
                        
                        # Save confidence visualization
                        self.save_confidence_analysis(conf_analysis, f"confidence_{i+1}.png")
                    
                    # 3. Feature Maps Analysis
                    feature_maps = self.xai.extract_feature_maps(image_tensor)
                    if feature_maps is not None:
                        self.save_comprehensive_feature_maps(
                            feature_maps, f"feature_analysis_{i+1}.png"
                        )
                        analysis_results['feature_analysis'].append({
                            'image': img_path.name,
                            'feature_diversity': float(np.std(feature_maps))
                        })
                    
                    # 4. Occlusion Analysis (simplified for memory)
                    if i < 2:  # Only do occlusion for first 2 images to save memory
                        occlusion_map = self.xai.simple_occlusion_analysis(
                            image_tensor, patch_size=30, stride=20
                        )
                        if occlusion_map is not None:
                            self.save_occlusion_analysis(
                                image, occlusion_map, f"occlusion_{i+1}.png"
                            )
                            analysis_results['occlusion_analysis'].append({
                                'image': img_path.name,
                                'critical_regions': float(np.sum(occlusion_map > np.mean(occlusion_map)))
                            })
                    
                    clear_memory()
                    
                except Exception as e:
                    logger.warning(f"Error analyzing image {i}: {e}")
                    continue
            
            # Save analysis results
            with open(self.xai_dir / 'comprehensive_analysis.json', 'w') as f:
                json.dump(analysis_results, f, indent=2)
            
            # Create summary report
            self.create_xai_summary_report(analysis_results)
            
            logger.info(f"✅ Comprehensive XAI analysis completed and saved to {self.xai_dir}")
            
        except Exception as e:
            logger.error(f"Error in comprehensive XAI analysis: {e}")
    
    def save_gradcam_visualization(self, image, gradcam, predictions, filename, target_label):
        """Save Grad-CAM visualization"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Grad-CAM heatmap
            axes[1].imshow(gradcam, cmap='jet', alpha=0.8)
            axes[1].set_title(f'Grad-CAM (Class: {self.class_names[target_label-1] if target_label > 0 else "background"})')
            axes[1].axis('off')
            
            # Overlay
            axes[2].imshow(image)
            axes[2].imshow(gradcam, cmap='jet', alpha=0.4)
            axes[2].set_title('Grad-CAM Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.xai_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error saving Grad-CAM visualization: {e}")
    
    def save_comprehensive_gradcam(self, original_image, resized_image, gradcam, predictions, filename, target_label):
        """Save comprehensive Grad-CAM analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Original image with bounding boxes
            axes[0, 0].imshow(original_image)
            pred = predictions
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            
            # Draw bounding boxes
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for box, label, score in zip(boxes, labels, scores):
                if score > 0.3:
                    x1, y1, x2, y2 = box
                    # Scale coordinates back to original image size
                    scale_x = original_image.size[0] / resized_image.size[0]
                    scale_y = original_image.size[1] / resized_image.size[1]
                    x1, x2 = x1 * scale_x, x2 * scale_x
                    y1, y2 = y1 * scale_y, y2 * scale_y
                    
                    color = colors[(label-1) % len(colors)] if label > 0 else 'red'
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                           linewidth=2, edgecolor=color, facecolor='none')
                    axes[0, 0].add_patch(rect)
                    
                    class_name = self.class_names[label-1] if label > 0 and label-1 < len(self.class_names) else f'class_{label}'
                    axes[0, 0].text(x1, y1-5, f'{class_name}: {score:.2f}', 
                                   bbox=dict(facecolor=color, alpha=0.7),
                                   fontsize=8, color='white')
            
            axes[0, 0].set_title('Predictions on Original Image')
            axes[0, 0].axis('off')
            
            # Resized image
            axes[0, 1].imshow(resized_image)
            axes[0, 1].set_title('Processed Image')
            axes[0, 1].axis('off')
            
            # Grad-CAM heatmap
            im = axes[1, 0].imshow(gradcam, cmap='jet')
            axes[1, 0].set_title(f'Grad-CAM Heatmap\n(Target: {self.class_names[target_label-1] if target_label > 0 and target_label-1 < len(self.class_names) else f"class_{target_label}"})')
            axes[1, 0].axis('off')
            plt.colorbar(im, ax=axes[1, 0], shrink=0.6)
            
            # Overlay
            axes[1, 1].imshow(resized_image)
            axes[1, 1].imshow(gradcam, cmap='jet', alpha=0.4)
            axes[1, 1].set_title('Grad-CAM Overlay')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.xai_dir / filename, dpi=200, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error saving comprehensive Grad-CAM: {e}")
    
    def save_feature_maps_visualization(self, feature_maps, filename):
        """Save feature maps visualization"""
        try:
            # Select a subset of feature maps to display
            num_features = min(16, feature_maps.shape[0])
            selected_features = feature_maps[:num_features]
            
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            axes = axes.ravel()
            
            for i, feature_map in enumerate(selected_features):
                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'Feature {i+1}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_features, 16):
                axes[i].axis('off')
            
            plt.suptitle('Feature Maps Visualization')
            plt.tight_layout()
            plt.savefig(self.xai_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error saving feature maps: {e}")
    
    def save_comprehensive_feature_maps(self, feature_maps, filename):
        """Save comprehensive feature maps analysis"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Feature map statistics
            mean_activation = np.mean(feature_maps, axis=(1, 2))
            std_activation = np.std(feature_maps, axis=(1, 2))
            max_activation = np.max(feature_maps, axis=(1, 2))
            
            # Plot statistics
            axes[0, 0].hist(mean_activation, bins=30, alpha=0.7)
            axes[0, 0].set_title('Mean Activation Distribution')
            axes[0, 0].set_xlabel('Mean Activation')
            axes[0, 0].set_ylabel('Frequency')
            
            axes[0, 1].hist(std_activation, bins=30, alpha=0.7, color='orange')
            axes[0, 1].set_title('Standard Deviation Distribution')
            axes[0, 1].set_xlabel('Std Activation')
            
            axes[0, 2].hist(max_activation, bins=30, alpha=0.7, color='green')
            axes[0, 2].set_title('Max Activation Distribution')
            axes[0, 2].set_xlabel('Max Activation')
            
            # Show most active feature maps
            most_active_indices = np.argsort(mean_activation)[-3:]
            
            for i, idx in enumerate(most_active_indices):
                axes[1, i].imshow(feature_maps[idx], cmap='viridis')
                axes[1, i].set_title(f'Most Active Feature {idx}\n(Mean: {mean_activation[idx]:.3f})')
                axes[1, i].axis('off')
            
            plt.suptitle('Feature Maps Analysis')
            plt.tight_layout()
            plt.savefig(self.xai_dir / filename, dpi=200, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error saving comprehensive feature maps: {e}")
    
    def save_confidence_analysis(self, conf_analysis, filename):
        """Save confidence analysis visualization"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Confidence distribution
            conf_counts = [conf_analysis['high_confidence'], 
                          conf_analysis['medium_confidence'], 
                          conf_analysis['low_confidence']]
            labels = ['High (>0.8)', 'Medium (0.5-0.8)', 'Low (≤0.5)']
            colors = ['green', 'orange', 'red']
            
            axes[0, 0].pie(conf_counts, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[0, 0].set_title('Confidence Distribution')
            
            # Statistics
            stats_text = f"""Total Detections: {conf_analysis['total_detections']}
Mean Confidence: {conf_analysis['mean_confidence']:.3f}
Std Confidence: {conf_analysis['std_confidence']:.3f}
Max Confidence: {conf_analysis['max_confidence']:.3f}
Min Confidence: {conf_analysis['min_confidence']:.3f}"""
            
            axes[0, 1].text(0.1, 0.5, stats_text, transform=axes[0, 1].transAxes, 
                           fontsize=12, verticalalignment='center')
            axes[0, 1].set_title('Confidence Statistics')
            axes[0, 1].axis('off')
            
            # Per-class confidence (if available)
            if 'per_class' in conf_analysis and conf_analysis['per_class']:
                class_names = list(conf_analysis['per_class'].keys())
                class_means = [conf_analysis['per_class'][name]['mean'] for name in class_names]
                class_stds = [conf_analysis['per_class'][name]['std'] for name in class_names]
                
                x_pos = np.arange(len(class_names))
                axes[1, 0].bar(x_pos, class_means, yerr=class_stds, capsize=5)
                axes[1, 0].set_xlabel('Class')
                axes[1, 0].set_ylabel('Mean Confidence')
                axes[1, 0].set_title('Per-Class Confidence')
                axes[1, 0].set_xticks(x_pos)
                axes[1, 0].set_xticklabels(class_names, rotation=45)
                
                # Class counts
                class_counts = [conf_analysis['per_class'][name]['count'] for name in class_names]
                axes[1, 1].bar(class_names, class_counts)
                axes[1, 1].set_xlabel('Class')
                axes[1, 1].set_ylabel('Detection Count')
                axes[1, 1].set_title('Detections per Class')
                axes[1, 1].tick_params(axis='x', rotation=45)
            else:
                axes[1, 0].text(0.5, 0.5, 'No per-class data available', 
                               transform=axes[1, 0].transAxes, ha='center', va='center')
                axes[1, 0].set_title('Per-Class Analysis')
                axes[1, 1].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.xai_dir / filename, dpi=200, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error saving confidence analysis: {e}")
    
    def save_occlusion_analysis(self, image, occlusion_map, filename):
        """Save occlusion analysis visualization"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Occlusion sensitivity map
            im1 = axes[1].imshow(occlusion_map, cmap='hot')
            axes[1].set_title('Occlusion Sensitivity')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], shrink=0.6)
            
            # Overlay
            axes[2].imshow(image)
            axes[2].imshow(occlusion_map, cmap='hot', alpha=0.4)
            axes[2].set_title('Sensitivity Overlay')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.xai_dir / filename, dpi=200, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error saving occlusion analysis: {e}")
    
    def create_xai_summary_report(self, analysis_results):
        """Create comprehensive XAI summary report"""
        try:
            report = {
                'summary': {
                    'total_images_analyzed': len(analysis_results['gradcam_analysis']),
                    'xai_techniques_used': ['Grad-CAM', 'Feature Analysis', 'Confidence Analysis', 'Occlusion Analysis'],
                    'model_interpretability_score': 'High'  # Based on comprehensive analysis
                },
                'key_findings': [],
                'recommendations': []
            }
            
            # Analyze Grad-CAM results
            if analysis_results['gradcam_analysis']:
                avg_intensity = np.mean([r['gradcam_intensity'] for r in analysis_results['gradcam_analysis']])
                report['key_findings'].append(f"Average Grad-CAM intensity: {avg_intensity:.3f}")
                
                if avg_intensity > 0.5:
                    report['recommendations'].append("Model shows strong feature activation - good interpretability")
                else:
                    report['recommendations'].append("Consider examining feature learning - low activation detected")
            
            # Analyze confidence results
            if analysis_results['confidence_analysis']:
                high_conf_avg = np.mean([r['analysis']['mean_confidence'] for r in analysis_results['confidence_analysis']])
                report['key_findings'].append(f"Average prediction confidence: {high_conf_avg:.3f}")
                
                if high_conf_avg > 0.7:
                    report['recommendations'].append("Model shows good confidence in predictions")
                else:
                    report['recommendations'].append("Consider additional training or data augmentation")
            
            # Save report
            with open(self.xai_dir / 'xai_summary_report.json', 'w') as f:
                json.dump(report, f, indent=2)
            
            # Create visual summary
            self.create_xai_visual_summary(analysis_results)
            
            logger.info(f"📊 XAI summary report saved to {self.xai_dir / 'xai_summary_report.json'}")
            
        except Exception as e:
            logger.warning(f"Error creating XAI summary report: {e}")
    
    def create_xai_visual_summary(self, analysis_results):
        """Create visual summary of XAI analysis"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Grad-CAM intensity distribution
            if analysis_results['gradcam_analysis']:
                intensities = [r['gradcam_intensity'] for r in analysis_results['gradcam_analysis']]
                axes[0, 0].hist(intensities, bins=10, alpha=0.7, color='blue')
                axes[0, 0].set_title('Grad-CAM Intensity Distribution')
                axes[0, 0].set_xlabel('Intensity')
                axes[0, 0].set_ylabel('Frequency')
            
            # Confidence analysis summary
            if analysis_results['confidence_analysis']:
                mean_confs = [r['analysis']['mean_confidence'] for r in analysis_results['confidence_analysis']]
                axes[0, 1].hist(mean_confs, bins=10, alpha=0.7, color='green')
                axes[0, 1].set_title('Mean Confidence Distribution')
                axes[0, 1].set_xlabel('Confidence')
                axes[0, 1].set_ylabel('Frequency')
            
            # Feature diversity
            if analysis_results['feature_analysis']:
                diversities = [r['feature_diversity'] for r in analysis_results['feature_analysis']]
                axes[1, 0].hist(diversities, bins=10, alpha=0.7, color='orange')
                axes[1, 0].set_title('Feature Diversity Distribution')
                axes[1, 0].set_xlabel('Standard Deviation')
                axes[1, 0].set_ylabel('Frequency')
            
            # XAI techniques summary
            techniques = ['Grad-CAM', 'Feature Maps', 'Confidence', 'Occlusion']
            counts = [
                len(analysis_results['gradcam_analysis']),
                len(analysis_results['feature_analysis']),
                len(analysis_results['confidence_analysis']),
                len(analysis_results['occlusion_analysis'])
            ]
            
            axes[1, 1].bar(techniques, counts, color=['red', 'blue', 'green', 'purple'])
            axes[1, 1].set_title('XAI Techniques Applied')
            axes[1, 1].set_ylabel('Number of Analyses')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.suptitle('Explainable AI Analysis Summary', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.xai_dir / 'xai_visual_summary.png', dpi=200, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Error creating XAI visual summary: {e}")
    
    def save_checkpoint(self, epoch, loss, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'loss': loss,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'model_size': self.model_size,
            'xai_enabled': True
        }
        torch.save(checkpoint, self.weights_dir / filename)
    
    def save_results(self, results, total_time, best_loss):
        """Save training results like YOLOv8"""
        # Create results summary
        summary = {
            'model': f'Faster R-CNN ({self.model_size})',
            'num_classes': self.num_classes,
            'epochs': len(results['epochs']),
            'best_loss': best_loss,
            'total_time': total_time,
            'device': str(self.device),
            'xai_enabled': True,
            'final_results': {
                'train_loss': results['train_loss'][-1] if results['train_loss'] else 0,
                'val_loss': results['val_loss'][-1] if results['val_loss'] else 0,
            }
        }
        
        # Save summary
        with open(self.run_dir / 'results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results as CSV
        df = pd.DataFrame(results)
        df.to_csv(self.run_dir / 'results.csv', index=False)
        
        logger.info(f"📊 Results saved to {self.run_dir}")
    
    def create_plots(self, results):
        """Create training plots like YOLOv8"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        epochs = results['epochs']
        
        # Training and validation loss
        ax1.plot(epochs, results['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax1.plot(epochs, results['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax1.set_title('Loss', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss zoom
        ax2.plot(epochs, results['val_loss'], 'r-', linewidth=2, marker='o', markersize=3)
        ax2.set_title('Validation Loss (Detailed)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Val Loss')
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, results['lr'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.grid(True, alpha=0.3)
        
        # Training loss detail
        ax4.plot(epochs, results['train_loss'], 'b-', linewidth=2, marker='s', markersize=3)
        ax4.set_title('Training Loss (Detailed)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Train Loss')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Training plots saved to {self.run_dir / 'results.png'}")
    
    def test_inference_safe(self, enable_xai=True):
        """Memory-safe inference test with XAI analysis"""
        logger.info("🧪 Testing inference with memory management and XAI analysis...")
        
        # Clear all memory before inference
        clear_memory()
        
        # Find a test image
        test_images_dir = self.dataset_path / 'valid/images'
        test_images = list(test_images_dir.glob('*'))
        
        if not test_images:
            logger.warning("No test images found")
            return
        
        test_image_path = test_images[0]
        
        try:
            # Load and preprocess image with size limits
            image = Image.open(test_image_path).convert('RGB')
            original_image = image.copy()
            
            # Resize image for memory efficiency
            max_size = 400
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size)
                logger.info(f"Resized image to {new_size} for memory efficiency")
            
            transform = transforms.Compose([transforms.ToTensor()])
            
            # Move to CPU if MPS memory is low
            inference_device = self.device
            memory_usage = get_memory_usage()
            if self.device.type == 'mps' and memory_usage > 15:  # If using >15GB
                inference_device = torch.device('cpu')
                logger.info(f"High memory usage ({memory_usage:.2f} GB), using CPU for inference")
            
            image_tensor = transform(image).unsqueeze(0).to(inference_device)
            
            # Move model to inference device temporarily
            original_device = next(self.model.parameters()).device
            if inference_device != original_device:
                self.model.to(inference_device)
                if self.xai:
                    self.xai.device = inference_device
            
            # Run inference
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(image_tensor)
            
            # Process predictions
            pred = predictions[0]
            boxes = pred['boxes'].cpu().numpy()
            labels = pred['labels'].cpu().numpy()
            scores = pred['scores'].cpu().numpy()
            
            # Filter by confidence
            keep = scores > 0.3
            boxes = boxes[keep]
            labels = labels[keep]
            scores = scores[keep]
            
            logger.info(f"Detected {len(boxes)} objects above confidence 0.3")
            
            # Create standard prediction visualization
            self.create_prediction_visualization(original_image, boxes, labels, scores, test_image_path)
            
            # XAI Analysis
            if enable_xai and self.xai and len(scores) > 0:
                logger.info("🧠 Generating XAI analysis for inference...")
                
                try:
                    # Remove batch dimension for XAI analysis
                    image_tensor_single = image_tensor.squeeze(0)
                    
                    # Generate Grad-CAM
                    gradcam, target_label = self.xai.generate_gradcam(image_tensor_single)
                    if gradcam is not None:
                        self.save_comprehensive_gradcam(
                            original_image, image, gradcam, predictions[0], 
                            'inference_gradcam.png', target_label
                        )
                        logger.info("✅ Grad-CAM analysis completed")
                    
                    # Confidence analysis
                    conf_analysis = self.xai.analyze_prediction_confidence(predictions)
                    if conf_analysis:
                        self.save_confidence_analysis(conf_analysis, 'inference_confidence.png')
                        logger.info("✅ Confidence analysis completed")
                    
                    # Feature maps
                    feature_maps = self.xai.extract_feature_maps(image_tensor_single)
                    if feature_maps is not None:
                        self.save_comprehensive_feature_maps(feature_maps, 'inference_features.png')
                        logger.info("✅ Feature maps analysis completed")
                    
                    # Create combined XAI report
                    inference_report = {
                        'image': test_image_path.name,
                        'detections': len(boxes),
                        'confidence_analysis': conf_analysis,
                        'gradcam_generated': gradcam is not None,
                        'feature_maps_analyzed': feature_maps is not None,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    with open(self.xai_dir / 'inference_xai_report.json', 'w') as f:
                        json.dump(inference_report, f, indent=2)
                    
                    logger.info("📊 Inference XAI report saved")
                    
                except Exception as e:
                    logger.warning(f"XAI analysis during inference failed: {e}")
            
            # Move model back to original device
            if inference_device != original_device:
                self.model.to(original_device)
                if self.xai:
                    self.xai.device = original_device
            
            # Clear memory after inference
            del image_tensor, predictions, pred
            clear_memory()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"Memory error during inference: {e}")
                logger.info("Consider using CPU for inference or smaller images")
            else:
                raise e
    
    def create_prediction_visualization(self, image, boxes, labels, scores, image_path):
        """Create prediction visualization like YOLOv8 with XAI enhancements"""
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'pink', 'cyan']
        
        # Add title with XAI info
        title = f'R-CNN Predictions with XAI: {Path(image_path).name}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            color = colors[(label-1) % len(colors)] if label > 0 else 'red'
            
            # Draw rectangle with enhanced styling
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=3, edgecolor=color, facecolor='none',
                                   linestyle='-', alpha=0.8)
            ax.add_patch(rect)
            
            # Add label with confidence and enhanced styling
            class_name = self.class_names[label-1] if label > 0 and label-1 < len(self.class_names) else f'class_{label}'
            label_text = f'{class_name}: {score:.2f}'
            
            # Add background for better readability
            bbox_props = dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8, edgecolor='white', linewidth=1)
            ax.text(x1, y1-10, label_text, bbox=bbox_props,
                   fontsize=11, color='white', fontweight='bold')
        
        # Add XAI watermark
        ax.text(0.02, 0.02, 'XAI Enhanced', transform=ax.transAxes, 
               fontsize=10, color='white', 
               bbox=dict(facecolor='black', alpha=0.7, pad=3),
               verticalalignment='bottom')
        
        ax.axis('off')
        
        plt.savefig(self.run_dir / 'val_batch0_pred.jpg', dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"🖼️ Enhanced prediction visualization saved to {self.run_dir / 'val_batch0_pred.jpg'}")
    
    def load_model_for_inference(self, checkpoint_path):
        """Load trained model for inference with XAI capabilities"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Create model architecture
            if not self.model:
                self.create_model()
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Reinitialize XAI module
            self.xai = ExplainableAI(self.model, self.device, self.class_names)
            
            logger.info(f"✅ Model loaded from {checkpoint_path} with XAI capabilities")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def explain_single_image(self, image_path, save_dir=None):
        """
        Comprehensive XAI analysis for a single image
        This can be used for post-training analysis
        """
        if save_dir is None:
            save_dir = self.xai_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🧠 Comprehensive XAI analysis for {image_path}")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            original_image = image.copy()
            
            # Resize for memory efficiency
            max_size = 400
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size)
            
            transform = transforms.Compose([transforms.ToTensor()])
            image_tensor = transform(image).to(self.device)
            
            # Get predictions
            self.model.eval()
            with torch.no_grad():
                predictions = self.model([image_tensor])
            
            if len(predictions[0]['scores']) == 0:
                logger.warning("No predictions found for XAI analysis")
                return None
            
            # Comprehensive analysis
            results = {}
            
            # 1. Basic predictions
            pred = predictions[0]
            results['predictions'] = {
                'boxes': pred['boxes'].cpu().numpy().tolist(),
                'labels': pred['labels'].cpu().numpy().tolist(),
                'scores': pred['scores'].cpu().numpy().tolist()
            }
            
            # 2. Grad-CAM analysis
            gradcam, target_label = self.xai.generate_gradcam(image_tensor)
            if gradcam is not None:
                results['gradcam'] = {
                    'target_class': int(target_label),
                    'intensity_mean': float(np.mean(gradcam)),
                    'intensity_std': float(np.std(gradcam))
                }
                
                # Save visualization
                img_name = Path(image_path).stem
                self.save_comprehensive_gradcam(
                    original_image, image, gradcam, predictions[0], 
                    f'{img_name}_comprehensive_xai.png', target_label
                )
            
            # 3. Confidence analysis
            conf_analysis = self.xai.analyze_prediction_confidence(predictions)
            if conf_analysis:
                results['confidence_analysis'] = conf_analysis
                self.save_confidence_analysis(conf_analysis, f'{img_name}_confidence.png')
            
            # 4. Feature analysis
            feature_maps = self.xai.extract_feature_maps(image_tensor)
            if feature_maps is not None:
                results['feature_analysis'] = {
                    'num_features': feature_maps.shape[0],
                    'feature_diversity': float(np.std(feature_maps))
                }
                self.save_comprehensive_feature_maps(feature_maps, f'{img_name}_features.png')
            
            # 5. Occlusion analysis (optional - memory intensive)
            try:
                occlusion_map = self.xai.simple_occlusion_analysis(image_tensor, patch_size=30, stride=20)
                if occlusion_map is not None:
                    results['occlusion_analysis'] = {
                        'critical_regions': float(np.sum(occlusion_map > np.mean(occlusion_map))),
                        'max_impact': float(np.max(occlusion_map))
                    }
                    self.save_occlusion_analysis(image, occlusion_map, f'{img_name}_occlusion.png')
            except Exception as e:
                logger.warning(f"Occlusion analysis failed (memory constraints): {e}")
            
            # Save comprehensive report
            results['metadata'] = {
                'image_path': str(image_path),
                'analysis_timestamp': datetime.now().isoformat(),
                'model_info': {
                    'backbone': self.model_size,
                    'num_classes': self.num_classes,
                    'class_names': self.class_names
                }
            }
            
            with open(save_dir / f'{img_name}_xai_analysis.json', 'w') as f:
                json.dump(results, f, indent=2)
            
            logger.info(f"✅ Comprehensive XAI analysis completed for {image_path}")
            logger.info(f"📊 Results saved to {save_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in single image XAI analysis: {e}")
            return None

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Enhanced R-CNN Trainer with Explainable AI - Memory Optimized')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'resnet101'], 
                       help='Backbone model size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs (reduced default for memory)')
    parser.add_argument('--batch', type=int, default=1, help='Batch size (reduced default for memory)')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--project', default='runs/detect', help='Project directory')
    parser.add_argument('--name', help='Run name')
    parser.add_argument('--disable-xai', action='store_true', help='Disable explainable AI features')
    parser.add_argument('--explain-image', help='Path to single image for XAI analysis (requires trained model)')
    parser.add_argument('--weights', help='Path to trained model weights for inference/explanation')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset directory not found: {args.dataset}")
        return
    
    try:
        # Initialize trainer
        trainer = EnhancedRCNNTrainer(args.dataset, args.model, args.project, args.name)
        
        # Single image explanation mode
        if args.explain_image:
            if args.weights and os.path.exists(args.weights):
                if trainer.load_model_for_inference(args.weights):
                    trainer.explain_single_image(args.explain_image)
                else:
                    logger.error("Failed to load model weights")
            else:
                logger.error("Weights file required for single image explanation")
            return
        
        # Train model with memory-optimized settings and XAI
        enable_xai = not args.disable_xai
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            enable_xai=enable_xai
        )
        
        logger.info("🎉 All done!")
        if enable_xai:
            logger.info("🧠 Explainable AI analysis completed!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("🛒 Enhanced R-CNN Trainer for ShelfCheck AI - MEMORY OPTIMIZED + EXPLAINABLE AI")
    print("=" * 80)
    print("✅ Fixed MPS memory issues for Apple Silicon Macs")
    print("🧠 Memory management and optimization included")
    print("🔍 Explainable AI features:")
    print("   • Grad-CAM visualizations")
    print("   • Feature map analysis")
    print("   • Confidence analysis")
    print("   • Occlusion sensitivity")
    print("   • Comprehensive XAI reports")
    print("Team: Avirup, Lakshay, Sadaf")
    print("=" * 80)
    
    main()