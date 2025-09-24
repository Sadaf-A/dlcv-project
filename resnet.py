#!/usr/bin/env python3
"""
Enhanced ResNet Trainer for ShelfCheck AI with Explainable AI
Organized output structure like YOLOv8 runs/
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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, resnet101, resnet18, resnet34
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

# Explainable AI imports
import torch.nn.functional as F
from torch.autograd import Variable

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GradCAM:
    """Grad-CAM implementation for ResNet visualization"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.register_hooks()
    
    def register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        """Generate Grad-CAM heatmap"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        # Generate CAM
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=[1, 2])
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().numpy(), output

class LayerCAM:
    """Layer-CAM implementation for better localization"""
    
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        self.register_hooks()
    
    def register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        for name, module in self.model.named_modules():
            if name == self.target_layer_name:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
    
    def generate_cam(self, input_tensor, class_idx=None):
        self.model.eval()
        
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        self.model.zero_grad()
        class_score = output[:, class_idx]
        class_score.backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        # Layer-CAM: element-wise multiplication
        cam = F.relu(gradients * activations)
        cam = torch.sum(cam, dim=0)
        
        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.detach().numpy(), output

class IntegratedGradients:
    """Integrated Gradients implementation"""
    
    def __init__(self, model):
        self.model = model
    
    def generate_integrated_gradients(self, input_tensor, class_idx=None, steps=50):
        """Generate integrated gradients"""
        self.model.eval()
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(input_tensor)
        
        if class_idx is None:
            output = self.model(input_tensor)
            class_idx = output.argmax(dim=1)
        
        # Generate path from baseline to input
        alphas = torch.linspace(0, 1, steps)
        integrated_gradients = torch.zeros_like(input_tensor)
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated_input = baseline + alpha * (input_tensor - baseline)
            interpolated_input.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated_input)
            
            # Backward pass
            self.model.zero_grad()
            class_score = output[:, class_idx]
            class_score.backward()
            
            # Accumulate gradients
            integrated_gradients += interpolated_input.grad
        
        # Average and scale
        integrated_gradients = integrated_gradients / steps
        integrated_gradients = integrated_gradients * (input_tensor - baseline)
        
        return integrated_gradients.detach()

class ResNetDataset(Dataset):
    """Custom dataset for ResNet training with YOLO format annotations converted to classification"""
    
    def __init__(self, images_dir, labels_dir, transforms=None, image_size=224):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.transforms = transforms
        self.image_size = image_size
        
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
        
        # Load corresponding label file
        label_name = Path(img_name).stem + '.txt'
        label_path = self.labels_dir / label_name
        
        # Extract class label (use first class found in annotation file)
        label = 0  # Default label
        
        if label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        if line:
                            parts = line.split()
                            if len(parts) >= 5:
                                label = int(parts[0])
                                break  # Use first class found
            except:
                label = 0  # Default to class 0 if error
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        return image, label, img_name

class EnhancedResNetTrainer:
    """Enhanced ResNet trainer with organized output structure and Explainable AI"""
    
    def __init__(self, dataset_path, model_size='resnet50', project='runs/classify', name=None):
        """
        Initialize trainer
        Args:
            dataset_path: Path to dataset directory
            model_size: ResNet model size ('resnet18', 'resnet34', 'resnet50', 'resnet101')
            project: Project directory (like YOLOv8)
            name: Run name (like YOLOv8)
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None
        
        # Create organized output structure like YOLOv8
        self.project = Path(project)
        if name is None:
            name = f'resnet_{model_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.run_dir = self.project / name
        self.weights_dir = self.run_dir / 'weights'
        self.xai_dir = self.run_dir / 'explainable_ai'  # New XAI directory
        
        # Create directories
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.xai_dir.mkdir(parents=True, exist_ok=True)
        
        # Create XAI subdirectories
        (self.xai_dir / 'gradcam').mkdir(exist_ok=True)
        (self.xai_dir / 'layercam').mkdir(exist_ok=True)
        (self.xai_dir / 'integrated_gradients').mkdir(exist_ok=True)
        (self.xai_dir / 'feature_importance').mkdir(exist_ok=True)
        
        # Setup file logging
        log_file = self.run_dir / 'train.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"💾 Output directory: {self.run_dir}")
        logger.info(f"🔍 Explainable AI directory: {self.xai_dir}")
        
        # Detect device
        self.device = self.detect_device()
        logger.info(f"Using device: {self.device}")
        
        # Validate dataset structure
        self.validate_dataset()
        
        # Detect classes
        self.class_names = self.detect_classes()
        self.num_classes = len(self.class_names)
        
        logger.info(f"Number of classes: {self.num_classes}")
        logger.info(f"Class names: {self.class_names}")
        
        # Save dataset info
        self.save_dataset_info()
    
    def detect_device(self):
        """Detect available device"""
        try:
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        except:
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
            'explainable_ai': True,
            'created': datetime.now().isoformat()
        }
        
        with open(self.run_dir / 'args.yaml', 'w') as f:
            import yaml
            yaml.dump(dataset_info, f, default_flow_style=False)
    
    def create_model(self):
        """Create ResNet model"""
        logger.info(f"Creating {self.model_size} model for classification...")
        
        # Load pre-trained ResNet model
        if self.model_size == 'resnet18':
            model = resnet18(pretrained=True)
        elif self.model_size == 'resnet34':
            model = resnet34(pretrained=True)
        elif self.model_size == 'resnet50':
            model = resnet50(pretrained=True)
        elif self.model_size == 'resnet101':
            model = resnet101(pretrained=True)
        else:
            raise ValueError(f"Unsupported model size: {self.model_size}")
        
        # Replace the final fully connected layer
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)
        
        # Move model to device
        model.to(self.device)
        
        self.model = model
        logger.info(f"✅ Model created with {self.num_classes} classes")
        return model
    
    def create_data_loaders(self, batch_size=32, image_size=224, return_filenames=False):
        """Create data loaders for training and validation"""
        # Data transforms
        train_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_transforms = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = ResNetDataset(
            images_dir=self.dataset_path / 'train/images',
            labels_dir=self.dataset_path / 'train/labels',
            transforms=train_transforms,
            image_size=image_size
        )
        
        val_dataset = ResNetDataset(
            images_dir=self.dataset_path / 'valid/images',
            labels_dir=self.dataset_path / 'valid/labels',
            transforms=val_transforms,
            image_size=image_size
        )
        
        # Create data loaders (num_workers=0 to avoid hanging)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0  # No hanging issues
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # No hanging issues
        )
        
        logger.info(f"Created data loaders: Train={len(train_dataset)}, Val={len(val_dataset)}")
        return train_loader, val_loader
    
    def train(self, epochs=5, batch_size=32, learning_rate=0.001, image_size=224):
        """
        Train ResNet with organized output like YOLOv8
        """
        logger.info("🚀 Starting ResNet training...")
        logger.info(f"Configuration: {epochs} epochs, batch size {batch_size}, lr {learning_rate}")
        logger.info(f"Image size: {image_size}, Device: {self.device}")
        
        start_time = time.time()
        
        # Create model
        if not self.model:
            self.create_model()
        
        # Adjust batch size for device
        if self.device.type == 'cpu':
            batch_size = min(batch_size, 16)
            logger.info(f"CPU detected - reducing batch size to {batch_size}")
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(batch_size, image_size)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # Training tracking
        best_acc = 0.0
        results = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': [],
            'lr': []
        }
        
        # Training loop
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            for batch_idx, (images, labels, _) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                
                if batch_idx % 10 == 0:
                    logger.info(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
            
            epoch_loss = running_loss / len(train_loader)
            train_acc = 100 * correct_train / total_train
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            correct_val = 0
            total_val = 0
            
            with torch.no_grad():
                for images, labels, _ in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct_val / total_val
            
            # Record results
            results['epochs'].append(epoch + 1)
            results['train_loss'].append(epoch_loss)
            results['train_acc'].append(train_acc)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)
            results['lr'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            logger.info(f'Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - '
                       f'Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                       f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                self.save_checkpoint(epoch, val_acc, 'best.pt')
                logger.info(f'✅ Best model saved with validation accuracy: {best_acc:.2f}%')
            
            # Save last model
            if epoch == epochs - 1:
                self.save_checkpoint(epoch, val_acc, 'last.pt')
            
            scheduler.step()
        
        total_time = time.time() - start_time
        
        # Save results and create plots
        self.save_results(results, total_time, best_acc)
        self.create_plots(results)
        
        # Final evaluation
        self.evaluate_model()
        
        # Generate explainable AI visualizations
        self.generate_explainable_ai()
        
        logger.info(f"✅ Training completed in {total_time:.1f}s!")
        logger.info(f"📊 Best validation accuracy: {best_acc:.2f}%")
        logger.info(f"🔍 Explainable AI visualizations generated")
        logger.info(f"💾 Results saved to: {self.run_dir}")
        
        return results
    
    def save_checkpoint(self, epoch, accuracy, filename):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'accuracy': accuracy,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'model_size': self.model_size
        }
        torch.save(checkpoint, self.weights_dir / filename)
    
    def save_results(self, results, total_time, best_acc):
        """Save training results like YOLOv8"""
        # Create results summary
        summary = {
            'model': self.model_size,
            'num_classes': self.num_classes,
            'epochs': len(results['epochs']),
            'best_accuracy': best_acc,
            'total_time': total_time,
            'device': str(self.device),
            'explainable_ai': True,
            'final_results': {
                'train_loss': results['train_loss'][-1],
                'train_acc': results['train_acc'][-1],
                'val_loss': results['val_loss'][-1],
                'val_acc': results['val_acc'][-1]
            }
        }
        
        # Save summary
        with open(self.run_dir / 'results.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results as CSV (like YOLOv8)
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
        
        # Training and validation accuracy
        ax2.plot(epochs, results['train_acc'], 'b-', label='Train Acc', linewidth=2)
        ax2.plot(epochs, results['val_acc'], 'r-', label='Val Acc', linewidth=2)
        ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(epochs, results['lr'], 'g-', linewidth=2)
        ax3.set_title('Learning Rate', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.grid(True, alpha=0.3)
        
        # Validation accuracy zoom
        ax4.plot(epochs, results['val_acc'], 'r-', linewidth=2, marker='o', markersize=3)
        ax4.set_title('Validation Accuracy (Detailed)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Val Accuracy (%)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / 'results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Training plots saved to {self.run_dir / 'results.png'}")
    
    def evaluate_model(self):
        """Create detailed evaluation like YOLOv8"""
        logger.info("📊 Creating detailed evaluation...")
        
        # Load validation data
        _, val_loader = self.create_data_loaders(batch_size=32)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_confidences = []
        
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidences, predicted = torch.max(probabilities, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
        
        # Create confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.run_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Classification report
        report = classification_report(all_labels, all_predictions, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        with open(self.run_dir / 'classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Evaluation completed - Confusion matrix and report saved")
    
    def get_target_layer(self):
        """Get target layer for CAM visualization based on model size"""
        if self.model_size in ['resnet18', 'resnet34']:
            return 'layer4.1.conv2'
        else:  # resnet50, resnet101
            return 'layer4.2.conv3'
    
    def normalize_image_for_display(self, img_tensor):
        """Normalize image tensor for display"""
        # Denormalize using ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        if img_tensor.device != mean.device:
            mean = mean.to(img_tensor.device)
            std = std.to(img_tensor.device)
        
        denorm = img_tensor * std + mean
        denorm = torch.clamp(denorm, 0, 1)
        
        # Convert to numpy
        img_np = denorm.squeeze().permute(1, 2, 0).cpu().numpy()
        return img_np
    
    def overlay_heatmap(self, img_np, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        # Resize heatmap to match image
        heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Normalize
        heatmap_colored = heatmap_colored.astype(np.float32) / 255
        
        # Overlay
        overlayed = (1 - alpha) * img_np + alpha * heatmap_colored
        overlayed = np.clip(overlayed, 0, 1)
        
        return overlayed
    
    def generate_explainable_ai(self, num_samples=20):
        """Generate comprehensive explainable AI visualizations"""
        logger.info("🔍 Generating Explainable AI visualizations...")
        
        # Load best model
        best_model_path = self.weights_dir / 'best.pt'
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("✅ Loaded best model for XAI analysis")
        
        # Create validation dataset without data augmentation
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        val_dataset = ResNetDataset(
            images_dir=self.dataset_path / 'valid/images',
            labels_dir=self.dataset_path / 'valid/labels',
            transforms=val_transforms,
            image_size=224
        )
        
        # Initialize XAI methods
        target_layer = self.get_target_layer()
        gradcam = GradCAM(self.model, target_layer)
        layercam = LayerCAM(self.model, target_layer)
        integrated_grads = IntegratedGradients(self.model)
        
        # Sample images from each class
        samples_per_class = max(1, num_samples // self.num_classes)
        class_samples = {i: [] for i in range(self.num_classes)}
        
        # Collect samples
        for idx, (image, label, filename) in enumerate(val_dataset):
            if len(class_samples[label]) < samples_per_class:
                class_samples[label].append((idx, image, label, filename))
            
            if sum(len(samples) for samples in class_samples.values()) >= num_samples:
                break
        
        # Generate XAI for selected samples
        xai_results = []
        
        for class_id, samples in class_samples.items():
            if not samples:
                continue
                
            logger.info(f"Processing class {class_id} ({self.class_names[class_id]})...")
            
            for idx, (sample_idx, image, label, filename) in enumerate(samples):
                try:
                    # Prepare input
                    input_tensor = image.unsqueeze(0).to(self.device)
                    
                    # Get model prediction
                    with torch.no_grad():
                        output = self.model(input_tensor)
                        probabilities = F.softmax(output, dim=1)
                        confidence, predicted = torch.max(probabilities, 1)
                        predicted_class = predicted.item()
                        confidence_score = confidence.item()
                    
                    # Generate Grad-CAM
                    gradcam_heatmap, _ = gradcam.generate_cam(input_tensor, class_idx=predicted_class)
                    
                    # Generate Layer-CAM
                    layercam_heatmap, _ = layercam.generate_cam(input_tensor, class_idx=predicted_class)
                    
                    # Generate Integrated Gradients
                    ig_result = integrated_grads.generate_integrated_gradients(input_tensor, class_idx=predicted_class)
                    ig_magnitude = torch.norm(ig_result, dim=1).squeeze().cpu().numpy()
                    ig_magnitude = (ig_magnitude - ig_magnitude.min()) / (ig_magnitude.max() - ig_magnitude.min())
                    
                    # Normalize original image for display
                    img_display = self.normalize_image_for_display(input_tensor)
                    
                    # Create visualizations
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    
                    # Original image
                    axes[0, 0].imshow(img_display)
                    axes[0, 0].set_title(f'Original Image\nTrue: {self.class_names[label]}\nPred: {self.class_names[predicted_class]}\nConf: {confidence_score:.3f}', 
                                        fontsize=12, fontweight='bold')
                    axes[0, 0].axis('off')
                    
                    # Grad-CAM
                    gradcam_overlay = self.overlay_heatmap(img_display, gradcam_heatmap)
                    axes[0, 1].imshow(gradcam_overlay)
                    axes[0, 1].set_title('Grad-CAM', fontsize=12, fontweight='bold')
                    axes[0, 1].axis('off')
                    
                    # Layer-CAM
                    layercam_overlay = self.overlay_heatmap(img_display, layercam_heatmap)
                    axes[0, 2].imshow(layercam_overlay)
                    axes[0, 2].set_title('Layer-CAM', fontsize=12, fontweight='bold')
                    axes[0, 2].axis('off')
                    
                    # Grad-CAM heatmap only
                    im1 = axes[1, 0].imshow(gradcam_heatmap, cmap='jet')
                    axes[1, 0].set_title('Grad-CAM Heatmap', fontsize=12, fontweight='bold')
                    axes[1, 0].axis('off')
                    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046)
                    
                    # Layer-CAM heatmap only
                    im2 = axes[1, 1].imshow(layercam_heatmap, cmap='jet')
                    axes[1, 1].set_title('Layer-CAM Heatmap', fontsize=12, fontweight='bold')
                    axes[1, 1].axis('off')
                    plt.colorbar(im2, ax=axes[1, 1], fraction=0.046)
                    
                    # Integrated Gradients
                    im3 = axes[1, 2].imshow(ig_magnitude, cmap='hot')
                    axes[1, 2].set_title('Integrated Gradients', fontsize=12, fontweight='bold')
                    axes[1, 2].axis('off')
                    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046)
                    
                    plt.tight_layout()
                    
                    # Save visualization
                    save_name = f'class_{class_id}_{self.class_names[class_id]}_sample_{idx}_{filename.split(".")[0]}'
                    plt.savefig(self.xai_dir / f'{save_name}_analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Save individual heatmaps
                    plt.figure(figsize=(8, 8))
                    plt.imshow(gradcam_overlay)
                    plt.title(f'Grad-CAM: {self.class_names[predicted_class]} (Conf: {confidence_score:.3f})', 
                             fontsize=14, fontweight='bold')
                    plt.axis('off')
                    plt.savefig(self.xai_dir / 'gradcam' / f'{save_name}_gradcam.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    plt.figure(figsize=(8, 8))
                    plt.imshow(layercam_overlay)
                    plt.title(f'Layer-CAM: {self.class_names[predicted_class]} (Conf: {confidence_score:.3f})', 
                             fontsize=14, fontweight='bold')
                    plt.axis('off')
                    plt.savefig(self.xai_dir / 'layercam' / f'{save_name}_layercam.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    plt.figure(figsize=(8, 8))
                    plt.imshow(ig_magnitude, cmap='hot')
                    plt.title(f'Integrated Gradients: {self.class_names[predicted_class]} (Conf: {confidence_score:.3f})', 
                             fontsize=14, fontweight='bold')
                    plt.axis('off')
                    plt.colorbar(fraction=0.046)
                    plt.savefig(self.xai_dir / 'integrated_gradients' / f'{save_name}_ig.png', dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    # Store results
                    xai_results.append({
                        'filename': filename,
                        'true_class': self.class_names[label],
                        'predicted_class': self.class_names[predicted_class],
                        'confidence': confidence_score,
                        'correct_prediction': label == predicted_class,
                        'gradcam_file': f'gradcam/{save_name}_gradcam.png',
                        'layercam_file': f'layercam/{save_name}_layercam.png',
                        'ig_file': f'integrated_gradients/{save_name}_ig.png',
                        'analysis_file': f'{save_name}_analysis.png'
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to generate XAI for {filename}: {e}")
                    continue
        
        # Generate feature importance analysis
        self.analyze_feature_importance()
        
        # Generate class activation summary
        self.generate_class_activation_summary(xai_results)
        
        # Save XAI results
        with open(self.xai_dir / 'xai_results.json', 'w') as f:
            json.dump(xai_results, f, indent=2)
        
        logger.info(f"🔍 Explainable AI analysis completed!")
        logger.info(f"Generated visualizations for {len(xai_results)} samples")
        logger.info(f"XAI results saved to: {self.xai_dir}")
    
    def analyze_feature_importance(self):
        """Analyze feature importance across layers"""
        logger.info("🧠 Analyzing feature importance...")
        
        # Get layer activations
        activation_stats = {}
        
        def get_activation_stats(name):
            def hook(model, input, output):
                if isinstance(output, torch.Tensor):
                    activation_stats[name] = {
                        'mean': output.mean().item(),
                        'std': output.std().item(),
                        'max': output.max().item(),
                        'min': output.min().item(),
                        'shape': list(output.shape)
                    }
            return hook
        
        # Register hooks for key layers
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and any(layer in name for layer in ['layer1', 'layer2', 'layer3', 'layer4', 'fc']):
                hook = module.register_forward_hook(get_activation_stats(name))
                hooks.append(hook)
        
        # Run inference on a few samples
        val_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_dataset = ResNetDataset(
            images_dir=self.dataset_path / 'valid/images',
            labels_dir=self.dataset_path / 'valid/labels',
            transforms=val_transforms
        )
        
        self.model.eval()
        with torch.no_grad():
            for i in range(min(10, len(val_dataset))):
                image, _, _ = val_dataset[i]
                input_tensor = image.unsqueeze(0).to(self.device)
                _ = self.model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Create feature importance visualization
        layer_names = list(activation_stats.keys())
        layer_means = [activation_stats[name]['mean'] for name in layer_names]
        layer_stds = [activation_stats[name]['std'] for name in layer_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Mean activations
        bars1 = ax1.bar(range(len(layer_names)), layer_means, alpha=0.7, color='skyblue')
        ax1.set_title('Mean Layer Activations', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Mean Activation')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Standard deviation
        bars2 = ax2.bar(range(len(layer_names)), layer_stds, alpha=0.7, color='lightcoral')
        ax2.set_title('Layer Activation Standard Deviation', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Standard Deviation')
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(self.xai_dir / 'feature_importance' / 'layer_activation_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save activation statistics
        with open(self.xai_dir / 'feature_importance' / 'activation_stats.json', 'w') as f:
            json.dump(activation_stats, f, indent=2)
        
        logger.info("✅ Feature importance analysis completed")
    
    def generate_class_activation_summary(self, xai_results):
        """Generate summary of class activations"""
        logger.info("📊 Generating class activation summary...")
        
        # Analyze results by class
        class_analysis = {}
        
        for result in xai_results:
            true_class = result['true_class']
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            correct = result['correct_prediction']
            
            if true_class not in class_analysis:
                class_analysis[true_class] = {
                    'total_samples': 0,
                    'correct_predictions': 0,
                    'confidences': [],
                    'misclassified_as': {}
                }
            
            class_analysis[true_class]['total_samples'] += 1
            class_analysis[true_class]['confidences'].append(confidence)
            
            if correct:
                class_analysis[true_class]['correct_predictions'] += 1
            else:
                if predicted_class not in class_analysis[true_class]['misclassified_as']:
                    class_analysis[true_class]['misclassified_as'][predicted_class] = 0
                class_analysis[true_class]['misclassified_as'][predicted_class] += 1
        
        # Calculate statistics
        for class_name in class_analysis:
            stats = class_analysis[class_name]
            stats['accuracy'] = stats['correct_predictions'] / stats['total_samples']
            stats['avg_confidence'] = np.mean(stats['confidences'])
            stats['confidence_std'] = np.std(stats['confidences'])
        
        # Create visualization
        classes = list(class_analysis.keys())
        accuracies = [class_analysis[cls]['accuracy'] for cls in classes]
        confidences = [class_analysis[cls]['avg_confidence'] for cls in classes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Class accuracies
        bars1 = ax1.bar(classes, accuracies, alpha=0.7, color='lightgreen')
        ax1.set_title('Class-wise Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars1):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Average confidences
        bars2 = ax2.bar(classes, confidences, alpha=0.7, color='orange')
        ax2.set_title('Average Prediction Confidence', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Confidence')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.xai_dir / 'class_activation_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed analysis
        with open(self.xai_dir / 'class_analysis.json', 'w') as f:
            json.dump(class_analysis, f, indent=2, default=str)
        
        # Create XAI summary report
        self.create_xai_report(class_analysis, xai_results)
        
        logger.info("✅ Class activation summary completed")
    
    def create_xai_report(self, class_analysis, xai_results):
        """Create comprehensive XAI report"""
        logger.info("📝 Creating XAI report...")
        
        report = {
            'model_info': {
                'model_size': self.model_size,
                'num_classes': self.num_classes,
                'class_names': self.class_names,
                'target_layer': self.get_target_layer()
            },
            'xai_methods': [
                'Grad-CAM',
                'Layer-CAM', 
                'Integrated Gradients',
                'Feature Importance Analysis'
            ],
            'summary_stats': {
                'total_samples_analyzed': len(xai_results),
                'overall_accuracy': np.mean([r['correct_prediction'] for r in xai_results]),
                'average_confidence': np.mean([r['confidence'] for r in xai_results])
            },
            'class_performance': class_analysis,
            'insights': {
                'best_performing_class': max(class_analysis.keys(), 
                                           key=lambda k: class_analysis[k]['accuracy']),
                'most_confident_class': max(class_analysis.keys(),
                                          key=lambda k: class_analysis[k]['avg_confidence']),
                'most_confused_pairs': []
            },
            'files_generated': {
                'individual_analyses': len(xai_results),
                'gradcam_visualizations': len([r for r in xai_results if 'gradcam_file' in r]),
                'layercam_visualizations': len([r for r in xai_results if 'layercam_file' in r]),
                'integrated_gradients': len([r for r in xai_results if 'ig_file' in r])
            }
        }
        
        # Find most confused pairs
        confusion_pairs = {}
        for class_name, stats in class_analysis.items():
            for misclass, count in stats.get('misclassified_as', {}).items():
                pair = tuple(sorted([class_name, misclass]))
                if pair not in confusion_pairs:
                    confusion_pairs[pair] = 0
                confusion_pairs[pair] += count
        
        # Sort by confusion frequency
        sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
        report['insights']['most_confused_pairs'] = [
            {'classes': list(pair), 'confusion_count': count}
            for pair, count in sorted_pairs[:5]
        ]
        
        # Save report
        with open(self.xai_dir / 'xai_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create markdown summary
        markdown_content = f"""# Explainable AI Report

## Model Information
- **Model**: {self.model_size}
- **Classes**: {self.num_classes}
- **Target Layer**: {self.get_target_layer()}

## Analysis Summary
- **Total Samples Analyzed**: {len(xai_results)}
- **Overall Accuracy**: {report['summary_stats']['overall_accuracy']:.3f}
- **Average Confidence**: {report['summary_stats']['average_confidence']:.3f}

## XAI Methods Used
- Grad-CAM (Gradient-weighted Class Activation Mapping)
- Layer-CAM (Layer-wise Class Activation Mapping) 
- Integrated Gradients
- Feature Importance Analysis

## Class Performance
"""
        
        for class_name, stats in class_analysis.items():
            markdown_content += f"""
### {class_name}
- **Accuracy**: {stats['accuracy']:.3f}
- **Average Confidence**: {stats['avg_confidence']:.3f}
- **Samples**: {stats['total_samples']}
"""
        
        markdown_content += f"""
## Key Insights
- **Best Performing Class**: {report['insights']['best_performing_class']}
- **Most Confident Class**: {report['insights']['most_confident_class']}

## Files Generated
- **Individual Analyses**: {report['files_generated']['individual_analyses']}
- **Grad-CAM Visualizations**: {report['files_generated']['gradcam_visualizations']}  
- **Layer-CAM Visualizations**: {report['files_generated']['layercam_visualizations']}
- **Integrated Gradients**: {report['files_generated']['integrated_gradients']}

## Directory Structure
```
explainable_ai/
├── gradcam/              # Grad-CAM visualizations
├── layercam/             # Layer-CAM visualizations  
├── integrated_gradients/ # Integrated Gradients
├── feature_importance/   # Feature analysis
├── xai_report.json      # Detailed report
├── class_analysis.json  # Class-wise analysis
└── *.png               # Combined visualizations
```
"""
        
        with open(self.xai_dir / 'README.md', 'w') as f:
            f.write(markdown_content)
        
        logger.info("✅ XAI report created")

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Enhanced ResNet Trainer with Explainable AI')
    parser.add_argument('--dataset', required=True, help='Path to dataset directory')
    parser.add_argument('--model', default='resnet50', choices=['resnet18', 'resnet34', 'resnet50', 'resnet101'], 
                       help='ResNet model size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--project', default='runs/classify', help='Project directory')
    parser.add_argument('--name', help='Run name')
    parser.add_argument('--xai-samples', type=int, default=20, help='Number of samples for XAI analysis')
    
    args = parser.parse_args()
    
    # Check if dataset exists
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset directory not found: {args.dataset}")
        return
    
    try:
        # Initialize trainer
        trainer = EnhancedResNetTrainer(args.dataset, args.model, args.project, args.name)
        
        # Train model
        results = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch,
            learning_rate=args.lr,
            image_size=args.image_size
        )
        
        logger.info("🎉 All done!")
        logger.info(f"📁 Check results in: {trainer.run_dir}")
        logger.info(f"🔍 XAI visualizations in: {trainer.xai_dir}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        raise

if __name__ == '__main__':
    print("🛒 Enhanced ResNet Trainer for ShelfCheck AI with Explainable AI")
    print("=" * 60)
    print("Organized output structure like YOLOv8 + Comprehensive XAI Analysis")
    print("Team: Avirup, Lakshay, Sadaf")
    print("=" * 60)
    
    main()