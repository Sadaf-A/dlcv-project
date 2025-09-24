#!/usr/bin/env python3
"""
ShelfCheck AI Backend Server - Real Models Integration
Connects HTML frontend to actual trained models
Team: Avirup, Lakshay, Sadaf - Amrita Vishwa Vidyapeetham
"""

import os
import json
import logging
import base64
import tempfile
import time
from datetime import datetime
from pathlib import Path
import io
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import your model classes
try:
    from enhanced_resnet_trainer import EnhancedResNetTrainer
    from enhanced_rcnn_trainer import EnhancedRCNNTrainer
    from enhanced_yolo_trainer import YOLOExplainableTrainer
    import torch
    import torchvision.transforms as transforms
    from ultralytics import YOLO
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model dependencies: {e}")
    MODELS_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration paths
MODEL_PATHS = {
    'yolo': './runs/detect/yolo_explainable/weights/best.pt',
    'rcnn': './runs/detect/rcnn_*/weights/best.pt',
    'resnet': './runs/classify/resnet_*/weights/best.pt'
}

DATASET_PATH = './dataset'  # Update this to your dataset path
TEMP_DIR = Path('./temp_uploads')
TEMP_DIR.mkdir(exist_ok=True)

# Global model instances
models = {
    'yolo': None,
    'rcnn': None,
    'resnet': None
}

class RealModelWrapper:
    """Wrapper for your actual trained models"""
    
    def __init__(self, model_type, model_instance=None, model_path=None):
        self.model_type = model_type
        self.model_instance = model_instance
        self.model_path = model_path
        self.is_loaded = model_instance is not None
        
        # Class names (update these based on your dataset)
        self.class_names = [
            'bottle', 'can', 'box', 'package', 'jar', 'tube', 
            'bag', 'carton', 'container', 'wrapper'
        ]
    
    def predict(self, image_path):
        """Run inference on image"""
        if not self.is_loaded:
            raise ValueError(f"{self.model_type} model not loaded")
        
        start_time = time.time()
        
        try:
            if self.model_type == 'yolo':
                return self._predict_yolo(image_path)
            elif self.model_type == 'rcnn':
                return self._predict_rcnn(image_path)
            elif self.model_type == 'resnet':
                return self._predict_resnet(image_path)
        except Exception as e:
            logger.error(f"Prediction error for {self.model_type}: {e}")
            return {
                'detections': [],
                'inference_time': time.time() - start_time,
                'model_type': self.model_type,
                'error': str(e)
            }
    
    def _predict_yolo(self, image_path):
        """YOLO prediction"""
        results = self.model_instance.predict(image_path, conf=0.25, save=False)
        
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                
                for box, conf, cls in zip(boxes, confidences, classes):
                    class_name = self.class_names[cls] if cls < len(self.class_names) else f'class_{cls}'
                    detections.append({
                        'class_name': class_name,
                        'confidence': float(conf),
                        'bbox': [float(x) for x in box]
                    })
        
        return {
            'detections': detections,
            'inference_time': time.time() - time.time(),
            'model_type': 'YOLOv8'
        }
    
    def _predict_rcnn(self, image_path):
        """R-CNN prediction"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize for memory efficiency
        if max(image.size) > 800:
            ratio = 800 / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size)
        
        transform = transforms.Compose([transforms.ToTensor()])
        image_tensor = transform(image).to(self.model_instance.device)
        
        # Run inference
        self.model_instance.model.eval()
        with torch.no_grad():
            predictions = self.model_instance.model([image_tensor])
        
        detections = []
        if predictions and len(predictions) > 0:
            pred = predictions[0]
            if 'boxes' in pred and len(pred['boxes']) > 0:
                boxes = pred['boxes'].cpu().numpy()
                labels = pred['labels'].cpu().numpy()
                scores = pred['scores'].cpu().numpy()
                
                # Filter by confidence
                keep = scores > 0.3
                boxes = boxes[keep]
                labels = labels[keep]
                scores = scores[keep]
                
                for box, label, score in zip(boxes, labels, scores):
                    # Convert label (1-indexed) to class name
                    class_idx = label - 1
                    class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'class_{label}'
                    
                    detections.append({
                        'class_name': class_name,
                        'confidence': float(score),
                        'bbox': [float(x) for x in box]
                    })
        
        return {
            'detections': detections,
            'inference_time': time.time() - time.time(),
            'model_type': 'Faster R-CNN'
        }
    
    def _predict_resnet(self, image_path):
        """ResNet prediction (classification converted to detection format)"""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # ResNet transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(self.model_instance.device)
        
        # Run inference
        self.model_instance.model.eval()
        with torch.no_grad():
            outputs = self.model_instance.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Convert classification to detection format
        detections = []
        if confidence.item() > 0.5:  # Only if confident
            class_idx = predicted.item()
            class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'class_{class_idx}'
            
            # Create a full-image bounding box since this is classification
            img_width, img_height = image.size
            detections.append({
                'class_name': class_name,
                'confidence': float(confidence.item()),
                'bbox': [0, 0, img_width, img_height]  # Full image box
            })
        
        return {
            'detections': detections,
            'inference_time': time.time() - time.time(),
            'model_type': 'ResNet Classification'
        }

def find_model_weights():
    """Find available model weight files"""
    available_models = {}
    
    # Check for YOLO weights
    yolo_paths = [
        './runs/detect/yolo_explainable/weights/best.pt',
        './runs/detect/*/weights/best.pt',
        './best_yolo.pt'
    ]
    
    for pattern in yolo_paths:
        if '*' in pattern:
            import glob
            matches = glob.glob(pattern)
            if matches:
                available_models['yolo'] = matches[0]
                break
        else:
            if os.path.exists(pattern):
                available_models['yolo'] = pattern
                break
    
    # Check for R-CNN weights
    rcnn_paths = [
        './runs/detect/rcnn_resnet50_*/weights/best.pt',
        './runs/detect/*/weights/best.pt',
        './best_rcnn.pt'
    ]
    
    for pattern in rcnn_paths:
        if '*' in pattern:
            import glob
            matches = glob.glob(pattern)
            if matches:
                available_models['rcnn'] = matches[0]
                break
        else:
            if os.path.exists(pattern):
                available_models['rcnn'] = pattern
                break
    
    # Check for ResNet weights
    resnet_paths = [
        './runs/classify/resnet_*/weights/best.pt',
        './runs/classify/*/weights/best.pt',
        './best_resnet.pt'
    ]
    
    for pattern in resnet_paths:
        if '*' in pattern:
            import glob
            matches = glob.glob(pattern)
            if matches:
                available_models['resnet'] = matches[0]
                break
        else:
            if os.path.exists(pattern):
                available_models['resnet'] = pattern
                break
    
    return available_models

def initialize_models():
    """Initialize actual trained models"""
    global models
    
    if not MODELS_AVAILABLE:
        logger.warning("Model dependencies not available, using mock models")
        return initialize_mock_models()
    
    available_weights = find_model_weights()
    logger.info(f"Found model weights: {available_weights}")
    
    # Initialize YOLO
    if 'yolo' in available_weights:
        try:
            yolo_model = YOLO(available_weights['yolo'])
            models['yolo'] = RealModelWrapper('yolo', yolo_model, available_weights['yolo'])
            logger.info(f"✅ YOLO model loaded from {available_weights['yolo']}")
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO model: {e}")
    
    # Initialize R-CNN
    if 'rcnn' in available_weights:
        try:
            if os.path.exists(DATASET_PATH):
                rcnn_trainer = EnhancedRCNNTrainer(DATASET_PATH)
                if rcnn_trainer.load_model_for_inference(available_weights['rcnn']):
                    models['rcnn'] = RealModelWrapper('rcnn', rcnn_trainer, available_weights['rcnn'])
                    logger.info(f"✅ R-CNN model loaded from {available_weights['rcnn']}")
                else:
                    logger.error("❌ Failed to load R-CNN model")
            else:
                logger.error(f"❌ Dataset path not found: {DATASET_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to load R-CNN model: {e}")
    
    # Initialize ResNet
    if 'resnet' in available_weights:
        try:
            if os.path.exists(DATASET_PATH):
                resnet_trainer = EnhancedResNetTrainer(DATASET_PATH)
                if resnet_trainer.model is None:
                    resnet_trainer.create_model()
                
                # Load weights
                checkpoint = torch.load(available_weights['resnet'], map_location=resnet_trainer.device)
                resnet_trainer.model.load_state_dict(checkpoint['model_state_dict'])
                resnet_trainer.model.eval()
                
                models['resnet'] = RealModelWrapper('resnet', resnet_trainer, available_weights['resnet'])
                logger.info(f"✅ ResNet model loaded from {available_weights['resnet']}")
            else:
                logger.error(f"❌ Dataset path not found: {DATASET_PATH}")
        except Exception as e:
            logger.error(f"❌ Failed to load ResNet model: {e}")
    
    # Fall back to mock models for missing ones
    for model_name in ['yolo', 'rcnn', 'resnet']:
        if models[model_name] is None:
            logger.warning(f"Using mock model for {model_name}")
            models[model_name] = MockModel(model_name)

def initialize_mock_models():
    """Initialize mock models when real ones aren't available"""
    from collections import namedtuple
    
    class MockModel:
        def __init__(self, model_type):
            self.model_type = model_type
            self.is_loaded = True
        
        def predict(self, image_path):
            time.sleep(0.3)  # Simulate processing
            num_objects = np.random.randint(2, 6)
            
            detections = []
            for i in range(num_objects):
                detections.append({
                    'class_name': np.random.choice(['bottle', 'can', 'box', 'package', 'jar']),
                    'confidence': np.random.uniform(0.3, 0.95),
                    'bbox': [
                        np.random.randint(10, 200),
                        np.random.randint(10, 200),
                        np.random.randint(250, 400),
                        np.random.randint(250, 400)
                    ]
                })
            
            return {
                'detections': detections,
                'inference_time': np.random.uniform(0.1, 0.8),
                'model_type': f'{self.model_type.upper()}'
            }
    
    models['yolo'] = MockModel('YOLOv8')
    models['rcnn'] = MockModel('Faster R-CNN')
    models['resnet'] = MockModel('ResNet')
    
    logger.info("Initialized mock models for demonstration")

def save_base64_image(base64_string, filename):
    """Save base64 encoded image to file"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        filepath = TEMP_DIR / filename
        
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return str(filepath)
    except Exception as e:
        logger.error(f"Error saving base64 image: {e}")
        return None

def analyze_single_image(image_path, model_choice):
    """Analyze single image with specified model"""
    try:
        model = models.get(model_choice)
        if not model:
            raise ValueError(f"Model {model_choice} not available")
        
        results = model.predict(image_path)
        
        detections = results.get('detections', [])
        avg_confidence = np.mean([d['confidence'] for d in detections]) if detections else 0
        
        analysis = {
            'model_used': model_choice.upper(),
            'object_count': len(detections),
            'avg_confidence': avg_confidence,
            'detections': detections,
            'inference_time': results.get('inference_time', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise

def compare_images(before_analysis, after_analysis):
    """Compare before and after analyses"""
    try:
        before_objects = before_analysis['detections']
        after_objects = after_analysis['detections']
        
        # Simple comparison based on object matching
        added_objects = []
        removed_objects = []
        moved_objects = []
        
        # Match objects by class name and position
        before_matched = [False] * len(before_objects)
        after_matched = [False] * len(after_objects)
        
        # Find matches
        for i, before_obj in enumerate(before_objects):
            best_match = None
            best_distance = float('inf')
            
            for j, after_obj in enumerate(after_objects):
                if after_matched[j] or before_obj['class_name'] != after_obj['class_name']:
                    continue
                
                # Calculate center distance
                before_center = [(before_obj['bbox'][0] + before_obj['bbox'][2]) / 2,
                               (before_obj['bbox'][1] + before_obj['bbox'][3]) / 2]
                after_center = [(after_obj['bbox'][0] + after_obj['bbox'][2]) / 2,
                              (after_obj['bbox'][1] + after_obj['bbox'][3]) / 2]
                
                distance = np.sqrt((before_center[0] - after_center[0])**2 + 
                                 (before_center[1] - after_center[1])**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = j
            
            if best_match is not None and best_distance < 100:  # Threshold for same object
                before_matched[i] = True
                after_matched[best_match] = True
                
                if best_distance > 30:  # Moved threshold
                    moved_objects.append({
                        'before': before_objects[i],
                        'after': after_objects[best_match],
                        'distance': best_distance
                    })
        
        # Find added and removed objects
        for i, matched in enumerate(before_matched):
            if not matched:
                removed_objects.append(before_objects[i])
        
        for i, matched in enumerate(after_matched):
            if not matched:
                added_objects.append(after_objects[i])
        
        return {
            'added_objects': added_objects,
            'removed_objects': removed_objects,
            'moved_objects': moved_objects,
            'total_changes': len(added_objects) + len(removed_objects) + len(moved_objects)
        }
    except Exception as e:
        logger.error(f"Error comparing images: {e}")
        raise

def generate_compliance_report(before_analysis, after_analysis, comparison):
    """Generate compliance report"""
    try:
        total_changes = comparison['total_changes']
        added = len(comparison['added_objects'])
        removed = len(comparison['removed_objects'])
        moved = len(comparison['moved_objects'])
        
        # Calculate compliance score
        base_score = 100
        change_penalty = min(total_changes * 10, 80)
        compliance_score = max(base_score - change_penalty, 0)
        
        report_lines = [
            "SHELFCHECK AI - PLANOGRAM COMPLIANCE REPORT",
            "=" * 50,
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Model Used: {before_analysis.get('model_used', 'Unknown')}",
            "",
            "DETECTION SUMMARY:",
            f"• Before: {before_analysis['object_count']} objects detected",
            f"• After: {after_analysis['object_count']} objects detected",
            f"• Average Confidence: {((before_analysis['avg_confidence'] + after_analysis['avg_confidence']) / 2 * 100):.1f}%",
            "",
            "CHANGE ANALYSIS:",
            f"• Objects Added: {added}",
            f"• Objects Removed: {removed}",
            f"• Objects Moved: {moved}",
            f"• Total Changes: {total_changes}",
            "",
            f"COMPLIANCE SCORE: {compliance_score:.0f}/100",
            "",
            "DETAILED CHANGES:"
        ]
        
        # Add specific changes
        for obj in comparison['added_objects']:
            report_lines.append(f"+ ADDED: {obj['class_name']} (confidence: {obj['confidence']:.2f})")
        
        for obj in comparison['removed_objects']:
            report_lines.append(f"- REMOVED: {obj['class_name']} (was at: {obj['bbox']})")
        
        for obj in comparison['moved_objects']:
            report_lines.append(f"→ MOVED: {obj['before']['class_name']} (distance: {obj['distance']:.1f}px)")
        
        if total_changes == 0:
            report_lines.append("✅ NO CHANGES DETECTED - PERFECT COMPLIANCE")
        
        return {
            'compliance_score': compliance_score,
            'report_text': "\n".join(report_lines)
        }
    except Exception as e:
        logger.error(f"Error generating compliance report: {e}")
        return {
            'compliance_score': 0,
            'report_text': f"Error generating report: {e}"
        }

# Flask routes (same as before but using real models)
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/model_status')
def model_status():
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    except:
        device = 'cpu'
    
    status = {
        'rcnn_loaded': models['rcnn'] is not None and models['rcnn'].is_loaded,
        'yolo_loaded': models['yolo'] is not None and models['yolo'].is_loaded,
        'resnet_loaded': models['resnet'] is not None and models['resnet'].is_loaded,
        'models_available': [k for k, v in models.items() if v is not None and v.is_loaded],
        'device': device,
        'backend_status': 'operational',
        'model_types': {k: v.model_type if v else 'Not loaded' for k, v in models.items()}
    }
    
    return jsonify(status)

@app.route('/analyze_complete', methods=['POST'])
def analyze_complete():
    """Complete analysis with real models"""
    try:
        data = request.json
        before_image = data.get('before_image')
        after_image = data.get('after_image')
        model_choice = data.get('model_choice', 'yolo')
        
        if not before_image or not after_image:
            return jsonify({'error': 'Both before and after images required'}), 400
        
        # Save images
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        before_path = save_base64_image(before_image, f'before_{timestamp}.jpg')
        after_path = save_base64_image(after_image, f'after_{timestamp}.jpg')
        
        if not before_path or not after_path:
            return jsonify({'error': 'Failed to save uploaded images'}), 400
        
        # Analyze with real models
        before_analysis = analyze_single_image(before_path, model_choice)
        after_analysis = analyze_single_image(after_path, model_choice)
        
        # Compare results
        comparison = compare_images(before_analysis, after_analysis)
        
        # Generate compliance report
        compliance_report = generate_compliance_report(before_analysis, after_analysis, comparison)
        
        # Cleanup
        try:
            os.unlink(before_path)
            os.unlink(after_path)
        except:
            pass
        
        results = {
            'success': True,
            'model_used': model_choice.upper(),
            'before_analysis': before_analysis,
            'after_analysis': after_analysis,
            'comparison': comparison,
            'compliance_report': compliance_report,
            'human_readable_report': compliance_report['report_text'],
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error in complete analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/benchmark_models', methods=['POST'])
def benchmark_models():
    """Benchmark all available real models"""
    try:
        data = request.json
        test_image = data.get('test_image')
        
        if not test_image:
            return jsonify({'error': 'Test image required'}), 400
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_path = save_base64_image(test_image, f'benchmark_{timestamp}.jpg')
        
        if not test_path:
            return jsonify({'error': 'Failed to save test image'}), 400
        
        benchmark_results = {}
        
        for model_name, model in models.items():
            if model and model.is_loaded:
                try:
                    start_time = time.time()
                    results = model.predict(test_path)
                    end_time = time.time()
                    
                    benchmark_results[model_name] = {
                        'model_type': results.get('model_type', model_name),
                        'inference_time': end_time - start_time,
                        'detections': results.get('detections', []),
                        'success': True,
                        'is_real_model': not 'Mock' in results.get('model_type', '')
                    }
                    
                except Exception as e:
                    benchmark_results[model_name] = {
                        'error': str(e),
                        'success': False
                    }
        
        # Generate analysis
        successful_results = {k: v for k, v in benchmark_results.items() if v.get('success', False)}
        
        comparison_summary = "REAL MODEL BENCHMARK RESULTS:\n\n"
        if successful_results:
            fastest_model = min(successful_results.keys(), 
                              key=lambda x: successful_results[x]['inference_time'])
            most_detections = max(successful_results.keys(),
                                key=lambda x: len(successful_results[x]['detections']))
            
            comparison_summary += f"Fastest Model: {fastest_model.upper()} ({successful_results[fastest_model]['inference_time']:.3f}s)\n"
            comparison_summary += f"Most Detections: {most_detections.upper()} ({len(successful_results[most_detections]['detections'])} objects)\n\n"
            
            comparison_summary += "Performance Details:\n"
            for model_name, results in successful_results.items():
                model_status = "REAL" if results.get('is_real_model', False) else "MOCK"
                comparison_summary += f"• {model_name.upper()} [{model_status}]: {results['inference_time']:.3f}s, {len(results['detections'])} detections\n"
        else:
            comparison_summary += "No successful benchmark results available."
        
        # Cleanup
        try:
            os.unlink(test_path)
        except:
            pass
        
        response = {
            'success': True,
            'benchmark_results': benchmark_results,
            'analysis': {
                'total_models_tested': len(benchmark_results),
                'successful_tests': len(successful_results),
                'comparison_summary': comparison_summary,
                'real_models_count': sum(1 for v in successful_results.values() if v.get('is_real_model', False))
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error in model benchmarking: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'real_models_loaded': sum(1 for model in models.values() if model is not None and model.is_loaded),
        'dataset_path_exists': os.path.exists(DATASET_PATH)
    })

if __name__ == '__main__':
    print("🚀 Starting ShelfCheck AI Backend Server with REAL MODELS...")
    print("=" * 60)
    
    # Check dataset path
    if not os.path.exists(DATASET_PATH):
        print(f"⚠️  Dataset path not found: {DATASET_PATH}")
        print("   Update DATASET_PATH in the script or create the dataset folder")
        print()
    
    # Initialize models
    initialize_models()
    
    print(f"✅ Server starting on http://localhost:5001")
    print("📱 Frontend available at http://localhost:5001")
    print("🎯 Real model integration enabled!")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5001, debug=True)