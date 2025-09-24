#!/usr/bin/env python3
"""
Quick fix to replace the DataLoader creation in your R-CNN script
"""

import sys
import re

def fix_rcnn_script(filename='r-cnn.py'):
    """Fix the hanging DataLoader issue"""
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
        
        old_pattern = r'num_workers=\d+ if self\.device\.type != \'cpu\' else \d+'
        new_pattern = 'num_workers=0  # Fixed: Set to 0 to avoid hanging'
        
        content = re.sub(old_pattern, new_pattern, content)
        
        content = re.sub(r'num_workers=\d+', 'num_workers=0', content)
        
        with open(filename, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed DataLoader hanging issue in {filename}")
        return True
        
    except FileNotFoundError:
        print(f"❌ File {filename} not found")
        return False
    except Exception as e:
        print(f"❌ Error fixing file: {e}")
        return False

if __name__ == '__main__':
    fix_rcnn_script('r-cnn.py')
    
    print("\n🚀 Now try running your R-CNN training again:")
    print("python simple_rcnn_trainer.py --dataset ./dataset --mode fast")