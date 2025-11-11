#!/usr/bin/env python3
"""
Test script to verify VisDrone dataset loads correctly in TransVOD++
"""
import sys
sys.path.insert(0, '/root/TemporalAttentionPlayground/TransVOD_plusplus')

import argparse
from pathlib import Path

# Create minimal args for testing
class Args:
    def __init__(self):
        self.vid_path = '/root/datasets/visdrone/transvod'
        self.masks = False
        self.cache_mode = False
        self.dataset_file = 'visdrone_single'

def main():
    print("=" * 60)
    print("Testing VisDrone Dataset Loader for TransVOD++")
    print("=" * 60)
    
    # Import dataset builder
    from datasets import build_dataset
    from pycocotools.coco import COCO
    
    args = Args()
    
    # Check annotations exist
    root = Path(args.vid_path)
    train_ann = root / "annotations" / "imagenet_vid_train.json"
    val_ann = root / "annotations" / "imagenet_vid_val.json"
    
    print(f"\n1. Checking paths:")
    print(f"   Root: {root} - {'✓ EXISTS' if root.exists() else '✗ MISSING'}")
    print(f"   Train ann: {train_ann} - {'✓ EXISTS' if train_ann.exists() else '✗ MISSING'}")
    print(f"   Val ann: {val_ann} - {'✓ EXISTS' if val_ann.exists() else '✗ MISSING'}")
    
    # Load COCO annotations to check format
    print(f"\n2. Loading COCO annotations:")
    try:
        coco_train = COCO(str(train_ann))
        coco_val = COCO(str(val_ann))
        
        print(f"   Train set:")
        print(f"      Images: {len(coco_train.imgs)}")
        print(f"      Annotations: {len(coco_train.anns)}")
        print(f"      Categories: {len(coco_train.cats)}")
        print(f"      Category IDs: {list(coco_train.cats.keys())}")
        
        print(f"   Val set:")
        print(f"      Images: {len(coco_val.imgs)}")
        print(f"      Annotations: {len(coco_val.anns)}")
        print(f"      Categories: {len(coco_val.cats)}")
        
        # Show sample categories
        print(f"\n3. Sample categories:")
        for cat_id, cat_info in list(coco_train.cats.items())[:5]:
            print(f"      ID {cat_id}: {cat_info['name']}")
            
    except Exception as e:
        print(f"   ✗ Error loading annotations: {e}")
        return 1
    
    # Try building dataset
    print(f"\n4. Building PyTorch dataset:")
    try:
        dataset_train = build_dataset(image_set='train_vid', args=args)
        dataset_val = build_dataset(image_set='val', args=args)
        
        print(f"   Train dataset: {len(dataset_train)} samples")
        print(f"   Val dataset: {len(dataset_val)} samples")
        
        # Try loading one sample
        print(f"\n5. Loading sample data:")
        img, target = dataset_train[0]
        print(f"   Image shape: {img.shape}")
        print(f"   Target keys: {target.keys()}")
        print(f"   Num boxes: {len(target['boxes'])}")
        print(f"   Labels: {target['labels'][:10]}...")  # Show first 10 labels
        
        print(f"\n{'=' * 60}")
        print("✓ Dataset test PASSED! Ready for training.")
        print(f"{'=' * 60}")
        return 0
        
    except Exception as e:
        print(f"   ✗ Error building dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
