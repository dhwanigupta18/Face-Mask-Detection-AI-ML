"""
dataset_prep.py
Optional helper to inspect dataset counts and create a small validation sample.
Usage:
    python src/dataset_prep.py --data_dir dataset
"""
import os
import argparse

def main(args):
    data_dir = args.data_dir
    for cls in os.listdir(data_dir):
        path = os.path.join(data_dir, cls)
        if os.path.isdir(path):
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg','.png','.jpeg'))])
            print(f"Class {cls}: {count} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    args = parser.parse_args()
    main(args)
