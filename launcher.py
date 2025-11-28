import os
import time

def run(model):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    cmd = f"python train.py --config configs/{model}.yaml"
    print(f"\n🚀 Launching experiment: {model} ({timestamp})")
    os.system(cmd)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python launcher.py <simple_cnn|resnet|convnext|vit>")
        exit()

    run(sys.argv[1])
