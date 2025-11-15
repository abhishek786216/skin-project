"""
GPU Verification and Optimization Script for Skin Cancer Classification
Tests CUDA availability and optimizes PyTorch for GPU training
"""

import torch
import torchvision
import numpy as np
import time
import sys
import os

def check_cuda_setup():
    """Comprehensive CUDA and GPU setup verification"""
    print("="*60)
    print("GPU VERIFICATION AND OPTIMIZATION CHECK")
    print("="*60)
    
    # Basic CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("\nWARNING: CUDA is not available!")
        print("Possible solutions:")
        print("1. Install CUDA-enabled PyTorch:")
        print("   pip uninstall torch torchvision")
        print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        print("2. Check NVIDIA driver installation")
        print("3. Verify CUDA toolkit installation")
        return False
    
    # Detailed GPU information
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        gpu_props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {gpu_props.name}")
        print(f"  - Total Memory: {gpu_props.total_memory / 1024**3:.2f} GB")
        print(f"  - Compute Capability: {gpu_props.major}.{gpu_props.minor}")
        print(f"  - Multi-processors: {gpu_props.multi_processor_count}")
        
        # Current memory usage
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  - Current Allocated: {allocated:.2f} GB")
        print(f"  - Current Reserved: {reserved:.2f} GB")
    
    return True

def test_gpu_performance():
    """Test GPU performance with a simple operation"""
    print("\n" + "="*60)
    print("GPU PERFORMANCE TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("Skipping GPU test - CUDA not available")
        return
    
    device = torch.device('cuda')
    
    # Test matrix multiplication
    print("Testing matrix multiplication performance...")
    
    size = 2048
    
    # CPU test
    print(f"CPU test ({size}x{size} matrix multiplication):")
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    start_time = time.time()
    c_cpu = torch.mm(a_cpu, b_cpu)
    cpu_time = time.time() - start_time
    print(f"  CPU Time: {cpu_time:.3f} seconds")
    
    # GPU test
    print(f"GPU test ({size}x{size} matrix multiplication):")
    a_gpu = torch.randn(size, size, device=device)
    b_gpu = torch.randn(size, size, device=device)
    
    # Warm up GPU
    torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    
    start_time = time.time()
    c_gpu = torch.mm(a_gpu, b_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    print(f"  GPU Time: {gpu_time:.3f} seconds")
    
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    print(f"  GPU Speedup: {speedup:.1f}x faster")
    
    if speedup < 2:
        print("  WARNING: Low GPU speedup. Check GPU drivers or hardware.")
    elif speedup > 10:
        print("  EXCELLENT: High GPU acceleration detected!")
    else:
        print("  GOOD: Reasonable GPU acceleration.")

def test_cnn_training():
    """Test CNN training on GPU"""
    print("\n" + "="*60)
    print("CNN TRAINING TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("Skipping CNN test - CUDA not available")
        return
    
    device = torch.device('cuda')
    
    # Simple CNN model
    class TestCNN(torch.nn.Module):
        def __init__(self):
            super(TestCNN, self).__init__()
            self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
            self.pool = torch.nn.MaxPool2d(2, 2)
            self.fc1 = torch.nn.Linear(64 * 56 * 56, 512)
            self.fc2 = torch.nn.Linear(512, 7)
            self.dropout = torch.nn.Dropout(0.5)
            self.relu = torch.nn.ReLU()
            
        def forward(self, x):
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = x.view(-1, 64 * 56 * 56)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    # Initialize model and data
    model = TestCNN().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Simulate batch of skin lesion images
    batch_size = 16
    images = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 7, (batch_size,), device=device)
    
    print(f"Testing CNN training with batch size {batch_size}...")
    
    # Training step
    model.train()
    start_time = time.time()
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize()
    training_time = time.time() - start_time
    
    print(f"  Training step time: {training_time:.3f} seconds")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    
    if training_time < 0.1:
        print("  EXCELLENT: Very fast CNN training!")
    elif training_time < 0.5:
        print("  GOOD: Reasonable CNN training speed.")
    else:
        print("  WARNING: Slow CNN training. Check GPU performance.")

def optimize_pytorch_settings():
    """Configure optimal PyTorch settings for GPU training"""
    print("\n" + "="*60)
    print("PYTORCH OPTIMIZATION SETUP")
    print("="*60)
    
    if torch.cuda.is_available():
        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Set memory growth
        torch.cuda.set_per_process_memory_fraction(0.9)
        
        print("GPU optimizations applied:")
        print("  ✓ CUDNN benchmark mode enabled")
        print("  ✓ Non-deterministic mode for speed")
        print("  ✓ GPU memory cache cleared")
        print("  ✓ Memory fraction set to 90%")
        
        # Test memory allocation
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"  ✓ Test GPU allocation successful: {test_tensor.device}")
            del test_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"  ✗ GPU allocation test failed: {e}")
    else:
        print("No GPU optimizations applied - CUDA not available")

def provide_recommendations():
    """Provide recommendations for optimal training"""
    print("\n" + "="*60)
    print("TRAINING OPTIMIZATION RECOMMENDATIONS")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("CRITICAL: Install CUDA-enabled PyTorch for GPU training!")
        print("Command: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        return
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"Based on your {gpu_memory:.1f}GB GPU:")
    
    if gpu_memory >= 12:
        print("  Recommended batch size: 64-96")
        print("  Recommended num_workers: 8-12")
        print("  Can handle large models (Xception, ResNet50+)")
    elif gpu_memory >= 8:
        print("  Recommended batch size: 32-48")
        print("  Recommended num_workers: 6-8")
        print("  Good for medium models (ResNet18, MobileNet)")
    elif gpu_memory >= 6:
        print("  Recommended batch size: 16-32")
        print("  Recommended num_workers: 4-6")
        print("  Suitable for lightweight models")
    else:
        print("  Recommended batch size: 8-16")
        print("  Recommended num_workers: 2-4")
        print("  Use smaller models only")
    
    print(f"\nExpected training time estimates:")
    print(f"  Full hyperparameter tuning: {2 if gpu_memory >= 8 else 4}-{6 if gpu_memory >= 8 else 12} hours")
    print(f"  Quick test: {3 if gpu_memory >= 8 else 8}-{10 if gpu_memory >= 8 else 20} minutes")

def main():
    """Run complete GPU verification and optimization"""
    print("Starting comprehensive GPU verification...")
    
    # Check CUDA setup
    cuda_ok = check_cuda_setup()
    
    if cuda_ok:
        # Test GPU performance
        test_gpu_performance()
        
        # Test CNN training
        test_cnn_training()
        
        # Apply optimizations
        optimize_pytorch_settings()
    
    # Provide recommendations
    provide_recommendations()
    
    print("\n" + "="*60)
    print("GPU VERIFICATION COMPLETE")
    print("="*60)
    
    if cuda_ok:
        print("✓ GPU setup is ready for training!")
        print("You can now run: python advanced_train_model.py")
    else:
        print("✗ GPU setup needs attention before training.")

if __name__ == "__main__":
    main()