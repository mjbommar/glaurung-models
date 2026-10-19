#!/usr/bin/env python3
"""Optimize trained model for CPU inference."""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict

import torch
import torch.quantization
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.transformers import optimizer
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.model.architecture import LightweightIOCModel, EfficientIOCTokenizer


class ModelOptimizer:
    """Optimize model for CPU deployment."""
    
    def __init__(self, model_path: str, output_dir: str):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        self.tokenizer = EfficientIOCTokenizer()
        
    def _load_model(self) -> LightweightIOCModel:
        """Load trained model from checkpoint."""
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        model = LightweightIOCModel(
            vocab_size=checkpoint['args'].vocab_size,
            hidden_dim=checkpoint['args'].hidden_dim,
            num_ioc_types=checkpoint['args'].num_ioc_types,
            nhead=checkpoint['args'].num_heads,
            num_layers=checkpoint['args'].num_layers
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"Loaded model from {self.model_path}")
        print(f"Model metrics: {checkpoint['metrics']}")
        
        return model
    
    def export_to_onnx(self) -> str:
        """Export PyTorch model to ONNX."""
        print("Exporting model to ONNX...")
        
        # Compile model first for optimizations if available
        if hasattr(torch, 'compile'):
            print("Compiling model with torch.compile before export...")
            self.model = torch.compile(self.model, mode='reduce-overhead')
        
        # Prepare model for export
        self.model.prepare_for_onnx_export()
        
        # Create dummy input
        dummy_input_ids = torch.randint(0, 256, (1, 512))
        dummy_attention_mask = torch.ones(1, 512, dtype=torch.long)
        
        # Export path
        onnx_path = self.output_dir / "model.onnx"
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            (dummy_input_ids, dummy_attention_mask),
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['ner_logits', 'ioc_logits', 'risk_score'],
            dynamic_axes={
                'input_ids': {0: 'batch_size'},
                'attention_mask': {0: 'batch_size'},
                'ner_logits': {0: 'batch_size'},
                'ioc_logits': {0: 'batch_size'},
                'risk_score': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        print(f"ONNX model saved to {onnx_path}")
        return str(onnx_path)
    
    def quantize_model(self, onnx_path: str) -> str:
        """Apply dynamic quantization to ONNX model."""
        print("Applying INT8 quantization...")
        
        quantized_path = self.output_dir / "model_int8.onnx"
        
        # Apply dynamic quantization
        quantize_dynamic(
            onnx_path,
            str(quantized_path),
            weight_type=QuantType.QInt8
        )
        
        # Calculate size reduction
        original_size = Path(onnx_path).stat().st_size / (1024 * 1024)  # MB
        quantized_size = quantized_path.stat().st_size / (1024 * 1024)  # MB
        reduction = (1 - quantized_size / original_size) * 100
        
        print(f"Original model size: {original_size:.2f} MB")
        print(f"Quantized model size: {quantized_size:.2f} MB")
        print(f"Size reduction: {reduction:.1f}%")
        
        return str(quantized_path)
    
    def optimize_graph(self, onnx_path: str) -> str:
        """Apply graph optimizations."""
        print("Applying graph optimizations...")
        
        optimized_path = self.output_dir / "model_optimized.onnx"
        
        # Load and optimize
        opt_model = optimizer.optimize_model(
            onnx_path,
            model_type='bert',  # Similar architecture
            num_heads=4,
            hidden_size=256,
            optimization_options=optimizer.FusionOptions('all')
        )
        
        # Save optimized model
        opt_model.save_model_to_file(str(optimized_path))
        
        print(f"Optimized model saved to {optimized_path}")
        return str(optimized_path)
    
    def benchmark_inference(self, onnx_path: str, num_samples: int = 1000):
        """Benchmark inference performance."""
        print(f"\nBenchmarking inference on {num_samples} samples...")
        
        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4  # Use 4 CPU threads
        
        session = ort.InferenceSession(onnx_path, sess_options)
        
        # Generate test samples
        test_texts = [
            f"Suspicious activity from IP 192.168.1.{i} detected" for i in range(num_samples)
        ]
        
        # Warm up
        for _ in range(10):
            sample = self.tokenizer.encode(test_texts[0])
            session.run(None, {
                'input_ids': sample['input_ids'].numpy().reshape(1, -1),
                'attention_mask': sample['attention_mask'].numpy().reshape(1, -1)
            })
        
        # Benchmark
        latencies = []
        
        for text in test_texts:
            sample = self.tokenizer.encode(text)
            
            start_time = time.perf_counter()
            outputs = session.run(None, {
                'input_ids': sample['input_ids'].numpy().reshape(1, -1),
                'attention_mask': sample['attention_mask'].numpy().reshape(1, -1)
            })
            end_time = time.perf_counter()
            
            latencies.append((end_time - start_time) * 1000)  # ms
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        stats = {
            'mean_latency_ms': np.mean(latencies),
            'p50_latency_ms': np.percentile(latencies, 50),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'throughput_samples_per_sec': 1000 / np.mean(latencies)
        }
        
        print("\nInference Performance:")
        print(f"Mean latency: {stats['mean_latency_ms']:.2f} ms")
        print(f"P50 latency: {stats['p50_latency_ms']:.2f} ms")
        print(f"P95 latency: {stats['p95_latency_ms']:.2f} ms")
        print(f"P99 latency: {stats['p99_latency_ms']:.2f} ms")
        print(f"Throughput: {stats['throughput_samples_per_sec']:.1f} samples/sec")
        
        return stats
    
    def compare_models(self, original_path: str, optimized_path: str):
        """Compare original and optimized model outputs."""
        print("\nComparing model outputs...")
        
        # Create sessions
        original_session = ort.InferenceSession(original_path)
        optimized_session = ort.InferenceSession(optimized_path)
        
        # Test samples
        test_samples = [
            "Malicious file detected: 5d41402abc4b2a76b9719d911017c592",
            "Connection from 192.168.1.100 to evil.com",
            "CVE-2021-44228 vulnerability exploited",
            "Normal system operation at 10:30 AM"
        ]
        
        max_diff = 0.0
        
        for text in test_samples:
            sample = self.tokenizer.encode(text)
            inputs = {
                'input_ids': sample['input_ids'].numpy().reshape(1, -1),
                'attention_mask': sample['attention_mask'].numpy().reshape(1, -1)
            }
            
            # Get outputs
            original_out = original_session.run(None, inputs)
            optimized_out = optimized_session.run(None, inputs)
            
            # Compare outputs
            for orig, opt in zip(original_out, optimized_out):
                diff = np.max(np.abs(orig - opt))
                max_diff = max(max_diff, diff)
        
        print(f"Maximum output difference: {max_diff:.6f}")
        
        if max_diff < 0.01:
            print("✓ Models produce nearly identical outputs")
        else:
            print("⚠ Significant difference detected between models")
            
    def run_full_optimization(self):
        """Run complete optimization pipeline."""
        print("=" * 50)
        print("Starting model optimization pipeline")
        print("=" * 50)
        
        # Step 1: Export to ONNX
        onnx_path = self.export_to_onnx()
        
        # Step 2: Quantize model
        quantized_path = self.quantize_model(onnx_path)
        
        # Step 3: Optimize graph
        try:
            optimized_path = self.optimize_graph(quantized_path)
        except Exception as e:
            print(f"Graph optimization failed: {e}")
            optimized_path = quantized_path
        
        # Step 4: Benchmark
        print("\n" + "=" * 50)
        print("Benchmarking original ONNX model:")
        original_stats = self.benchmark_inference(onnx_path)
        
        print("\n" + "=" * 50)
        print("Benchmarking optimized model:")
        optimized_stats = self.benchmark_inference(optimized_path)
        
        # Step 5: Compare outputs
        self.compare_models(onnx_path, optimized_path)
        
        # Calculate speedup
        speedup = original_stats['mean_latency_ms'] / optimized_stats['mean_latency_ms']
        print(f"\n✨ Achieved {speedup:.2f}x speedup with optimization!")
        
        # Save optimization report
        report = {
            'original_model': {
                'path': onnx_path,
                'stats': original_stats
            },
            'optimized_model': {
                'path': optimized_path,
                'stats': optimized_stats
            },
            'speedup': speedup
        }
        
        report_path = self.output_dir / "optimization_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nOptimization report saved to {report_path}")
        print("\n✅ Optimization complete!")


def main():
    parser = argparse.ArgumentParser(description='Optimize IOC model for CPU')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='optimized_models',
                       help='Directory to save optimized models')
    parser.add_argument('--num_benchmark_samples', type=int, default=1000,
                       help='Number of samples for benchmarking')
    
    args = parser.parse_args()
    
    # Run optimization
    optimizer = ModelOptimizer(args.model_path, args.output_dir)
    optimizer.run_full_optimization()


if __name__ == "__main__":
    main()