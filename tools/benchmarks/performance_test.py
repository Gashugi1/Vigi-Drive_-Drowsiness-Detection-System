"""
Performance benchmarks for drowsiness detection system
Measures FPS, inference latency, and identifies bottlenecks
"""

import time
import cv2
import numpy as np
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from realtime_ml_frame import DrowsinessDetector
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)


class PerformanceBenchmark:
    """Benchmark suite for detection pipeline"""
    
    def __init__(self, duration_seconds=10):
        self.duration = duration_seconds
        self.timings = defaultdict(list)
        
    def benchmark_fps(self):
        """Measure average frames per second"""
        print("\n" + "="*60)
        print("BENCHMARK: Frames Per Second (FPS)")
        print("="*60)
        
        try:
            detector = DrowsinessDetector(config_path="config.json")
            
            start_time = time.time()
            frame_count = 0
            
            print(f"Recording for {self.duration} seconds...")
            
            while time.time() - start_time < self.duration:
                ok, frame = detector.cap.read()
                if not ok:
                    continue
                
                # Process frame (full pipeline)
                frame_start = time.time()
                
                # Low-light preprocessing
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                l_channel_eq = detector.clahe.apply(l_channel)
                lab_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
                frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
                
                # MediaPipe detection
                H, W = frame_eq.shape[:2]
                rgb = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)
                res = detector.mesh.process(rgb)
                
                frame_time = time.time() - frame_start
                self.timings['frame_processing'].append(frame_time * 1000)  # ms
                
                frame_count += 1
            
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            avg_latency = np.mean(self.timings['frame_processing'])
            
            detector.release_resources()
            
            print(f"\n📊 Results:")
            print(f"  Total frames: {frame_count}")
            print(f"  Duration: {elapsed:.2f}s")
            print(f"  Average FPS: {fps:.2f}")
            print(f"  Avg frame latency: {avg_latency:.2f}ms")
            print(f"  Min latency: {np.min(self.timings['frame_processing']):.2f}ms")
            print(f"  Max latency: {np.max(self.timings['frame_processing']):.2f}ms")
            
            # Performance assessment
            if fps >= 20:
                print(f"\n✅ PASS: FPS >= 20 (target met)")
            else:
                print(f"\n⚠️  WARNING: FPS < 20 (target: >=20)")
            
            return fps, avg_latency
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return None, None
    
    def benchmark_inference_latency(self):
        """Measure ML model inference time"""
        print("\n" + "="*60)
        print("BENCHMARK: ML Inference Latency")
        print("="*60)
        
        try:
            detector = DrowsinessDetector(config_path="config.json")
            
            # Create sample inputs
            samples = []
            for _ in range(100):
                ear = np.random.uniform(0.15, 0.35)
                mar = np.random.uniform(0.30, 0.70)
                eye_close = 1.0 if ear < 0.25 else 0.0
                mouth_open = 1.0 if mar > 0.60 else 0.0
                samples.append([ear, mar, eye_close, mouth_open])
            
            inference_times = []
            
            for sample in samples:
                x = np.array([sample])
                
                start = time.time()
                x_scaled = detector.scaler.transform(x)
                p_drowsy = detector.clf.predict_proba(x_scaled)[0, 1]
                elapsed = (time.time() - start) * 1000  # ms
                
                inference_times.append(elapsed)
            
            avg_inference = np.mean(inference_times)
            p95_inference = np.percentile(inference_times, 95)
            
            detector.release_resources()
            
            print(f"\n📊 Results (n=100 samples):")
            print(f"  Average latency: {avg_inference:.2f}ms")
            print(f"  95th percentile: {p95_inference:.2f}ms")
            print(f"  Min: {np.min(inference_times):.2f}ms")
            print(f"  Max: {np.max(inference_times):.2f}ms")
            
            if avg_inference < 30:
                print(f"\n✅ PASS: Inference latency < 30ms")
            else:
                print(f"\n⚠️  WARNING: Inference latency >= 30ms")
            
            return avg_inference
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return None
    
    def benchmark_component_breakdown(self):
        """Measure individual component timings"""
        print("\n" + "="*60)
        print("BENCHMARK: Component Breakdown")
        print("="*60)
        
        try:
            detector = DrowsinessDetector(config_path="config.json")
            
            timings = {
                'capture': [],
                'clahe': [],
                'mediapipe': [],
                'feature_calc': [],
                'inference': [],
            }
            
            n_frames = 100
            print(f"Analyzing {n_frames} frames...")
            
            for i in range(n_frames):
                # Capture
                t0 = time.time()
                ok, frame = detector.cap.read()
                if not ok:
                    continue
                timings['capture'].append((time.time() - t0) * 1000)
                
                # CLAHE preprocessing
                t1 = time.time()
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                l_channel_eq = detector.clahe.apply(l_channel)
                lab_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
                frame_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
                timings['clahe'].append((time.time() - t1) * 1000)
                
                # MediaPipe
                t2 = time.time()
                H, W = frame_eq.shape[:2]
                rgb = cv2.cvtColor(frame_eq, cv2.COLOR_BGR2RGB)
                res = detector.mesh.process(rgb)
                timings['mediapipe'].append((time.time() - t2) * 1000)
                
                if res.multi_face_landmarks:
                    from realtime_ml_frame import calculate_ear, calculate_mar, LEFT_EYE_LANDMARKS, RIGHT_EYE_LANDMARKS
                    lm = res.multi_face_landmarks[0].landmark
                    
                    # Feature calculation
                    t3 = time.time()
                    left_ear = calculate_ear(lm, LEFT_EYE_LANDMARKS, W, H)
                    right_ear = calculate_ear(lm, RIGHT_EYE_LANDMARKS, W, H)
                    e_val = (left_ear + right_ear) / 2.0
                    m_val = calculate_mar(lm, W, H)
                    timings['feature_calc'].append((time.time() - t3) * 1000)
                    
                    # ML inference
                    t4 = time.time()
                    feats = {"ear": e_val, "mar": m_val, "eye_close": 1.0 if e_val < 0.25 else 0.0, "mouth_open": 1.0 if m_val > 0.6 else 0.0}
                    x = np.array([[feats[k] for k in detector.feat_order]])
                    x_scaled = detector.scaler.transform(x)
                    p_d = detector.clf.predict_proba(x_scaled)[0, 1]
                    timings['inference'].append((time.time() - t4) * 1000)
            
            detector.release_resources()
            
            print(f"\n📊 Average Timings:")
            total = 0
            for component, times in timings.items():
                if times:
                    avg = np.mean(times)
                    total += avg
                    print(f"  {component:15s}: {avg:6.2f}ms ({avg/total*100 if total > 0 else 0:.1f}%)")
            
            print(f"  {'TOTAL':15s}: {total:6.2f}ms")
            print(f"\n🔍 Bottleneck Analysis:")
            max_component = max(timings.items(), key=lambda x: np.mean(x[1]) if x[1] else 0)
            print(f"  Slowest component: {max_component[0]} ({np.mean(max_component[1]):.2f}ms)")
            
            return timings
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return None
    
    def benchmark_memory_usage(self):
        """Estimate memory footprint"""
        print("\n" + "="*60)
        print("BENCHMARK: Memory Usage")
        print("="*60)
        
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            
            # Baseline memory
            baseline = process.memory_info().rss / 1024 / 1024  # MB
            
            # Load detector
            detector = DrowsinessDetector(config_path="config.json")
            
            # After init memory
            after_init = process.memory_info().rss / 1024 / 1024
            
            # Process some frames
            for _ in range(100):
                ok, frame = detector.cap.read()
                if ok:
                    H, W = frame.shape[:2]
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    res = detector.mesh.process(rgb)
            
            # After processing memory
            after_processing = process.memory_info().rss / 1024 / 1024
            
            detector.release_resources()
            
            print(f"\n📊 Memory Usage:")
            print(f"  Baseline: {baseline:.1f} MB")
            print(f"  After init: {after_init:.1f} MB (+{after_init-baseline:.1f} MB)")
            print(f"  After 100 frames: {after_processing:.1f} MB (+{after_processing-after_init:.1f} MB)")
            
            if after_processing < 200:
                print(f"\n✅ PASS: Memory usage < 200 MB")
            else:
                print(f"\n⚠️  WARNING: Memory usage >= 200 MB")
            
            return after_processing
            
        except ImportError:
            print("⚠️  psutil not installed, skipping memory benchmark")
            return None
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            return None


def main():
    """Run all benchmarks"""
    print("\n" + "="*60)
    print("DROWSINESS DETECTION - PERFORMANCE BENCHMARKS")
    print("="*60)
    
    benchmark = PerformanceBenchmark(duration_seconds=10)
    
    # Run benchmarks
    fps, latency = benchmark.benchmark_fps()
    inference_time = benchmark.benchmark_inference_latency()
    component_timings = benchmark.benchmark_component_breakdown()
    memory = benchmark.benchmark_memory_usage()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results = []
    if fps:
        results.append(f"  FPS: {fps:.1f} {'✅' if fps >= 20 else '⚠️'}")
    if latency:
        results.append(f"  Frame Latency: {latency:.1f}ms")
    if inference_time:
        results.append(f"  Inference: {inference_time:.2f}ms {'✅' if inference_time < 30 else '⚠️'}")
    if memory:
        results.append(f"  Memory: {memory:.0f}MB {'✅' if memory < 200 else '⚠️'}")
    
    for result in results:
        print(result)
    
    print("\n✅ = Target met, ⚠️ = Needs optimization")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
