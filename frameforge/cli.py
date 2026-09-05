import argparse
import sys
from tqdm import tqdm
import time
import logging

from .config import load_config, FrameForgeConfig
from .pipeline import Pipeline
from .utils.system import setup_logging

def main():
    parser = argparse.ArgumentParser(description="FrameForge AI - Production AI Video Frame Interpolation")
    parser.add_argument("--input", required=True, help="Input video file path")
    parser.add_argument("--output", required=True, help="Output video file path")
    parser.add_argument("--fps", type=float, default=60.0, help="Target FPS (default: 60.0)")
    parser.add_argument("--scale", type=float, default=1.0, help="Processing scale for VRAM optimization (default: 1.0)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use (default: 0)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Load base config
    if args.config:
        config = load_config(args.config)
    else:
        config = FrameForgeConfig()
        
    # CLI overrides
    config.processing.target_fps = args.fps
    config.processing.scale = args.scale
    config.processing.gpu_id = args.gpu
    
    pipeline = Pipeline(args.input, args.output, config)
    
    try:
        pipeline.prepare()
        
        # Setup TQDM progress bar
        pbar = None
        start_time = time.time()
        
        for progress in pipeline.process():
            if pbar is None:
                # We know the total frames only after reading metadata
                total = progress["total_frames"]
                # If frame_count is unknown, we use duration * fps as an estimate
                if total == 0 and progress["duration_secs"] > 0:
                    total = int(progress["duration_secs"] * (progress["frames_in"] / max(0.001, progress["current_pts"])))
                pbar = tqdm(total=total, desc="Interpolating", unit="frames", dynamic_ncols=True)
                
            pbar.update(1) # We update by 1 source frame read
            pbar.set_postfix({"out": progress["frames_out"], "pts": f"{progress['current_pts']:.2f}s"})
            
        if pbar:
            pbar.close()
            
        elapsed = time.time() - start_time
        logging.info(f"Successfully finished processing in {elapsed:.2f} seconds.")
        logging.info(f"Output saved to {args.output}")
            
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
