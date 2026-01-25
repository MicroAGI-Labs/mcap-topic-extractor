"""
MCAP Preview Extractor - Extract last N seconds of color images

Extracts only the last few seconds of color camera images from an MCAP file.
Useful for quick previews without extracting everything.

Usage:
    python extract_preview.py <mcap_file> [seconds]
    python extract_preview.py recording.mcap        # Default: 5 seconds
    python extract_preview.py recording.mcap 10     # Custom: 10 seconds
"""

import os
import sys
from pathlib import Path
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
import cv2
import numpy as np


# Default settings
DEFAULT_SECONDS = 5
COLOR_IMAGE_TOPIC = '/camera/color/image'


def decode_image(msg):
    """Decode Foxglove CompressedImage or RawImage."""
    # CompressedImage (JPEG/PNG)
    if hasattr(msg, 'data') and hasattr(msg, 'format'):
        data = msg.data
        if isinstance(data, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    
    # RawImage
    if hasattr(msg, 'data') and hasattr(msg, 'width') and hasattr(msg, 'height'):
        data = msg.data
        width = msg.width
        height = msg.height
        encoding = getattr(msg, 'encoding', 'rgb8')
        
        if isinstance(data, (bytes, bytearray)) and width > 0 and height > 0:
            try:
                if encoding in ['rgb8', 'RGB8']:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                elif encoding in ['bgr8', 'BGR8']:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                elif encoding in ['rgba8', 'RGBA8']:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif encoding in ['bgra8', 'BGRA8']:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 4))
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                elif encoding in ['mono8', 'MONO8', '8UC1']:
                    img = np.frombuffer(data, dtype=np.uint8).reshape((height, width))
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                else:
                    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                return img
            except:
                pass
    
    return None


def extract_preview(mcap_path: str, seconds: float = DEFAULT_SECONDS):
    """Extract last N seconds of color images from MCAP."""
    mcap_path = Path(mcap_path)
    
    if not mcap_path.exists():
        print(f"ERROR: File not found: {mcap_path}")
        return
    
    # Output folder
    output_dir = mcap_path.parent / f"{mcap_path.stem}_last_{seconds}s"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MCAP Preview Extractor (Last N Seconds)")
    print("=" * 60)
    print(f"Input:    {mcap_path}")
    print(f"Output:   {output_dir}")
    print(f"Duration: Last {seconds} seconds")
    print(f"Topic:    {COLOR_IMAGE_TOPIC}")
    print("=" * 60)
    
    # First pass: get the time range from summary
    print("\n[1/2] Reading file summary...")
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        
        if not summary or not summary.statistics:
            print("ERROR: Could not read file summary")
            return
        
        stats = summary.statistics
        end_time_ns = stats.message_end_time
        start_time_ns = stats.message_start_time
        total_duration_sec = (end_time_ns - start_time_ns) / 1e9
        
        print(f"Total duration: {total_duration_sec:.2f} seconds")
        print(f"Total messages: {stats.message_count}")
    
    # Calculate cutoff time (last N seconds)
    duration_ns = int(seconds * 1e9)
    cutoff_time_ns = end_time_ns - duration_ns
    
    if cutoff_time_ns < start_time_ns:
        cutoff_time_ns = start_time_ns
        actual_duration = total_duration_sec
        print(f"Note: File is shorter than {seconds}s, extracting entire file ({actual_duration:.2f}s)")
    
    # Second pass: extract images from last N seconds
    print(f"\n[2/2] Extracting frames from last {seconds} seconds...")
    
    frame_count = 0
    first_frame_time_ns = None
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
        
        for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
            if channel.topic != COLOR_IMAGE_TOPIC:
                continue
            
            timestamp_ns = message.log_time
            
            # Skip frames before cutoff
            if timestamp_ns < cutoff_time_ns:
                continue
            
            # Record first extracted frame time
            if first_frame_time_ns is None:
                first_frame_time_ns = timestamp_ns
                time_from_end = (end_time_ns - timestamp_ns) / 1e9
                print(f"Starting extraction at {time_from_end:.2f}s from end")
            
            # Decode and save image
            img = decode_image(decoded_msg)
            if img is not None:
                # Time relative to start of extracted segment
                elapsed_ns = timestamp_ns - first_frame_time_ns
                elapsed_sec = elapsed_ns / 1e9
                
                filename = f"{frame_count:04d}_{elapsed_sec:.3f}s.png"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), img)
                
                if frame_count == 0:
                    print(f"Image size: {img.shape[1]}x{img.shape[0]}")
                
                frame_count += 1
                print(f"  [{frame_count}] {filename}")
    
    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE!")
    print("=" * 60)
    print(f"Extracted: {frame_count} frames")
    print(f"Output:    {output_dir}")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_preview.py <mcap_file> [seconds]")
        print("")
        print("Examples:")
        print("  python extract_preview.py recording.mcap      # Last 5 seconds")
        print("  python extract_preview.py recording.mcap 10   # Last 10 seconds")
        sys.exit(1)
    
    mcap_file = sys.argv[1]
    seconds = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SECONDS
    
    extract_preview(mcap_file, seconds)


if __name__ == "__main__":
    main()
