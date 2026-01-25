"""
MCAP Topic Extractor - Extract all topics to folders/JSON/TXT

Extracts all topics from an MCAP file into organized folders:
- Images → PNG files
- Structured data → JSON files  
- Raw bytes → .bin files
- Text/metadata → .txt files

Usage:
    python extract_mcap_topics.py <mcap_file>
    python extract_mcap_topics.py pii-blurring_100_7331910_23_common.mcap
"""

import os
import sys
import json
import re
from pathlib import Path
from datetime import datetime
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
import cv2
import numpy as np


def sanitize_topic_name(topic: str) -> str:
    """Convert topic name to valid folder name."""
    # Remove leading slash, replace slashes with underscores
    name = topic.lstrip('/')
    name = name.replace('/', '_')
    # Remove any invalid characters
    name = re.sub(r'[<>:"|?*]', '_', name)
    return name


def decode_image(msg):
    """Try to decode various image formats."""
    # Foxglove CompressedImage
    if hasattr(msg, 'data') and hasattr(msg, 'format'):
        data = msg.data
        if isinstance(data, (bytes, bytearray)):
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
    
    # Foxglove RawImage
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
                elif encoding in ['jpeg', 'png']:
                    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                else:
                    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                return img
            except Exception:
                pass
    
    return None


def msg_to_dict(msg, depth=0):
    """Convert protobuf message to dictionary - clean, readable output."""
    if depth > 20:
        return "<max depth reached>"
    
    if msg is None:
        return None
    
    # Handle primitive types first
    if isinstance(msg, bool):  # Must check before int (bool is subclass of int)
        return msg
    if isinstance(msg, (int, float, str)):
        return msg
    
    # Handle bytes
    if isinstance(msg, (bytes, bytearray)):
        if len(msg) > 1000:
            return f"<bytes: {len(msg)} bytes>"
        try:
            return msg.decode('utf-8')
        except:
            return f"<bytes: {len(msg)} bytes>"
    
    # Handle numpy arrays
    if isinstance(msg, np.ndarray):
        if msg.size > 100:
            return f"<ndarray: shape={msg.shape}, dtype={msg.dtype}>"
        return msg.tolist()
    
    # Handle dict
    if isinstance(msg, dict):
        return {k: msg_to_dict(v, depth + 1) for k, v in msg.items()}
    
    # Handle lists/tuples
    if isinstance(msg, (list, tuple)):
        return [msg_to_dict(item, depth + 1) for item in msg[:200]]
    
    # Try protobuf MessageToDict first (best for proper protobuf messages)
    try:
        from google.protobuf.json_format import MessageToDict
        return MessageToDict(msg, preserving_proto_field_name=True)
    except:
        pass
    
    # Try ListFields() for protobuf messages (gets only set fields, no DESCRIPTOR)
    if hasattr(msg, 'ListFields'):
        try:
            result = {}
            for field, value in msg.ListFields():
                result[field.name] = msg_to_dict(value, depth + 1)
            return result
        except:
            pass
    
    # Check if iterable (handles RepeatedCompositeContainer, etc.)
    if hasattr(msg, '__iter__') and not isinstance(msg, (str, bytes, bytearray)):
        try:
            items = list(msg)
            return [msg_to_dict(item, depth + 1) for item in items[:200]]
        except:
            pass
    
    # Last resort: extract non-internal attributes
    skip_patterns = {'DESCRIPTOR', 'Extensions', 'ByteSize', 'Clear', 'Copy', 
                     'Discard', 'Find', 'Has', 'IsInitialized', 'ListFields',
                     'Merge', 'Parse', 'Register', 'Serialize', 'Set', 'Unknown', 'Which'}
    
    result = {}
    for attr in dir(msg):
        if attr.startswith('_'):
            continue
        if any(attr.startswith(p) or attr == p for p in skip_patterns):
            continue
        try:
            val = getattr(msg, attr)
            if callable(val):
                continue
            # Skip objects that look like protobuf internals
            val_type = type(val).__name__
            if 'Descriptor' in val_type or 'GenericSequence' in val_type:
                continue
            if isinstance(val, (bytes, bytearray)) and len(val) > 1000:
                result[attr] = f"<binary: {len(val)} bytes>"
            else:
                result[attr] = msg_to_dict(val, depth + 1)
        except:
            pass
    
    return result if result else str(msg)


def is_image_topic(schema_name: str, topic: str) -> bool:
    """Check if topic likely contains images."""
    image_keywords = ['image', 'camera', 'rgb', 'depth', 'color', 'compressed']
    image_schemas = ['foxglove.CompressedImage', 'foxglove.RawImage', 
                     'sensor_msgs/Image', 'sensor_msgs/CompressedImage']
    
    if schema_name in image_schemas:
        return True
    
    topic_lower = topic.lower()
    return any(kw in topic_lower for kw in image_keywords)


def extract_mcap(mcap_path: str, output_base: str = None):
    """Extract all topics from MCAP file."""
    mcap_path = Path(mcap_path)
    
    if not mcap_path.exists():
        print(f"ERROR: File not found: {mcap_path}")
        return
    
    # Create output directory
    if output_base is None:
        output_base = mcap_path.parent / f"{mcap_path.stem}_extracted"
    else:
        output_base = Path(output_base)
    
    output_base.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MCAP Topic Extractor")
    print("=" * 70)
    print(f"Input:  {mcap_path}")
    print(f"Output: {output_base}")
    print(f"Size:   {mcap_path.stat().st_size / (1024*1024):.2f} MB")
    print("=" * 70)
    
    # First pass: discover all topics and their schemas
    print("\n[1/3] Discovering topics...")
    topic_info = {}
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
        
        # Get summary info
        summary = reader.get_summary()
        if summary and summary.statistics:
            stats = summary.statistics
            duration_sec = (stats.message_end_time - stats.message_start_time) / 1e9
            print(f"Duration: {duration_sec:.2f} seconds")
            print(f"Total messages: {stats.message_count}")
        
        # Iterate to get topic details
        for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
            topic = channel.topic
            if topic not in topic_info:
                schema_name = schema.name if schema else 'unknown'
                topic_info[topic] = {
                    'schema': schema_name,
                    'count': 0,
                    'is_image': is_image_topic(schema_name, topic),
                    'folder': sanitize_topic_name(topic)
                }
            topic_info[topic]['count'] += 1
    
    # Print discovered topics
    print(f"\nFound {len(topic_info)} topics:")
    for topic, info in sorted(topic_info.items()):
        img_marker = " [IMAGE]" if info['is_image'] else ""
        print(f"  {topic}: {info['count']} msgs ({info['schema']}){img_marker}")
    
    # Create directories for each topic
    print("\n[2/3] Creating directories...")
    for topic, info in topic_info.items():
        topic_dir = output_base / info['folder']
        topic_dir.mkdir(parents=True, exist_ok=True)
    
    # Second pass: extract all data
    print("\n[3/3] Extracting data...")
    
    counters = {topic: 0 for topic in topic_info}
    
    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
        
        for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
            topic = channel.topic
            info = topic_info[topic]
            idx = counters[topic]
            counters[topic] += 1
            
            topic_dir = output_base / info['folder']
            timestamp_ns = message.log_time
            
            # Handle image topics
            if info['is_image']:
                img = decode_image(decoded_msg)
                if img is not None:
                    img_path = topic_dir / f"{idx:06d}_{timestamp_ns}.png"
                    cv2.imwrite(str(img_path), img)
                else:
                    # Save raw message as JSON
                    json_path = topic_dir / f"{idx:06d}_{timestamp_ns}.json"
                    data = {
                        'timestamp_ns': timestamp_ns,
                        'index': idx,
                        'data': msg_to_dict(decoded_msg)
                    }
                    with open(json_path, 'w') as jf:
                        json.dump(data, jf, indent=2, default=str)
            else:
                # Non-image topic: save as JSON
                json_path = topic_dir / f"{idx:06d}_{timestamp_ns}.json"
                data = {
                    'timestamp_ns': timestamp_ns,
                    'index': idx,
                    'data': msg_to_dict(decoded_msg)
                }
                with open(json_path, 'w') as jf:
                    json.dump(data, jf, indent=2, default=str)
            
            # Progress update
            total_processed = sum(counters.values())
            if total_processed % 500 == 0:
                print(f"  Processed {total_processed} messages...")
    
    # Write summary file
    summary_path = output_base / "_summary.txt"
    with open(summary_path, 'w') as f:
        f.write(f"MCAP Extraction Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Source: {mcap_path}\n")
        f.write(f"Extracted: {datetime.now().isoformat()}\n")
        f.write(f"\nTopics:\n")
        for topic, info in sorted(topic_info.items()):
            img_marker = " [IMAGE]" if info['is_image'] else ""
            f.write(f"  {topic}: {info['count']} messages{img_marker}\n")
            f.write(f"    Schema: {info['schema']}\n")
            f.write(f"    Folder: {info['folder']}/\n")
    
    # Write topic index as JSON
    index_path = output_base / "_topics.json"
    with open(index_path, 'w') as f:
        json.dump(topic_info, f, indent=2)
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"Output directory: {output_base}")
    print(f"\nExtracted {sum(counters.values())} total messages from {len(topic_info)} topics:")
    for topic, count in sorted(counters.items()):
        print(f"  {topic}: {count} files")
    print(f"\nSummary files:")
    print(f"  {summary_path.name} - Human readable summary")
    print(f"  {index_path.name} - Topic metadata as JSON")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        # Default to the file mentioned by user
        mcap_file = r"O:\Micro-AGI Projects\randombullshitgo\pii-blurring_100_7331910_23_common.mcap"
        if not os.path.exists(mcap_file):
            print("Usage: python extract_mcap_topics.py <mcap_file> [output_dir]")
            print("\nExample:")
            print("  python extract_mcap_topics.py myfile.mcap")
            print("  python extract_mcap_topics.py myfile.mcap ./extracted_data")
            sys.exit(1)
    else:
        mcap_file = sys.argv[1]
    
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    extract_mcap(mcap_file, output_dir)


if __name__ == "__main__":
    main()

