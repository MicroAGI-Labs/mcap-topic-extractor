# MCAP Tools

Python utilities for working with MCAP (Foxglove) files.

## 📚 Quick Navigation

| Tool | Description |
|------|-------------|
| [**MCAP Topic Extractor**](#mcap-topic-extractor) | Extract all topics from MCAP files into organized folders (PNG images, JSON data) |
| [**MCAP Preview Extractor**](#mcap-preview-extractor) | Extract last N seconds of color images for quick previews |

---

# MCAP Topic Extractor

A Python tool for extracting all topics from MCAP (Foxglove) files into organized folders with human-readable formats (PNG images, JSON data files).

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [List Topics](#list-topics)
  - [Extract All Topics](#extract-all-topics)
- [Output Structure](#output-structure)
- [How the Code Works](#how-the-code-works)
  - [Architecture Overview](#architecture-overview)
  - [Function-by-Function Breakdown](#function-by-function-breakdown)
  - [Processing Pipeline](#processing-pipeline)
- [Supported Formats](#supported-formats)
- [File Naming Convention](#file-naming-convention)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

---

## Overview

MCAP is a container file format for storing heterogeneous timestamped data, commonly used in robotics for recording sensor data, images, poses, transforms, and metadata. This tool extracts all topics from an MCAP file into a well-organized folder structure where:

- **Images** are saved as PNG files
- **Structured data** (poses, transforms, metadata) are saved as clean, readable JSON files
- **Summary files** provide an overview of extracted content

This makes it easy to inspect, analyze, or process individual messages without specialized MCAP tooling.

---

## Features

- **Automatic Topic Discovery**: Scans the MCAP file to identify all topics and their schemas
- **Smart Image Detection**: Automatically identifies image topics by schema type and topic name
- **Multiple Image Format Support**: Handles compressed (JPEG/PNG) and raw images (RGB8, BGR8, RGBA8, mono8, etc.)
- **Clean Protobuf Decoding**: Converts protobuf messages to human-readable JSON (no internal metadata or object references)
- **Organized Output**: Creates a folder per topic with sequentially numbered files
- **Progress Reporting**: Shows extraction progress for large files
- **Summary Generation**: Creates both human-readable (`.txt`) and machine-readable (`.json`) summaries

---

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Requirements Breakdown

| Package | Version | Purpose |
|---------|---------|---------|
| `mcap` | ≥1.1.0 | Core MCAP file reading library |
| `mcap-protobuf-support` | ≥0.5.0 | Protobuf message decoding for MCAP |
| `foxglove-schemas-protobuf` | ≥0.3.0 | Foxglove-specific protobuf schemas |
| `opencv-python` | ≥4.8.0 | Image decoding and saving |
| `numpy` | ≥1.24.0 | Array manipulation for image processing |

---

## Usage

### List Topics

Quickly inspect what topics are in an MCAP file without extracting:

```bash
python list_mcap_topics.py <mcap_file>
```

**Example output**:
```
Scanning: recording.mcap

Duration: 13.72s | Messages: 10438
------------------------------------------------------------
/body/upper: 385 msgs (foxglove.PosesInFrame)
/camera/color/image: 385 msgs (foxglove.CompressedImage)
/camera/depth/image: 384 msgs (foxglove.CompressedImage)
/hands/left: 385 msgs (foxglove.PosesInFrame)
/hands/right: 385 msgs (foxglove.PosesInFrame)
/imu/accel/sample: 2716 msgs (foxglove.PoseInFrame)
/imu/gyro/sample: 2716 msgs (foxglove.PoseInFrame)
/meta: 1 msgs (microagi.Meta)
/task: 385 msgs (microagi.TaskEvent)
/tf/body: 385 msgs (foxglove.FrameTransforms)
/tf/camera: 383 msgs (foxglove.FrameTransforms)
/tf/hands: 385 msgs (foxglove.FrameTransforms)
------------------------------------------------------------
Total: 21 topics
```

### Extract All Topics

```bash
python extract_mcap_topics.py <mcap_file>
```

This creates an output folder named `<mcap_file>_extracted/` in the same directory.

#### Custom Output Directory

```bash
python extract_mcap_topics.py <mcap_file> <output_directory>
```

#### Examples

```bash
# Extract to default location (creates recording_extracted/)
python extract_mcap_topics.py recording.mcap

# Extract to specific folder
python extract_mcap_topics.py recording.mcap ./my_output

# Full path example
python extract_mcap_topics.py "C:\Data\session.mcap" "D:\Extracted\session"
```

---

## Output Structure

```
<mcap_file>_extracted/
├── _summary.txt              # Human-readable extraction summary
├── _topics.json              # Topic metadata as JSON
├── camera_color_image/       # Topic: /camera/color/image
│   ├── 000000_1762370549054834000.png
│   ├── 000001_1762370549088237000.png
│   └── ...
├── camera_depth_image/       # Topic: /camera/depth/image
│   └── ...
├── body_upper/               # Topic: /body/upper
│   ├── 000000_1762370549054834000.json
│   └── ...
├── hands_left/               # Topic: /hands/left
│   └── ...
├── imu_accel_sample/         # Topic: /imu/accel/sample
│   └── ...
├── meta/                     # Topic: /meta
│   └── 000000_1762370548872110000.json
└── tf_body/                  # Topic: /tf/body
    └── ...
```

### Summary Files

**`_summary.txt`** - Human readable overview:
```
MCAP Extraction Summary
==================================================
Source: recording.mcap
Extracted: 2026-01-25T22:53:09.766104

Topics:
  /camera/color/image: 385 messages [IMAGE]
    Schema: foxglove.CompressedImage
    Folder: camera_color_image/
  /body/upper: 385 messages
    Schema: foxglove.PosesInFrame
    Folder: body_upper/
  ...
```

**`_topics.json`** - Machine readable metadata:
```json
{
  "/camera/color/image": {
    "schema": "foxglove.CompressedImage",
    "count": 385,
    "is_image": true,
    "folder": "camera_color_image"
  },
  ...
}
```

---

## How the Code Works

### Architecture Overview

The extractor uses a **two-pass approach**:

```
┌─────────────────────────────────────────────────────────────────┐
│                         MCAP FILE                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASS 1: DISCOVERY                                               │
│  • Read file summary (duration, message count)                   │
│  • Iterate all messages to discover topics                       │
│  • Record schema names and message counts                        │
│  • Detect which topics contain images                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  CREATE DIRECTORIES                                              │
│  • Create output base folder                                     │
│  • Create subfolder for each topic                               │
│  • Sanitize topic names for filesystem compatibility             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PASS 2: EXTRACTION                                              │
│  • Re-read file from beginning                                   │
│  • For each message:                                             │
│    ├─ Image topic? → Decode & save as PNG                        │
│    └─ Data topic?  → Convert to dict & save as JSON              │
│  • Track progress and update console                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  GENERATE SUMMARIES                                              │
│  • Write _summary.txt (human readable)                           │
│  • Write _topics.json (machine readable)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

### Function-by-Function Breakdown

#### `sanitize_topic_name(topic: str) -> str`

**Purpose**: Converts ROS-style topic names to valid filesystem folder names.

**How it works**:
1. Strips leading `/` from topic name
2. Replaces all `/` with `_` (folder separator → underscore)
3. Removes Windows-invalid characters: `< > : " | ? *`

**Example transformations**:
| Input | Output |
|-------|--------|
| `/camera/color/image` | `camera_color_image` |
| `/tf/body` | `tf_body` |
| `/hands/left/health` | `hands_left_health` |

```python
def sanitize_topic_name(topic: str) -> str:
    name = topic.lstrip('/')           # "/camera/color" → "camera/color"
    name = name.replace('/', '_')      # "camera/color" → "camera_color"
    name = re.sub(r'[<>:"|?*]', '_', name)  # Remove invalid chars
    return name
```

---

#### `decode_image(msg) -> np.ndarray | None`

**Purpose**: Attempts to decode image data from various Foxglove/ROS image message formats.

**Supported formats**:

| Schema | Format | Handling |
|--------|--------|----------|
| `foxglove.CompressedImage` | JPEG/PNG bytes | Decode with `cv2.imdecode()` |
| `foxglove.RawImage` | Raw pixels | Reshape numpy array based on encoding |

**Raw image encoding support**:
- `rgb8` / `RGB8` - 3-channel RGB, converted to BGR for OpenCV
- `bgr8` / `BGR8` - 3-channel BGR (native OpenCV format)
- `rgba8` / `RGBA8` - 4-channel with alpha, converted to BGR
- `bgra8` / `BGRA8` - 4-channel with alpha, converted to BGR
- `mono8` / `MONO8` / `8UC1` - Grayscale, converted to 3-channel BGR
- `jpeg` / `png` - Compressed within RawImage container

**Algorithm**:
```
1. Check if message has 'data' and 'format' attributes (CompressedImage)
   → YES: Decode compressed bytes with cv2.imdecode()
   
2. Check if message has 'data', 'width', 'height' attributes (RawImage)
   → YES: 
      a. Get encoding (default: 'rgb8')
      b. Reshape byte array to (height, width, channels)
      c. Convert color space to BGR if needed
      
3. Return decoded image or None if decoding fails
```

---

#### `msg_to_dict(msg, depth=0) -> dict | str | primitive`

**Purpose**: Recursively converts protobuf messages to clean, human-readable Python dictionaries for JSON serialization.

**Conversion strategy** (in order of preference):

1. **`MessageToDict`** - Uses Google's protobuf library for proper conversion
2. **`ListFields()`** - Falls back to iterating only set fields (skips DESCRIPTOR)
3. **Iterable handling** - Converts repeated fields and sequences to lists
4. **Attribute extraction** - Last resort, filters out internal protobuf objects

**Handles**:
- `None` values → `null`
- Primitive types (`str`, `int`, `float`, `bool`) → direct values
- Bytes → UTF-8 decode if small, placeholder if large
- NumPy arrays → list if small, placeholder if large
- Lists/tuples → recursive conversion (max 200 items)
- Dictionaries → recursive conversion
- Protobuf messages → clean dict with only data fields

**Filters out** (never appears in output):
- `DESCRIPTOR` and other protobuf metadata
- Internal objects like `_GenericSequence`, `FieldDescriptor`
- Method references and callables

**Example output**:
```json
{
  "timestamp_ns": 1762370549188422000,
  "index": 3,
  "data": {
    "transforms": [
      {
        "timestamp": "2025-11-05T19:22:29.188422Z",
        "parent_frame_id": "world",
        "child_frame_id": "camera",
        "translation": {
          "x": 0.02035904515518046,
          "y": 0.0032828756596984423,
          "z": 0.002191608694142282
        },
        "rotation": {
          "x": -0.39623632452558827,
          "y": 0.2782342033471385,
          "z": 0.6789959547645192,
          "w": 0.5518577684763118
        }
      }
    ]
  }
}
```

---

#### `is_image_topic(schema_name: str, topic: str) -> bool`

**Purpose**: Determines if a topic likely contains image data (for routing to image vs JSON extraction).

**Detection methods**:

1. **Schema-based detection** (high confidence):
   - `foxglove.CompressedImage`
   - `foxglove.RawImage`
   - `sensor_msgs/Image`
   - `sensor_msgs/CompressedImage`

2. **Topic name heuristics** (fallback):
   - Contains: `image`, `camera`, `rgb`, `depth`, `color`, `compressed`

```python
def is_image_topic(schema_name: str, topic: str) -> bool:
    image_schemas = ['foxglove.CompressedImage', 'foxglove.RawImage', ...]
    if schema_name in image_schemas:
        return True
    
    image_keywords = ['image', 'camera', 'rgb', 'depth', 'color', 'compressed']
    topic_lower = topic.lower()
    return any(kw in topic_lower for kw in image_keywords)
```

**Note**: Topics marked as `[IMAGE]` that fail image decoding will fallback to JSON output.

---

#### `extract_mcap(mcap_path: str, output_base: str = None)`

**Purpose**: Main extraction function that orchestrates the entire process.

**Detailed flow**:

```python
# SETUP
mcap_path = Path(mcap_path)
output_base = mcap_path.parent / f"{mcap_path.stem}_extracted"
output_base.mkdir(parents=True, exist_ok=True)

# PASS 1: DISCOVERY
with open(mcap_path, "rb") as f:
    reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
    
    # Get file statistics
    summary = reader.get_summary()
    duration = (summary.statistics.message_end_time - 
                summary.statistics.message_start_time) / 1e9
    
    # Discover all topics
    for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
        topic = channel.topic
        if topic not in topic_info:
            topic_info[topic] = {
                'schema': schema.name,
                'count': 0,
                'is_image': is_image_topic(schema.name, topic),
                'folder': sanitize_topic_name(topic)
            }
        topic_info[topic]['count'] += 1

# CREATE DIRECTORIES
for topic, info in topic_info.items():
    (output_base / info['folder']).mkdir(parents=True, exist_ok=True)

# PASS 2: EXTRACTION
with open(mcap_path, "rb") as f:
    reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
    
    for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
        topic = channel.topic
        info = topic_info[topic]
        idx = counters[topic]
        counters[topic] += 1
        
        topic_dir = output_base / info['folder']
        timestamp_ns = message.log_time
        
        if info['is_image']:
            img = decode_image(decoded_msg)
            if img is not None:
                cv2.imwrite(str(topic_dir / f"{idx:06d}_{timestamp_ns}.png"), img)
            else:
                # Fallback to JSON if image decode fails
                save_as_json(...)
        else:
            save_as_json(...)

# GENERATE SUMMARIES
write_summary_txt(...)
write_topics_json(...)
```

---

### Processing Pipeline

```
                    ┌──────────────┐
                    │  MCAP File   │
                    └──────┬───────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   make_reader()       │
               │   + ProtobufDecoder   │
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │  iter_decoded_msgs()  │
               └───────────┬───────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌─────────────────┐      ┌─────────────────┐
     │  Image Topic?   │      │  Data Topic     │
     │  is_image=True  │      │  is_image=False │
     └────────┬────────┘      └────────┬────────┘
              │                        │
              ▼                        │
     ┌─────────────────┐               │
     │ decode_image()  │               │
     └────────┬────────┘               │
              │                        │
     ┌────────┴────────┐               │
     ▼                 ▼               ▼
┌─────────┐      ┌──────────┐    ┌──────────┐
│  PNG    │      │   JSON   │    │   JSON   │
│ cv2.im  │      │ fallback │    │ msg_to_  │
│ write() │      │          │    │ dict()   │
└─────────┘      └──────────┘    └──────────┘
```

---

## Supported Formats

### Image Schemas
| Schema | Description |
|--------|-------------|
| `foxglove.CompressedImage` | JPEG/PNG compressed images |
| `foxglove.RawImage` | Uncompressed pixel data |
| `sensor_msgs/Image` | ROS1/2 image messages |
| `sensor_msgs/CompressedImage` | ROS1/2 compressed images |

### Data Schemas (saved as JSON)
| Schema | Description |
|--------|-------------|
| `foxglove.PosesInFrame` | Multiple poses with frame ID |
| `foxglove.PoseInFrame` | Single pose with frame ID |
| `foxglove.FrameTransforms` | TF transform data |
| `foxglove.CameraCalibration` | Camera intrinsics/extrinsics |
| `microagi.Health` | Health/status messages |
| `microagi.Meta` | Recording metadata |
| `microagi.TaskEvent` | Task annotations |
| Any protobuf | Generic protobuf → JSON conversion |

---

## File Naming Convention

All extracted files follow this pattern:

```
{index:06d}_{timestamp_ns}.{ext}
```

| Component | Description | Example |
|-----------|-------------|---------|
| `index` | Zero-padded 6-digit sequence number | `000042` |
| `timestamp_ns` | Message log time in nanoseconds | `1762370549054834000` |
| `ext` | File extension (`.png` or `.json`) | `.png` |

**Full example**: `000042_1762370549054834000.png`

This naming ensures:
- Files sort correctly by sequence
- Timestamps allow precise time correlation
- No naming conflicts within a topic

---

## Examples

### Example JSON Output (Transform Data)

**File**: `tf_camera/000003_1762370549188422000.json`

```json
{
  "timestamp_ns": 1762370549188422000,
  "index": 3,
  "data": {
    "transforms": [
      {
        "timestamp": "2025-11-05T19:22:29.188422Z",
        "parent_frame_id": "world",
        "child_frame_id": "camera",
        "translation": {
          "x": 0.02035904515518046,
          "y": 0.0032828756596984423,
          "z": 0.002191608694142282
        },
        "rotation": {
          "x": -0.39623632452558827,
          "y": 0.2782342033471385,
          "z": 0.6789959547645192,
          "w": 0.5518577684763118
        }
      }
    ]
  }
}
```

### Example JSON Output (Task Event)

**File**: `task/000005_1762370549221825000.json`

```json
{
  "timestamp_ns": 1762370549221825000,
  "index": 5,
  "data": {
    "task_title": "Cleaning the stovetop",
    "start_time": "00:00",
    "end_time": "00:11",
    "grouped_tasks": "stovetop wiped",
    "confidence": 1.0
  }
}
```

### Example JSON Output (Camera Calibration)

**File**: `camera_color_info/000000_1762370548872110000.json`

```json
{
  "timestamp_ns": 1762370548872110000,
  "index": 0,
  "data": {
    "timestamp": "2025-11-05T19:22:28.872110Z",
    "width": 1920,
    "height": 1080,
    "distortion_model": "rational_polynomial",
    "D": [-3.11, 4.26, -0.0006, -0.0002, -1.96, -3.11, 4.27, -1.97],
    "K": [1042.56, 0.0, 969.29, 0.0, 1042.52, 532.93, 0.0, 0.0, 1.0],
    "R": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    "P": [1042.56, 0.0, 969.29, 0.0, 0.0, 1042.52, 532.93, 0.0, 0.0, 0.0, 1.0, 0.0],
    "frame_id": "color"
  }
}
```

### Example JSON Output (Metadata)

**File**: `meta/000000_1762370548872110000.json`

```json
{
  "timestamp_ns": 1762370548872110000,
  "index": 0,
  "data": {
    "device_id": "7331910",
    "operator_height_in_m": 1.65,
    "customer": "skild",
    "customer_metadata": "{\"task_id\":100,\"task_name\":{\"name\":\"Cleaning stove top\"}}"
  }
}
```

---

## Troubleshooting

### "No module named 'mcap'"
```bash
pip install mcap mcap-protobuf-support
```

### "cv2 not found"
```bash
pip install opencv-python
```

### Images not extracting (saved as JSON instead)
- The image schema may not be recognized
- The image encoding may be unsupported
- Check the JSON file's `data` field for clues about the format

### Large binary data showing as placeholders
This is intentional. Binary data >1000 bytes is replaced with:
```json
"data": "<bytes: 921600 bytes>"
```
This prevents JSON files from becoming extremely large.

### Out of memory on large files
The script processes messages sequentially and doesn't load the entire file into memory. However, if you have many topics with many messages, the output folder may require significant disk space.

---

## Scripts Included

| Script | Purpose |
|--------|---------|
| `extract_mcap_topics.py` | Full extraction of all topics to folders |
| `list_mcap_topics.py` | Quick listing of topics without extraction |

---

## License

MIT License - Feel free to use and modify.

---

# MCAP Preview Extractor

A minimal, educational example showing how to extract color images from our MCAP files. This script extracts only the **last N seconds** of video frames, making it perfect for quick previews of the end of a recording.

**This README serves as a tutorial for anyone who wants to programmatically read our MCAP files.**

---

## Table of Contents

- [Quick Start](#quick-start)
- [Understanding MCAP Files](#understanding-mcap-files)
- [Our MCAP Structure](#our-mcap-structure)
- [Step-by-Step Code Walkthrough](#step-by-step-code-walkthrough)
- [Working with Different Topics](#working-with-different-topics)
- [Common Patterns](#common-patterns)
- [Full API Reference](#full-api-reference)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract last 5 seconds of color images (default)
python extract_preview.py recording.mcap

# Extract last 10 seconds
python extract_preview.py recording.mcap 10
```

Output: `recording_last_5s/` folder with PNG images named `0000_0.000s.png`, `0001_0.033s.png`, etc.

---

## Understanding MCAP Files

### What is MCAP?

MCAP is a container file format designed for storing **heterogeneous, timestamped data**. Think of it like a video file, but instead of just video frames, it can store:

- Camera images (RGB, depth)
- Sensor readings (IMU, GPS)
- Robot poses and transforms
- Metadata and annotations
- Any other timestamped data

### Key Concepts

| Concept | Description |
|---------|-------------|
| **Topic** | A named data stream (e.g., `/camera/color/image`, `/imu/accel`) |
| **Message** | A single data point on a topic with a timestamp |
| **Schema** | The data type/format of messages on a topic (e.g., `foxglove.CompressedImage`) |
| **Channel** | The binding of a topic to a schema |

### Why MCAP?

- **Self-describing**: Schema information is embedded in the file
- **Efficient**: Supports chunked reading, seeking, and compression
- **Flexible**: Can store any serialization format (Protobuf, JSON, ROS messages)
- **Indexed**: Fast random access to any time range

---

## Our MCAP Structure

Our MCAP files use **Protobuf** serialization with **Foxglove schemas**. Here's what you'll find:

### Topics Overview

| Topic | Schema | Description | Typical Rate |
|-------|--------|-------------|--------------|
| `/camera/color/image` | `foxglove.CompressedImage` | RGB camera frames (JPEG) | ~30 Hz |
| `/camera/depth/image` | `foxglove.CompressedImage` | Depth images (PNG) | ~30 Hz |
| `/camera/color/info` | `foxglove.CameraCalibration` | Camera intrinsics | 1 msg |
| `/body/upper` | `foxglove.PosesInFrame` | Upper body joint poses | ~30 Hz |
| `/hands/left` | `foxglove.PosesInFrame` | Left hand joint poses | ~30 Hz |
| `/hands/right` | `foxglove.PosesInFrame` | Right hand joint poses | ~30 Hz |
| `/tf/body` | `foxglove.FrameTransforms` | Body transform tree | ~30 Hz |
| `/tf/camera` | `foxglove.FrameTransforms` | Camera pose in world | ~30 Hz |
| `/tf/hands` | `foxglove.FrameTransforms` | Hand transforms | ~30 Hz |
| `/imu/accel/sample` | `foxglove.PoseInFrame` | Accelerometer readings | ~200 Hz |
| `/imu/gyro/sample` | `foxglove.PoseInFrame` | Gyroscope readings | ~200 Hz |
| `/meta` | `microagi.Meta` | Recording metadata | 1 msg |
| `/task` | `microagi.TaskEvent` | Task annotations | ~30 Hz |
| `/**/health` | `microagi.Health` | Health/status messages | ~30 Hz |

### Coordinate Frames

```
world (fixed)
  └── body (operator torso)
        ├── left_wrist
        │     └── left_hand_joints[0..20]
        ├── right_wrist
        │     └── right_hand_joints[0..20]
        └── camera
```

### Timestamps

All timestamps are in **nanoseconds since Unix epoch**. To convert:

```python
timestamp_ns = 1762370549054834000
timestamp_sec = timestamp_ns / 1e9  # 1762370549.054834
from datetime import datetime
dt = datetime.fromtimestamp(timestamp_sec)  # 2025-11-05 19:22:29.054834
```

---

## Step-by-Step Code Walkthrough

### Step 1: Import Required Libraries

```python
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
import cv2
import numpy as np
```

| Library | Purpose |
|---------|---------|
| `mcap` | Core MCAP reading functionality |
| `mcap_protobuf.decoder` | Decodes Protobuf messages embedded in MCAP |
| `cv2` (OpenCV) | Image decoding and manipulation |
| `numpy` | Array operations for image data |

### Step 2: Open the MCAP File

```python
with open("recording.mcap", "rb") as f:
    reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
```

**What's happening:**
1. Open the file in binary read mode (`"rb"`)
2. Create an MCAP reader with a Protobuf decoder factory
3. The decoder factory tells the reader how to deserialize messages

**Important:** The `ProtobufDecoderFactory` is essential for our files. Without it, you'll get raw bytes instead of decoded Python objects.

### Step 3: Iterate Over Messages

```python
for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
    # schema: Information about the message type
    # channel: Information about the topic
    # message: Raw message with timestamp
    # decoded_msg: The actual Python object with data
    
    topic = channel.topic           # e.g., "/camera/color/image"
    timestamp_ns = message.log_time # e.g., 1762370549054834000
    schema_name = schema.name       # e.g., "foxglove.CompressedImage"
```

**The four return values:**

| Variable | Type | Contains |
|----------|------|----------|
| `schema` | `Schema` | `.name`, `.encoding`, `.data` (schema definition) |
| `channel` | `Channel` | `.topic`, `.schema_id`, `.metadata` |
| `message` | `Message` | `.log_time`, `.publish_time`, `.data` (raw bytes) |
| `decoded_msg` | `object` | The deserialized Protobuf message (Python object) |

### Step 4: Filter by Topic

```python
COLOR_IMAGE_TOPIC = '/camera/color/image'

for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
    if channel.topic != COLOR_IMAGE_TOPIC:
        continue  # Skip non-image messages
    
    # Process only color images here
```

**Alternative: Filter at read time (more efficient)**

```python
for schema, channel, message, decoded_msg in reader.iter_decoded_messages(
    topics=['/camera/color/image']
):
    # Only color images will be yielded
```

### Step 5: Decode Image Data

Our images use `foxglove.CompressedImage` which contains JPEG/PNG bytes:

```python
def decode_image(msg):
    """Decode a Foxglove CompressedImage message."""
    # Check if it's a CompressedImage (has 'data' and 'format')
    if hasattr(msg, 'data') and hasattr(msg, 'format'):
        data = msg.data  # Raw JPEG/PNG bytes
        
        # Convert bytes to numpy array
        img_array = np.frombuffer(data, dtype=np.uint8)
        
        # Decode JPEG/PNG to BGR image (OpenCV format)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        return img  # Shape: (height, width, 3), dtype: uint8
    
    return None
```

**Understanding the decode process:**

```
CompressedImage message
    │
    ├── .data = b'\xff\xd8\xff\xe0...' (JPEG bytes)
    ├── .format = "jpeg"
    └── .timestamp = {...}
           │
           ▼
    np.frombuffer(data, dtype=np.uint8)
           │
           ▼
    array([255, 216, 255, 224, ...], dtype=uint8)  # 1D array
           │
           ▼
    cv2.imdecode(array, cv2.IMREAD_COLOR)
           │
           ▼
    array([[[B, G, R], ...]], dtype=uint8)  # Shape: (1080, 1920, 3)
```

### Step 6: Handle Timestamps for Duration Filtering (Last N Seconds)

To extract the **last N seconds**, we need a two-pass approach:

1. **First pass**: Read the summary to get the end time
2. **Second pass**: Extract only frames after the cutoff time

```python
# First pass: Get end time from summary
with open("recording.mcap", "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()
    end_time_ns = summary.statistics.message_end_time

# Calculate cutoff (5 seconds before end)
duration_ns = int(5 * 1e9)  # 5 seconds in nanoseconds
cutoff_time_ns = end_time_ns - duration_ns

# Second pass: Extract frames after cutoff
first_frame_time_ns = None

with open("recording.mcap", "rb") as f:
    reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
    
    for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
        if channel.topic != '/camera/color/image':
            continue
        
        timestamp_ns = message.log_time
        
        # Skip frames before cutoff
        if timestamp_ns < cutoff_time_ns:
            continue
        
        # Record first extracted frame time
        if first_frame_time_ns is None:
            first_frame_time_ns = timestamp_ns
        
        # Calculate elapsed time from start of extraction
        elapsed_ns = timestamp_ns - first_frame_time_ns
        elapsed_sec = elapsed_ns / 1e9
        print(f"Frame at {elapsed_sec:.3f}s")
```

### Step 7: Save Images

```python
frame_count = 0

for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
    if channel.topic != '/camera/color/image':
        continue
    
    img = decode_image(decoded_msg)
    if img is not None:
        # Create filename with frame number and timestamp
        elapsed_sec = (message.log_time - start_time_ns) / 1e9
        filename = f"{frame_count:04d}_{elapsed_sec:.3f}s.png"
        
        # Save image
        cv2.imwrite(filename, img)
        
        frame_count += 1
```

---

## Working with Different Topics

### Reading Pose Data (`foxglove.PosesInFrame`)

```python
for schema, channel, message, decoded_msg in reader.iter_decoded_messages(
    topics=['/hands/left']
):
    # decoded_msg is a foxglove.PosesInFrame
    frame_id = decoded_msg.frame_id  # e.g., "left_wrist"
    
    for i, pose in enumerate(decoded_msg.poses):
        position = pose.position
        orientation = pose.orientation
        
        print(f"Joint {i}:")
        print(f"  Position: ({position.x}, {position.y}, {position.z})")
        print(f"  Orientation: ({orientation.x}, {orientation.y}, {orientation.z}, {orientation.w})")
```

**PosesInFrame structure:**
```
PosesInFrame
├── timestamp: Timestamp
├── frame_id: str (e.g., "left_wrist")
└── poses: List[Pose]
      └── Pose
            ├── position: Vector3 {x, y, z}
            └── orientation: Quaternion {x, y, z, w}
```

### Reading Transform Data (`foxglove.FrameTransforms`)

```python
for schema, channel, message, decoded_msg in reader.iter_decoded_messages(
    topics=['/tf/camera']
):
    for transform in decoded_msg.transforms:
        parent = transform.parent_frame_id  # e.g., "world"
        child = transform.child_frame_id    # e.g., "camera"
        
        pos = transform.translation
        rot = transform.rotation
        
        print(f"{parent} → {child}")
        print(f"  Translation: ({pos.x}, {pos.y}, {pos.z})")
        print(f"  Rotation: ({rot.x}, {rot.y}, {rot.z}, {rot.w})")
```

### Reading Metadata (`microagi.Meta`)

```python
for schema, channel, message, decoded_msg in reader.iter_decoded_messages(
    topics=['/meta']
):
    device_id = decoded_msg.device_id
    operator_height = decoded_msg.operator_height_in_m
    customer = decoded_msg.customer
    metadata_json = decoded_msg.customer_metadata  # JSON string
    
    import json
    metadata = json.loads(metadata_json)
    task_name = metadata['task_name']['name']
    
    print(f"Device: {device_id}")
    print(f"Operator height: {operator_height}m")
    print(f"Task: {task_name}")
```

### Reading IMU Data (`foxglove.PoseInFrame`)

```python
for schema, channel, message, decoded_msg in reader.iter_decoded_messages(
    topics=['/imu/accel/sample']
):
    # IMU data is stored in the position field of a Pose
    accel = decoded_msg.pose.position
    
    print(f"Acceleration: x={accel.x:.3f}, y={accel.y:.3f}, z={accel.z:.3f} m/s²")
```

---

## Common Patterns

### Get File Summary Without Reading All Messages

```python
with open("recording.mcap", "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()
    
    if summary and summary.statistics:
        stats = summary.statistics
        
        # Duration
        duration_ns = stats.message_end_time - stats.message_start_time
        duration_sec = duration_ns / 1e9
        print(f"Duration: {duration_sec:.2f} seconds")
        
        # Total messages
        print(f"Total messages: {stats.message_count}")
        
        # Messages per channel
        for channel_id, count in stats.channel_message_counts.items():
            channel = summary.channels[channel_id]
            print(f"  {channel.topic}: {count} messages")
```

### List All Topics and Schemas

```python
with open("recording.mcap", "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()
    
    channels = summary.channels if summary else {}
    schemas = summary.schemas if summary else {}
    
    for channel_id, channel in channels.items():
        schema = schemas.get(channel.schema_id)
        schema_name = schema.name if schema else "unknown"
        print(f"{channel.topic}: {schema_name}")
```

### Convert Protobuf to Dict (for JSON export)

```python
from google.protobuf.json_format import MessageToDict

for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
    # Convert to Python dict
    data_dict = MessageToDict(decoded_msg, preserving_proto_field_name=True)
    
    # Now you can serialize to JSON
    import json
    json_str = json.dumps(data_dict, indent=2)
```

### Time-Based Seeking (Jump to Specific Time)

```python
# Note: Requires indexed MCAP files
target_time_ns = int(5.0 * 1e9)  # Jump to 5 seconds

with open("recording.mcap", "rb") as f:
    reader = make_reader(f)
    
    # Get start time
    summary = reader.get_summary()
    start_time = summary.statistics.message_start_time
    
    # Seek to target time
    for schema, channel, message, decoded_msg in reader.iter_decoded_messages(
        start_time=start_time + target_time_ns
    ):
        # Messages from 5 seconds onwards
        pass
```

---

## Full API Reference

### MCAP Reader Methods

| Method | Description |
|--------|-------------|
| `make_reader(file, decoder_factories=[])` | Create a reader for an MCAP file |
| `reader.get_summary()` | Get file metadata without reading messages |
| `reader.iter_messages()` | Iterate raw messages (not decoded) |
| `reader.iter_decoded_messages()` | Iterate decoded messages |
| `reader.iter_decoded_messages(topics=[...])` | Filter by topic names |
| `reader.iter_decoded_messages(start_time=ns)` | Start from timestamp |
| `reader.iter_decoded_messages(end_time=ns)` | End at timestamp |

### Summary Object

```python
summary = reader.get_summary()
summary.statistics.message_count      # Total messages
summary.statistics.message_start_time # First timestamp (ns)
summary.statistics.message_end_time   # Last timestamp (ns)
summary.statistics.channel_message_counts  # Dict[channel_id, count]
summary.channels  # Dict[channel_id, Channel]
summary.schemas   # Dict[schema_id, Schema]
```

### Foxglove Schema Types

| Schema | Key Fields |
|--------|------------|
| `foxglove.CompressedImage` | `.data`, `.format`, `.timestamp` |
| `foxglove.RawImage` | `.data`, `.width`, `.height`, `.encoding`, `.timestamp` |
| `foxglove.PosesInFrame` | `.frame_id`, `.poses[]`, `.timestamp` |
| `foxglove.PoseInFrame` | `.frame_id`, `.pose`, `.timestamp` |
| `foxglove.FrameTransforms` | `.transforms[]` |
| `foxglove.FrameTransform` | `.parent_frame_id`, `.child_frame_id`, `.translation`, `.rotation`, `.timestamp` |
| `foxglove.CameraCalibration` | `.width`, `.height`, `.K`, `.D`, `.R`, `.P`, `.distortion_model` |

---

## Troubleshooting

### "No decoder for schema"

You're missing the protobuf decoder factory:

```python
# Wrong
reader = make_reader(f)

# Correct
reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
```

### "ModuleNotFoundError: No module named 'mcap'"

Install dependencies:

```bash
pip install mcap mcap-protobuf-support foxglove-schemas-protobuf
```

### Images are None / Not decoding

Check the image format:

```python
print(f"Format: {decoded_msg.format}")  # Should be "jpeg" or "png"
print(f"Data length: {len(decoded_msg.data)}")  # Should be > 0
```

### Getting raw bytes instead of decoded objects

Use `iter_decoded_messages()` not `iter_messages()`:

```python
# Wrong - gives raw bytes
for schema, channel, message in reader.iter_messages():
    data = message.data  # Raw bytes

# Correct - gives Python objects
for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
    data = decoded_msg  # Decoded Protobuf message
```

### Timestamps look weird (huge numbers)

Timestamps are in **nanoseconds**, not seconds or milliseconds:

```python
timestamp_ns = 1762370549054834000
timestamp_sec = timestamp_ns / 1e9  # Divide by 1 billion
# Result: 1762370549.054834
```

---

## Example: Complete Minimal Script

```python
"""Minimal example: Extract last 5 seconds of color frames from an MCAP file."""

from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory as ProtobufDecoderFactory
import cv2
import numpy as np
import os

MCAP_FILE = "recording.mcap"
OUTPUT_DIR = "frames"
LAST_SECONDS = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# First pass: get end time
with open(MCAP_FILE, "rb") as f:
    reader = make_reader(f)
    summary = reader.get_summary()
    end_time_ns = summary.statistics.message_end_time
    cutoff_time_ns = end_time_ns - int(LAST_SECONDS * 1e9)

# Second pass: extract frames after cutoff
frame_count = 0

with open(MCAP_FILE, "rb") as f:
    reader = make_reader(f, decoder_factories=[ProtobufDecoderFactory()])
    
    for schema, channel, message, decoded_msg in reader.iter_decoded_messages(
        topics=['/camera/color/image']
    ):
        if message.log_time < cutoff_time_ns:
            continue  # Skip frames before cutoff
        
        # Decode JPEG to image
        img = cv2.imdecode(
            np.frombuffer(decoded_msg.data, dtype=np.uint8),
            cv2.IMREAD_COLOR
        )
        
        if img is not None:
            cv2.imwrite(f"{OUTPUT_DIR}/frame_{frame_count:04d}.png", img)
            frame_count += 1
            print(f"Saved frame {frame_count}")

print(f"Done! Extracted {frame_count} frames to {OUTPUT_DIR}/")
```

---

## License

MIT License - Feel free to use, modify, and learn from this code.

