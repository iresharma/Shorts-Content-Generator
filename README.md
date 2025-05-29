# Shorts Content Generator - Usage Guide
Completely free CPU first youtube shorts generator.

## Feature
- Fast 🔥
- Optimised for short content video 
- Local first
- No paid APIs (except pexels for images)
- Human sounding
- CPU first

## Setup Instructions

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Create directories
```bash
mkdir -p data output/videos temp
```

### 3. Create .env file with your settings
Copy the template below and add your Pexels API key

### 4. Create topics.json file in data/ directory
Use the sample below as a starting point

### 5. Run all scripts in ./setup_scripts
```bash
chmod 777 setup_scripts/*

./setup_scripts/CoquiSetup.sh
....
....
```

### 6. Test the system
```bash
python main.py --mode status
```

### 6. Generate your first video
```bash
python main.py --mode single --verbose
```

### 8. Set up cron job for automation
Add to crontab for every 30 minutes:
```bash
*/30 * * * * cd /path/to/your/project && python main.py --mode single
```

## Sample data/topics.json
```json
[
  {
    "title": "Abstraction",
    "description": "Abstraction is the process of reducing complexity by focusing on the essential characteristics of an object or system, while hiding irrelevant details. It allows developers to manage complexity by creating simplified models or interfaces that represent real-world entities, thereby enabling easier understanding, design, and maintenance of software systems.",
    "complete": false
  },
  {
    "title": "Algorithms",
    "description": "Algorithms are step-by-step procedures or formulas for solving problems. In computer science, they define a sequence of computational steps that transform input data into desired output. Good algorithms are efficient, clear, and correct, forming the foundation of all software development and computational thinking.",
    "complete": false
  },
  {
    "title": "API Design",
    "description": "API design involves creating interfaces that allow different software applications to communicate effectively. Well-designed APIs are intuitive, consistent, and provide clear documentation. They serve as contracts between different parts of a system, enabling modularity and integration between diverse software components.",
    "complete": false
  }
]
```
### Sample .env file

```dotenv
# Pexels API Configuration
PEXELS_API_KEY=your_actual_pexels_api_key_here

# File Paths
OUTPUT_DIRECTORY=./output/videos
TEMP_DIRECTORY=./temp
TOPICS_FILE=./data/topics.json
PROCESSED_TOPICS_FILE=./data/processed_topics.json

# TTS Configuration (Enhanced for better voice quality)
TTS_RATE=160
TTS_VOLUME=0.85
INTRO_TEMPLATE=Today, we will talk about {title}.

# Video Settings (YouTube Shorts - 9:16 aspect ratio)
VIDEO_WIDTH=1080
VIDEO_HEIGHT=1920
VIDEO_FPS=30
MIN_DURATION=30
IMAGES_PER_VIDEO=6
IMAGE_DURATION=5.0

# Pexels Settings
PEXELS_IMAGES_PER_REQUEST=10
PEXELS_IMAGE_SIZE=large
PEXELS_SEARCH_TAGS=dark,computer,coding,technology,abstract,digital,programming,software
```

### Cron Job Setup

```bash
# Edit crontab
crontab -e

# Add this line for every 30 minutes
*/30 * * * * cd /path/to/youtube-shorts-generator && /usr/bin/python3 main.py --mode single >> logs/cron.log 2>&1

# Or for every hour at minute 0
0 * * * * cd /path/to/youtube-shorts-generator && /usr/bin/python3 main.py --mode single >> logs/cron.log 2>&1
```


### Usage Examples
Check out the scripts in the example directory


### Troubleshooting

If TTS fails, install additional dependencies:
```bash
sudo apt-get install espeak espeak-data libespeak-dev ffmpeg
```

For macOS:
```bash
brew install espeak ffmpeg
```

If Pexels API fails, check:
- API key is correct in .env file
- Internet connection is working
- API rate limits haven't been exceeded

If video generation fails:
- Ensure ffmpeg is installed
- Check temp directory permissions
- Verify output directory is writable

## 🎬 Two Ways to Generate Videos

### Method 1: File-Based Generation (Original)
Use topics from JSON file with automatic progression.

```bash
# Generate one video from topics file
python main.py --mode single --verbose

# Generate multiple videos
python main.py --mode continuous --max-iterations 5

# Check system status
python main.py --mode status
```

### Method 2: Direct Data Generation (NEW!)
Generate videos directly from topic data in your code.

```python
from main import MainOrchestrator

# Define your topic
data = {
    "title": "Abstraction",
    "description": "Abstraction is the process of reducing complexity by focusing on the essential characteristics of an object or system, while hiding irrelevant details. It allows developers to manage complexity by creating simplified models or interfaces that represent real-world entities, thereby enabling easier understanding, design, and maintenance of software systems.",
    "complete": False  # Optional field
}

# Create orchestrator and generate video
mo = MainOrchestrator.from_topic_data(data)
result = mo.generate()

# Check result
if result["success"]:
    print(f"✅ Video: {result['video_path']}")
    print(f"⏱️  Total time: {result['timing']['total']:.2f}s")
else:
    print(f"❌ Error: {result['error']}")
```

## ⏱️ Timing Information

Both methods now provide detailed timing breakdowns:

```python
result = {
    "success": True,
    "video_path": "output/videos/shorts_001_Abstraction_1748476582.mp4",
    "audio_duration": 23.66,
    "images_count": 6,
    "timing": {
        "validation": 0.05,        # Topic validation
        "audio_generation": 4.68,  # TTS generation  
        "image_fetching": 2.34,    # Pexels API calls
        "video_creation": 15.23,   # Video composition
        "file_update": 0.12,       # Mark topic complete (file-based only)
        "total": 22.42             # Total generation time
    },
    "stats": {...}
}
```

## 🔄 Batch Processing Example

```python
topics = [
    {"title": "API Design", "description": "..."},
    {"title": "Algorithms", "description": "..."},
    {"title": "Data Structures", "description": "..."}
]

results = []
for topic_data in topics:
    mo = MainOrchestrator.from_topic_data(topic_data)
    result = mo.generate()
    results.append(result)
    
    if result["success"]:
        print(f"✅ {topic_data['title']}: {result['timing']['total']:.1f}s")
    else:
        print(f"❌ {topic_data['title']}: {result['error']}")
```

## 🛠️ Integration Examples

### Web API Integration
```python
from flask import Flask, request, jsonify
from main import MainOrchestrator

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_video():
    data = request.json
    
    try:
        mo = MainOrchestrator.from_topic_data(data)
        result = mo.generate()
        
        return jsonify({
            "success": result["success"],
            "video_url": result["video_path"] if result["success"] else None,
            "duration": result["audio_duration"],
            "processing_time": result["timing"]["total"],
            "error": result.get("error")
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

### Queue Processing
```python
import queue
import threading
from main import MainOrchestrator

def video_worker(topic_queue, result_queue):
    """Worker thread for processing video generation."""
    while True:
        topic_data = topic_queue.get()
        if topic_data is None:
            break
        
        try:
            mo = MainOrchestrator.from_topic_data(topic_data)
            result = mo.generate()
            result_queue.put(result)
        except Exception as e:
            result_queue.put({"success": False, "error": str(e)})
        finally:
            topic_queue.task_done()

# Usage
topic_queue = queue.Queue()
result_queue = queue.Queue()

# Start worker threads
for i in range(2):  # 2 concurrent workers
    t = threading.Thread(target=video_worker, args=(topic_queue, result_queue))
    t.start()

# Add topics to queue
topics = [...]  # Your topic list
for topic in topics:
    topic_queue.put(topic)

# Collect results
for i in range(len(topics)):
    result = result_queue.get()
    print(f"Result: {result}")
```

## 📊 Performance Monitoring

```python
import time
from main import MainOrchestrator

def benchmark_generation(topic_data, runs=3):
    """Benchmark video generation performance."""
    times = []
    
    for i in range(runs):
        mo = MainOrchestrator.from_topic_data(topic_data)
        result = mo.generate()
        
        if result["success"]:
            times.append(result["timing"]["total"])
            print(f"Run {i+1}: {result['timing']['total']:.2f}s")
        else:
            print(f"Run {i+1}: FAILED - {result['error']}")
    
    if times:
        avg_time = sum(times) / len(times)
        print(f"\nAverage generation time: {avg_time:.2f}s")
        print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s")

# Run benchmark
topic = {"title": "Test", "description": "Test description..."}
benchmark_generation(topic)
```

## 🎯 Best Practices

1. **Reuse Orchestrator**: For batch processing, create one orchestrator per topic
2. **Monitor Timing**: Use timing data to optimize your workflow
3. **Handle Errors**: Always check `result["success"]` before using output
4. **Resource Management**: Generation is resource-intensive - limit concurrent processes
5. **Cleanup**: Temporary files are cleaned up automatically after each generation
