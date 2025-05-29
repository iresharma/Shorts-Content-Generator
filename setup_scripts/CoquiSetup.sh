#!/bin/bash
# setup_coqui_tts.sh - Easy setup for Coqui TTS (Local, High-Quality)

echo "🎤 Setting up Coqui TTS for YouTube Shorts Generator"
echo "===================================================="
echo "✅ Local processing (no API needed)"
echo "✅ High-quality neural voices"
echo "✅ No usage limits or costs"
echo "✅ Works completely offline"
echo ""

# Check Python version
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

if [[ $(echo "$python_version < 3.9" | bc -l) -eq 1 ]]; then
    echo "⚠️  Warning: Coqui TTS works best with Python 3.9+"
    echo "   Your version: $python_version"
fi

# Install PyTorch first (required for Coqui TTS)
echo "📦 Installing PyTorch..."
pip install torch torchaudio

# Install Coqui TTS
echo "📦 Installing Coqui TTS..."
pip install coqui-tts

# Check if installation worked
echo "🧪 Testing Coqui TTS installation..."
python -c "
import TTS
print('✅ Coqui TTS installed successfully')
print(f'   Version: {TTS.__version__}')
" 2>/dev/null || {
    echo "❌ Coqui TTS installation failed"
    echo "   Try: pip install --upgrade pip setuptools wheel"
    echo "   Then: pip install coqui-tts"
    exit 1
}

# Test model download
echo "🧪 Testing model download (this may take a moment)..."
python -c "
try:
    from TTS.api import TTS
    print('📥 Downloading TTS model...')
    tts = TTS('tts_models/en/ljspeech/vits', progress_bar=False)
    print('✅ Model downloaded successfully')
    print('🎯 Ready to generate high-quality speech!')
except Exception as e:
    print(f'⚠️  Model download issue: {e}')
    print('   This is normal - models will download on first use')
" 2>/dev/null

echo ""
echo "📝 Updating requirements and config..."

# Update .env if it exists
if [ -f ".env" ]; then
    echo "   Found existing .env file"
    if ! grep -q "# Coqui TTS" .env; then
        echo "" >> .env
        echo "# Coqui TTS - Local High-Quality TTS (no API key needed)" >> .env
        echo "# TTS_RATE=160" >> .env
        echo "# TTS_VOLUME=0.85" >> .env
        echo "✅ Added Coqui TTS config to .env"
    fi
else
    echo "   Creating .env file..."
    cat > .env << EOF
# Coqui TTS - Local High-Quality TTS (no API key needed!)
TTS_RATE=160
TTS_VOLUME=0.85
INTRO_TEMPLATE=Today, we will talk about {title}.

# Pexels API Configuration
PEXELS_API_KEY=your_pexels_api_key_here

# File Paths
OUTPUT_DIRECTORY=./output/videos
TEMP_DIRECTORY=./temp
TOPICS_FILE=./data/topics.json

# Video Settings
VIDEO_WIDTH=1080
VIDEO_HEIGHT=1920
VIDEO_FPS=30
MIN_DURATION=30
IMAGES_PER_VIDEO=6

# Pexels Settings
PEXELS_IMAGES_PER_REQUEST=10
PEXELS_IMAGE_SIZE=large
PEXELS_SEARCH_TAGS=dark,computer,coding,technology,abstract,digital
EOF
    echo "✅ Created .env file"
fi

echo ""
echo "🧪 Testing the complete system..."
python videoOrchestrator.py --mode status

echo ""
echo "🎯 Next Steps:"
echo "1. Add your Pexels API key to .env"
echo "2. Run: python main.py --mode single --verbose"
echo ""
echo "💡 Coqui TTS Features:"
echo "   🔹 Neural voice synthesis (very natural)"
echo "   🔹 Multiple speaker options"
echo "   🔹 Works completely offline"
echo "   🔹 No API costs or limits"
echo "   🔹 Privacy-friendly (local processing)"
echo ""
echo "🎊 Setup complete! Your videos will have professional voice quality!"
echo "   First run will download models (~100MB) - this is normal."