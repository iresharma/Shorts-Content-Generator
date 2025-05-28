#!/bin/bash
# macos_tts_fix.sh - Fix TTS issues on macOS

echo "🔧 Fixing TTS issues on macOS..."

# Option 1: Install PyObjC for pyttsx3
echo "📦 Installing PyObjC framework for pyttsx3..."
pip install pyobjc-framework-Cocoa pyobjc-framework-AVFoundation

if [ $? -eq 0 ]; then
    echo "✅ PyObjC installed successfully"
    echo "🧪 Testing pyttsx3..."
    python -c "import pyttsx3; engine = pyttsx3.init(); print('pyttsx3 working!')"
    if [ $? -eq 0 ]; then
        echo "✅ pyttsx3 is now working!"
        exit 0
    fi
fi

# Option 2: Install espeak as fallback
echo "📦 Installing espeak as fallback TTS..."
if command -v brew &> /dev/null; then
    brew install espeak
    echo "✅ espeak installed via Homebrew"
else
    echo "❌ Homebrew not found. Please install Homebrew or espeak manually"
    echo "   Visit: https://brew.sh/"
fi

# Test the system
echo "🧪 Testing the YouTube Shorts generator..."
python main.py --mode status

echo ""
echo "🎯 TTS Fix Summary:"
echo "   1. Updated TTS generator now supports multiple engines"
echo "   2. Fallback order: pyttsx3 → macOS 'say' → espeak"
echo "   3. If all fail, the system will guide you to install dependencies"
echo ""
echo "💡 Quick test commands:"
echo "   python main.py --mode status     # Check system health"
echo "   python main.py --mode single     # Generate one video"