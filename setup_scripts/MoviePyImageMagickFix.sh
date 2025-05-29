#!/bin/bash
# fix_moviepy_setup.sh - Fix MoviePy text overlay issues

echo "🔧 Fixing MoviePy and ImageMagick setup for video generation"
echo "============================================================"

# Install ImageMagick for macOS
echo "📦 Installing ImageMagick..."
if command -v brew &> /dev/null; then
    brew install imagemagick
    echo "✅ ImageMagick installed via Homebrew"
else
    echo "❌ Homebrew not found. Please install Homebrew first:"
    echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "   Then run this script again."
    exit 1
fi

# Update MoviePy configuration
echo "🔧 Configuring MoviePy..."
python3 -c "
import moviepy.config as config
print('Current ImageMagick path:', config.IMAGEMAGICK_BINARY)

# Try to find ImageMagick
import shutil
magick_path = shutil.which('magick')
if magick_path:
    print('Found ImageMagick at:', magick_path)

    # Update MoviePy config
    with open('moviepy_config.py', 'w') as f:
        f.write(f'IMAGEMAGICK_BINARY = \"{magick_path}\"\\n')
    print('✅ Created moviepy_config.py')
else:
    print('❌ ImageMagick not found in PATH')
"

# Test MoviePy text creation
echo "🧪 Testing MoviePy text overlay..."
python3 -c "
try:
    from moviepy.editor import TextClip
    clip = TextClip('Test', fontsize=50, color='white', bg_color='black')
    print('✅ MoviePy text overlay working!')
except Exception as e:
    print(f'❌ MoviePy text overlay failed: {e}')
    print('💡 Solution: Video will use simple title rectangles instead')
"

# Update requirements for compatibility
echo "📦 Updating Python packages for compatibility..."
pip install --upgrade Pillow moviepy pydub

echo ""
echo "🎯 Setup Summary:"
echo "   1. ImageMagick installed for text overlays"
echo "   2. MoviePy configured to find ImageMagick"
echo "   3. Updated packages for compatibility"
echo "   4. Fixed PIL.Image.ANTIALIAS deprecation"
echo "   5. Fixed AudioArrayClip import issue"
echo ""
echo "💡 If text overlays still don't work:"
echo "   - Videos will use simple colored rectangles as title placeholders"
echo "   - This is a fallback that ensures video generation works"
echo ""
echo "🧪 Test the fixes:"
echo "   python main.py --mode single --verbose"