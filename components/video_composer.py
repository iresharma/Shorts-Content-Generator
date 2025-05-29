# components/video_composer.py

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from moviepy.editor import (
    VideoFileClip, ImageClip, AudioFileClip, CompositeVideoClip,
    TextClip, concatenate_videoclips, ColorClip
)
from PIL import Image, ImageEnhance
import math


class VideoComposer:
    """Handles video composition combining audio, images, and title text."""

    def __init__(self, config_manager):
        """Initialize video composer with configuration."""
        self.config = config_manager
        self.video_settings = self.config.get_video_settings()
        self.output_dir = self.config.get_output_directory()
        self.temp_dir = self.config.get_temp_directory()

        # Video dimensions (9:16 aspect ratio for YouTube Shorts)
        self.width = self.video_settings['width']
        self.height = self.video_settings['height']
        self.fps = self.video_settings['fps']
        self.min_duration = self.video_settings['min_duration']

        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        logging.info(f"Video composer initialized - {self.width}x{self.height} @ {self.fps}fps")

    def create_video(self, audio_path: str, image_paths: List[str], title: str,
                     topic_index: int) -> Tuple[bool, Optional[str]]:
        """
        Create a complete video from audio, images, and title with perfect timing.

        Args:
            audio_path: Path to the audio file
            image_paths: List of paths to background images
            title: Video title text
            topic_index: Index of the topic for filename

        Returns:
            Tuple of (success: bool, video_path: Optional[str])
        """
        try:
            # Load and validate audio
            audio_clip = self._load_audio(audio_path)
            if not audio_clip:
                return False, None

            audio_duration = audio_clip.duration
            logging.info(f"🎵 Audio loaded - Duration: {audio_duration:.2f}s")

            # Ensure minimum duration
            target_duration = max(audio_duration, self.min_duration)
            if target_duration > audio_duration:
                logging.info(f"Extending video to minimum duration: {target_duration:.2f}s")

            # Validate and prepare images
            valid_images = self._validate_images(image_paths)
            if not valid_images:
                logging.error("No valid images available for video")
                return False, None

            logging.info(f"🖼️  Using {len(valid_images)} valid images")

            # Create background video from images
            logging.info(f"🎬 Creating slideshow for {target_duration:.2f}s...")
            background_video = self._create_background_slideshow(valid_images, target_duration)
            if not background_video:
                return False, None

            # Verify slideshow duration
            slideshow_duration = background_video.duration
            logging.info(f"✅ Slideshow created - Duration: {slideshow_duration:.2f}s")

            if abs(slideshow_duration - target_duration) > 0.1:
                logging.warning(
                    f"⚠️  Slideshow duration mismatch: Expected {target_duration:.2f}s, Got {slideshow_duration:.2f}s")
                # Force correct duration
                background_video = background_video.set_duration(target_duration)
                logging.info(f"🔧 Corrected slideshow duration to {target_duration:.2f}s")

            # Create title overlay
            logging.info("📝 Creating title overlay...")
            title_overlay = self._create_title_overlay(title, target_duration)
            if title_overlay:
                logging.info("✅ Title overlay created successfully")
            else:
                logging.warning("⚠️  Title overlay creation failed")

            # Extend audio if needed
            final_audio = audio_clip
            if target_duration > audio_duration:
                # Add silence to match target duration using pydub
                from pydub import AudioSegment
                from moviepy.editor import AudioFileClip

                silence_duration = target_duration - audio_duration
                logging.info(f"🔇 Adding {silence_duration:.2f}s of silence to audio")

                # Load original audio with pydub
                audio_segment = AudioSegment.from_mp3(audio_path)

                # Generate silence using the correct pydub method
                silence_ms = int(silence_duration * 1000)  # Convert to milliseconds
                silence = AudioSegment.silent(duration=silence_ms)

                # Concatenate audio with silence
                extended_audio = audio_segment + silence

                # Save extended audio
                extended_path = Path(self.temp_dir) / f"extended_{int(time.time())}.mp3"
                extended_audio.export(extended_path, format="mp3")

                # Load the extended audio with moviepy
                final_audio = AudioFileClip(str(extended_path))

            logging.info(f"🎵 Final audio duration: {final_audio.duration:.2f}s")

            # Combine all elements
            logging.info("🎭 Composing final video...")
            final_video = self._compose_final_video(background_video, title_overlay, final_audio)
            if not final_video:
                return False, None

            # Final duration check
            final_duration = final_video.duration
            final_audio_duration = final_audio.duration
            logging.info(f"🎯 Final check - Video: {final_duration:.2f}s, Audio: {final_audio_duration:.2f}s")

            if abs(final_duration - final_audio_duration) > 0.05:
                logging.error(f"❌ Critical timing error - Video/Audio duration mismatch!")
                return False, None

            # Generate output filename
            output_path = self._generate_output_filename(title, topic_index)

            # Render and save video
            logging.info(f"🎬 Rendering video to: {output_path}")
            success = self._render_video(final_video, output_path)

            # Cleanup
            self._cleanup_clips([audio_clip, final_audio, background_video, title_overlay, final_video])

            if success:
                logging.info(f"🎉 Video created successfully: {output_path}")
                logging.info(f"📊 Final stats - Duration: {final_duration:.2f}s, Images: {len(valid_images)}")
                return True, output_path
            else:
                return False, None

        except Exception as e:
            logging.error(f"💥 Error creating video: {e}")
            return False, None

    def _load_audio(self, audio_path: str) -> Optional[AudioFileClip]:
        """Load and validate audio file."""
        try:
            if not Path(audio_path).exists():
                logging.error(f"Audio file not found: {audio_path}")
                return None

            audio_clip = AudioFileClip(audio_path)

            if audio_clip.duration <= 0:
                logging.error("Audio file has no duration")
                return None

            return audio_clip

        except Exception as e:
            logging.error(f"Error loading audio: {e}")
            return None

    def _validate_images(self, image_paths: List[str]) -> List[str]:
        """Validate image files and return list of valid ones with detailed logging."""
        valid_images = []

        logging.info(f"🔍 Validating {len(image_paths)} images...")

        for i, image_path in enumerate(image_paths):
            try:
                if not Path(image_path).exists():
                    logging.warning(f"❌ Image {i + 1} not found: {image_path}")
                    continue

                # Try to open with PIL to validate
                with Image.open(image_path) as img:
                    if img.size[0] > 100 and img.size[1] > 100:  # Minimum size check
                        valid_images.append(image_path)
                        logging.debug(f"✅ Image {i + 1} valid: {img.size[0]}x{img.size[1]} - {Path(image_path).name}")
                    else:
                        logging.warning(f"❌ Image {i + 1} too small ({img.size}): {image_path}")

            except Exception as e:
                logging.warning(f"❌ Image {i + 1} invalid ({e}): {image_path}")

        logging.info(f"✅ Validated {len(valid_images)} out of {len(image_paths)} images")

        if len(valid_images) == 0:
            logging.error("❌ No valid images found! Cannot create video.")
        elif len(valid_images) < 3:
            logging.warning(f"⚠️  Only {len(valid_images)} valid images found. Video quality may be reduced.")

        return valid_images

    def _create_background_slideshow(self, image_paths: List[str], duration: float) -> Optional[VideoFileClip]:
        """Create a slideshow video from background images that exactly matches audio duration."""
        try:
            if not image_paths:
                return None

            logging.info(f"Creating slideshow for exact duration: {duration:.2f} seconds")

            # Ensure we have enough images by cycling if needed
            min_images_needed = max(2, int(duration / 4))  # At least 4 seconds per image for readability

            # Cycle through images if we don't have enough
            original_count = len(image_paths)
            while len(image_paths) < min_images_needed and len(image_paths) < 20:  # Cap at 20 to avoid too many
                image_paths.extend(image_paths[:min(original_count, min_images_needed - len(image_paths))])

            num_images = len(image_paths)
            logging.info(f"Using {num_images} images (cycled from {original_count} original images)")

            # Calculate precise timing - NO transitions to avoid timing issues
            base_time_per_image = duration / num_images

            # Ensure minimum 3 seconds per image for readability
            if base_time_per_image < 3.0:
                # Use fewer images if duration is short
                num_images = max(2, int(duration / 3.0))
                image_paths = image_paths[:num_images]
                base_time_per_image = duration / num_images
                logging.info(f"Adjusted to {num_images} images for better timing ({base_time_per_image:.2f}s each)")

            video_clips = []
            cumulative_time = 0.0

            logging.info(f"Target slideshow duration: {duration:.2f}s with {num_images} images")

            for i, image_path in enumerate(image_paths):
                try:
                    # Calculate exact duration for this clip
                    if i == len(image_paths) - 1:
                        # Last image gets ALL remaining time to ensure exact total duration
                        clip_duration = duration - cumulative_time
                        logging.info(f"Last image gets remaining time: {clip_duration:.2f}s")
                    else:
                        clip_duration = base_time_per_image

                    # Ensure minimum duration
                    if clip_duration < 0.5:
                        logging.warning(f"Clip duration too short: {clip_duration:.2f}s, adjusting...")
                        clip_duration = 0.5

                    # Process image
                    processed_image = self._process_image_for_video(image_path)
                    if not processed_image:
                        logging.warning(f"Failed to process image {i}, skipping...")
                        continue

                    # Create image clip with EXACT duration
                    img_clip = ImageClip(processed_image, duration=clip_duration)
                    img_clip = img_clip.set_fps(self.fps)

                    # Add SUBTLE visual effects (that don't affect timing)
                    effect_type = i % 4
                    if effect_type == 0:
                        # Slow zoom in (1.0 to 1.05)
                        img_clip = img_clip.resize(lambda t: 1 + 0.05 * (t / clip_duration))
                    elif effect_type == 1:
                        # Slow zoom out (1.05 to 1.0)
                        img_clip = img_clip.resize(lambda t: 1.05 - 0.05 * (t / clip_duration))
                    elif effect_type == 2:
                        # Slight pan left to right
                        img_clip = img_clip.set_position(lambda t: (-20 + 40 * (t / clip_duration), 'center'))
                    # effect_type == 3: Static (no effect)

                    video_clips.append(img_clip)
                    cumulative_time += clip_duration

                    logging.debug(f"Image {i + 1}/{num_images}: {clip_duration:.2f}s (total: {cumulative_time:.2f}s)")

                except Exception as e:
                    logging.warning(f"Error processing image {i} ({image_path}): {e}")
                    continue

            if not video_clips:
                logging.error("No valid video clips created from images")
                return None

            # Concatenate clips WITHOUT transitions to maintain exact timing
            logging.info(f"Concatenating {len(video_clips)} clips...")
            slideshow = concatenate_videoclips(video_clips, method="compose")

            # Force exact duration match
            actual_duration = slideshow.duration
            logging.info(f"Slideshow created - Target: {duration:.2f}s, Actual: {actual_duration:.2f}s")

            if abs(actual_duration - duration) > 0.1:  # If off by more than 0.1 seconds
                logging.warning(f"Duration mismatch detected, forcing exact duration...")
                slideshow = slideshow.set_duration(duration)
                logging.info(f"Duration corrected to: {duration:.2f}s")

            return slideshow

        except Exception as e:
            logging.error(f"Error creating background slideshow: {e}")
            return None

    def _process_image_for_video(self, image_path: str) -> Optional[str]:
        """Process image to fit video dimensions and enhance for video use."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Calculate scaling to fill frame while maintaining aspect ratio
                img_width, img_height = img.size
                scale_w = self.width / img_width
                scale_h = self.height / img_height
                scale = max(scale_w, scale_h)  # Scale to fill

                # Resize image - use proper Pillow resampling method
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)

                # Use the correct resampling method for newer Pillow versions
                try:
                    # Try new Pillow API first
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                except AttributeError:
                    # Fallback for older Pillow versions
                    try:
                        img = img.resize((new_width, new_height), Image.LANCZOS)
                    except AttributeError:
                        # Final fallback
                        img = img.resize((new_width, new_height))

                # Center crop to exact dimensions
                left = (new_width - self.width) // 2
                top = (new_height - self.height) // 2
                img = img.crop((left, top, left + self.width, top + self.height))

                # Enhance image slightly
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(0.9)  # Slightly darker for text readability

                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.1)  # Slightly more contrast

                # Save processed image
                processed_path = Path(self.temp_dir) / f"processed_{int(time.time())}_{Path(image_path).name}"
                img.save(processed_path, 'JPEG', quality=85)

                logging.debug(f"Successfully processed image: {processed_path}")
                return str(processed_path)

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return None

    def _create_title_overlay(self, title: str, duration: float) -> Optional[TextClip]:
        """Create title text overlay - with ImageMagick fallback for systems without it."""
        try:
            # First, let's try without advanced fonts (simple approach)
            logging.info(f"Creating title overlay for: '{title[:30]}...'")

            # Clean and prepare title text
            clean_title = title.strip()
            if len(clean_title) > 40:
                # Split long titles into multiple lines
                words = clean_title.split()
                if len(words) > 4:
                    mid = len(words) // 2
                    clean_title = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])

            # Try simple TextClip first (most compatible)
            try:
                title_clip = TextClip(
                    clean_title,
                    fontsize=60,
                    color='white',
                    bg_color='black',
                    size=(self.width - 100, None)
                )

                # Position and set duration
                title_clip = title_clip.set_position(('center', 80))
                title_clip = title_clip.set_duration(duration)

                logging.info("✅ Created simple title overlay successfully")
                return title_clip

            except Exception as e:
                logging.warning(f"Simple TextClip failed: {e}")

                # If TextClip fails completely, create a colored rectangle as placeholder
                logging.info("Creating title placeholder (text overlay not available)")

                # Create a simple colored rectangle as a title placeholder
                from moviepy.editor import ColorClip
                title_bg = ColorClip(
                    size=(self.width, 120),
                    color=(0, 0, 0),  # Black background
                    duration=duration
                ).set_position((0, 60)).set_opacity(0.7)

                logging.info("Created title placeholder rectangle")
                return title_bg

        except Exception as e:
            logging.error(f"All title overlay methods failed: {e}")
            logging.info("Video will be created without title overlay")
            return None

    def _compose_final_video(self, background: VideoFileClip, title: Optional[TextClip],
                             audio: AudioFileClip) -> Optional[CompositeVideoClip]:
        """Compose all elements into final video with perfect timing synchronization."""
        try:
            audio_duration = audio.duration
            background_duration = background.duration

            logging.info(
                f"Composing final video - Audio: {audio_duration:.2f}s, Background: {background_duration:.2f}s")

            # Ensure background exactly matches audio duration
            if abs(background_duration - audio_duration) > 0.05:  # More than 50ms difference
                logging.warning(
                    f"Duration mismatch detected - adjusting background from {background_duration:.2f}s to {audio_duration:.2f}s")
                background = background.set_duration(audio_duration)

            # Start with background
            video_clips = [background]

            # Add title overlay if available (ensure it matches duration too)
            if title:
                # Ensure title duration matches audio
                title = title.set_duration(audio_duration)
                video_clips.append(title)
                logging.info("Title overlay added to video composition with matching duration")
            else:
                logging.warning("No title overlay available - video will have no title text")

            # Composite video with proper layering
            final_video = CompositeVideoClip(video_clips, size=(self.width, self.height))

            # Set audio and ensure exact duration synchronization
            final_video = final_video.set_audio(audio)
            final_video = final_video.set_duration(audio_duration)

            # Verify final timing
            final_duration = final_video.duration
            logging.info(
                f"Final video composed - Audio: {audio_duration:.2f}s, Video: {final_duration:.2f}s, Clips: {len(video_clips)}")

            if abs(final_duration - audio_duration) > 0.01:
                logging.warning(f"Final duration mismatch: Video={final_duration:.2f}s, Audio={audio_duration:.2f}s")
            else:
                logging.info("✅ Perfect audio/video synchronization achieved")

            return final_video

        except Exception as e:
            logging.error(f"Error composing final video: {e}")
            return None

    def _generate_output_filename(self, title: str, topic_index: int) -> str:
        """Generate output filename for the video."""
        # Create safe filename
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')[:30]

        timestamp = int(time.time())
        filename = f"shorts_{topic_index:03d}_{safe_title}_{timestamp}.mp4"

        return str(Path(self.output_dir) / filename)

    def _render_video(self, video: CompositeVideoClip, output_path: str) -> bool:
        """Render and save the final video."""
        try:
            logging.info(f"Rendering video to: {output_path}")

            # Render with optimized settings for YouTube Shorts
            video.write_videofile(
                output_path,
                fps=self.fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(Path(self.temp_dir) / 'temp_audio.m4a'),
                remove_temp=True,
                bitrate="2000k",  # Good quality for 1080p
                preset='medium',  # Balance between speed and compression
                verbose=False,
                logger=None  # Suppress moviepy logs
            )

            # Verify output file
            if Path(output_path).exists() and Path(output_path).stat().st_size > 1024:
                file_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
                logging.info(f"Video rendered successfully - Size: {file_size_mb:.1f}MB")
                return True
            else:
                logging.error("Video file was not created or is too small")
                return False

        except Exception as e:
            logging.error(f"Error rendering video: {e}")
            return False

    def _cleanup_clips(self, clips: List):
        """Clean up video clips to free memory."""
        for clip in clips:
            try:
                if clip and hasattr(clip, 'close'):
                    clip.close()
            except:
                pass

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary processed images."""
        try:
            temp_path = Path(self.temp_dir)
            if not temp_path.exists():
                return

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0

            # Clean processed images
            for pattern in ["processed_*.jpg", "processed_*.png"]:
                for file_path in temp_path.glob(pattern):
                    if current_time - file_path.stat().st_mtime > max_age_seconds:
                        file_path.unlink()
                        cleaned_count += 1

            if cleaned_count > 0:
                logging.info(f"Cleaned up {cleaned_count} temporary video files")

        except Exception as e:
            logging.warning(f"Error cleaning up temp video files: {e}")

    def get_video_info(self, video_path: str) -> Dict:
        """Get information about a created video."""
        try:
            if not Path(video_path).exists():
                return {"error": "Video file not found"}

            stat = Path(video_path).stat()

            # Try to get video duration
            try:
                with VideoFileClip(video_path) as video:
                    duration = video.duration
                    fps = video.fps
                    size = video.size
            except:
                duration = None
                fps = None
                size = None

            return {
                "file_size_mb": stat.st_size / (1024 * 1024),
                "created_time": stat.st_mtime,
                "duration": duration,
                "fps": fps,
                "dimensions": size,
                "path": video_path
            }

        except Exception as e:
            return {"error": str(e)}