# utils/video_composer.py

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
        Create a complete video from audio, images, and title.

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
            logging.info(f"Audio duration: {audio_duration:.2f}s")

            # Ensure minimum duration
            target_duration = max(audio_duration, self.min_duration)

            # Validate and prepare images
            valid_images = self._validate_images(image_paths)
            if not valid_images:
                logging.error("No valid images available for video")
                return False, None

            # Create background video from images
            background_video = self._create_background_slideshow(valid_images, target_duration)
            if not background_video:
                return False, None

            # Create title overlay
            title_overlay = self._create_title_overlay(title, target_duration)

            # Combine all elements
            final_video = self._compose_final_video(background_video, title_overlay, audio_clip)
            if not final_video:
                return False, None

            # Generate output filename
            output_path = self._generate_output_filename(title, topic_index)

            # Render and save video
            success = self._render_video(final_video, output_path)

            # Cleanup
            self._cleanup_clips([audio_clip, background_video, title_overlay, final_video])

            if success:
                logging.info(f"Video created successfully: {output_path}")
                return True, output_path
            else:
                return False, None

        except Exception as e:
            logging.error(f"Error creating video: {e}")
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
        """Validate image files and return list of valid ones."""
        valid_images = []

        for image_path in image_paths:
            try:
                if not Path(image_path).exists():
                    logging.warning(f"Image file not found: {image_path}")
                    continue

                # Try to open with PIL to validate
                with Image.open(image_path) as img:
                    if img.size[0] > 100 and img.size[1] > 100:  # Minimum size check
                        valid_images.append(image_path)
                    else:
                        logging.warning(f"Image too small: {image_path}")

            except Exception as e:
                logging.warning(f"Invalid image {image_path}: {e}")

        logging.info(f"Validated {len(valid_images)} out of {len(image_paths)} images")
        return valid_images

    def _create_background_slideshow(self, image_paths: List[str], duration: float) -> Optional[VideoFileClip]:
        """Create a slideshow video from background images that exactly matches audio duration."""
        try:
            if not image_paths:
                return None

            # Ensure we have enough images by cycling if needed
            min_images_needed = max(3, int(duration / 3))  # At least 3 seconds per image minimum

            # Cycle through images if we don't have enough
            while len(image_paths) < min_images_needed:
                image_paths.extend(image_paths[:min(len(image_paths), min_images_needed - len(image_paths))])

            # Calculate optimal timing
            num_images = len(image_paths)
            base_time_per_image = duration / num_images

            # Ensure minimum 2 seconds per image for readability
            if base_time_per_image < 2.0:
                # Use fewer images if duration is short
                num_images = max(2, int(duration / 2.0))
                image_paths = image_paths[:num_images]
                base_time_per_image = duration / num_images

            logging.info(
                f"Creating slideshow: {num_images} images, {base_time_per_image:.2f}s each, total: {duration:.2f}s")

            video_clips = []
            cumulative_time = 0.0

            for i, image_path in enumerate(image_paths):
                try:
                    # Calculate exact duration for this clip
                    if i == len(image_paths) - 1:
                        # Last image gets remaining time to ensure exact total duration
                        clip_duration = duration - cumulative_time
                    else:
                        clip_duration = base_time_per_image

                    # Process image
                    processed_image = self._process_image_for_video(image_path)
                    if not processed_image:
                        continue

                    # Create image clip with exact duration
                    img_clip = ImageClip(processed_image, duration=clip_duration)
                    img_clip = img_clip.set_fps(self.fps)

                    # Add subtle effects for visual interest
                    if i % 3 == 0:
                        # Slow zoom in
                        img_clip = img_clip.resize(lambda t: 1 + 0.03 * t / clip_duration)
                    elif i % 3 == 1:
                        # Slow zoom out
                        img_clip = img_clip.resize(lambda t: 1.03 - 0.03 * t / clip_duration)
                    # Every 3rd image stays static

                    video_clips.append(img_clip)
                    cumulative_time += clip_duration

                    logging.debug(f"Image {i + 1}: {clip_duration:.2f}s (cumulative: {cumulative_time:.2f}s)")

                except Exception as e:
                    logging.warning(f"Error processing image {image_path}: {e}")
                    continue

            if not video_clips:
                logging.error("No valid video clips created from images")
                return None

            # Concatenate all clips with crossfade transitions
            if len(video_clips) > 1:
                # Add crossfade transitions between clips
                transition_duration = 0.5  # Half second crossfade

                for i in range(len(video_clips) - 1):
                    # Adjust clip durations to account for transitions
                    if i == 0:
                        video_clips[i] = video_clips[i].fadeout(transition_duration)
                    else:
                        video_clips[i] = video_clips[i].fadein(transition_duration).fadeout(transition_duration)

                # Last clip only fades in
                video_clips[-1] = video_clips[-1].fadein(transition_duration)

            # Concatenate with smooth transitions
            slideshow = concatenate_videoclips(video_clips, method="compose")

            # Ensure exact duration match
            slideshow = slideshow.set_duration(duration)

            logging.info(f"Slideshow created successfully - Exact duration: {duration:.2f}s")
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

                # Resize image
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

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

                return str(processed_path)

        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")
            return None

    def _create_title_overlay(self, title: str, duration: float) -> Optional[TextClip]:
        """Create title text overlay with enhanced styling and visibility."""
        try:
            # Clean and prepare title text
            clean_title = title.strip()
            if len(clean_title) > 40:
                # Split long titles into multiple lines
                words = clean_title.split()
                if len(words) > 4:
                    mid = len(words) // 2
                    clean_title = ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])

            # Dynamic font size based on title length and screen width
            base_font_size = 70
            if len(clean_title) > 30:
                font_size = max(50, base_font_size - len(clean_title))
            else:
                font_size = base_font_size

            # Try different font options in order of preference
            font_options = [
                'Arial-Bold',
                'Helvetica-Bold',
                'DejaVu-Sans-Bold',
                'Liberation-Sans-Bold',
                None  # System default
            ]

            title_clip = None
            for font in font_options:
                try:
                    # Create text with enhanced styling
                    title_clip = TextClip(
                        clean_title,
                        fontsize=font_size,
                        color='white',
                        font=font,
                        stroke_color='black',
                        stroke_width=4,
                        method='caption',
                        size=(self.width - 120, None),  # Leave margins on sides
                        align='center',
                        interline=-5  # Tighter line spacing for multi-line
                    )
                    break  # Success, exit loop

                except Exception as e:
                    logging.debug(f"Font {font} failed: {e}")
                    continue

            if not title_clip:
                # Final fallback - simple text without advanced features
                title_clip = TextClip(
                    clean_title,
                    fontsize=font_size,
                    color='white'
                )

            # Position title at the top with padding
            title_height = title_clip.h if hasattr(title_clip, 'h') else 100
            y_position = 60  # Top padding

            # Ensure title doesn't go off screen
            if y_position + title_height > self.height * 0.3:
                y_position = 30
                # Make font smaller if still too big
                if title_height > self.height * 0.25:
                    font_size = int(font_size * 0.7)
                    title_clip = TextClip(
                        clean_title,
                        fontsize=font_size,
                        color='white',
                        stroke_color='black',
                        stroke_width=3
                    )

            # Set position and duration
            title_clip = title_clip.set_position(('center', y_position))
            title_clip = title_clip.set_duration(duration)

            # Enhanced visibility effects
            # Add semi-transparent background box for better readability
            from moviepy.editor import ColorClip
            bg_height = title_clip.h + 40 if hasattr(title_clip, 'h') else 120
            bg_clip = ColorClip(
                size=(self.width, bg_height),
                color=(0, 0, 0),  # Black background
                duration=duration
            ).set_opacity(0.6).set_position((0, y_position - 20))

            # Combine background and text
            title_with_bg = CompositeVideoClip([bg_clip, title_clip])

            # Add fade effects for smooth appearance
            title_with_bg = title_with_bg.fadeout(1.0).fadein(1.0)

            logging.info(f"Created enhanced title overlay: '{clean_title[:30]}...' at position (center, {y_position})")
            return title_with_bg

        except Exception as e:
            logging.error(f"Error creating title overlay: {e}")
            # Emergency fallback
            try:
                simple_title = TextClip(
                    title[:30],  # Truncate if too long
                    fontsize=50,
                    color='white'
                ).set_position(('center', 50)).set_duration(duration)

                logging.info("Created simple fallback title overlay")
                return simple_title
            except Exception as e2:
                logging.error(f"Even simple title overlay failed: {e2}")
                return None

    def _compose_final_video(self, background: VideoFileClip, title: Optional[TextClip],
                             audio: AudioFileClip) -> Optional[CompositeVideoClip]:
        """Compose all elements into final video with proper layering."""
        try:
            # Start with background
            video_clips = [background]

            # Add title overlay if available (ensure it's on top)
            if title:
                video_clips.append(title)
                logging.info("Title overlay added to video composition")
            else:
                logging.warning("No title overlay available - video will have no title text")

            # Composite video with proper layering
            final_video = CompositeVideoClip(video_clips, size=(self.width, self.height))

            # Set audio
            final_video = final_video.set_audio(audio)

            # Ensure duration matches audio exactly
            final_video = final_video.set_duration(audio.duration)

            logging.info(
                f"Final video composed successfully - Duration: {audio.duration:.2f}s, Clips: {len(video_clips)}")
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