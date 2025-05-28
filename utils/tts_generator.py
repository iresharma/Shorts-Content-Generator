# utils/tts_generator.py

import os
import logging
import pyttsx3
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple
from pydub import AudioSegment
import time


class TextToSpeechGenerator:
    """Handles text-to-speech conversion for video narration."""

    def __init__(self, config_manager):
        """Initialize TTS generator with configuration."""
        self.config = config_manager
        self.tts_settings = self.config.get_tts_settings()
        self.intro_template = self.config.get_intro_template()
        self.temp_dir = self.config.get_temp_directory()

        # Initialize TTS engine
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self):
        """Initialize the pyttsx3 TTS engine with settings."""
        try:
            self.engine = pyttsx3.init()

            # Set rate (words per minute)
            self.engine.setProperty('rate', self.tts_settings['rate'])

            # Set volume (0.0 to 1.0)
            self.engine.setProperty('volume', self.tts_settings['volume'])

            # Set voice if specified
            if self.tts_settings.get('voice_id'):
                voices = self.engine.getProperty('voices')
                for voice in voices:
                    if self.tts_settings['voice_id'] in voice.id:
                        self.engine.setProperty('voice', voice.id)
                        break

            logging.info("TTS engine initialized successfully")
            self._log_available_voices()

        except Exception as e:
            logging.error(f"Failed to initialize TTS engine: {e}")
            raise

    def _log_available_voices(self):
        """Log available TTS voices for debugging."""
        try:
            voices = self.engine.getProperty('voices')
            logging.info(f"Available TTS voices: {len(voices)}")
            for i, voice in enumerate(voices[:3]):  # Log first 3 voices
                logging.debug(f"Voice {i}: {voice.id} - {voice.name}")
        except Exception as e:
            logging.warning(f"Could not retrieve voice information: {e}")

    def generate_narration_text(self, title: str, description: str) -> str:
        """Generate the complete narration text with intro and description."""
        try:
            # Generate intro statement
            intro = self.intro_template.format(title=title)

            # Add pause between intro and description
            pause = "... "

            # Combine intro and description
            full_text = f"{intro}{pause}{description}"

            logging.info(f"Generated narration text ({len(full_text)} characters)")
            return full_text

        except Exception as e:
            logging.error(f"Error generating narration text: {e}")
            return description  # Fallback to just description

    def text_to_speech(self, text: str, output_filename: str) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Convert text to speech and save as audio file.

        Returns:
            Tuple of (success: bool, file_path: Optional[str], duration: Optional[float])
        """
        try:
            # Create output path
            output_path = Path(self.temp_dir) / f"{output_filename}.wav"

            # Ensure temp directory exists
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)

            # Generate speech
            logging.info(f"Generating speech for text ({len(text)} characters)...")

            # Save to file
            self.engine.save_to_file(text, str(output_path))
            self.engine.runAndWait()

            # Verify file was created
            if not output_path.exists():
                raise Exception("Audio file was not created")

            # Convert to MP3 and get duration
            mp3_path, duration = self._convert_to_mp3(output_path, output_filename)

            # Clean up WAV file
            try:
                output_path.unlink()
            except:
                pass  # Ignore cleanup errors

            logging.info(f"TTS generation successful: {mp3_path} ({duration:.2f}s)")
            return True, mp3_path, duration

        except Exception as e:
            logging.error(f"TTS generation failed: {e}")
            return False, None, None

    def _convert_to_mp3(self, wav_path: Path, base_filename: str) -> Tuple[str, float]:
        """Convert WAV to MP3 and return path and duration."""
        try:
            # Load WAV file
            audio = AudioSegment.from_wav(str(wav_path))

            # Get duration in seconds
            duration = len(audio) / 1000.0

            # Export as MP3
            mp3_path = Path(self.temp_dir) / f"{base_filename}.mp3"
            audio.export(str(mp3_path), format="mp3", bitrate="192k")

            return str(mp3_path), duration

        except Exception as e:
            logging.error(f"Error converting to MP3: {e}")
            raise

    def generate_audio_for_topic(self, topic_data: Dict) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Generate audio file for a complete topic.

        Args:
            topic_data: Dictionary containing title and description

        Returns:
            Tuple of (success: bool, file_path: Optional[str], duration: Optional[float])
        """
        try:
            title = topic_data.get('title', '')
            description = topic_data.get('description', '')

            if not title or not description:
                logging.error("Topic missing title or description")
                return False, None, None

            # Generate narration text
            narration_text = self.generate_narration_text(title, description)

            # Create safe filename
            safe_title = self._create_safe_filename(title)
            timestamp = int(time.time())
            filename = f"{safe_title}_{timestamp}"

            # Generate audio
            success, audio_path, duration = self.text_to_speech(narration_text, filename)

            if success and duration:
                logging.info(f"Audio generated successfully for '{title}' - Duration: {duration:.2f}s")

            return success, audio_path, duration

        except Exception as e:
            logging.error(f"Error generating audio for topic: {e}")
            return False, None, None

    def _create_safe_filename(self, title: str) -> str:
        """Create a safe filename from topic title."""
        # Remove special characters and limit length
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_')
        return safe_title[:30]  # Limit length

    def estimate_speech_duration(self, text: str) -> float:
        """Estimate speech duration without generating audio."""
        try:
            # Average speaking rate is about 150-200 words per minute
            # We'll use the configured rate
            words = len(text.split())
            rate = self.tts_settings['rate']  # words per minute
            duration = (words / rate) * 60  # convert to seconds

            # Add small buffer for pauses and processing
            duration += 2.0

            return duration

        except Exception as e:
            logging.warning(f"Could not estimate speech duration: {e}")
            return 30.0  # Default fallback

    def test_tts_engine(self) -> bool:
        """Test the TTS engine with a short phrase."""
        try:
            test_text = "Testing text to speech engine."
            test_filename = "tts_test"

            success, file_path, duration = self.text_to_speech(test_text, test_filename)

            if success and file_path:
                # Clean up test file
                try:
                    Path(file_path).unlink()
                except:
                    pass
                logging.info("TTS engine test successful")
                return True
            else:
                logging.error("TTS engine test failed")
                return False

        except Exception as e:
            logging.error(f"TTS engine test error: {e}")
            return False

    def cleanup_temp_files(self, max_age_hours: int = 24):
        """Clean up temporary audio files older than specified hours."""
        try:
            temp_path = Path(self.temp_dir)
            if not temp_path.exists():
                return

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600

            for file_path in temp_path.glob("*.mp3"):
                if current_time - file_path.stat().st_mtime > max_age_seconds:
                    file_path.unlink()
                    logging.debug(f"Cleaned up old temp file: {file_path}")

        except Exception as e:
            logging.warning(f"Error cleaning up temp files: {e}")

    def __del__(self):
        """Cleanup TTS engine on destruction."""
        try:
            if self.engine:
                self.engine.stop()
        except:
            pass