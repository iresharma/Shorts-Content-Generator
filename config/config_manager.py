# config/config_manager.py

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Dict, Any


class ConfigManager:
    """Manages configuration settings, API keys, and file paths for the YouTube Shorts generator."""

    def __init__(self, env_file: str = ".env", pexels_api_key: str = None):
        """Initialize configuration manager."""
        self.env_file = env_file
        self.pexels_api_key = pexels_api_key
        self._load_environment()
        self._setup_directories()
        self._setup_logging()

    def _load_environment(self):
        """Load environment variables from .env file."""
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            logging.info(f"Loaded environment variables from {self.env_file}")
        else:
            logging.warning(f"Environment file {self.env_file} not found. Using defaults.")

    def _setup_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.get_output_directory(),
            self.get_temp_directory(),
            Path(self.get_topics_file()).parent,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('youtube_shorts_generator.log'),
                logging.StreamHandler()
            ]
        )

    # API Configuration
    def get_pexels_api_key(self) -> str:
        """Get Pexels API key from environment."""
        if self.pexels_api_key is not None:
            print("+" * 60)
            print(self.pexels_api_key)
            print("+" * 60)
            return self.pexels_api_key
        api_key = os.getenv("PEXELS_API_KEY")
        if not api_key:
            raise ValueError("PEXELS_API_KEY not found in environment variables")
        return api_key

    # File Paths
    def get_topics_file(self) -> str:
        """Get path to topics JSON file."""
        return os.getenv("TOPICS_FILE", "./data/topics.json")

    def get_processed_topics_file(self) -> str:
        """Get path to processed topics backup file."""
        return os.getenv("PROCESSED_TOPICS_FILE", "./data/processed_topics.json")

    def get_output_directory(self) -> str:
        """Get output directory for generated videos."""
        return os.getenv("OUTPUT_DIRECTORY", "./output/videos")

    def get_temp_directory(self) -> str:
        """Get temporary directory for intermediate files."""
        return os.getenv("TEMP_DIRECTORY", "./temp")

    # TTS Configuration
    def get_tts_settings(self) -> Dict[str, Any]:
        """Get text-to-speech configuration."""
        return {
            "rate": int(os.getenv("TTS_RATE", "150")),
            "volume": float(os.getenv("TTS_VOLUME", "0.9")),
            "voice_id": os.getenv("TTS_VOICE_ID", None),  # None for default voice
        }

    # Video Configuration
    def get_video_settings(self) -> Dict[str, Any]:
        """Get video generation configuration."""
        return {
            "width": int(os.getenv("VIDEO_WIDTH", "1080")),
            "height": int(os.getenv("VIDEO_HEIGHT", "1920")),
            "fps": int(os.getenv("VIDEO_FPS", "30")),
            "min_duration": int(os.getenv("MIN_DURATION", "30")),
            "images_per_video": int(os.getenv("IMAGES_PER_VIDEO", "8")),
            "image_duration": float(os.getenv("IMAGE_DURATION", "4.0")),  # seconds per image
        }

    # Pexels Configuration
    def get_pexels_settings(self) -> Dict[str, Any]:
        """Get Pexels API configuration."""
        return {
            "images_per_request": int(os.getenv("PEXELS_IMAGES_PER_REQUEST", "10")),
            "image_size": os.getenv("PEXELS_IMAGE_SIZE", "large"),  # original, large, medium, small
            "search_tags": os.getenv("PEXELS_SEARCH_TAGS", "dark,computer,coding,technology,abstract").split(","),
        }

    # Intro Configuration
    def get_intro_template(self) -> str:
        """Get intro statement template."""
        return os.getenv("INTRO_TEMPLATE", "Today we will talk about {title}.")

    def validate_configuration(self) -> bool:
        """Validate that all required configuration is present."""
        try:
            # Check required API keys
            self.get_pexels_api_key()

            # Check required directories exist or can be created
            self._setup_directories()

            # Validate video settings
            video_settings = self.get_video_settings()
            if video_settings["width"] <= 0 or video_settings["height"] <= 0:
                raise ValueError("Invalid video dimensions")

            if video_settings["min_duration"] <= 0:
                raise ValueError("Invalid minimum duration")

            logging.info("Configuration validation successful")
            return True

        except Exception as e:
            logging.error(f"Configuration validation failed: {e}")
            return False

    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information for debugging."""
        return {
            "topics_file": self.get_topics_file(),
            "output_directory": self.get_output_directory(),
            "temp_directory": self.get_temp_directory(),
            "video_settings": self.get_video_settings(),
            "tts_settings": self.get_tts_settings(),
            "pexels_settings": self.get_pexels_settings(),
        }