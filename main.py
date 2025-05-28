# main.py
import os
import sys
import logging
import traceback
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config_manager import ConfigManager
from utils.topic_manager import TopicManager
from utils.tts_generator import TextToSpeechGenerator
from utils.pexels_fetcher import PexelsImageFetcher
from utils.video_composer import VideoComposer


class MainOrchestrator:
    """Main orchestrator for YouTube Shorts generation workflow."""

    def __init__(self, config_file: str = ".env"):
        """Initialize the orchestrator with all components."""
        self.config = None
        self.topic_manager = None
        self.tts_generator = None
        self.image_fetcher = None
        self.video_composer = None

        try:
            # Initialize configuration
            self.config = ConfigManager(config_file)

            # Validate configuration
            if not self.config.validate_configuration():
                raise Exception("Configuration validation failed")

            # Initialize components
            self._initialize_components()

            logging.info("MainOrchestrator initialized successfully")

        except Exception as e:
            logging.error(f"Failed to initialize MainOrchestrator: {e}")
            raise

    def _initialize_components(self):
        """Initialize all utility components."""
        try:
            self.topic_manager = TopicManager(self.config)
            self.tts_generator = TextToSpeechGenerator(self.config)
            self.image_fetcher = PexelsImageFetcher(self.config)
            self.video_composer = VideoComposer(self.config)

            # Test TTS engine
            if not self.tts_generator.test_tts_engine():
                logging.warning("TTS engine test failed, but continuing...")

            logging.info("All components initialized successfully")

        except Exception as e:
            logging.error(f"Error initializing components: {e}")
            raise

    def run_single_generation(self) -> Dict[str, Any]:
        """
        Run a single video generation cycle.

        Returns:
            Dictionary with generation results and status
        """
        result = {
            "success": False,
            "topic": None,
            "video_path": None,
            "audio_duration": None,
            "images_count": 0,
            "error": None,
            "stats": {}
        }

        try:
            logging.info("Starting video generation cycle")

            # Step 1: Get next topic
            topic = self.topic_manager.get_next_topic()
            if not topic:
                result["error"] = "No incomplete topics available"
                logging.warning("No incomplete topics found")
                return result

            result["topic"] = topic
            logging.info(f"Processing topic: '{topic['title']}'")

            # Validate topic content
            if not self.topic_manager.validate_topic_content(topic):
                result["error"] = "Topic content validation failed"
                return result

            # Step 2: Generate audio narration
            audio_success, audio_path, audio_duration = self.tts_generator.generate_audio_for_topic(topic)
            if not audio_success or not audio_path:
                result["error"] = "Failed to generate audio narration"
                return result

            result["audio_duration"] = audio_duration
            logging.info(f"Audio generated successfully - Duration: {audio_duration:.2f}s")

            # Step 3: Fetch background images
            images = self.image_fetcher.fetch_images_for_topic(
                topic['title'],
                self.config.get_video_settings()['images_per_video']
            )

            if not images:
                result["error"] = "Failed to fetch background images"
                return result

            # Validate images
            valid_images = self.image_fetcher.validate_downloaded_images(images)
            if len(valid_images) < 2:  # Need at least 2 images
                result["error"] = f"Not enough valid images ({len(valid_images)} found)"
                return result

            result["images_count"] = len(valid_images)
            logging.info(f"Fetched {len(valid_images)} valid background images")

            # Step 4: Create video
            video_success, video_path = self.video_composer.create_video(
                audio_path, valid_images, topic['title'], topic['index']
            )

            if not video_success or not video_path:
                result["error"] = "Failed to create video"
                return result

            result["video_path"] = video_path
            logging.info(f"Video created successfully: {video_path}")

            # Step 5: Mark topic as complete
            if self.topic_manager.mark_topic_complete(topic['index']):
                result["success"] = True
                logging.info(f"Topic '{topic['title']}' marked as complete")
            else:
                logging.warning("Failed to mark topic as complete")
                result["error"] = "Video created but failed to mark topic complete"

            # Get final stats
            result["stats"] = self._get_generation_stats()

            return result

        except Exception as e:
            logging.error(f"Error in generation cycle: {e}")
            logging.error(traceback.format_exc())
            result["error"] = str(e)
            return result

        finally:
            # Cleanup temporary files
            self._cleanup_temp_files()

    def _get_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about the current generation state."""
        try:
            completion_stats = self.topic_manager.get_completion_stats()
            api_usage = self.image_fetcher.get_api_usage_info()

            return {
                "topics": completion_stats,
                "api_usage": api_usage,
                "video_settings": self.config.get_video_settings(),
            }
        except Exception as e:
            logging.warning(f"Error getting stats: {e}")
            return {}

    def _cleanup_temp_files(self):
        """Clean up temporary files from all components."""
        try:
            self.tts_generator.cleanup_temp_files()
            self.image_fetcher.cleanup_old_images()
            self.video_composer.cleanup_temp_files()
            logging.debug("Temporary files cleaned up")
        except Exception as e:
            logging.warning(f"Error during cleanup: {e}")

    def run_continuous_mode(self, max_iterations: int = None):
        """
        Run in continuous mode for processing multiple topics.

        Args:
            max_iterations: Maximum number of topics to process (None for all)
        """
        iteration = 0

        while True:
            if max_iterations and iteration >= max_iterations:
                logging.info(f"Reached maximum iterations: {max_iterations}")
                break

            result = self.run_single_generation()

            if not result["success"]:
                if "No incomplete topics" in str(result.get("error", "")):
                    logging.info("All topics completed!")
                    break
                else:
                    logging.error(f"Generation failed: {result['error']}")
                    break

            iteration += 1
            logging.info(f"Completed iteration {iteration}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status and health check."""
        status = {
            "config_valid": False,
            "tts_working": False,
            "api_accessible": False,
            "output_dir_writable": False,
            "topics_available": False,
            "stats": {}
        }

        try:
            # Check configuration
            status["config_valid"] = self.config.validate_configuration()

            # Check TTS
            status["tts_working"] = self.tts_generator.test_tts_engine()

            # Check API (simple test)
            test_results = self.image_fetcher.search_images("technology", per_page=1)
            status["api_accessible"] = test_results is not None

            # Check output directory
            output_dir = Path(self.config.get_output_directory())
            status["output_dir_writable"] = output_dir.exists() and os.access(output_dir, os.W_OK)

            # Check topics
            topic = self.topic_manager.get_next_topic()
            status["topics_available"] = topic is not None

            # Get stats
            status["stats"] = self._get_generation_stats()

        except Exception as e:
            status["error"] = str(e)

        return status


def main():
    """Main entry point for the script."""
    import argparse

    parser = argparse.ArgumentParser(description="YouTube Shorts Generator")
    parser.add_argument("--mode", choices=["single", "continuous", "status"],
                        default="single", help="Operation mode")
    parser.add_argument("--max-iterations", type=int,
                        help="Maximum iterations for continuous mode")
    parser.add_argument("--config", default=".env",
                        help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")

    args = parser.parse_args()

    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Initialize orchestrator
        orchestrator = MainOrchestrator(args.config)

        if args.mode == "single":
            # Run single generation
            result = orchestrator.run_single_generation()

            if result["success"]:
                print(f"✅ Video generated successfully!")
                print(f"   Topic: {result['topic']['title']}")
                print(f"   Video: {result['video_path']}")
                print(f"   Duration: {result['audio_duration']:.1f}s")
                print(f"   Images: {result['images_count']}")

                # Print stats
                stats = result.get("stats", {})
                if "topics" in stats:
                    topics_stats = stats["topics"]
                    print(f"   Progress: {topics_stats['completed']}/{topics_stats['total']} topics completed")

                sys.exit(0)
            else:
                print(f"❌ Generation failed: {result['error']}")
                sys.exit(1)

        elif args.mode == "continuous":
            # Run continuous mode
            print("🔄 Running in continuous mode...")
            orchestrator.run_continuous_mode(args.max_iterations)
            print("✅ Continuous mode completed")

        elif args.mode == "status":
            # Check system status
            status = orchestrator.get_system_status()

            print("📊 System Status:")
            print(f"   Config valid: {'✅' if status['config_valid'] else '❌'}")
            print(f"   TTS working: {'✅' if status['tts_working'] else '❌'}")
            print(f"   API accessible: {'✅' if status['api_accessible'] else '❌'}")
            print(f"   Output writable: {'✅' if status['output_dir_writable'] else '❌'}")
            print(f"   Topics available: {'✅' if status['topics_available'] else '❌'}")

            if "stats" in status and "topics" in status["stats"]:
                topics = status["stats"]["topics"]
                print(
                    f"   Topics: {topics['completed']}/{topics['total']} completed ({topics['completion_percentage']}%)")

            # Exit with error code if any critical component is failing
            critical_checks = [status['config_valid'], status['tts_working'],
                               status['api_accessible'], status['output_dir_writable']]
            if not all(critical_checks):
                sys.exit(1)

    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        sys.exit(1)

    except Exception as e:
        logging.error(f"Fatal error: {e}")
        logging.error(traceback.format_exc())
        print(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

# Example usage commands:
# python main.py --mode single                    # Generate one video
# python main.py --mode continuous --max-iterations 5  # Generate 5 videos
# python main.py --mode status                    # Check system status
# python main.py --mode single --verbose          # Verbose logging