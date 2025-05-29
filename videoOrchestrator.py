# videoOrchestrator.py

import sys
import os
import logging
import traceback
import time
from pathlib import Path
from typing import Optional, Dict, Any

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from config.config_manager import ConfigManager
from components.topic_manager import TopicManager
from components.tts_generator import TextToSpeechGenerator
from components.pexels_fetcher import PexelsImageFetcher
from components.video_composer import VideoComposer


class VideoOrchestrator:
    """Main orchestrator for YouTube Shorts generation workflow."""

    def __init__(self, topic_data=None, pexels_api_key: str = None, config_file: str = ".env"):
        """
        Initialize the orchestrator with all components.

        Args:
            topic_data: Optional dict with topic data for direct generation
            config_file: Path to configuration file
        """
        self.config = None
        self.topic_manager = None
        self.tts_generator = None
        self.image_fetcher = None
        self.video_composer = None
        self.topic_data = topic_data
        self.start_time = None

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

    @classmethod
    def from_topic_data(cls, topic_data: Dict[str, Any], config_file: str = ".env"):
        """
        Create MainOrchestrator instance for direct topic generation.

        Args:
            topic_data: Dictionary containing 'title' and 'description'
            config_file: Path to configuration file

        Returns:
            MainOrchestrator instance ready for direct generation
        """
        return cls(topic_data=topic_data, config_file=config_file)

    def _initialize_components(self):
        """Initialize all utility components."""
        try:
            if not self.topic_data:  # Only initialize topic_manager if not using direct data
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

    def _log_step_time(self, step_name: str, start_time: float) -> float:
        """Log the time taken for a step and return current time."""
        current_time = time.time()
        elapsed = current_time - start_time
        logging.info(f"⏱️  {step_name}: {elapsed:.2f}s")
        return current_time

    def generate(self) -> Dict[str, Any]:
        """
        Generate video from the provided topic data.

        Returns:
            Dictionary with generation results and status
        """
        if not self.topic_data:
            raise ValueError("No topic data provided. Use run_single_generation() for file-based topics.")

        # Validate topic data
        if not isinstance(self.topic_data, dict):
            raise ValueError("Topic data must be a dictionary")

        required_fields = ["title", "description"]
        for field in required_fields:
            if field not in self.topic_data:
                raise ValueError(f"Topic data missing required field: {field}")

        # Create topic format compatible with existing workflow
        topic = {
            "index": -1,  # Not from file
            "title": self.topic_data["title"],
            "description": self.topic_data["description"],
            "complete": self.topic_data.get("complete", False)
        }

        return self._generate_video_from_topic(topic, mark_complete=False)

    def _generate_video_from_topic(self, topic: Dict[str, Any], mark_complete: bool = True) -> Dict[str, Any]:
        """
        Core video generation logic with timing tracking.

        Args:
            topic: Topic dictionary with title, description, index, complete
            mark_complete: Whether to mark topic as complete in file

        Returns:
            Dictionary with generation results and timing
        """
        result = {
            "success": False,
            "topic": topic,
            "video_path": None,
            "audio_duration": None,
            "images_count": 0,
            "error": None,
            "stats": {},
            "timing": {}
        }

        try:
            # Start total timing
            total_start_time = time.time()
            self.start_time = total_start_time
            logging.info(f"🚀 Starting video generation for: '{topic['title']}'")

            # Step 1: Validate topic content
            step_start = time.time()
            if self.topic_manager and not self.topic_manager.validate_topic_content(topic):
                result["error"] = "Topic content validation failed"
                return result
            result["timing"]["validation"] = self._log_step_time("Topic validation", step_start)

            # Step 2: Generate audio narration
            step_start = time.time()
            audio_success, audio_path, audio_duration = self.tts_generator.generate_audio_for_topic(topic)
            if not audio_success or not audio_path:
                result["error"] = "Failed to generate audio narration"
                return result

            result["audio_duration"] = audio_duration
            result["timing"]["audio_generation"] = self._log_step_time(f"Audio generation ({audio_duration:.2f}s)",
                                                                       step_start)

            # Step 3: Fetch background images
            step_start = time.time()
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
            result["timing"]["image_fetching"] = self._log_step_time(f"Image fetching ({len(valid_images)} images)",
                                                                     step_start)

            # Step 4: Create video
            step_start = time.time()
            video_success, video_path = self.video_composer.create_video(
                audio_path, valid_images, topic['title'], topic.get('index', 0)
            )

            if not video_success or not video_path:
                result["error"] = "Failed to create video"
                return result

            result["video_path"] = video_path
            result["timing"]["video_creation"] = self._log_step_time("Video creation", step_start)

            # Step 5: Mark topic as complete (only if from file)
            if mark_complete and self.topic_manager and topic.get('index', -1) >= 0:
                step_start = time.time()
                if self.topic_manager.mark_topic_complete(topic['index']):
                    logging.info(f"✅ Topic '{topic['title']}' marked as complete")
                else:
                    logging.warning("Failed to mark topic as complete")
                    result["error"] = "Video created but failed to mark topic complete"
                result["timing"]["file_update"] = self._log_step_time("File update", step_start)

            # Calculate total time
            total_time = time.time() - total_start_time
            result["timing"]["total"] = total_time

            # Success!
            result["success"] = True
            result["stats"] = self._get_generation_stats()

            # Log final summary
            logging.info("=" * 60)
            logging.info(f"🎉 VIDEO GENERATION COMPLETED SUCCESSFULLY!")
            logging.info(f"📹 Video: {video_path}")
            logging.info(
                f"📊 Duration: {audio_duration:.2f}s | Images: {len(valid_images)} | Total time: {total_time:.2f}s")
            logging.info("=" * 60)

            return result

        except Exception as e:
            total_time = time.time() - (self.start_time or time.time())
            result["timing"]["total"] = total_time
            result["error"] = str(e)

            logging.error("=" * 60)
            logging.error(f"💥 VIDEO GENERATION FAILED!")
            logging.error(f"❌ Error: {e}")
            logging.error(f"⏱️  Total time: {total_time:.2f}s")
            logging.error("=" * 60)
            logging.error(traceback.format_exc())

            return result

        finally:
            # Cleanup temporary files
            self._cleanup_temp_files()

    def run_single_generation(self) -> Dict[str, Any]:
        """
        Run a single video generation cycle from topics file.

        Returns:
            Dictionary with generation results and status
        """
        if not self.topic_manager:
            raise ValueError("Topic manager not initialized. Use generate() for direct topic data.")

        try:
            logging.info("🔍 Starting video generation cycle from topics file")

            # Get next topic
            topic = self.topic_manager.get_next_topic()
            if not topic:
                result = {
                    "success": False,
                    "topic": None,
                    "video_path": None,
                    "audio_duration": None,
                    "images_count": 0,
                    "error": "No incomplete topics available",
                    "stats": {},
                    "timing": {}
                }
                logging.warning("No incomplete topics found")
                return result

            logging.info(f"🎯 Selected topic from file: '{topic['title']}'")

            # Use the core generation logic
            return self._generate_video_from_topic(topic, mark_complete=True)

        except Exception as e:
            logging.error(f"Error in generation cycle: {e}")
            logging.error(traceback.format_exc())
            return {
                "success": False,
                "topic": None,
                "video_path": None,
                "audio_duration": None,
                "images_count": 0,
                "error": str(e),
                "stats": {},
                "timing": {}
            }

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
            "tts_method": None,
            "api_accessible": False,
            "output_dir_writable": False,
            "topics_available": False,
            "stats": {}
        }

        try:
            # Check configuration
            status["config_valid"] = self.config.validate_configuration()

            # Check TTS and get method info
            status["tts_working"] = self.tts_generator.test_tts_engine()
            status["tts_method"] = self.tts_generator.tts_method

            # Add TTS quality info
            if status["tts_method"] == "coqui_tts":
                status["tts_quality"] = "⭐⭐⭐⭐ Neural (Local)"
                if hasattr(self.tts_generator, 'coqui_model_name'):
                    status["tts_model"] = self.tts_generator.coqui_model_name
            elif status["tts_method"] == "elevenlabs":
                status["tts_quality"] = "⭐⭐⭐⭐⭐ Premium (API)"
            elif status["tts_method"] == "system_say":
                status["tts_quality"] = "⭐⭐⭐ Good (System)"
            elif status["tts_method"] == "espeak":
                status["tts_quality"] = "⭐⭐ Basic (Robotic)"
            else:
                status["tts_quality"] = "Unknown"

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
        orchestrator = VideoOrchestrator()

        if args.mode == "single":
            # Run single generation
            result = orchestrator.run_single_generation()

            if result["success"]:
                print(f"✅ Video generated successfully!")
                print(f"   📹 Topic: {result['topic']['title']}")
                print(f"   🎬 Video: {result['video_path']}")
                print(f"   ⏱️  Duration: {result['audio_duration']:.1f}s")
                print(f"   🖼️  Images: {result['images_count']}")

                # Display timing breakdown
                if "timing" in result and result["timing"]:
                    print(f"\n⏱️  Timing Breakdown:")
                    timing = result["timing"]
                    if "validation" in timing:
                        print(f"   📋 Validation: {timing['validation']:.2f}s")
                    if "audio_generation" in timing:
                        print(f"   🎤 Audio generation: {timing['audio_generation']:.2f}s")
                    if "image_fetching" in timing:
                        print(f"   🖼️  Image fetching: {timing['image_fetching']:.2f}s")
                    if "video_creation" in timing:
                        print(f"   🎬 Video creation: {timing['video_creation']:.2f}s")
                    if "file_update" in timing:
                        print(f"   📝 File update: {timing['file_update']:.2f}s")
                    print(f"   🎯 TOTAL TIME: {timing.get('total', 0):.2f}s")

                # Print stats
                stats = result.get("stats", {})
                if "topics" in stats:
                    topics_stats = stats["topics"]
                    print(f"\n📊 Progress: {topics_stats['completed']}/{topics_stats['total']} topics completed")

                sys.exit(0)
            else:
                print(f"❌ Generation failed: {result['error']}")

                # Show timing even for failures
                if "timing" in result and "total" in result["timing"]:
                    print(f"⏱️  Total time: {result['timing']['total']:.2f}s")

                sys.exit(1)

        elif args.mode == "continuous":
            # Run continuous mode
            print("🔄 Running in continuous mode...")

            iteration = 0
            total_start_time = time.time()
            successful_videos = []

            while True:
                if args.max_iterations and iteration >= args.max_iterations:
                    logging.info(f"Reached maximum iterations: {args.max_iterations}")
                    break

                result = orchestrator.run_single_generation()

                if not result["success"]:
                    if "No incomplete topics" in str(result.get("error", "")):
                        logging.info("All topics completed!")
                        break
                    else:
                        logging.error(f"Generation failed: {result['error']}")
                        break

                iteration += 1
                successful_videos.append(result)

                # Show progress
                timing = result.get("timing", {})
                total_time = timing.get("total", 0)
                print(f"✅ Video {iteration}: {result['topic']['title']} ({total_time:.1f}s)")

            # Final summary for continuous mode
            total_batch_time = time.time() - total_start_time
            print(f"\n📊 CONTINUOUS MODE SUMMARY:")
            print(f"   ✅ Videos generated: {len(successful_videos)}")
            print(f"   ⏱️  Total batch time: {total_batch_time:.1f}s")
            if successful_videos:
                avg_time = sum(v.get("timing", {}).get("total", 0) for v in successful_videos) / len(successful_videos)
                print(f"   📈 Average per video: {avg_time:.1f}s")

            print("✅ Continuous mode completed")

        elif args.mode == "status":
            # Check system status
            status = orchestrator.get_system_status()

            print("📊 System Status:")
            print(f"   Config valid: {'✅' if status['config_valid'] else '❌'}")

            # Enhanced TTS status display
            tts_status = '✅' if status['tts_working'] else '❌'
            tts_method = status.get('tts_method', 'unknown')
            tts_quality = status.get('tts_quality', '')
            print(f"   TTS working: {tts_status} ({tts_method}) {tts_quality}")

            if status.get('tts_model'):
                print(f"   TTS model: {status['tts_model']}")

            print(f"   API accessible: {'✅' if status['api_accessible'] else '❌'}")
            print(f"   Output writable: {'✅' if status['output_dir_writable'] else '❌'}")
            print(f"   Topics available: {'✅' if status['topics_available'] else '❌'}")

            if "stats" in status and "topics" in status["stats"]:
                topics = status["stats"]["topics"]
                print(
                    f"   Topics: {topics['completed']}/{topics['total']} completed ({topics['completion_percentage']}%)")

            # Show TTS recommendation if not using best option
            if status.get('tts_method') == 'espeak':
                print("\n💡 Recommendation: Install Coqui TTS for much better voice quality:")
                print("   pip install coqui-tts")
            elif status.get('tts_method') not in ['coqui_tts', 'elevenlabs']:
                print("\n💡 Consider upgrading TTS for better voice quality:")
                print("   Coqui TTS (local): pip install coqui-tts")
                print("   ElevenLabs (cloud): pip install elevenlabs")

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
# python videoOrchestrator.py --mode single                    # Generate one video
# python videoOrchestrator.py --mode continuous --max-iterations 5  # Generate 5 videos
# python videoOrchestrator.py --mode status                    # Check system status
# python videoOrchestrator.py --mode single --verbose          # Verbose logging