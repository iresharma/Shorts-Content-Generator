# utils/topic_manager.py

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime


class TopicManager:
    """Manages topic selection and completion tracking for YouTube Shorts generation."""

    def __init__(self, config_manager):
        """Initialize topic manager with configuration."""
        self.config = config_manager
        self.topics_file = self.config.get_topics_file()
        self.processed_file = self.config.get_processed_topics_file()
        self.topics_data = []
        self._load_topics()

    def _load_topics(self):
        """Load topics from JSON file."""
        try:
            if not Path(self.topics_file).exists():
                logging.error(f"Topics file not found: {self.topics_file}")
                raise FileNotFoundError(f"Topics file not found: {self.topics_file}")

            with open(self.topics_file, 'r', encoding='utf-8') as file:
                self.topics_data = json.load(file)

            logging.info(f"Loaded {len(self.topics_data)} topics from {self.topics_file}")

            # Validate topic structure
            self._validate_topics()

        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in topics file: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading topics: {e}")
            raise

    def _validate_topics(self):
        """Validate that topics have required fields."""
        required_fields = ["title", "description", "complete"]

        for i, topic in enumerate(self.topics_data):
            if not isinstance(topic, dict):
                raise ValueError(f"Topic at index {i} is not a dictionary")

            for field in required_fields:
                if field not in topic:
                    raise ValueError(f"Topic at index {i} missing required field: {field}")

            # Ensure complete is boolean
            if not isinstance(topic["complete"], bool):
                topic["complete"] = bool(topic["complete"])

        logging.info("Topic validation successful")

    def get_next_topic(self) -> Optional[Dict[str, Any]]:
        """Get the next incomplete topic in sequential order."""
        for i, topic in enumerate(self.topics_data):
            if not topic["complete"]:
                logging.info(f"Selected topic: '{topic['title']}'")
                return {
                    "index": i,
                    "title": topic["title"],
                    "description": topic["description"],
                    "complete": topic["complete"]
                }

        logging.warning("No incomplete topics found")
        return None

    def mark_topic_complete(self, topic_index: int) -> bool:
        """Mark a topic as complete and save to file."""
        try:
            if topic_index < 0 or topic_index >= len(self.topics_data):
                logging.error(f"Invalid topic index: {topic_index}")
                return False

            # Create backup before modifying
            self._create_backup()

            # Mark as complete with timestamp
            self.topics_data[topic_index]["complete"] = True
            self.topics_data[topic_index]["completed_at"] = datetime.now().isoformat()

            # Save updated topics
            self._save_topics()

            logging.info(f"Marked topic '{self.topics_data[topic_index]['title']}' as complete")
            return True

        except Exception as e:
            logging.error(f"Error marking topic as complete: {e}")
            return False

    def _create_backup(self):
        """Create backup of current topics file."""
        try:
            if Path(self.topics_file).exists():
                shutil.copy2(self.topics_file, self.processed_file)
                logging.debug("Created backup of topics file")
        except Exception as e:
            logging.warning(f"Could not create backup: {e}")

    def _save_topics(self):
        """Save topics data back to JSON file."""
        try:
            with open(self.topics_file, 'w', encoding='utf-8') as file:
                json.dump(self.topics_data, file, indent=2, ensure_ascii=False)
            logging.debug("Topics file saved successfully")
        except Exception as e:
            logging.error(f"Error saving topics file: {e}")
            raise

    def get_completion_stats(self) -> Dict[str, int]:
        """Get statistics about topic completion."""
        total = len(self.topics_data)
        completed = sum(1 for topic in self.topics_data if topic["complete"])
        remaining = total - completed

        return {
            "total": total,
            "completed": completed,
            "remaining": remaining,
            "completion_percentage": round((completed / total) * 100, 2) if total > 0 else 0
        }

    def reset_all_topics(self) -> bool:
        """Reset all topics to incomplete (use with caution)."""
        try:
            self._create_backup()

            for topic in self.topics_data:
                topic["complete"] = False
                if "completed_at" in topic:
                    del topic["completed_at"]

            self._save_topics()
            logging.info("All topics reset to incomplete")
            return True

        except Exception as e:
            logging.error(f"Error resetting topics: {e}")
            return False

    def add_topic(self, title: str, description: str) -> bool:
        """Add a new topic to the list."""
        try:
            new_topic = {
                "title": title.strip(),
                "description": description.strip(),
                "complete": False,
                "added_at": datetime.now().isoformat()
            }

            self._create_backup()
            self.topics_data.append(new_topic)
            self._save_topics()

            logging.info(f"Added new topic: '{title}'")
            return True

        except Exception as e:
            logging.error(f"Error adding topic: {e}")
            return False

    def get_all_topics(self) -> List[Dict[str, Any]]:
        """Get all topics with their current status."""
        return self.topics_data.copy()

    def get_incomplete_topics(self) -> List[Dict[str, Any]]:
        """Get all incomplete topics."""
        return [topic for topic in self.topics_data if not topic["complete"]]

    def validate_topic_content(self, topic: Dict[str, Any]) -> bool:
        """Validate that a topic has sufficient content for video generation."""
        try:
            title = topic.get("title", "").strip()
            description = topic.get("description", "").strip()

            if len(title) < 3:
                logging.warning(f"Topic title too short: '{title}'")
                return False

            if len(description) < 20:
                logging.warning(f"Topic description too short for topic '{title}'")
                return False

            # Check for reasonable length (not too long for TTS)
            if len(description) > 1000:
                logging.warning(f"Topic description might be too long for topic '{title}'")
                # Don't return False, just warn

            return True

        except Exception as e:
            logging.error(f"Error validating topic content: {e}")
            return False