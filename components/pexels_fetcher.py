# components/pexels_fetcher.py

import os
import requests
import logging
import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from urllib.parse import urlparse
import random


class PexelsImageFetcher:
    """Handles fetching background images from Pexels API for video generation."""

    def __init__(self, config_manager):
        """Initialize Pexels fetcher with configuration."""
        self.config = config_manager
        self.api_key = self.config.get_pexels_api_key()
        self.pexels_settings = self.config.get_pexels_settings()
        self.temp_dir = self.config.get_temp_directory()

        # API endpoints
        self.base_url = "https://api.pexels.com/v1"
        self.search_endpoint = f"{self.base_url}/search"

        # Headers for API requests
        self.headers = {
            "Authorization": self.api_key,
            "User-Agent": "YouTube-Shorts-Generator/1.0"
        }

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # Minimum seconds between requests

        # Cache for recent searches to avoid duplicate API calls
        self.search_cache = {}
        self.cache_max_age = 3600  # 1 hour cache

        logging.info("Pexels image fetcher initialized")

    def _wait_for_rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _is_cache_valid(self, search_term: str) -> bool:
        """Check if cached search results are still valid."""
        if search_term not in self.search_cache:
            return False

        cache_time = self.search_cache[search_term].get('timestamp', 0)
        return (time.time() - cache_time) < self.cache_max_age

    def search_images(self, query: str, per_page: int = None) -> Optional[Dict]:
        """
        Search for images on Pexels.

        Args:
            query: Search query string
            per_page: Number of images to fetch (default from config)

        Returns:
            Dictionary containing search results or None if failed
        """
        try:
            if per_page is None:
                per_page = self.pexels_settings['images_per_request']

            # Check cache first
            if self._is_cache_valid(query):
                logging.info(f"Using cached results for query: {query}")
                return self.search_cache[query]['data']

            # Prepare search parameters
            params = {
                'query': query,
                'per_page': min(per_page, 80),  # Pexels max is 80
                'page': 1,
                'size': self.pexels_settings['image_size']
            }

            # Wait for rate limit
            self._wait_for_rate_limit()

            # Make API request
            logging.info(f"Searching Pexels for: '{query}' ({per_page} images)")
            response = requests.get(self.search_endpoint, headers=self.headers, params=params)

            if response.status_code == 200:
                data = response.json()

                # Cache the results
                self.search_cache[query] = {
                    'data': data,
                    'timestamp': time.time()
                }

                logging.info(f"Found {len(data.get('photos', []))} images for query: {query}")
                return data

            elif response.status_code == 429:
                logging.warning("Pexels API rate limit exceeded")
                return None

            else:
                logging.error(f"Pexels API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logging.error(f"Error searching Pexels: {e}")
            return None

    def download_image(self, image_url: str, filename: str) -> Optional[str]:
        """
        Download an image from URL to local file.

        Args:
            image_url: URL of the image to download
            filename: Local filename to save as

        Returns:
            Local file path if successful, None if failed
        """
        try:
            # Create full path
            file_path = Path(self.temp_dir) / filename

            # Ensure directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download image
            response = requests.get(image_url, stream=True, timeout=30)
            response.raise_for_status()

            # Save to file
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logging.debug(f"Downloaded image: {filename}")
            return str(file_path)

        except Exception as e:
            logging.error(f"Error downloading image {filename}: {e}")
            return None

    def fetch_images_for_topic(self, topic_title: str, num_images: int = None) -> List[str]:
        """
        Fetch background images suitable for a topic.

        Args:
            topic_title: Title of the topic for contextual search
            num_images: Number of images to fetch (default from config)

        Returns:
            List of local file paths to downloaded images
        """
        try:
            if num_images is None:
                num_images = self.pexels_settings['images_per_video']

            # Generate search queries
            search_queries = self._generate_search_queries(topic_title)

            downloaded_images = []

            for query in search_queries:
                if len(downloaded_images) >= num_images:
                    break

                # Search for images
                search_results = self.search_images(query, per_page=10)

                if not search_results or 'photos' not in search_results:
                    continue

                # Download images from this search
                images_needed = num_images - len(downloaded_images)
                photos = search_results['photos'][:images_needed]

                for i, photo in enumerate(photos):
                    try:
                        # Get image URL (prefer large size)
                        image_url = self._get_best_image_url(photo)
                        if not image_url:
                            continue

                        # Generate filename
                        timestamp = int(time.time())
                        safe_query = "".join(c for c in query if c.isalnum())[:20]
                        filename = f"{safe_query}_{timestamp}_{i}.jpg"

                        # Download image
                        local_path = self.download_image(image_url, filename)
                        if local_path:
                            downloaded_images.append(local_path)

                        # Small delay between downloads
                        time.sleep(0.2)

                    except Exception as e:
                        logging.warning(f"Error processing image {i}: {e}")
                        continue

            logging.info(f"Successfully downloaded {len(downloaded_images)} images for topic: {topic_title}")
            return downloaded_images

        except Exception as e:
            logging.error(f"Error fetching images for topic '{topic_title}': {e}")
            return []

    def _generate_search_queries(self, topic_title: str) -> List[str]:
        """Generate relevant search queries based on topic and default tags."""
        queries = []

        # Default tech/coding related queries
        default_tags = self.pexels_settings['search_tags']

        # Add some basic tech queries
        tech_queries = [
            "technology abstract",
            "computer programming",
            "digital technology",
            "coding screen",
            "dark technology"
        ]

        # Combine default tags for variety
        combined_queries = [
            f"{tag1} {tag2}" for tag1 in default_tags[:3] for tag2 in default_tags[3:6]
        ]

        # Add all queries
        queries.extend(default_tags)
        queries.extend(tech_queries)
        queries.extend(combined_queries[:3])  # Limit combined queries

        # Shuffle for variety
        random.shuffle(queries)

        return queries[:5]  # Limit to 5 different searches to manage API usage

    def _get_best_image_url(self, photo: Dict) -> Optional[str]:
        """Get the best quality image URL from photo data."""
        try:
            src = photo.get('src', {})

            # Preferred order: original, large, medium, small
            for size in ['original', 'large', 'medium', 'small']:
                if size in src and src[size]:
                    return src[size]

            # Fallback to any available URL
            for url in src.values():
                if url:
                    return url

            logging.warning("No valid image URL found in photo data")
            return None

        except Exception as e:
            logging.error(f"Error extracting image URL: {e}")
            return None

    def get_api_usage_info(self) -> Dict[str, int]:
        """Get information about API usage (estimated)."""
        # Note: Pexels doesn't provide usage info in free tier
        # This is an estimation based on our tracking
        return {
            "estimated_requests_today": len(self.search_cache),
            "cache_entries": len(self.search_cache),
            "requests_per_run": self.pexels_settings['images_per_video'] // 2,  # Estimate
        }

    def clear_cache(self):
        """Clear the search results cache."""
        self.search_cache.clear()
        logging.info("Pexels search cache cleared")

    def validate_downloaded_images(self, image_paths: List[str]) -> List[str]:
        """Validate that downloaded images are valid and accessible."""
        valid_images = []

        for image_path in image_paths:
            try:
                if Path(image_path).exists() and Path(image_path).stat().st_size > 1024:  # At least 1KB
                    valid_images.append(image_path)
                else:
                    logging.warning(f"Invalid or missing image: {image_path}")
            except Exception as e:
                logging.warning(f"Error validating image {image_path}: {e}")

        logging.info(f"Validated {len(valid_images)} out of {len(image_paths)} images")
        return valid_images

    def cleanup_old_images(self, max_age_hours: int = 24):
        """Clean up old downloaded images."""
        try:
            temp_path = Path(self.temp_dir)
            if not temp_path.exists():
                return

            current_time = time.time()
            max_age_seconds = max_age_hours * 3600
            cleaned_count = 0

            for file_path in temp_path.glob("*.jpg"):
                if current_time - file_path.stat().st_mtime > max_age_seconds:
                    file_path.unlink()
                    cleaned_count += 1

            if cleaned_count > 0:
                logging.info(f"Cleaned up {cleaned_count} old image files")

        except Exception as e:
            logging.warning(f"Error cleaning up old images: {e}")