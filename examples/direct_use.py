#!/usr/bin/env python3
# example_direct_generation.py - Example of using MainOrchestrator with direct data

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Change to project directory for relative imports
os.chdir(project_root)

from videoOrchestrator import VideoOrchestrator


def example_single_generation():
    """Example of generating a single video from topic data."""

    # Define your topic data
    data = {
        "title": "Abstraction",
        "description": "Abstraction is the process of reducing complexity by focusing on the essential characteristics of an object or system, while hiding irrelevant details. It allows developers to manage complexity by creating simplified models or interfaces that represent real-world entities, thereby enabling easier understanding, design, and maintenance of software systems.",
        "complete": False
    }

    print("🚀 Starting direct video generation...")
    print(f"📋 Topic: {data['title']}")
    print()

    try:
        # Create orchestrator with direct data
        mo = VideoOrchestrator.from_topic_data(data)

        # Generate the video
        result = mo.generate()

        # Display results
        if result["success"]:
            print("🎉 SUCCESS! Video generated!")
            print(f"   📹 Video: {result['video_path']}")
            print(f"   ⏱️  Duration: {result['audio_duration']:.1f}s")
            print(f"   🖼️  Images: {result['images_count']}")

            # Show detailed timing
            if "timing" in result:
                timing = result["timing"]
                print(f"\n⏱️  Detailed Timing:")
                for step, time_taken in timing.items():
                    if step != "total":
                        print(f"   📌 {step.replace('_', ' ').title()}: {time_taken:.2f}s")
                print(f"   🎯 TOTAL: {timing.get('total', 0):.2f}s")

            return True

        else:
            print(f"❌ FAILED: {result['error']}")
            if "timing" in result and "total" in result["timing"]:
                print(f"⏱️  Time spent: {result['timing']['total']:.2f}s")
            return False

    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False


def example_batch_generation():
    """Example of generating multiple videos from a list of topics."""

    topics = [
        {
            "title": "API Design",
            "description": "API design involves creating interfaces that allow different software applications to communicate effectively. Well-designed APIs are intuitive, consistent, and provide clear documentation.",
        },
        {
            "title": "Algorithms",
            "description": "Algorithms are step-by-step procedures for solving problems. In computer science, they define a sequence of computational steps that transform input data into desired output.",
        },
        {
            "title": "Data Structures",
            "description": "Data structures are ways of organizing and storing data in computer memory to enable efficient access and modification. Common examples include arrays, linked lists, trees, and hash tables.",
        }
    ]

    print("🚀 Starting batch video generation...")
    print(f"📋 Topics: {len(topics)}")
    print()

    results = []
    total_time = 0

    try:
        for i, topic_data in enumerate(topics, 1):
            print(f"📹 Generating video {i}/{len(topics)}: {topic_data['title']}")

            # Create orchestrator for this topic
            mo = VideoOrchestrator.from_topic_data(topic_data)

            # Generate video
            result = mo.generate()
            results.append(result)

            if result["success"]:
                time_taken = result.get("timing", {}).get("total", 0)
                total_time += time_taken
                print(f"   ✅ Success! ({time_taken:.1f}s) - {result['video_path']}")
            else:
                print(f"   ❌ Failed: {result['error']}")

            print()

        # Summary
        successful = [r for r in results if r["success"]]
        print("📊 BATCH SUMMARY:")
        print(f"   ✅ Successful: {len(successful)}/{len(topics)}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   📈 Average: {total_time / len(topics):.1f}s per video")

        if successful:
            print(f"\n🎬 Generated videos:")
            for result in successful:
                print(f"   - {result['video_path']}")

        return len(successful) == len(topics)

    except Exception as e:
        print(f"💥 BATCH ERROR: {e}")
        return False


if __name__ == "__main__":
    print("🎬 YouTube Shorts Generator - Direct Generation Examples")
    print("=" * 60)

    # Example 1: Single generation
    print("\n1️⃣  SINGLE VIDEO GENERATION")
    print("-" * 30)
    success1 = example_single_generation()

    print("\n" + "=" * 60)

    # Example 2: Batch generation (optional)
    print("\n2️⃣  BATCH VIDEO GENERATION")
    print("-" * 30)

    # Ask user if they want to run batch example
    try:
        response = input("\nRun batch generation example? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            success2 = example_batch_generation()
        else:
            print("Skipping batch generation example.")
            success2 = True
    except KeyboardInterrupt:
        print("\n👋 Exiting...")
        success2 = True

    print("\n" + "=" * 60)
    print("🎯 EXAMPLES COMPLETED")

    if not success1:
        print("❌ Single generation failed")
        sys.exit(1)
    else:
        print("✅ All examples completed successfully!")
        sys.exit(0)