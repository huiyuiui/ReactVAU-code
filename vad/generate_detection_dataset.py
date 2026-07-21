# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Generate Detection Dataset for UCF-Crime and XD-Violence
Correctly handles the case where 'events' field exists but 'label' is empty (normal videos)
"""
import json
import os
import math

# Path configuration
HIVAU_ROOT = os.environ.get("HIVAU_ROOT", "/path/to/HIVAU-70k")
raw_annotation_dir = os.environ.get(
    "HIVAU_RAW_ANNOTATION_DIR",
    os.path.join(HIVAU_ROOT, "raw_annotations"),
)
output_dir = os.environ.get(
    "HIVAU_DETECTION_OUTPUT_DIR",
    os.path.join(HIVAU_ROOT, "instruction", "detection"),
)

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)


def seconds_to_frames(seconds, fps):
    """Convert seconds to frame index."""
    return int(math.floor(seconds * fps))


def process_dataset(input_file, dataset_name):
    """
    Process a single dataset and generate detection JSON.

    Key fix: Only use 'events' as anomaly time intervals when 'label' is non-empty.
    For normal videos (label=[]), events_seconds and events_frames are empty.
    """
    with open(input_file, "r") as f:
        data = json.load(f)
    
    detection_data = {}
    
    for video_key, video_info in data.items():
        n_frames = video_info.get("n_frames", 0)
        fps = video_info.get("fps", 30.0)
        label = video_info.get("label", [])
        events = video_info.get("events", [])
        
        # Determine whether the video is anomalous (binary label)
        # Key: only when label is non-empty does the video contain anomalies.
        has_anomaly = 1 if label and len(label) > 0 else 0
        
        # IMPORTANT FIX: Only use events as anomaly intervals when has_anomaly=1.
        # For normal videos, events may contain "interesting segments" but not anomalies.
        if has_anomaly:
            events_seconds = events
            # Convert event times from seconds to frames.
            events_in_frames = []
            for event in events:
                if len(event) == 2:
                    start_sec, end_sec = event
                    start_frame = seconds_to_frames(start_sec, fps)
                    end_frame = seconds_to_frames(end_sec, fps)
                    # Clamp frame indices to valid range.
                    start_frame = max(0, start_frame)
                    end_frame = min(n_frames - 1, end_frame)
                    events_in_frames.append([start_frame, end_frame])
        else:
            # Normal video: no anomaly events.
            events_seconds = []
            events_in_frames = []
        
        # Build output record.
        detection_data[video_key] = {
            "video_name": video_key,
            "n_frames": n_frames,
            "fps": fps,
            "label": label,  # Anomaly type labels (e.g., ["Arson"]).
            "anomaly": has_anomaly,  # Binary anomaly flag (0 or 1).
            "events_seconds": events_seconds,  # Anomaly time intervals in seconds.
            "events_frames": events_in_frames,  # Anomaly frame intervals.
        }
    
    return detection_data


def process_and_save(input_file, output_file, dataset_name, split_name):
    """Process and save the dataset."""
    print(f"Processing {dataset_name} {split_name} dataset...")
    detection_data = process_dataset(input_file, dataset_name)
    
    with open(output_file, "w") as f:
        json.dump(detection_data, f, indent=2)
    
    # 統計
    anomaly_count = sum(1 for v in detection_data.values() if v["anomaly"] == 1)
    normal_count = sum(1 for v in detection_data.values() if v["anomaly"] == 0)
    print(f"{dataset_name} ({split_name}): {len(detection_data)} videos ({anomaly_count} anomaly, {normal_count} normal)")
    print(f"Saved to: {output_file}")
    
    # Verify fix: check Normal videos have empty events
    normal_with_events = sum(1 for v in detection_data.values() 
                             if v["anomaly"] == 0 and len(v["events_frames"]) > 0)
    print(f"  Normal videos with events (should be 0): {normal_with_events}")
    
    return detection_data


def main():
    # ==================== UCF-Crime ====================
    # UCF-Crime Test
    ucf_test_input = os.path.join(raw_annotation_dir, "ucf_database_test.json")
    ucf_test_output = os.path.join(output_dir, "ucf_crime_detection_test.json")
    process_and_save(ucf_test_input, ucf_test_output, "UCF-Crime", "test")
    
    # UCF-Crime Train
    ucf_train_input = os.path.join(raw_annotation_dir, "ucf_database_train.json")
    ucf_train_output = os.path.join(output_dir, "ucf_crime_detection_train.json")
    process_and_save(ucf_train_input, ucf_train_output, "UCF-Crime", "train")
    
    # ==================== XD-Violence ====================
    # XD-Violence Test
    xd_test_input = os.path.join(raw_annotation_dir, "xd_database_test.json")
    xd_test_output = os.path.join(output_dir, "xd_violence_detection_test.json")
    process_and_save(xd_test_input, xd_test_output, "XD-Violence", "test")
    
    # XD-Violence Train
    xd_train_input = os.path.join(raw_annotation_dir, "xd_database_train.json")
    xd_train_output = os.path.join(output_dir, "xd_violence_detection_train.json")
    process_and_save(xd_train_input, xd_train_output, "XD-Violence", "train")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
