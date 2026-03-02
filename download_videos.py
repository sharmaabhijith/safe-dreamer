"""Download Kinetics 400 'driving car' videos for use as background distractors.

Downloads the official Kinetics 400 CSV annotations from AWS S3, filters by
label (default: 'driving car'), and downloads matching YouTube clips via yt-dlp
Python API. Videos are downloaded whole (no ffmpeg needed).

Adapted from https://github.com/burchim/MuDreamer/blob/main/download_videos.py

Prerequisites:
    pip install yt-dlp tqdm

Usage:
    python download_videos.py                          # defaults
    python download_videos.py --dest kinetics400/videos --label "driving car" --max-videos 50
"""

import argparse
import csv
import io
import os
import urllib.request

from tqdm import tqdm

# Official Kinetics 400 annotation CSVs on AWS S3
TRAIN_CSV_URL = "https://s3.amazonaws.com/kinetics/400/annotations/train.csv"
VAL_CSV_URL = "https://s3.amazonaws.com/kinetics/400/annotations/val.csv"
TEST_CSV_URL = "https://s3.amazonaws.com/kinetics/400/annotations/test.csv"


def download_csv(url):
    """Download a CSV from a URL and return rows as list of dicts."""
    print(f"Downloading annotations from {url} ...")
    response = urllib.request.urlopen(url)
    text = response.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def filter_by_label(rows, label):
    """Filter annotation rows by label (case-insensitive)."""
    label_lower = label.lower()
    entries = []
    for row in rows:
        if row.get("label", "").lower() == label_lower:
            youtube_id = row.get("youtube_id", "")
            if youtube_id:
                entries.append({
                    "youtube_id": youtube_id,
                    "url": f"https://www.youtube.com/watch?v={youtube_id}",
                })
    return entries


def download_videos(video_entries, dest_path, max_videos=None):
    """Download videos using yt-dlp Python API."""
    import yt_dlp

    os.makedirs(dest_path, exist_ok=True)
    entries = video_entries[:max_videos] if max_videos else video_entries
    succeeded = 0
    skipped = 0

    ydl_opts = {
        "format": "best[height<=360][ext=mp4]/best[height<=480][ext=mp4]/best[ext=mp4]/best",
        "outtmpl": os.path.join(dest_path, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
    }

    pbar = tqdm(entries, desc="Downloading videos")
    for entry in pbar:
        output_path = os.path.join(dest_path, f"{entry['youtube_id']}.mp4")
        # Also check for webm etc. since yt-dlp may pick a different ext
        existing = [
            f for f in os.listdir(dest_path)
            if f.startswith(entry["youtube_id"] + ".")
        ]
        if existing:
            succeeded += 1
            pbar.set_postfix(ok=succeeded, skip=skipped)
            continue
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([entry["url"]])
            # Verify file was created
            new_files = [
                f for f in os.listdir(dest_path)
                if f.startswith(entry["youtube_id"] + ".")
            ]
            if new_files:
                succeeded += 1
            else:
                skipped += 1
        except Exception:
            skipped += 1
        pbar.set_postfix(ok=succeeded, skip=skipped)

    print(f"Downloaded {succeeded}/{len(entries)} videos to {dest_path} "
          f"({skipped} unavailable)")


def main():
    parser = argparse.ArgumentParser(
        description="Download Kinetics 400 distractor videos for DMC evaluation"
    )
    parser.add_argument("--dest", type=str, default="kinetics400/videos",
                        help="Destination directory for downloaded videos")
    parser.add_argument("--label", type=str, default="driving car",
                        help="Kinetics label to filter (default: 'driving car')")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Max videos to download per split (default: all)")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"],
                        help="Which split to download (default: train)")
    args = parser.parse_args()

    splits = {
        "train": TRAIN_CSV_URL,
        "val": VAL_CSV_URL,
        "test": TEST_CSV_URL,
    }
    if args.split == "all":
        splits_to_download = splits
    else:
        splits_to_download = {args.split: splits[args.split]}

    for split_name, csv_url in splits_to_download.items():
        rows = download_csv(csv_url)
        entries = filter_by_label(rows, args.label)
        print(f"Found {len(entries)} '{args.label}' videos in {split_name} split")
        if entries:
            download_videos(
                entries,
                os.path.join(args.dest, split_name),
                max_videos=args.max_videos,
            )


if __name__ == "__main__":
    main()
