#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse
import os
import subprocess
import tarfile
import pandas as pd
from tqdm import tqdm
import random
from zipfile import ZipFile
import hashlib
import json

def is_within_directory(directory, target):
    abs_directory = os.path.abspath(directory)
    abs_target = os.path.abspath(target)
    prefix = os.path.commonprefix([abs_directory, abs_target])
    return prefix == abs_directory

def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    for member in tar.getmembers():
        member_path = os.path.join(path, member.name)
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path, members, numeric_owner=numeric_owner)

def full_extract(archive_path, out_path):
    print(f'Extracting {archive_path} to {out_path}')
    if archive_path.endswith(".tar.gz"):
        with tarfile.open(archive_path, "r:gz") as tar:
            safe_extract(tar, out_path)
    elif archive_path.endswith(".zip"):
        with ZipFile(archive_path, 'r') as zf:
            zf.extractall(out_path)

def create_short_speaker_id(client_id):
    """Create a shorter speaker ID to avoid Windows path length issues"""
    # Use MD5 hash to create a shorter, consistent identifier
    return hashlib.md5(client_id.encode()).hexdigest()[:16]

def get_audio_duration(file_path):
    """Get audio duration using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'json', file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            return duration
        else:
            return None
    except Exception as e:
        return None

def convert_and_organize(extracted_path, save_path, tsv_filename='validated.tsv', duration=None):
    tsv_path = None
    for root, dirs, files in os.walk(extracted_path):
        if tsv_filename in files:
            tsv_path = os.path.join(root, tsv_filename)
            break
    
    if tsv_path is None:
        raise FileNotFoundError(f"{tsv_filename} not found in the extracted archive at {extracted_path}")

    print(f"Reading metadata from {tsv_path}")
    df = pd.read_csv(tsv_path, sep='\t')
    
    clips_path = os.path.join(os.path.dirname(tsv_path), 'clips')
    output_wav_path = os.path.join(save_path, 'common_voice_wav')
    os.makedirs(output_wav_path, exist_ok=True)

    # Create mapping file for speaker IDs
    speaker_mapping_file = os.path.join(save_path, 'speaker_mapping.txt')
    speaker_mapping = {}
    
    print('Converting and organizing files...')
    skipped_duration = 0
    total_processed = 0
    
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        original_speaker_id = row['client_id']
        clip_filename = row['path']
        
        in_file = os.path.join(clips_path, clip_filename)
        if not os.path.exists(in_file):
            print(f"Warning: {in_file} does not exist, skipping.")
            continue
        
        if duration is not None:
            audio_duration = get_audio_duration(in_file)
            if audio_duration is None:
                print(f"Warning: Could not get duration for {in_file}, skipping.")
                continue
            elif audio_duration < duration:
                skipped_duration += 1
                continue
        
        # Create short speaker ID and maintain mapping
        if original_speaker_id not in speaker_mapping:
            short_speaker_id = create_short_speaker_id(original_speaker_id)
            speaker_mapping[original_speaker_id] = short_speaker_id
        else:
            short_speaker_id = speaker_mapping[original_speaker_id]
        
        speaker_dir = os.path.join(output_wav_path, short_speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)
        
        out_file = os.path.join(speaker_dir, os.path.splitext(clip_filename)[0] + '.wav')

        # Build ffmpeg command
        cmd = [
            'ffmpeg', '-y', '-i', in_file, 
            '-ac', '1', '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '16000'
        ]
        
        if duration is not None:
            cmd.extend(['-t', str(duration)])
            
        cmd.append(out_file)
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            total_processed += 1
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to convert {in_file}, skipping.")
            continue
    
    # Save speaker mapping
    print(f"Saving speaker mapping to {speaker_mapping_file}")
    with open(speaker_mapping_file, 'w') as f:
        for original_id, short_id in speaker_mapping.items():
            f.write(f"{original_id}\t{short_id}\n")
    
    print(f"Conversion complete:")
    print(f"  Total files processed: {total_processed}")
    if duration is not None:
        print(f"  Files skipped due to duration < {duration}s: {skipped_duration}")

def create_lists(save_path, train_split=0.9, num_verification_trials=1000, max_speakers=None, files_per_speaker=None):
    print("Creating train and test lists...")
    wav_path = os.path.join(save_path, 'common_voice_wav')
    speakers = os.listdir(wav_path)
    
    valid_speakers = []
    for speaker in speakers:
        speaker_dir = os.path.join(wav_path, speaker)
        if os.path.isdir(speaker_dir):
            files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
            
            if files_per_speaker is not None:
                if len(files) >= files_per_speaker:
                    valid_speakers.append(speaker)
            else:
                if len(files) >= 2:  
                    valid_speakers.append(speaker)
    
    if files_per_speaker is not None:
        print(f"Found {len(valid_speakers)} speakers with at least {files_per_speaker} audio files")
    else:
        print(f"Found {len(valid_speakers)} speakers with at least 2 audio files")
    
    if max_speakers is not None and len(valid_speakers) > max_speakers:
        speaker_file_counts = []
        for speaker in valid_speakers:
            speaker_dir = os.path.join(wav_path, speaker)
            file_count = len([f for f in os.listdir(speaker_dir) if f.endswith('.wav')])
            speaker_file_counts.append((speaker, file_count))
        
        speaker_file_counts.sort(key=lambda x: x[1], reverse=True)
        valid_speakers = [speaker for speaker, _ in speaker_file_counts[:max_speakers]]
        print(f"Limited to {max_speakers} speakers with most audio files")
    
    random.shuffle(valid_speakers)

    split_idx = int(len(valid_speakers) * train_split)
    train_speakers = valid_speakers[:split_idx]
    test_speakers = valid_speakers[split_idx:]

    print(f"Split: {len(train_speakers)} training speakers, {len(test_speakers)} test speakers")

    # Create train list
    def write_train_list(speaker_list, list_path):
        total_files = 0
        with open(list_path, 'w') as f:
            for speaker_id in tqdm(speaker_list, desc="Writing train list"):
                speaker_dir = os.path.join(wav_path, speaker_id)
                if not os.path.isdir(speaker_dir):
                    continue
                
                audio_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]
                
                if files_per_speaker is not None:
                    audio_files = audio_files[:files_per_speaker]
                
                for utt_file in audio_files:
                    file_path = os.path.join(speaker_id, utt_file)
                    f.write(f"{speaker_id} {file_path}\n")
                    total_files += 1
        
        if files_per_speaker is not None:
            print(f"Train list created with {total_files} files (max {files_per_speaker} files per speaker)")
        else:
            print(f"Train list created with {total_files} files")

    def write_verification_list(speaker_list, list_path, num_trials):
        verification_trials = []
        
        speaker_files = {}
        for speaker_id in speaker_list:
            speaker_dir = os.path.join(wav_path, speaker_id)
            if not os.path.isdir(speaker_dir):
                continue
            files = [os.path.join(speaker_id, f) for f in os.listdir(speaker_dir) if f.endswith('.wav')]
            speaker_files[speaker_id] = files
        
        valid_speakers = list(speaker_files.keys())
        
        # Generate same-speaker trials (label = 1)
        same_speaker_trials = []
        for speaker_id in valid_speakers:
            files = speaker_files[speaker_id]
            if len(files) >= 2:
                # Create pairs within the same speaker
                for i in range(len(files)):
                    for j in range(i + 1, min(i + 5, len(files))):  # Limit pairs per speaker
                        same_speaker_trials.append((1, files[i], files[j]))
        
        # Generate different-speaker trials (label = 0)
        different_speaker_trials = []
        for i in range(len(valid_speakers)):
            for j in range(i + 1, len(valid_speakers)):
                speaker1_files = speaker_files[valid_speakers[i]]
                speaker2_files = speaker_files[valid_speakers[j]]
                
                for k in range(min(5, len(speaker1_files))):
                    for m in range(min(5, len(speaker2_files))):
                        different_speaker_trials.append((0, speaker1_files[k], speaker2_files[m]))  
        
        # Balance and limit trials
        num_same = min(len(same_speaker_trials), num_trials // 2)
        num_diff = min(len(different_speaker_trials), num_trials // 2)
        
        selected_trials = (random.sample(same_speaker_trials, num_same) + 
                          random.sample(different_speaker_trials, num_diff))
        random.shuffle(selected_trials)
        
        # Write verification trials
        with open(list_path, 'w') as f:
            for label, file1, file2 in selected_trials:
                f.write(f"{label} {file1} {file2}\n")
        
        print(f"Created {len(selected_trials)} verification trials ({num_same} same-speaker, {num_diff} different-speaker)")

    write_train_list(train_speakers, os.path.join(save_path, 'train_list.txt'))
    write_verification_list(test_speakers, os.path.join(save_path, 'test_list.txt'), num_verification_trials)
    print("Train and test lists created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Common Voice data preparation")
    parser.add_argument('--save_path', type=str, default="data", help='Target directory for processed data and lists')
    parser.add_argument('--archive_path', type=str, required=True, help='Path to Common Voice tar.gz archive')
    parser.add_argument('--extract', dest='extract', action='store_true', help='Enable extract')
    parser.add_argument('--convert', dest='convert', action='store_true', help='Enable convert and organize')
    parser.add_argument('--create_lists', dest='create_lists', action='store_true', help='Enable creation of train/test lists')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of verification trials to generate')
    parser.add_argument('--max_speakers', type=int, default=None, help='Maximum number of speakers to include')
    parser.add_argument('--duration', type=float, default=None, help='Maximum duration in seconds (files shorter than this will be skipped, longer will be trimmed)')
    parser.add_argument('--files_per_speaker', type=int, default=None, help='Exact number of files per speaker (speakers with fewer files will be excluded, speakers with more files will be limited to this number)')

    args = parser.parse_args()

    extracted_path = os.path.join(args.save_path, 'cv-corpus')
    
    if args.extract:
        os.makedirs(extracted_path, exist_ok=True)
        full_extract(args.archive_path, extracted_path)

    if args.convert:
        convert_and_organize(extracted_path, args.save_path, duration=args.duration)

    if args.create_lists:
        create_lists(args.save_path, num_verification_trials=args.num_trials, max_speakers=args.max_speakers, files_per_speaker=args.files_per_speaker)
