#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Dataset Preparation Script")
    parser.add_argument('--dataset', type=str, required=True, choices=['voxceleb', 'commonvoice'], help='Dataset to prepare')

    # Common arguments
    parser.add_argument('--save_path', type=str, default="data", help='Target directory')
    
    # VoxCeleb specific arguments
    parser.add_argument('--user', type=str, default="user", help='VoxCeleb download username')
    parser.add_argument('--password', type=str, default="pass", help='VoxCeleb download password')
    parser.add_argument('--download', dest='download', action='store_true', help='Enable download for VoxCeleb')
    parser.add_argument('--augment', dest='augment', action='store_true', help='Download and extract augmentation files for VoxCeleb')

    # Common Voice specific arguments
    parser.add_argument('--archive_path', type=str, help='Path to Common Voice tar.gz archive')
    parser.add_argument('--max_speakers', type=int, default=None, help='Maximum number of speakers for Common Voice dataset')
    parser.add_argument('--duration', type=float, default=None, help='Audio duration in seconds for Common Voice (shorter files skipped, longer trimmed)')
    parser.add_argument('--files_per_speaker', type=int, default=None, help='Exact number of files per speaker (speakers with fewer files excluded, more files limited to this number)')

    # Common actions
    parser.add_argument('--extract', dest='extract', action='store_true', help='Enable extract')
    parser.add_argument('--convert', dest='convert', action='store_true', help='Enable convert')
    parser.add_argument('--create_lists', dest='create_lists', action='store_true', help='Enable list creation for Common Voice')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of verification trials for Common Voice')

    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.dataset == 'voxceleb':
        print("Preparing VoxCeleb dataset...")
        command = [
            sys.executable, 'dataprep.py',
            '--save_path', args.save_path,
            '--user', args.user,
            '--password', args.password,
        ]
        if args.download: command.append('--download')
        if args.extract: command.append('--extract')
        if args.convert: command.append('--convert')
        if args.augment: command.append('--augment')
        
        subprocess.run(command, check=True)

    elif args.dataset == 'commonvoice':
        print("Preparing Common Voice dataset...")
        if not args.archive_path:
            raise ValueError("--archive_path is required for commonvoice dataset")
        
        command = [
            sys.executable, 'dataprep_common_voice.py',
            '--save_path', args.save_path,
            '--archive_path', args.archive_path,
            '--num_trials', str(args.num_trials),
        ]
        
        if args.max_speakers is not None:
            command.extend(['--max_speakers', str(args.max_speakers)])
        if args.duration is not None:
            command.extend(['--duration', str(args.duration)])
        if args.files_per_speaker is not None:
            command.extend(['--files_per_speaker', str(args.files_per_speaker)])
        if args.extract: command.append('--extract')
        if args.convert: command.append('--convert')
        if args.create_lists: command.append('--create_lists')

        subprocess.run(command, check=True)
        
        # Print training command for Common Voice
        print("\n" + "="*50)
        print("Dataset preparation complete!")
        print("To train with Common Voice dataset, use:")
        print(f"python trainSpeakerNet.py \\")
        print(f"  --train_path {args.save_path}/common_voice_wav \\")
        print(f"  --test_path {args.save_path}/common_voice_wav \\")
        print(f"  --train_list {args.save_path}/train_list.txt \\")
        print(f"  --test_list {args.save_path}/test_list.txt \\")
        print(f"  --model ResNetSE34V2 \\")
        print(f"  --save_path exps/common_voice_exp \\")
        print(f"  --max_epoch 100")
        
        if args.max_speakers:
            print(f"\nDataset limited to {args.max_speakers} speakers")
        if args.duration:
            print(f"Audio duration set to {args.duration} seconds")
        if args.files_per_speaker:
            print(f"Files per speaker limited to {args.files_per_speaker}")
        print("="*50)

if __name__ == "__main__":
    main()
