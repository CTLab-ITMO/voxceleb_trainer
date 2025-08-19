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
from itertools import combinations
from collections import defaultdict
import soundfile as sf

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

def convert_and_organize(extracted_path, save_path, tsv_filename='other.tsv'):
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
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        original_speaker_id = row['client_id']
        clip_filename = row['path']

        # Create short speaker ID and maintain mapping
        if original_speaker_id not in speaker_mapping:
            short_speaker_id = create_short_speaker_id(original_speaker_id)
            speaker_mapping[original_speaker_id] = short_speaker_id
        else:
            short_speaker_id = speaker_mapping[original_speaker_id]

        speaker_dir = os.path.join(output_wav_path, short_speaker_id)
        os.makedirs(speaker_dir, exist_ok=True)

        in_file = os.path.join(clips_path, clip_filename)
        out_file = os.path.join(speaker_dir, os.path.splitext(clip_filename)[0] + '.wav')

        if not os.path.exists(in_file):
            print(f"Warning: {in_file} does not exist, skipping.")
            continue

        # Use subprocess with shell=False for better cross-platform compatibility
        cmd = [
            'ffmpeg', '-y', '-i', in_file,
            '-ac', '1', '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', out_file
        ]

        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except subprocess.CalledProcessError:
            print(f"Warning: Failed to convert {in_file}, skipping.")
            continue

    # Save speaker mapping
    print(f"Saving speaker mapping to {speaker_mapping_file}")
    with open(speaker_mapping_file, 'w') as f:
        for original_id, short_id in speaker_mapping.items():
            f.write(f"{original_id}\t{short_id}\n")

def create_lists(save_path, train_split=0.9, len_train_speakers=0, len_test_speakers=0, min_seg_per_spk=0, min_frames=600, min_eval_frames=300):
    print("Creating train and test lists...")
    wav_path = os.path.join(save_path, 'common_voice_wav')
    speakers = [spk for spk in os.listdir(wav_path) if os.path.isdir(os.path.join(wav_path, spk))]
    if len_test_speakers == 0: # Если отдельное ограничение по числу тестовых спикеров не задано...
        len_test_speakers = int(len(speakers) * (1 - train_split))  # Берем по заданному разбиению

    def get_stats(min_frames): # Собираем статистику по спикерам и группируем по количеству файлов
        speaker_stats = defaultdict(list)
        speaker_segs = defaultdict(list)
        for speaker in speakers:
            speaker_dir = os.path.join(wav_path, speaker)
            wav_files = [f for f in os.listdir(speaker_dir) if f.endswith('.wav')]

            # ФИЛЬТРУЕМ файлы по длине
            valid_files = []
            for wav_file in wav_files:
                file_path = os.path.join(speaker_dir, wav_file)
                audio, sr = sf.read(file_path)
                # Проверяем, что файл не короче eval_frames
                if len(audio) >= min_frames * 160 + 240:
                    valid_files.append(wav_file)
                    speaker_segs[speaker].append(wav_file)

            num_files = len(valid_files)
            speaker_stats[num_files].append(speaker)
        # Сортируем ключи (количество файлов) по возрастанию
        sorted_file_counts = sorted(speaker_stats.keys())
        return speaker_stats, speaker_segs, sorted_file_counts

    train_speaker_stats, train_speaker_segs, train_sorted_file_counts = get_stats(min_frames) # Допускаются файлы не короче указанного значения
    test_speaker_stats, test_speaker_segs, test_sorted_file_counts = get_stats(min_eval_frames)

    '''
    Формируем test_speakers, начиная со спикеров, у которых 2 файла
    Потому что разбиение тестовой и обучающей выборки идет по спикерам (чтобы в тестовой не оказались спикеры из трейна — это может
    искусственно завысить качество на тесте), и логичнее брать спикеров с наименьшим количеством дублирующихся записей (2),
    добавляя их по возрастанию, чтобы из обучающей выборки не убирались спикеры, у которых много записей (что сильно бы срезало объем
    обучающей выборки)
    Для ложных примеров (0) просто берутся сочетания спикера с любым другим спикером из тестовой выборки
    При этом так, чтобы один и тот же "напарник" в ложной паре выбирался всего лишь один раз (для повышения разнообразия тестирования)
    Для этого не допускаются дублирования и отслеживается used_negative_elems
    '''
    test_speakers = []
    current_file_count = 2  # Начинаем с 2 файлов

    while len(test_speakers) < len_test_speakers and current_file_count <= max(test_sorted_file_counts):
        if current_file_count in test_speaker_stats:
            # Берем всех спикеров с current_file_count файлами
            available_speakers = test_speaker_stats[current_file_count]
            needed = len_test_speakers - len(test_speakers)
            # Добавляем либо всех доступных, либо столько, сколько нужно
            test_speakers.extend(available_speakers[:needed])

        current_file_count += 1

    # Формируем train_speakers (все остальные)
    if len_train_speakers == 0:
        train_speakers = [spk for spk in speakers if spk not in test_speakers] # Если ограничения на число спикеров нет, берем всех
    else: # Если есть ограничение, начинаем с наибольшего количества записей для каждого спикера
        current_file_count = max(train_sorted_file_counts)
        train_speakers = []
        while len(train_speakers) < len_train_speakers and current_file_count > min_seg_per_spk: # Число записей каждого спикера не меньше заданного
            if current_file_count in train_speaker_stats:
                available_speakers = train_speaker_stats[current_file_count]
                available_speakers = [spk for spk in available_speakers if spk not in test_speakers]
                needed = len_train_speakers - len(train_speakers)

                # Добавляем либо всех доступных, либо столько, сколько нужно
                train_speakers.extend(available_speakers[:needed])

            current_file_count -= 1

    print(f"Total speakers: {len(speakers)}")
    print(f"Test speakers {len(test_speakers)}")
    print(f"Train speakers {len(train_speakers)}")

    # Формируем файлы-списки
    test_file_name, train_file_name = f'test_list_{len(test_speakers)}', f'train_list_{len(train_speakers)}'
    if min_seg_per_spk > 0:
        train_file_name += f'_min_seg_{min_seg_per_spk}'
    if min_frames > 0:
        train_file_name += f'_min_frames_{min_frames}'
    if min_eval_frames > 0:
        test_file_name += f'_min_frames_{min_eval_frames}'
    test_file_name += '.txt'
    train_file_name += '.txt'

    # Формируем test_list.txt
    other_speakers = test_speakers
    random.shuffle(other_speakers)  # Перемешиваем для случайного выбора
    used_negative_elems = set()  # Для отслеживания использованных элементов в отрицательных парах
    with open(os.path.join(save_path, test_file_name), 'w') as test_file:
        for speaker in test_speakers:
            wav_files = test_speaker_segs[speaker]

            # Положительный пример (1)
            pos_line = f"1 {speaker}/{wav_files[0]} {speaker}/{wav_files[1]}\n"
            test_file.write(pos_line)

            # Отрицательный пример (0) с уникальным файлом
            other_wav = None
            for other_spk in other_speakers:
                if other_spk == speaker:
                    continue

                other_files = test_speaker_segs[other_spk]

                if other_files:
                    other_wav = random.choice(other_files)
                    used_negative_elems.add(other_wav)
                    break

            if other_wav:
                    neg_line = f"0 {speaker}/{wav_files[0]} {other_spk}/{other_wav}\n"
                    test_file.write(neg_line)

    # Формируем train_list.txt (из трейн-спикеров)
    random.shuffle(train_speakers)
    with open(os.path.join(save_path, train_file_name), 'w') as train_file:
        for speaker in tqdm(train_speakers):
            for wav_file in train_speaker_segs[speaker]:
                if wav_file.endswith('.wav'):
                    train_file.write(f"{speaker} {speaker}/{wav_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Common Voice data preparation")
    parser.add_argument('--save_path', type=str, default="data", help='Target directory for processed data and lists')
    parser.add_argument('--archive_path', type=str, required=True, help='Path to Common Voice tar.gz archive')
    parser.add_argument('--extract', dest='extract', action='store_true', help='Enable extract')
    parser.add_argument('--convert', dest='convert', action='store_true', help='Enable convert and organize')
    parser.add_argument('--create_lists', dest='create_lists', action='store_true', help='Enable creation of train/test lists')
    parser.add_argument('--len_train_speakers', type=int, default=0, help='Num of train speakers for Common Voice')
    parser.add_argument('--len_test_speakers', type=int, default=0, help='Num of test speakers for Common Voice')
    parser.add_argument('--min_seg_per_spk', type=int, default=0, help='Min num of segments per speaker for Common Voice')
    parser.add_argument('--min_frames', type=int, default=0, help='Min num of frames in train set for Common Voice')
    parser.add_argument('--min_eval_frames', type=int, default=0, help='Min num of frames in test set for Common Voice')

    args = parser.parse_args()

    extracted_path = os.path.join(args.save_path, 'cv-corpus')

    if args.extract:
        os.makedirs(extracted_path, exist_ok=True)
        full_extract(args.archive_path, extracted_path)

    if args.convert:
        convert_and_organize(extracted_path, args.save_path)

    if args.create_lists:
        create_lists(args.save_path, len_train_speakers=args.len_train_speakers, len_test_speakers=args.len_test_speakers,
                     min_seg_per_spk=args.min_seg_per_spk, min_frames=args.min_frames, min_eval_frames=args.min_eval_frames)