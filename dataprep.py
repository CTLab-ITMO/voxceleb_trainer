#!/usr/bin/python
#-*- coding: utf-8 -*-
# The script downloads the VoxCeleb datasets and converts all files to WAV.
# Requirement: ffmpeg and wget running on a Linux system.

import argparse
import os
import subprocess
import pdb
import hashlib
import time
import glob
import tarfile
from zipfile import ZipFile
from tqdm import tqdm
from scipy.io import wavfile
import random
import soundfile

## ========== ===========
## Parse input arguments
## ========== ===========
parser = argparse.ArgumentParser(description = "VoxCeleb downloader");

parser.add_argument('--save_path',     type=str, default="data", help='Target directory');
parser.add_argument('--user',         type=str, default="user", help='Username');
parser.add_argument('--password',     type=str, default="pass", help='Password');

parser.add_argument('--download', dest='download', action='store_true', help='Enable download')
parser.add_argument('--extract',  dest='extract',  action='store_true', help='Enable extract')
parser.add_argument('--convert',  dest='convert',  action='store_true', help='Enable convert')
parser.add_argument('--augment',  dest='augment',  action='store_true', help='Download and extract augmentation files')
parser.add_argument('--create_lists',  dest='create_lists', action='store_true', help='Create train/test lists for VoxCeleb1 after extraction/conversion')

parser.add_argument('--train_split',   type=float, default=0.9, help='Train split ratio when generating lists')
parser.add_argument('--num_trials',    type=int,   default=1000, help='Number of verification trials (balanced same/diff)')
parser.add_argument('--max_speakers',  type=int,   default=None, help='Maximum number of speakers to include when creating lists')
parser.add_argument('--files_per_speaker', type=int, default=None, help='Limit (and equalize) number of files per speaker (exclude speakers with fewer)')
parser.add_argument('--duration',      type=float, default=None, help='Minimum duration in seconds (exclude files shorter than this)')
parser.add_argument('--archive',       action='append', help='Path to a local VoxCeleb zip/tar archive (can be repeated). Skips download logic.')

args = parser.parse_args();

## ========== ===========
## MD5SUM
## ========== ===========
def md5(fname):

    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

## ========== ===========
## Download with wget
## ========== ===========
def download(args, lines):

    for line in lines:
        url     = line.split()[0]
        md5gt     = line.split()[1]
        outfile = url.split('/')[-1]

        ## Download files
        out     = subprocess.call('wget %s --user %s --password %s -O %s/%s'%(url,args.user,args.password,args.save_path,outfile), shell=True)
        if out != 0:
            raise ValueError('Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'%url)

        ## Check MD5
        md5ck     = md5('%s/%s'%(args.save_path,outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.'%outfile)
        else:
            raise Warning('Checksum failed %s.'%outfile)

## ========== ===========
## Concatenate file parts
## ========== ===========
def concatenate(args,lines):

    for line in lines:
        infile     = line.split()[0]
        outfile    = line.split()[1]
        md5gt     = line.split()[2]

        ## Concatenate files
        out     = subprocess.call('cat %s/%s > %s/%s' %(args.save_path,infile,args.save_path,outfile), shell=True)

        ## Check MD5
        md5ck     = md5('%s/%s'%(args.save_path,outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.'%outfile)
        else:
            raise Warning('Checksum failed %s.'%outfile)

        out     = subprocess.call('rm %s/%s' %(args.save_path,infile), shell=True)

## ========== ===========
## Extract zip files
## ========== ===========
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

def full_extract(args, fname):

    print('Extracting %s'%fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            safe_extract(tar, args.save_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, 'r') as zf:
            zf.extractall(args.save_path)


## ========== ===========
## Partially extract zip files
## ========== ===========
def part_extract(args, fname, target):

    print('Extracting %s'%fname)
    with ZipFile(fname, 'r') as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile,args.save_path)
            # pdb.set_trace()
            # zf.extractall(args.save_path)

## ========== ===========
## Convert
## ========== ===========
def convert(args):

    files     = glob.glob('%s/voxceleb2/*/*/*.m4a'%args.save_path)
    files.sort()

    print('Converting files from AAC to WAV')
    for fname in tqdm(files):
        outfile = fname.replace('.m4a','.wav')
        out = subprocess.call('ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null' %(fname,outfile), shell=True)
        if out != 0:
            raise ValueError('Conversion failed %s.'%fname)

## ========== ===========
## Split MUSAN for faster random access
## ========== ===========
def split_musan(args):

    files = glob.glob('%s/musan/*/*/*.wav'%args.save_path)

    audlen = 16000*5
    audstr = 16000*3

    for idx,file in enumerate(files):
        fs,aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/','/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0,len(aud)-audlen,audstr):
            wavfile.write(writedir+'/%05d.wav'%(st/fs),fs,aud[st:st+audlen])

        print(idx,file)

def get_wav_duration(path):
    try:
        f = soundfile.SoundFile(path)
        return len(f) / float(f.samplerate)
    except:
        return None

def iter_speaker_wavs(root_dir, speaker_id):
    spk_root = os.path.join(root_dir, speaker_id)
    for dirpath, _, filenames in os.walk(spk_root):
        for fn in filenames:
            if fn.lower().endswith('.wav'):
                full = os.path.join(dirpath, fn)
                rel_inside = os.path.relpath(full, root_dir)
                yield full, rel_inside

def create_voxceleb_lists(save_path, source_dir='voxceleb1', train_split=0.9,
                          num_trials=1000, max_speakers=None,
                          files_per_speaker=None, duration=None):
    print("Creating VoxCeleb lists...")
    vox_path = os.path.join(save_path, source_dir)
    if not os.path.isdir(vox_path):
        raise ValueError(f"Source directory {vox_path} not found")

    speakers = [d for d in os.listdir(vox_path) if os.path.isdir(os.path.join(vox_path, d))]
    speakers.sort()
    print(f"Found {len(speakers)} speaker folders")

    # Collect files per speaker
    spk_files = {}
    for spk in speakers:
        collected = []
        for full, rel in iter_speaker_wavs(vox_path, spk):
            if duration is not None:
                dur = get_wav_duration(full)
                if dur is None or dur < duration:
                    continue
            collected.append(rel.replace('\\', '/'))
        if files_per_speaker is not None:
            if len(collected) >= files_per_speaker:
                collected = collected[:files_per_speaker]
            else:
                # skip speaker if not enough
                collected = []
        if len(collected) >= 2:
            spk_files[spk] = collected

    print(f"Speakers with >=2 usable files (after filters): {len(spk_files)}")

    # Limit by max_speakers
    if max_speakers is not None and len(spk_files) > max_speakers:
        ranked = sorted(spk_files.items(), key=lambda x: len(x[1]), reverse=True)[:max_speakers]
        spk_files = dict(ranked)
        print(f"Limited to top {max_speakers} speakers by file count")

    all_speakers = list(spk_files.keys())
    random.shuffle(all_speakers)
    split_idx = int(len(all_speakers) * train_split)
    train_speakers = all_speakers[:split_idx]
    test_speakers  = all_speakers[split_idx:]

    print(f"Split: {len(train_speakers)} train speakers, {len(test_speakers)} test speakers")

    train_list_path = os.path.join(save_path, 'train_list.txt')
    test_list_path  = os.path.join(save_path, 'test_list.txt')

    total_train = 0
    with open(train_list_path, 'w') as f:
        for spk in train_speakers:
            for rel in spk_files[spk]:
                f.write(f"{spk} {rel}\n")
                total_train += 1
    print(f"Train list written: {total_train} lines")

    test_files_by_spk = {spk: spk_files[spk] for spk in test_speakers if len(spk_files[spk]) >= 2}

    same_trials = []
    for spk, files in test_files_by_spk.items():
        limit_inner = min(len(files), 10)
        for i in range(limit_inner):
            for j in range(i + 1, limit_inner):
                same_trials.append((1, files[i], files[j]))

    diff_trials = []
    test_spk_list = list(test_files_by_spk.keys())
    for i in range(len(test_spk_list)):
        for j in range(i + 1, len(test_spk_list)):
            f1s = test_files_by_spk[test_spk_list[i]][:5]
            f2s = test_files_by_spk[test_spk_list[j]][:5]
            for a in f1s:
                for b in f2s:
                    diff_trials.append((0, a, b))

    num_same = min(len(same_trials), num_trials // 2)
    num_diff = min(len(diff_trials), num_trials // 2)
    chosen = random.sample(same_trials, num_same) + random.sample(diff_trials, num_diff)
    random.shuffle(chosen)

    with open(test_list_path, 'w') as f:
        for lab, f1, f2 in chosen:
            f.write(f"{lab} {f1} {f2}\n")
    print(f"Test list written: {len(chosen)} trials ({num_same} same / {num_diff} diff)")

## ========== ===========
## Main script
## ========== ===========
if __name__ == "__main__":

    if not os.path.exists(args.save_path):
        raise ValueError('Target directory does not exist.')

    if args.archive:
        for arch in args.archive:
            if not os.path.isfile(arch):
                raise ValueError(f"Archive not found: {arch}")
            full_extract(args, arch)
    else:
        f = open('lists/fileparts.txt','r')
        fileparts = f.readlines()
        f.close()

        f = open('lists/files.txt','r')
        files = f.readlines()
        f.close()

        f = open('lists/augment.txt','r')
        augfiles = f.readlines()
        f.close()

        if args.augment:
            download(args,augfiles)
            part_extract(args,os.path.join(args.save_path,'rirs_noises.zip'),['RIRS_NOISES/simulated_rirs/mediumroom','RIRS_NOISES/simulated_rirs/smallroom'])
            full_extract(args,os.path.join(args.save_path,'musan.tar.gz'))
            split_musan(args)

        if args.download:
            download(args,fileparts)

        if args.extract:
            concatenate(args, files)
            for file in files:
                full_extract(args,os.path.join(args.save_path,file.split()[1]))
            out = subprocess.call('mv %s/dev/aac/* %s/aac/ && rm -r %s/dev' %(args.save_path,args.save_path,args.save_path), shell=True)
            out = subprocess.call('mv %s/wav %s/voxceleb1' %(args.save_path,args.save_path), shell=True)
            out = subprocess.call('mv %s/aac %s/voxceleb2' %(args.save_path,args.save_path), shell=True)

        if args.convert:
            convert(args)

    if args.create_lists:
        source_dir = 'voxceleb1'
        if not os.path.isdir(os.path.join(args.save_path, source_dir)):
            if os.path.isdir(os.path.join(args.save_path, 'wav')):
                source_dir = 'wav'
        create_voxceleb_lists(save_path=args.save_path,
                              source_dir=source_dir,
                              train_split=args.train_split,
                              num_trials=args.num_trials,
                              max_speakers=args.max_speakers,
                              files_per_speaker=args.files_per_speaker,
                              duration=args.duration)

