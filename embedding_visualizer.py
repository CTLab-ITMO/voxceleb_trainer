#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import os
from collections import defaultdict
import time

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Visualization will be limited.")


class EmbeddingVisualizer:    
    def __init__(self, save_path, max_samples_per_speaker=50):
        self.save_path = save_path
        self.max_samples_per_speaker = max_samples_per_speaker
        os.makedirs(save_path, exist_ok=True)
        
        plt.style.use('default')
        
    def extract_embeddings_with_labels(self, model, test_loader, num_speakers=None):

        model.eval()
        
        speaker_embeddings = defaultdict(list)
        
        print("Extracting embeddings for visualization...")
        
        with torch.no_grad():
            for idx, (data, filename) in enumerate(test_loader):
                if idx % 100 == 0:
                    print(f"Processing {idx}/{len(test_loader)}")
                
                speaker_name = filename[0].split('\\')[0] if '\\' in filename[0] else filename[0].split('/')[0]
                
                if len(speaker_embeddings[speaker_name]) >= self.max_samples_per_speaker:
                    continue
                
                data = data[0].cuda()
                embedding = model(data)
                
                embedding = F.normalize(embedding, p=2, dim=1)
                embedding_mean = torch.mean(embedding, dim=0).cpu().numpy()
                speaker_embeddings[speaker_name].append(embedding_mean)
        
        embeddings = []
        labels = []
        speaker_names = []
        
        speakers_to_visualize = list(speaker_embeddings.keys())
        if num_speakers is not None:
            speakers_to_visualize = speakers_to_visualize[:num_speakers]
        
        for speaker_idx, speaker_name in enumerate(speakers_to_visualize):
            for embedding in speaker_embeddings[speaker_name]:
                embeddings.append(embedding)
                labels.append(speaker_idx)
                speaker_names.append(speaker_name)
        
        embeddings = np.array(embeddings)
        
        print(f"Collected {len(embeddings)} embeddings from {len(speakers_to_visualize)} speakers")
        
        return embeddings, labels, speaker_names, speakers_to_visualize
    
    def visualize_with_tsne(self, embeddings, labels, speaker_names, title_suffix="", perplexity=30):
        print("Computing t-SNE...")
        start_time = time.time()
        
        if len(embeddings) < perplexity * 3:
            perplexity = max(5, len(embeddings) // 3)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        tsne_time = time.time() - start_time
        print(f"t-SNE completed in {tsne_time:.2f} seconds")
        
        plt.figure(figsize=(12, 8))
        
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            speaker_name = speaker_names[np.where(mask)[0][0]]
            
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=f"Speaker {label} ({speaker_name[:8]}...)", 
                       alpha=0.7, s=50)
        
        plt.title(f"t-SNE Visualization of Speaker Embeddings{title_suffix}", fontsize=16)
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"tsne_embeddings{title_suffix.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(self.save_path, filename), dpi=300, bbox_inches='tight')
        print(f"t-SNE plot saved to {os.path.join(self.save_path, filename)}")
        plt.close()
        
        return embeddings_2d
    
    def visualize_with_pca(self, embeddings, labels, speaker_names, title_suffix=""):
        print("Computing PCA...")
        
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        
        unique_labels = list(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            speaker_name = speaker_names[np.where(mask)[0][0]]
            
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[colors[i]], label=f"Speaker {label} ({speaker_name[:8]}...)", 
                       alpha=0.7, s=50)
        
        plt.title(f"PCA Visualization of Speaker Embeddings{title_suffix}", fontsize=16)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"pca_embeddings{title_suffix.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(self.save_path, filename), dpi=300, bbox_inches='tight')
        print(f"PCA plot saved to {os.path.join(self.save_path, filename)}")
        plt.close()
        
        return embeddings_2d
    
    def create_embedding_stats_plot(self, embeddings, title_suffix=""):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Гистограмма норм эмбеддингов
        norms = np.linalg.norm(embeddings, axis=1)
        axes[0, 0].hist(norms, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title("Distribution of Embedding Norms")
        axes[0, 0].set_xlabel("L2 Norm")
        axes[0, 0].set_ylabel("Frequency")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Гистограмма значений эмбеддингов
        axes[0, 1].hist(embeddings.flatten(), bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title("Distribution of Embedding Values")
        axes[0, 1].set_xlabel("Embedding Value")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Средние значения по измерениям
        mean_values = np.mean(embeddings, axis=0)
        axes[1, 0].plot(mean_values, alpha=0.7, color='green')
        axes[1, 0].set_title("Mean Values Across Embedding Dimensions")
        axes[1, 0].set_xlabel("Dimension")
        axes[1, 0].set_ylabel("Mean Value")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Стандартные отклонения по измерениям
        std_values = np.std(embeddings, axis=0)
        axes[1, 1].plot(std_values, alpha=0.7, color='orange')
        axes[1, 1].set_title("Standard Deviation Across Embedding Dimensions")
        axes[1, 1].set_xlabel("Dimension")
        axes[1, 1].set_ylabel("Standard Deviation")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f"Embedding Statistics{title_suffix}", fontsize=16)
        plt.tight_layout()
    
        filename = f"embedding_stats{title_suffix.replace(' ', '_').lower()}.png"
        plt.savefig(os.path.join(self.save_path, filename), dpi=300, bbox_inches='tight')
        print(f"Statistics plot saved to {os.path.join(self.save_path, filename)}")
        plt.close()
    
    def visualize_embeddings(self, model, test_loader, num_speakers=10, title_suffix="", methods=['all']):
        print(f"Starting embedding visualization{title_suffix}...")
        
        embeddings, labels, speaker_names, speakers_list = self.extract_embeddings_with_labels(
            model, test_loader, num_speakers
        )
        
        if len(embeddings) == 0:
            print("No embeddings extracted. Skipping visualization.")
            return
        
        # Создать статистики
        if 'all' in methods or 'stats' in methods:
            self.create_embedding_stats_plot(embeddings, title_suffix)
        if 'all' in methods or 'pca' in methods:
            self.visualize_with_pca(embeddings, labels, speaker_names, title_suffix)
        if 'all' in methods or 'tsne' in methods:
            self.visualize_with_tsne(embeddings, labels, speaker_names, title_suffix)
        
        print(f"\nVisualized speakers{title_suffix}:")
        for i, speaker in enumerate(speakers_list):
            count = labels.count(i)
            print(f"  Speaker {i}: {speaker} ({count} samples)")
        
        print(f"Visualization complete{title_suffix}!")
