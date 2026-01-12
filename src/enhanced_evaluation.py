"""
Enhanced Evaluation Module - Medium Task
Includes: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI, NMI
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score
)


def evaluate_clustering_comprehensive(features, clusters, true_labels, method_name):
    """
    Comprehensive evaluation of clustering with all metrics.
    
    Args:
        features: Feature vectors
        clusters: Cluster labels
        true_labels: Ground truth labels
        method_name: Name of the clustering method
    
    Returns:
        results: Dictionary of metric scores
    """
    results = {'Method': method_name}
    
    # Check if clustering is valid
    n_unique = len(set(clusters))
    if n_unique < 2:
        # Return NaN for invalid clustering
        results['Silhouette Score'] = np.nan
        results['Calinski-Harabasz'] = np.nan
        results['Davies-Bouldin'] = np.nan
        results['Adjusted Rand Index'] = np.nan
        results['Normalized Mutual Info'] = np.nan
        return results
    
    # Silhouette Score (unsupervised)
    try:
        results['Silhouette Score'] = silhouette_score(features, clusters)
    except Exception as e:
        results['Silhouette Score'] = np.nan
        print(f"  Warning: Silhouette score failed for {method_name}: {e}")
    
    # Calinski-Harabasz Index (unsupervised)
    try:
        results['Calinski-Harabasz'] = calinski_harabasz_score(features, clusters)
    except Exception as e:
        results['Calinski-Harabasz'] = np.nan
        print(f"  Warning: Calinski-Harabasz failed for {method_name}: {e}")
    
    # Davies-Bouldin Index (unsupervised, lower is better)
    try:
        results['Davies-Bouldin'] = davies_bouldin_score(features, clusters)
    except Exception as e:
        results['Davies-Bouldin'] = np.nan
        print(f"  Warning: Davies-Bouldin failed for {method_name}: {e}")
    
    # Adjusted Rand Index (supervised)
    try:
        results['Adjusted Rand Index'] = adjusted_rand_score(true_labels, clusters)
    except Exception as e:
        results['Adjusted Rand Index'] = np.nan
        print(f"  Warning: ARI failed for {method_name}: {e}")
    
    # Normalized Mutual Information (supervised)
    try:
        results['Normalized Mutual Info'] = normalized_mutual_info_score(true_labels, clusters)
    except Exception as e:
        results['Normalized Mutual Info'] = np.nan
        print(f"  Warning: NMI failed for {method_name}: {e}")
    
    return results


def compare_all_methods(clustering_results, true_labels):
    """
    Compare all clustering methods.
    
    Args:
        clustering_results: Dictionary of clustering results from AdvancedClusteringPipeline
        true_labels: Ground truth labels
    
    Returns:
        results_df: DataFrame with comparison results
    """
    results_list = []
    
    for method_key, result_dict in clustering_results.items():
        if result_dict['clusters'] is not None:
            features = result_dict['features']
            clusters = result_dict['clusters']
            name = result_dict['name']
            
            metrics = evaluate_clustering_comprehensive(features, clusters, true_labels, name)
            results_list.append(metrics)
    
    results_df = pd.DataFrame(results_list)
    
    return results_df


class EnhancedMetricsEvaluator:
    """
    Enhanced metrics evaluator with all clustering metrics.
    """
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = None
    
    def evaluate_all(self, clustering_results, true_labels):
        """
        Evaluate all clustering methods.
        
        Args:
            clustering_results: Dictionary of clustering results
            true_labels: Ground truth labels
        
        Returns:
            results_df: DataFrame with evaluation results
        """
        self.results = compare_all_methods(clustering_results, true_labels)
        return self.results
    
    def print_results(self):
        """Print evaluation results in formatted table."""
        if self.results is not None:
            print("\\n" + "="*120)
            print("COMPREHENSIVE EVALUATION RESULTS")
            print("="*120)
            print(self.results.to_string(index=False))
            print("="*120)
            print("\\nMetric Interpretation:")
            print("  Silhouette Score: Higher is better (range: -1 to 1)")
            print("  Calinski-Harabasz: Higher is better")
            print("  Davies-Bouldin: Lower is better")
            print("  Adjusted Rand Index: Higher is better (range: -1 to 1, 1 = perfect)")
            print("  Normalized Mutual Info: Higher is better (range: 0 to 1)")
            print("="*120)
        else:
            print("No results to display. Run evaluate_all() first.")
    
    def save_results(self, filepath):
        """
        Save results to CSV file.
        
        Args:
            filepath: Path to save CSV file
        """
        if self.results is not None:
            self.results.to_csv(filepath, index=False)
            print(f"\\n✅ Results saved to '{filepath}'")
        else:
            print("No results to save. Run evaluate_all() first.")
    
    def get_results(self):
        """Get evaluation results."""
        return self.results
    
    def get_best_method(self, metric='Silhouette Score'):
        """
        Get the best performing method for a given metric.
        
        Args:
            metric: Metric name to use for ranking
        
        Returns:
            best_method: Name of best performing method
            best_score: Score of best performing method
        """
        if self.results is None:
            return None, None
        
        # Handle metrics where lower is better
        if metric == 'Davies-Bouldin':
            idx = self.results[metric].idxmin()
        else:
            idx = self.results[metric].idxmax()
        
        best_method = self.results.loc[idx, 'Method']
        best_score = self.results.loc[idx, metric]
        
        return best_method, best_score
    
    def analyze_results(self):
        """
        Provide analysis of clustering results.
        
        Returns:
            analysis: Dictionary with key findings
        """
        if self.results is None:
            return None
        
        analysis = {}
        
        # Best methods for each metric
        metrics = ['Silhouette Score', 'Calinski-Harabasz', 'Adjusted Rand Index', 'Normalized Mutual Info']
        for metric in metrics:
            method, score = self.get_best_method(metric)
            analysis[f'Best_{metric}'] = {'method': method, 'score': score}
        
        # Davies-Bouldin (lower is better)
        method, score = self.get_best_method('Davies-Bouldin')
        analysis['Best_Davies-Bouldin'] = {'method': method, 'score': score}
        
        # Compare hybrid vs audio-only
        try:
            hybrid_sil = self.results[self.results['Method'].str.contains('Hybrid')]['Silhouette Score'].max()
            audio_sil = self.results[self.results['Method'].str.contains('Audio')]['Silhouette Score'].max()
            analysis['Hybrid_vs_Audio'] = {
                'hybrid_better': hybrid_sil > audio_sil,
                'hybrid_score': hybrid_sil,
                'audio_score': audio_sil
            }
        except:
            pass
        
        # Compare with baseline
        try:
            baseline_sil = self.results[self.results['Method'].str.contains('Baseline')]['Silhouette Score'].values[0]
            best_vae_sil = self.results[~self.results['Method'].str.contains('Baseline')]['Silhouette Score'].max()
            analysis['VAE_vs_Baseline'] = {
                'vae_better': best_vae_sil > baseline_sil,
                'vae_score': best_vae_sil,
                'baseline_score': baseline_sil
            }
        except:
            pass
        
        return analysis
    
    def print_analysis(self):
        """Print analysis of results."""
        analysis = self.analyze_results()
        
        if analysis is None:
            print("No analysis available. Run evaluate_all() first.")
            return
        
        print("\\n" + "="*80)
        print("CLUSTERING ANALYSIS")
        print("="*80)
        
        print("\\n1. Best Methods by Metric:")
        metrics = ['Silhouette Score', 'Calinski-Harabasz', 'Davies-Bouldin', 
                  'Adjusted Rand Index', 'Normalized Mutual Info']
        for metric in metrics:
            key = f'Best_{metric}'
            if key in analysis:
                info = analysis[key]
                print(f"   {metric}: {info['method']} (score: {info['score']:.4f})")
        
        if 'Hybrid_vs_Audio' in analysis:
            print("\\n2. Hybrid vs Audio-Only Features:")
            info = analysis['Hybrid_vs_Audio']
            if info['hybrid_better']:
                print(f"   ✓ Hybrid features outperform audio-only")
                print(f"     Hybrid: {info['hybrid_score']:.4f}, Audio: {info['audio_score']:.4f}")
            else:
                print(f"   • Audio-only features perform better")
                print(f"     Hybrid: {info['hybrid_score']:.4f}, Audio: {info['audio_score']:.4f}")
        
        if 'VAE_vs_Baseline' in analysis:
            print("\\n3. VAE vs PCA Baseline:")
            info = analysis['VAE_vs_Baseline']
            if info['vae_better']:
                print(f"   ✓ VAE significantly outperforms PCA baseline")
                print(f"     Best VAE: {info['vae_score']:.4f}, Baseline: {info['baseline_score']:.4f}")
            else:
                print(f"   • PCA baseline performs better")
                print(f"     Best VAE: {info['vae_score']:.4f}, Baseline: {info['baseline_score']:.4f}")
        
        print("="*80)
