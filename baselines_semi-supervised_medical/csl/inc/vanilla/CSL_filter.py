import torch
import torch.nn.functional as F
import logging

class CSLPseudoLabelFilter:
    """
    Confidence Separable Learning (CSL) for adaptive pseudo-label filtering.
    Replaces fixed threshold methods with distribution-based clustering.
    """
    
    def __init__(self, alpha=2.0, logger=None):
        """
        Args:
            alpha: Temperature parameter for Gaussian weighting (default: 2.0)
            logger: Logger instance for tracking filtering process
        """
        self.alpha = alpha
        self.logger = logger or logging.getLogger(__name__)
        
        # Statistics for monitoring
        self.stats = {
            'total_pixels': 0,
            'valid_pixels': 0,
            'high_conf_pixels': 0,
            'cluster_separations': []
        }
    
    @torch.no_grad()
    def filter_pseudo_labels(self, logits, ignore_mask=None):
        """
        Main filtering function using CSL approach.
        
        Args:
            logits: (B, C, H, W, D) or (B, C, H, W) - model predictions
            ignore_mask: (B, H, W, D) or (B, H, W) - pixels to ignore (255 = ignore)
        
        Returns:
            pseudo_labels: (B, H, W, D) or (B, H, W) - filtered pseudo-labels
            weights: (B, H, W, D) or (B, H, W) - confidence weights [0, 1]
            metrics: dict with filtering statistics
        """
        self.logger.info("=" * 50)
        self.logger.info("CSL Pseudo-Label Filtering Started")
        
        # Get predictions
        probs = F.softmax(logits, dim=1)
        pseudo_labels = torch.argmax(probs, dim=1)
        
        # Handle ignore mask
        if ignore_mask is None:
            ignore_mask = torch.zeros_like(pseudo_labels)
        valid_mask = (ignore_mask != 255)
        
        # Step 1: Compute max confidence and residual variance
        self.logger.info("Step 1: Computing confidence metrics...")
        max_conf, res_var = self._get_max_confidence_and_residual_variance(
            probs, valid_mask
        )
        
        # Step 2: Perform spectral clustering per image
        self.logger.info("Step 2: Performing spectral clustering...")
        weights = self._compute_adaptive_weights(
            max_conf, res_var, valid_mask
        )
        
        # Step 3: Generate metrics
        metrics = self._compute_metrics(weights, valid_mask)
        
        self.logger.info(f"Filtering complete: {metrics['high_conf_ratio']:.2%} high-confidence pixels")
        self.logger.info("=" * 50)
        
        return pseudo_labels, weights, metrics
    
    @torch.no_grad()
    def _get_max_confidence_and_residual_variance(self, predictions, valid_mask):
        """
        Compute max confidence and residual variance for each pixel.
        
        Args:
            predictions: (B, C, H, W, D) or (B, C, H, W) - probability distributions
            valid_mask: (B, H, W, D) or (B, H, W) - valid pixels
        
        Returns:
            max_confidence: (B, H, W, D) or (B, H, W)
            residual_variance: (B, H, W, D) or (B, H, W)
        """
        # Expand valid mask
        valid_mask_expanded = valid_mask.unsqueeze(1).expand_as(predictions)
        
        # Set invalid pixels to NaN
        predictions_masked = torch.where(
            valid_mask_expanded, 
            predictions, 
            torch.tensor(float('nan'), device=predictions.device)
        )
        
        # Get max confidence
        max_confidence, max_indices = torch.max(predictions_masked, dim=1)
        
        # Create one-hot mask for max class
        num_classes = predictions.shape[1]
        one_hot_max = F.one_hot(max_indices, num_classes=num_classes)
        one_hot_max = one_hot_max.permute(0, -1, *range(1, len(predictions.shape)-1))
        
        # Compute residual variance (variance of non-max classes)
        remaining_predictions = predictions_masked * (1 - one_hot_max)
        sum_remaining = torch.sum(remaining_predictions, dim=1)
        num_remaining = num_classes - 1
        mean_remaining = sum_remaining / num_remaining
        
        # Variance calculation
        diff = remaining_predictions - mean_remaining.unsqueeze(1)
        squared_diff = diff ** 2
        sum_squared_diff = torch.sum(squared_diff, dim=1)
        residual_variance = sum_squared_diff / num_remaining
        
        return max_confidence, residual_variance
    
    @torch.no_grad()
    def _compute_adaptive_weights(self, max_conf, res_var, valid_mask):
        """
        Compute adaptive weights using spectral clustering and z-score normalization.
        
        Args:
            max_conf: (B, H, W, D) or (B, H, W) - maximum confidence per pixel
            res_var: (B, H, W, D) or (B, H, W) - residual variance per pixel
            valid_mask: (B, H, W, D) or (B, H, W) - valid pixels
        
        Returns:
            weights: (B, H, W, D) or (B, H, W) - adaptive weights [0, 1]
        """
        batch_size = max_conf.shape[0]
        weights = torch.zeros_like(max_conf)
        
        for b in range(batch_size):
            # Extract features for this image
            features = torch.stack([max_conf[b], res_var[b]], dim=-1)
            features_flat = features.view(-1, 2)
            valid_flat = valid_mask[b].view(-1)
            valid_features = features_flat[valid_flat]
            
            if valid_features.size(0) == 0:
                self.logger.warning(f"Batch {b}: No valid pixels!")
                continue
            
            # Perform spectral clustering
            cluster_means, cluster_vars = self._spectral_clustering(valid_features)
            
            # Select "good" cluster (highest mean confidence)
            good_cluster_idx = torch.argmax(cluster_means[:, 0])
            conf_mean = cluster_means[good_cluster_idx, 0]
            res_mean = cluster_means[good_cluster_idx, 1]
            conf_var = cluster_vars[good_cluster_idx, 0]
            res_var_stat = cluster_vars[good_cluster_idx, 1]
            
            self.logger.debug(
                f"Batch {b}: Good cluster - conf_mean={conf_mean:.3f}, "
                f"res_mean={res_mean:.3f}, conf_var={conf_var:.3f}"
            )
            
            # Compute z-scores
            epsilon = 1e-8
            conf_z = (max_conf[b] - conf_mean) / torch.sqrt(conf_var + epsilon)
            res_z = (res_mean - res_var[b]) / torch.sqrt(res_var_stat + epsilon)
            
            # Compute Gaussian-like weights
            weight_conf = torch.exp(-(conf_z ** 2) / self.alpha)
            weight_res = torch.exp(-(res_z ** 2) / self.alpha)
            weight = weight_conf * weight_res
            
            # Set high-confidence pixels to weight 1.0
            confident_mask = (conf_z > 0) | (res_z > 0)
            weight = torch.where(confident_mask, torch.ones_like(weight), weight)
            
            # Apply valid mask
            weights[b] = torch.where(valid_mask[b], weight, torch.zeros_like(weight))
        
        return weights
    
    @torch.no_grad()
    def _spectral_clustering(self, features):
        """
        Perform spectral clustering using SVD to separate good/bad predictions.
        
        Args:
            features: (N, 2) - [max_confidence, residual_variance] for N valid pixels
        
        Returns:
            cluster_means: (2, 2) - mean of each cluster for each feature
            cluster_vars: (2, 2) - variance of each cluster for each feature
        """
        num_clusters = 2
        
        # Remove NaN values
        valid_mask = ~torch.isnan(features).any(dim=-1)
        features = features[valid_mask]
        
        if features.size(0) < num_clusters:
            # Fallback: return default statistics
            return (
                torch.tensor([[1.0, 0.0], [0.5, 0.5]], device=features.device),
                torch.tensor([[1.0, 1.0], [1.0, 1.0]], device=features.device)
            )
        
        # Perform SVD for dimensionality reduction
        U, S, Vt = torch.linalg.svd(features.T, full_matrices=False)
        eigvals = S ** 2
        idx = torch.argsort(-eigvals)
        eigvecs = Vt.T[:, idx[:num_clusters]]
        
        # Assign clusters based on eigenvector projections
        cluster_assignments = torch.argmax(torch.abs(eigvecs), dim=1)
        
        # Compute statistics for each cluster
        means = []
        vars = []
        for cluster_id in range(num_clusters):
            cluster_mask = cluster_assignments == cluster_id
            cluster_points = features[cluster_mask]
            
            if cluster_points.size(0) == 0:
                mean = torch.zeros(2, device=features.device)
                var = torch.ones(2, device=features.device)
            elif cluster_points.size(0) == 1:
                mean = cluster_points.squeeze(0)
                var = torch.ones(2, device=features.device)
            else:
                mean = cluster_points.mean(dim=0)
                var = cluster_points.var(dim=0, unbiased=True)
                var = torch.clamp(var, min=1e-6)  # Prevent zero variance
            
            means.append(mean)
            vars.append(var)
        
        return torch.stack(means), torch.stack(vars)
    
    @torch.no_grad()
    def _compute_metrics(self, weights, valid_mask):
        """Compute filtering statistics for logging."""
        total_valid = valid_mask.sum().item()
        high_conf = (weights > 0.9).sum().item()
        
        metrics = {
            'total_valid_pixels': total_valid,
            'high_conf_pixels': high_conf,
            'high_conf_ratio': high_conf / max(total_valid, 1),
            'mean_weight': weights[valid_mask].mean().item() if total_valid > 0 else 0.0,
            'median_weight': weights[valid_mask].median().item() if total_valid > 0 else 0.0
        }
        
        return metrics


# Integration Checklist Logger
class CSLIntegrationLogger:
    """Tracks CSL integration progress with checklist."""
    
    def __init__(self, logger):
        self.logger = logger
        self.checklist = {
            '1_max_conf_residual_var': False,
            '2_spectral_clustering': False,
            '3_adaptive_weighting': False,
            '4_loss_computation': False,
            '5_training_integration': False
        }
    
    def mark_complete(self, step):
        """Mark a step as complete."""
        if step in self.checklist:
            self.checklist[step] = True
            self.logger.info(f"✓ CSL Step Complete: {step}")
    
    def print_status(self):
        """Print checklist status."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("CSL INTEGRATION CHECKLIST")
        self.logger.info("=" * 60)
        for step, complete in self.checklist.items():
            status = "✓ DONE" if complete else "✗ TODO"
            self.logger.info(f"{status} | {step}")
        self.logger.info("=" * 60 + "\n")
