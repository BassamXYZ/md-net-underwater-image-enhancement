import torch
import torch.nn as nn


class TotalLoss(nn.Module):
    def __init__(self, lambda_color=1.0, lambda_mse=1.0):
        """
        Total Loss Function
        Args:
            lambda_color: Weight for color correction loss
            lambda_mse: Weight for mean square error loss (if ground truth available)
        """
        super(TotalLoss, self).__init__()
        self.lambda_color = lambda_color
        self.lambda_mse = lambda_mse

        # Using MSE loss for adversarial reconstruction and supervised loss
        self.mse_loss = nn.MSELoss()

    def adversarial_reconstruction_loss(self, original_images, reconstructed_images):
        """
        Adversarial Reconstruction Loss (L_A)
        MSE between original and reconstructed images
        """
        return self.mse_loss(original_images, reconstructed_images)

    def color_correction_loss(self, enhanced_images):
        """
        Color Correction Loss (L_C)
        Based on gray world assumption - each channel mean should be 0.5
        """

        # Calculate mean for each color channel for each image in batch
        channel_means = torch.mean(enhanced_images, dim=[2, 3])

        # Compute L2 norm of (mean - 0.5) for each channel
        color_loss = torch.sum((channel_means - 0.5) ** 2, dim=1)
        color_loss = torch.mean(torch.sqrt(color_loss))

        return color_loss

    def mean_square_error_loss(self, predicted_radiance, ground_truth):
        """
        Mean Square Error Loss (L_M)
        Only used when ground truth is available
        """
        return self.mse_loss(predicted_radiance, ground_truth)

    def forward(self, original_images, reconstructed_images, enhanced_images,
                ground_truth=None, use_supervision=False):
        """
        Compute total loss

        Args:
            original_images: Original degraded underwater images (I)
            reconstructed_images: Reconstructed images from physical model (I')
            enhanced_images: Predicted scene radiance (J)
            ground_truth: Ground truth clear images (if available)
            use_supervision: Whether to use supervised loss L_M
        """
        # Adversarial reconstruction loss
        L_A = self.adversarial_reconstruction_loss(
            original_images, reconstructed_images)

        # Color correction loss
        L_C = self.color_correction_loss(enhanced_images)

        # Initialize MSE loss
        L_M = torch.tensor(0.0, device=original_images.device)

        # Use supervised loss if ground truth is available and supervision is enabled
        if use_supervision and ground_truth is not None:
            L_M = self.mean_square_error_loss(enhanced_images, ground_truth)

        # Total loss
        L_total = L_A + self.lambda_color * L_C + self.lambda_mse * L_M

        return {
            'total_loss': L_total,
            'adversarial_loss': L_A,
            'color_loss': L_C,
            'mse_loss': L_M,
            'loss_components': {
                'L_A': L_A.item(),
                'L_C': L_C.item(),
                'L_M': L_M.item() if use_supervision else 0.0
            }
        }
