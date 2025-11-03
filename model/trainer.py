import os
import torch
from tqdm import tqdm
from model import TotalLoss


class Trainer:
    def __init__(self, model, optimizer, device, checkpoint_dir='checkpoints'):
        """
        Args:
            model: MD-Net model
            optimizer: Optimizer (Adam recommended)
            device: Training device (cuda/cpu)
            checkpoint_dir: Directory to save checkpoints
        """
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.loss_fn = TotalLoss()

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Training history
        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader, use_supervision=False):
        """
        Train for one epoch

        Args:
            dataloader: Training data loader
            use_supervision: Whether to use ground truth supervision
        """
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'adversarial': 0.0,
            'color': 0.0,
            'mse': 0.0
        }

        num_batches = len(dataloader)

        progress_bar = tqdm(dataloader, desc='Training')

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            if use_supervision:
                underwater_imgs, clear_imgs = batch
                underwater_imgs = underwater_imgs.to(self.device)
                clear_imgs = clear_imgs.to(self.device)
                ground_truth = clear_imgs
            else:
                underwater_imgs = batch.to(self.device)
                ground_truth = None

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            j_out, i_rec = self.model(underwater_imgs)

            # Compute loss
            loss_dict = self.loss_fn(
                original_images=underwater_imgs,
                reconstructed_images=i_rec,
                enhanced_images=j_out,
                ground_truth=ground_truth,
                use_supervision=use_supervision
            )

            # Backward pass
            loss_dict['total_loss'].backward()
            self.optimizer.step()

            # Update metrics
            epoch_losses['total'] += loss_dict['total_loss'].item()
            epoch_losses['adversarial'] += loss_dict['adversarial_loss'].item()
            epoch_losses['color'] += loss_dict['color_loss'].item()
            epoch_losses['mse'] += loss_dict['mse_loss'].item()

            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss_dict['total_loss'].item():.4f}",
                'L_A': f"{loss_dict['adversarial_loss'].item():.4f}",
                'L_C': f"{loss_dict['color_loss'].item():.4f}",
                'L_M': f"{loss_dict['mse_loss'].item():.4f}"
            })

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches

        return epoch_losses

    def validate(self, dataloader, use_supervision=False):
        """
        Validate the model

        Args:
            dataloader: Validation data loader
            use_supervision: Whether to use ground truth for evaluation
        """
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'adversarial': 0.0,
            'color': 0.0,
            'mse': 0.0
        }

        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Validation'):
                # Move data to device
                if use_supervision:
                    underwater_imgs, clear_imgs = batch
                    underwater_imgs = underwater_imgs.to(self.device)
                    clear_imgs = clear_imgs.to(self.device)
                    ground_truth = clear_imgs
                else:
                    underwater_imgs = batch.to(self.device)
                    ground_truth = None

                # Forward pass
                j_out, i_rec = self.model(underwater_imgs)

                # Compute loss
                loss_dict = self.loss_fn(
                    original_images=underwater_imgs,
                    reconstructed_images=i_rec,
                    enhanced_images=j_out,
                    ground_truth=ground_truth,
                    use_supervision=use_supervision
                )

                # Update metrics
                val_losses['total'] += loss_dict['total_loss'].item()
                val_losses['adversarial'] += loss_dict['adversarial_loss'].item()
                val_losses['color'] += loss_dict['color_loss'].item()
                val_losses['mse'] += loss_dict['mse_loss'].item()

        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches

        return val_losses

    def train(self, train_loader, val_loader, num_epochs,
              use_supervision=False, save_interval=10):
        """
        Complete training loop

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            use_supervision: Whether to use ground truth supervision
            save_interval: Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Using supervision: {use_supervision}")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 30)

            # Train for one epoch
            train_losses = self.train_epoch(train_loader, use_supervision)
            self.train_losses.append(train_losses)

            # Validate
            val_losses = self.validate(val_loader, use_supervision)
            self.val_losses.append(val_losses)

            # Print epoch summary
            print(f"Train Loss: {train_losses['total']:.4f} "
                  f"(L_A: {train_losses['adversarial']:.4f}, "
                  f"L_C: {train_losses['color']:.4f}, "
                  f"L_M: {train_losses['mse']:.4f})")
            print(f"Val Loss: {val_losses['total']:.4f} "
                  f"(L_A: {val_losses['adversarial']:.4f}, "
                  f"L_C: {val_losses['color']:.4f}, "
                  f"L_M: {val_losses['mse']:.4f})")

            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, train_losses['total'])

    def save_checkpoint(self, epoch, loss):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        checkpoint_path = os.path.join(
            self.checkpoint_dir,
            f'mdnet_checkpoint_epoch_{epoch+1}.pth'
        )

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore training history if available
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            self.val_losses = checkpoint['val_losses']

        print(f"Checkpoint loaded from {checkpoint_path}")
        return checkpoint['epoch']
