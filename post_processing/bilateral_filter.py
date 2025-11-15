"""
Bilateral Filter Post-Processing

Classic bilateral filtering for edge-preserving noise reduction.

Reference: Bilateral Filtering for Gray and Color Images, Tomasi & Manduchi, ICCV 1998
"""

import torch
import cv2
import numpy as np
from typing import Optional


class BilateralFilter:
    """Bilateral filtering for edge-preserving smoothing."""

    def __init__(
        self,
        d: int = 9,
        sigma_color: float = 75,
        sigma_space: float = 75,
    ):
        """
        Args:
            d: Diameter of each pixel neighborhood
            sigma_color: Filter sigma in the color space
            sigma_space: Filter sigma in the coordinate space
        """
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def match_to_reference(
        self,
        image: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply bilateral filtering.

        Args:
            image: Input image [H, W, 3] in range [0, 1]
            reference: Not used (for interface compatibility)

        Returns:
            Filtered image [H, W, 3]
        """
        device = image.device

        # Convert to numpy
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)

        # Apply bilateral filter
        try:
            filtered_np = cv2.bilateralFilter(
                img_np,
                self.d,
                self.sigma_color,
                self.sigma_space
            )
        except Exception as e:
            print(f"[BilateralFilter] Filtering failed: {e}")
            return image

        # Convert back
        filtered = torch.from_numpy(filtered_np).float() / 255.0
        return filtered.to(device)

    def match_video_frames(
        self,
        frames: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        progress_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Apply bilateral filtering to video."""
        num_frames = frames.shape[0]
        filtered_frames = []

        for i in range(num_frames):
            filtered = self.match_to_reference(frames[i], reference)
            filtered_frames.append(filtered)

            if progress_callback:
                progress_callback(i + 1, num_frames)

        return torch.stack(filtered_frames, dim=0)


if __name__ == "__main__":
    print("Testing BilateralFilter...")
    test_img = torch.rand(256, 256, 3)
    bf = BilateralFilter(d=9, sigma_color=75, sigma_space=75)
    filtered = bf.match_to_reference(test_img)
    print(f"✓ Filtered: {test_img.shape} -> {filtered.shape}")
