"""
WLS (Weighted Least Squares) Filter Post-Processing

Edge-aware smoothing filter that preserves edges while smoothing color transitions.
Currently implements the existing WLS filter from the original code.

Reference: Edge-Preserving Decompositions for Multi-Scale Tone and Detail Manipulation
           Farbman et al., SIGGRAPH 2008
"""

import torch
import cv2
import numpy as np
from typing import Optional


class WLSFilter:
    """Weighted Least Squares edge-aware filter."""

    def __init__(
        self,
        lambda_value: float = 500.0,
        sigma_color: float = 4.0,
        available: bool = None,
    ):
        """
        Args:
            lambda_value: Smoothing strength (higher = smoother)
            sigma_color: Color sensitivity
            available: Whether WLS filter is available (auto-detect if None)
        """
        self.lambda_value = lambda_value
        self.sigma_color = sigma_color

        # Check if WLS filter is available
        if available is None:
            self.available = hasattr(cv2.ximgproc, 'createFastGlobalSmootherFilter')
        else:
            self.available = available

        if not self.available:
            print("[WLSFilter] Warning: opencv-contrib-python not installed or WLS not available")
            print("Install with: pip install opencv-contrib-python")

    def match_to_reference(
        self,
        image: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        guide: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply WLS filtering.

        Args:
            image: Input image [H, W, 3] in range [0, 1]
            reference: Not used (for interface compatibility)
            guide: Optional guide image for edge-aware filtering (uses image if None)

        Returns:
            Filtered image [H, W, 3]
        """
        if not self.available:
            print("[WLSFilter] Skipping: WLS filter not available")
            return image

        device = image.device

        # Convert to numpy
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)

        if guide is None:
            # Use luminance as guide
            guide_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            guide_np = (guide.cpu().numpy() * 255).astype(np.uint8)
            if guide_np.ndim == 3:
                guide_np = cv2.cvtColor(guide_np, cv2.COLOR_RGB2GRAY)

        # Apply WLS filter
        try:
            wls = cv2.ximgproc.createFastGlobalSmootherFilter(
                guide_np,
                self.lambda_value,
                self.sigma_color
            )

            # Filter each channel
            filtered_np = np.zeros_like(img_np)
            for c in range(3):
                filtered_np[:, :, c] = wls.filter(img_np[:, :, c])

        except Exception as e:
            print(f"[WLSFilter] Filtering failed: {e}")
            return image

        # Convert back to tensor
        filtered = torch.from_numpy(filtered_np).float() / 255.0
        return filtered.to(device)

    def match_video_frames(
        self,
        frames: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        progress_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Apply WLS filtering to video frames.

        Args:
            frames: Video frames [N, H, W, 3]
            reference: Not used (for interface compatibility)
            progress_callback: Optional callback(current, total)

        Returns:
            Filtered frames [N, H, W, 3]
        """
        if not self.available:
            return frames

        num_frames = frames.shape[0]
        filtered_frames = []

        for i in range(num_frames):
            filtered = self.match_to_reference(frames[i], reference)
            filtered_frames.append(filtered)

            if progress_callback is not None:
                progress_callback(i + 1, num_frames)

        return torch.stack(filtered_frames, dim=0)


if __name__ == "__main__":
    print("Testing WLSFilter...")

    # Test
    test_img = torch.rand(256, 256, 3)
    wls = WLSFilter(lambda_value=500.0, sigma_color=4.0)

    if wls.available:
        filtered = wls.match_to_reference(test_img)
        print(f"✓ Filtered: {test_img.shape} -> {filtered.shape}")
    else:
        print("✗ WLS filter not available")
