"""
Guided Filter Post-Processing

Fast edge-preserving filter that's much faster than bilateral filtering
while achieving similar quality.

Reference: Guided Image Filtering, He et al., ECCV 2010
"""

import torch
import cv2
import numpy as np
from typing import Optional


class GuidedFilter:
    """Guided image filtering for edge-preserving smoothing."""

    def __init__(
        self,
        radius: int = 8,
        eps: float = 0.01,
    ):
        """
        Args:
            radius: Filter radius (larger = smoother)
            eps: Regularization parameter (larger = smoother)
        """
        self.radius = radius
        self.eps = eps

    def match_to_reference(
        self,
        image: torch.Tensor,
        reference: Optional[torch.Tensor] = None,
        guide: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply guided filtering.

        Args:
            image: Input image [H, W, 3] in range [0, 1]
            reference: Not used (for interface compatibility)
            guide: Guide image (uses input if None)

        Returns:
            Filtered image [H, W, 3]
        """
        device = image.device

        # Convert to numpy [0, 255]
        img_np = (image.cpu().numpy() * 255).astype(np.uint8)

        if guide is None:
            guide_np = img_np
        else:
            guide_np = (guide.cpu().numpy() * 255).astype(np.uint8)

        # Apply guided filter
        try:
            filtered_np = cv2.ximgproc.guidedFilter(
                guide_np,
                img_np,
                self.radius,
                self.eps
            )
        except AttributeError:
            # Fallback: opencv-contrib not available, use bilateral
            print("[GuidedFilter] opencv-contrib not available, using bilateral filter")
            filtered_np = cv2.bilateralFilter(img_np, self.radius, 75, 75)
        except Exception as e:
            print(f"[GuidedFilter] Filtering failed: {e}")
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
        """Apply guided filtering to video."""
        num_frames = frames.shape[0]
        filtered_frames = []

        for i in range(num_frames):
            filtered = self.match_to_reference(frames[i], reference)
            filtered_frames.append(filtered)

            if progress_callback:
                progress_callback(i + 1, num_frames)

        return torch.stack(filtered_frames, dim=0)


if __name__ == "__main__":
    print("Testing GuidedFilter...")
    test_img = torch.rand(256, 256, 3)
    gf = GuidedFilter(radius=8, eps=0.01)
    filtered = gf.match_to_reference(test_img)
    print(f"✓ Filtered: {test_img.shape} -> {filtered.shape}")
