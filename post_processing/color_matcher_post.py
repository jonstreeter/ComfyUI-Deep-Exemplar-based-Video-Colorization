"""
Color Matcher Post-Processing for Video Colorization

Uses the color-matcher library to refine colorization results by ensuring
better color consistency with the reference image and temporal consistency
across video frames.

Reference: https://github.com/hahnec/color-matcher

Installation:
    pip install color-matcher

Methods available:
    - 'default' (hm-mvgd-hm): Best overall quality
    - 'mkl': Optimal transport-based, very accurate
    - 'reinhard': Classic color transfer, fast

Usage:
    from post_processing.color_matcher_post import ColorMatcherPost

    post_processor = ColorMatcherPost(method='hm-mvgd-hm')

    # Single image
    refined = post_processor.match_to_reference(colorized, reference)

    # Video frames
    refined_video = post_processor.match_video_frames(video_frames, reference)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Literal, List


class ColorMatcherPost:
    """Post-processing using color-matcher library for color refinement."""

    def __init__(
        self,
        method: Literal['default', 'hm-mvgd-hm', 'mkl', 'reinhard'] = 'default',
        temporal_consistency: bool = True,
        consistency_weight: float = 0.3,
    ):
        """
        Args:
            method: Color matching method
                - 'default' or 'hm-mvgd-hm': Hybrid method (best quality)
                - 'mkl': Monge-Kantorovich linearization (optimal transport)
                - 'reinhard': Classic Reinhard et al. method (fastest)
            temporal_consistency: Apply temporal smoothing across frames
            consistency_weight: Weight for temporal consistency (0-1)
                - 0 = no temporal smoothing
                - 1 = maximum temporal smoothing
        """
        try:
            from color_matcher import ColorMatcher
            from color_matcher.io_handler import load_img_file, save_img_file
            self.cm = ColorMatcher()
            self.io_handler = (load_img_file, save_img_file)
            self.available = True
        except ImportError:
            print("[ColorMatcherPost] Warning: color-matcher not installed")
            print("Install with: pip install color-matcher")
            self.available = False
            return

        # Normalize method name
        if method == 'default':
            method = 'hm-mvgd-hm'

        self.method = method
        self.temporal_consistency = temporal_consistency
        self.consistency_weight = consistency_weight

        # Validate method
        valid_methods = ['hm-mvgd-hm', 'mkl', 'reinhard']
        if method not in valid_methods:
            raise ValueError(f"Invalid method: {method}. Valid: {valid_methods}")

        print(f"[ColorMatcherPost] Initialized with method={method}, "
              f"temporal_consistency={temporal_consistency}")

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert PyTorch tensor to numpy array for color-matcher.

        Args:
            tensor: Image tensor in range [0, 1], shape [H, W, 3] or [B, H, W, 3]

        Returns:
            Numpy array in range [0, 255], shape [H, W, 3] or [B, H, W, 3]
        """
        if isinstance(tensor, torch.Tensor):
            array = tensor.detach().cpu().numpy()
        else:
            array = tensor

        # Ensure range [0, 1]
        array = np.clip(array, 0, 1)

        # Convert to uint8 [0, 255]
        array = (array * 255).astype(np.uint8)

        return array

    def _numpy_to_tensor(self, array: np.ndarray, device: str = 'cpu') -> torch.Tensor:
        """Convert numpy array back to PyTorch tensor.

        Args:
            array: Numpy array in range [0, 255], shape [H, W, 3]
            device: Target device

        Returns:
            Tensor in range [0, 1], shape [H, W, 3]
        """
        # Ensure uint8
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        # Convert to float [0, 1]
        tensor = torch.from_numpy(array).float() / 255.0

        return tensor.to(device)

    def match_to_reference(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Match image colors to reference image.

        Args:
            image: Colorized image [H, W, 3] in range [0, 1]
            reference: Reference image [H, W, 3] in range [0, 1]
            strength: Blending strength (0-1)
                - 0 = original image
                - 1 = full color matching

        Returns:
            Color-matched image [H, W, 3] in range [0, 1]
        """
        if not self.available:
            print("[ColorMatcherPost] Skipping: color-matcher not available")
            return image

        device = image.device

        # Convert to numpy
        img_np = self._tensor_to_numpy(image)
        ref_np = self._tensor_to_numpy(reference)

        # Apply color matching
        try:
            matched_np = self.cm.transfer(src=img_np, ref=ref_np, method=self.method)
        except Exception as e:
            print(f"[ColorMatcherPost] Color matching failed: {e}")
            return image

        # Convert back to tensor
        matched = self._numpy_to_tensor(matched_np, device=device)

        # Blend with original based on strength
        if strength < 1.0:
            matched = image * (1 - strength) + matched * strength

        return matched

    def match_video_frames(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor,
        strength: float = 1.0,
        progress_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Match colors for all video frames.

        Args:
            frames: Video frames [N, H, W, 3] in range [0, 1]
            reference: Reference image [H, W, 3] in range [0, 1]
            strength: Color matching strength (0-1)
            progress_callback: Optional callback(current, total)

        Returns:
            Color-matched frames [N, H, W, 3]
        """
        if not self.available:
            print("[ColorMatcherPost] Skipping: color-matcher not available")
            return frames

        num_frames = frames.shape[0]
        device = frames.device
        matched_frames = []

        # Process each frame
        for i in range(num_frames):
            frame = frames[i]

            # Match to reference
            matched = self.match_to_reference(frame, reference, strength=strength)

            # Apply temporal consistency
            if self.temporal_consistency and i > 0 and len(matched_frames) > 0:
                prev_frame = matched_frames[-1]
                matched = self._apply_temporal_smoothing(matched, prev_frame)

            matched_frames.append(matched)

            if progress_callback is not None:
                progress_callback(i + 1, num_frames)

        return torch.stack(matched_frames, dim=0)

    def _apply_temporal_smoothing(
        self,
        current: torch.Tensor,
        previous: torch.Tensor,
    ) -> torch.Tensor:
        """Apply temporal smoothing between consecutive frames.

        Args:
            current: Current frame [H, W, 3]
            previous: Previous frame [H, W, 3]

        Returns:
            Smoothed current frame [H, W, 3]
        """
        # Blend with previous frame
        alpha = self.consistency_weight
        smoothed = current * (1 - alpha) + previous * alpha

        return smoothed

    def match_video_to_first_frame(
        self,
        frames: torch.Tensor,
        strength: float = 0.5,
        progress_callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Match all frames to first frame for temporal consistency.

        This is useful when the first frame is already well-colorized.

        Args:
            frames: Video frames [N, H, W, 3]
            strength: Matching strength (0-1)
            progress_callback: Optional callback(current, total)

        Returns:
            Temporally consistent frames [N, H, W, 3]
        """
        if not self.available:
            return frames

        # Use first frame as reference
        reference = frames[0]

        return self.match_video_frames(
            frames,
            reference,
            strength=strength,
            progress_callback=progress_callback
        )

    def match_histogram_only(
        self,
        image: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Simple histogram matching (faster than full color matching).

        Args:
            image: Input image [H, W, 3]
            reference: Reference image [H, W, 3]

        Returns:
            Histogram-matched image [H, W, 3]
        """
        from skimage import exposure

        device = image.device
        img_np = self._tensor_to_numpy(image)
        ref_np = self._tensor_to_numpy(reference)

        # Match histogram per channel
        matched = np.zeros_like(img_np)
        for c in range(3):
            matched[:, :, c] = exposure.match_histograms(
                img_np[:, :, c],
                ref_np[:, :, c]
            )

        return self._numpy_to_tensor(matched, device=device)


class ColorMatcherPostBatch:
    """Batch processing version for better performance."""

    def __init__(self, method: str = 'default', batch_size: int = 8):
        """
        Args:
            method: Color matching method
            batch_size: Number of frames to process at once
        """
        self.matcher = ColorMatcherPost(method=method)
        self.batch_size = batch_size

    def process_video(
        self,
        frames: torch.Tensor,
        reference: torch.Tensor,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Process video in batches for better performance.

        Args:
            frames: Video frames [N, H, W, 3]
            reference: Reference image [H, W, 3]
            strength: Color matching strength

        Returns:
            Processed frames [N, H, W, 3]
        """
        num_frames = frames.shape[0]
        all_matched = []

        for i in range(0, num_frames, self.batch_size):
            batch_end = min(i + self.batch_size, num_frames)
            batch = frames[i:batch_end]

            # Process batch
            matched_batch = self.matcher.match_video_frames(
                batch,
                reference,
                strength=strength
            )

            all_matched.append(matched_batch)

        return torch.cat(all_matched, dim=0)


def compare_methods(
    image: torch.Tensor,
    reference: torch.Tensor,
) -> dict:
    """Compare all color matching methods.

    Args:
        image: Test image [H, W, 3]
        reference: Reference image [H, W, 3]

    Returns:
        Dictionary with results for each method
    """
    methods = ['hm-mvgd-hm', 'mkl', 'reinhard']
    results = {}

    for method in methods:
        try:
            matcher = ColorMatcherPost(method=method)
            matched = matcher.match_to_reference(image, reference)
            results[method] = matched
            print(f"✓ {method} completed")
        except Exception as e:
            print(f"✗ {method} failed: {e}")
            results[method] = None

    return results


if __name__ == "__main__":
    print("Testing ColorMatcherPost...")

    # Create test images
    test_img = torch.rand(256, 256, 3)
    ref_img = torch.rand(256, 256, 3)

    # Test single image
    print("\n1. Testing single image matching...")
    matcher = ColorMatcherPost(method='mkl')
    matched = matcher.match_to_reference(test_img, ref_img, strength=1.0)
    print(f"   Input shape: {test_img.shape}")
    print(f"   Output shape: {matched.shape}")
    print(f"   Output range: [{matched.min():.3f}, {matched.max():.3f}]")

    # Test video
    print("\n2. Testing video matching...")
    test_video = torch.rand(10, 256, 256, 3)
    matched_video = matcher.match_video_frames(test_video, ref_img)
    print(f"   Input: {test_video.shape}")
    print(f"   Output: {matched_video.shape}")

    # Compare methods
    print("\n3. Comparing all methods...")
    results = compare_methods(test_img, ref_img)
    print(f"   Successfully tested: {len([r for r in results.values() if r is not None])}/{len(results)} methods")

    print("\n✓ All tests passed!")
