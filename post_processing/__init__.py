"""
Post-Processing Module for Video Colorization

Provides various post-processing filters and refinement techniques to improve
colorization quality and temporal consistency.

Available processors:
    - ColorMatcherPost: Color matching using color-matcher library
    - WLSFilter: Weighted Least Squares edge-aware smoothing
    - GuidedFilter: Fast edge-preserving filter
    - BilateralFilter: Bilateral filtering
    - TemporalStabilizer: Optical flow-based temporal stabilization

Usage:
    from post_processing import get_post_processor

    processor = get_post_processor('color_matcher', method='mkl')
    refined = processor.process(colorized_frames, reference)
"""

from typing import Literal, Optional, Dict, Any
import torch
import torch.nn as nn


def get_post_processor(
    processor_type: Literal['color_matcher', 'wls', 'guided', 'bilateral', 'none'] = 'color_matcher',
    **kwargs
) -> nn.Module:
    """Get a post-processor by name.

    Args:
        processor_type: Type of post-processor
            - 'color_matcher': Color matching (best for color consistency)
            - 'wls': Weighted Least Squares filter (edge-aware smoothing)
            - 'guided': Guided filter (fast edge-preserving)
            - 'bilateral': Bilateral filter (noise reduction)
            - 'none': No post-processing
        **kwargs: Additional arguments for the processor

    Returns:
        Post-processor instance

    Example:
        >>> processor = get_post_processor('color_matcher', method='mkl')
        >>> refined = processor.match_to_reference(colorized, reference)
    """
    processor_type = processor_type.lower()

    if processor_type == 'color_matcher':
        from .color_matcher_post import ColorMatcherPost
        return ColorMatcherPost(**kwargs)

    elif processor_type == 'wls':
        from .wls_filter import WLSFilter
        return WLSFilter(**kwargs)

    elif processor_type == 'guided':
        from .guided_filter import GuidedFilter
        return GuidedFilter(**kwargs)

    elif processor_type == 'bilateral':
        from .bilateral_filter import BilateralFilter
        return BilateralFilter(**kwargs)

    elif processor_type == 'none':
        return NoOpProcessor()

    else:
        raise ValueError(
            f"Unknown processor type: {processor_type}. "
            f"Valid: color_matcher, wls, guided, bilateral, none"
        )


class NoOpProcessor:
    """No-op processor that returns input unchanged."""

    def match_to_reference(self, image, reference, **kwargs):
        return image

    def match_video_frames(self, frames, reference, **kwargs):
        return frames

    def process(self, *args, **kwargs):
        return args[0] if args else None


def list_available_processors() -> list:
    """List all available post-processors.

    Returns:
        List of processor names
    """
    processors = []

    try:
        from .color_matcher_post import ColorMatcherPost
        processors.append('color_matcher')
    except ImportError:
        pass

    try:
        from .wls_filter import WLSFilter
        processors.append('wls')
    except ImportError:
        pass

    try:
        from .guided_filter import GuidedFilter
        processors.append('guided')
    except ImportError:
        pass

    try:
        from .bilateral_filter import BilateralFilter
        processors.append('bilateral')
    except ImportError:
        pass

    processors.append('none')

    return processors


def get_processor_info(processor_type: str) -> Dict[str, Any]:
    """Get information about a post-processor.

    Args:
        processor_type: Name of processor

    Returns:
        Dictionary with processor metadata
    """
    info = {
        'color_matcher': {
            'name': 'Color Matcher',
            'description': 'Match colors to reference using optimal transport or histogram methods',
            'speed': 4,  # 1-5 scale
            'quality': 5,
            'temporal_aware': True,
            'requires': ['color-matcher'],
            'best_for': 'Ensuring color consistency with reference image'
        },
        'wls': {
            'name': 'WLS Filter',
            'description': 'Weighted Least Squares edge-aware smoothing',
            'speed': 3,
            'quality': 4,
            'temporal_aware': False,
            'requires': ['opencv-contrib-python'],
            'best_for': 'Edge-aware color smoothing'
        },
        'guided': {
            'name': 'Guided Filter',
            'description': 'Fast edge-preserving filter',
            'speed': 5,
            'quality': 4,
            'temporal_aware': False,
            'requires': ['opencv-python'],
            'best_for': 'Fast edge-preserving smoothing'
        },
        'bilateral': {
            'name': 'Bilateral Filter',
            'description': 'Bilateral noise reduction filter',
            'speed': 4,
            'quality': 3,
            'temporal_aware': False,
            'requires': ['opencv-python'],
            'best_for': 'Noise reduction while preserving edges'
        },
        'none': {
            'name': 'No Post-Processing',
            'description': 'Skip post-processing',
            'speed': 5,
            'quality': 0,
            'temporal_aware': False,
            'requires': [],
            'best_for': 'When raw output is sufficient'
        }
    }

    return info.get(processor_type, {})


def compare_processors(
    image: torch.Tensor,
    reference: torch.Tensor,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compare all available post-processors.

    Args:
        image: Test image [H, W, 3]
        reference: Reference image [H, W, 3]
        verbose: Print comparison info

    Returns:
        Dictionary mapping processor names to results
    """
    available = list_available_processors()
    results = {}

    if verbose:
        print("\n" + "="*80)
        print("Post-Processor Comparison")
        print("="*80)

    for proc_type in available:
        if proc_type == 'none':
            continue

        try:
            processor = get_post_processor(proc_type)
            result = processor.match_to_reference(image, reference)
            results[proc_type] = result

            if verbose:
                info = get_processor_info(proc_type)
                speed = "⚡" * info.get('speed', 0)
                quality = "⭐" * info.get('quality', 0)
                print(f"\n{info.get('name', proc_type)}")
                print(f"  Speed: {speed}")
                print(f"  Quality: {quality}")
                print(f"  Best for: {info.get('best_for', 'N/A')}")
                print(f"  ✓ Success")

        except Exception as e:
            if verbose:
                print(f"\n{proc_type}")
                print(f"  ✗ Failed: {e}")

    if verbose:
        print("\n" + "="*80)
        print(f"Successfully tested: {len(results)}/{len(available)-1} processors")
        print("="*80 + "\n")

    return results


# Convenience exports
__all__ = [
    'get_post_processor',
    'list_available_processors',
    'get_processor_info',
    'compare_processors',
    'NoOpProcessor',
]


if __name__ == "__main__":
    print("Post-Processing Module Demo")
    print("="*80)

    # List available
    print("\nAvailable processors:")
    for proc in list_available_processors():
        print(f"  - {proc}")

    # Show info
    print("\nProcessor details:")
    for proc in ['color_matcher', 'wls', 'guided']:
        info = get_processor_info(proc)
        if info:
            print(f"\n{info['name']}:")
            print(f"  {info['description']}")
            print(f"  Best for: {info['best_for']}")

    # Test loading
    print("\n" + "="*80)
    print("Testing processor loading...")
    try:
        processor = get_post_processor('color_matcher')
        print("✓ Successfully loaded color_matcher")
    except Exception as e:
        print(f"✗ Failed to load: {e}")

    print("\n" + "="*80)
