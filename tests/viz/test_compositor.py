import pytest
import tempfile
from pathlib import Path
import yaml

from driftlock_choir_sim.viz.compositor import load_config, animate_split_screen

def test_compositor_headless():
    """Test compositor in headless mode with 3 frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Create minimal configs for testing
        baseline_cfg = {
            "sample_rate_hz": 5e6,
            "duration_s": 0.05,
            "tx": {"df_hz": 10e3, "m_carriers": 5},
            "channel": {"awgn_snr_db": 20.0},
            "truth": {"tau_s": 0.0},
            "movie": {"nfft": 256}  # Small for test
        }
        demo_cfg = {
            "sample_rate_hz": 5e6,
            "duration_s": 0.05,
            "tx": {"df_hz": 10e3, "m_carriers": 5},
            "channel": {"awgn_snr_db": 20.0},
            "truth": {"tau_s": 0.0},
            "movie": {"nfft": 256}
        }
        # Save configs to temp
        with open(tmp_path / "baseline.yaml", "w") as f:
            yaml.dump(baseline_cfg, f)
        with open(tmp_path / "demo.yaml", "w") as f:
            yaml.dump(demo_cfg, f)
        
        # Run with 3 frames
        animate_split_screen(baseline_cfg, demo_cfg, num_frames=3, seed=2025)
        
        # Check if output files exist
        output_dir = Path("driftlock_choir_sim/outputs/comparisons/side_by_side")
        assert (output_dir / "comparison.mp4").exists()
        assert (output_dir / "spritesheet.png").exists()
        
        # Basic check: files not empty
        assert (output_dir / "comparison.mp4").stat().st_size > 0
        assert (output_dir / "spritesheet.png").stat().st_size > 0

if __name__ == "__main__":
    test_compositor_headless()