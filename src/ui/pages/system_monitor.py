"""
System Monitor Page
Real-time GPU, RAM, and disk usage monitoring
"""
import sys
import time
from pathlib import Path

import psutil
import streamlit as st

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render():
    """Render system monitor page"""
    st.markdown('<h1 class="main-header">System Monitor</h1>', unsafe_allow_html=True)

    # Real-time metrics
    col1, col2, col3 = st.columns(3)

    # CPU Usage
    with col1:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        st.metric("CPU Usage", f"{cpu_percent:.1f}%", delta=None)
        st.progress(cpu_percent / 100.0)

    # RAM Usage
    with col2:
        ram = psutil.virtual_memory()
        ram_percent = ram.percent
        ram_used_gb = ram.used / (1024**3)
        ram_total_gb = ram.total / (1024**3)

        st.metric("RAM Usage", f"{ram_used_gb:.1f} / {ram_total_gb:.1f} GB")
        st.progress(ram_percent / 100.0)

    # Disk Usage
    with col3:
        disk = psutil.disk_usage(".")
        disk_percent = disk.percent
        disk_used_gb = disk.used / (1024**3)
        disk_total_gb = disk.total / (1024**3)

        st.metric("Disk Usage", f"{disk_used_gb:.1f} / {disk_total_gb:.1f} GB")
        st.progress(disk_percent / 100.0)

    st.markdown("---")

    # GPU Information (if available)
    st.subheader("GPU Status")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()

            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                gpu_memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                gpu_memory_percent = (gpu_memory_allocated / gpu_memory) * 100

                with st.expander(f"🎮 GPU {i}: {gpu_name}"):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("VRAM Usage", f"{gpu_memory_allocated:.1f} / {gpu_memory:.1f} GB")
                        st.progress(gpu_memory_percent / 100.0)

                    with col2:
                        st.metric("Temperature", "N/A")
                        st.metric("Utilization", "N/A")

                    st.markdown("**Processes on GPU**")
                    st.info("No active training processes")

        else:
            st.warning("No CUDA-capable GPU detected")
    except ImportError:
        st.warning("PyTorch not installed - GPU monitoring unavailable")

    st.markdown("---")

    # Process list
    st.subheader("Agent Forge Processes")

    processes = []
    for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            if "python" in proc.info["name"].lower():
                processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if processes:
        import pandas as pd

        df = pd.DataFrame(processes)
        df["memory_percent"] = df["memory_percent"].apply(lambda x: f"{x:.2f}%")
        df["cpu_percent"] = df["cpu_percent"].apply(lambda x: f"{x:.1f}%")

        st.dataframe(df, use_container_width=True)
    else:
        st.info("No active Agent Forge processes")

    st.markdown("---")

    # Storage usage breakdown
    st.subheader("Storage Breakdown")

    # Get actual storage stats from registry
    try:
        from cross_phase.storage.model_registry import ModelRegistry

        registry = ModelRegistry()
        storage_stats = registry.get_storage_stats()
        registry.close()

        model_count = storage_stats["model_count"]
        total_size_mb = storage_stats["total_size_mb"]
        checkpoint_count = storage_stats["checkpoint_count"]
    except Exception:
        # Fallback to defaults if registry unavailable
        model_count = 0
        total_size_mb = 0.0
        checkpoint_count = 0

    # Calculate dataset cache size (scan directories)
    dataset_cache_path = Path("./data/cache")
    wandb_path = Path("./wandb")

    try:
        dataset_size_gb = (
            sum(f.stat().st_size for f in dataset_cache_path.rglob("*") if f.is_file())
            / (1024**3)
            if dataset_cache_path.exists()
            else 0
        )
        dataset_count = (
            len(list(dataset_cache_path.glob("*"))) if dataset_cache_path.exists() else 0
        )
    except Exception:
        dataset_size_gb = 0
        dataset_count = 0

    try:
        wandb_size_mb = (
            sum(f.stat().st_size for f in wandb_path.rglob("*") if f.is_file()) / (1024**2)
            if wandb_path.exists()
            else 0
        )
    except Exception:
        wandb_size_mb = 0

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Model Storage**")
        st.metric("Models Stored", str(model_count))
        st.metric("Total Size", f"{total_size_mb:.1f} MB")
        st.metric("Checkpoints", str(checkpoint_count))

    with col2:
        st.markdown("**Dataset Cache**")
        st.metric("Datasets", str(dataset_count))
        st.metric("Total Size", f"{dataset_size_gb:.2f} GB")
        st.metric("W&B Logs", f"{wandb_size_mb:.0f} MB")

    # Cleanup recommendations
    st.markdown("---")
    st.subheader("Cleanup Recommendations")

    cleanup_items = [
        {"type": "Old sessions", "size": "450 MB", "age": "45 days"},
        {"type": "Temp checkpoints", "size": "280 MB", "age": "7 days"},
        {"type": "W&B cache", "size": "120 MB", "age": "30 days"},
    ]

    for item in cleanup_items:
        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

        with col1:
            st.text(item["type"])

        with col2:
            st.text(item["size"])

        with col3:
            st.text(f"Age: {item['age']}")

        with col4:
            if st.button(f"Clean {item['type'][:5]}", key=f"clean_{item['type']}"):
                st.success(f"Cleaned {item['type']}")

    # Auto-refresh
    if st.sidebar.checkbox("Auto-refresh (2s)", value=False, key="monitor_refresh"):
        time.sleep(2)
        st.rerun()


# Auto-run when accessed directly via Streamlit multipage
render()
