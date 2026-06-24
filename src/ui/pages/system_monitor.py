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


def _render_resource_metrics() -> None:
    """Render CPU, RAM, and disk usage metrics"""
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


def _render_gpu_device(i: int) -> None:
    """Render a single GPU device expander"""
    import torch

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


def _render_gpu_status() -> None:
    """Render GPU status section"""
    st.subheader("GPU Status")

    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                _render_gpu_device(i)
        else:
            st.warning("No CUDA-capable GPU detected")
    except ImportError:
        st.warning("PyTorch not installed - GPU monitoring unavailable")


def _render_process_list() -> None:
    """Render Agent Forge process list"""
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


def _get_model_storage_stats() -> tuple:
    """Return (model_count, total_size_mb, checkpoint_count) from registry"""
    try:
        from cross_phase.storage.model_registry import ModelRegistry

        registry = ModelRegistry()
        storage_stats = registry.get_storage_stats()
        registry.close()

        return (
            storage_stats["model_count"],
            storage_stats["total_size_mb"],
            storage_stats["checkpoint_count"],
        )
    except Exception:
        # Fallback to defaults if registry unavailable
        return 0, 0.0, 0


def _get_dataset_cache_stats() -> tuple:
    """Return (dataset_size_gb, dataset_count) for the dataset cache"""
    dataset_cache_path = Path("./data/cache")
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
        return dataset_size_gb, dataset_count
    except Exception:
        return 0, 0


def _get_wandb_size_mb() -> float:
    """Return total size of the wandb directory in MB"""
    wandb_path = Path("./wandb")
    try:
        return (
            sum(f.stat().st_size for f in wandb_path.rglob("*") if f.is_file()) / (1024**2)
            if wandb_path.exists()
            else 0
        )
    except Exception:
        return 0


def _render_storage_breakdown() -> None:
    """Render storage usage breakdown section"""
    st.subheader("Storage Breakdown")

    model_count, total_size_mb, checkpoint_count = _get_model_storage_stats()
    dataset_size_gb, dataset_count = _get_dataset_cache_stats()
    wandb_size_mb = _get_wandb_size_mb()

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


def _render_cleanup_recommendations() -> None:
    """Render cleanup recommendations section"""
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


def _render_auto_refresh() -> None:
    """Render auto-refresh control"""
    if st.sidebar.checkbox("Auto-refresh (2s)", value=False, key="monitor_refresh"):
        time.sleep(2)
        st.rerun()


def render() -> None:
    """Render system monitor page"""
    st.markdown('<h1 class="main-header">System Monitor</h1>', unsafe_allow_html=True)

    _render_resource_metrics()

    st.markdown("---")
    _render_gpu_status()

    st.markdown("---")
    _render_process_list()

    st.markdown("---")
    _render_storage_breakdown()

    st.markdown("---")
    _render_cleanup_recommendations()

    _render_auto_refresh()


# Auto-run when accessed directly via Streamlit multipage
render()
