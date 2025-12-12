"""Utility functions for model comparison scripts."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional


def here():
    """Get the directory containing this script or current working directory."""
    try:
        if os.getcwd() == "/workspace":
            return Path("/workspace")
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def _get_latest_from_numbered_files(folder, files, index_from_end=0, exact_step=None):
    """Helper function to get latest file from numbered files (e.g., 0_rna_2025-01-01-12-00-00.h5ad, 1_rna.h5ad)."""

    def get_step_number(filename):
        # Extract step number from filename like "3_rna_2025-01-01-12-00-00.h5ad" or "3_rna.h5ad"
        step_part = filename.split("_")[0]
        try:
            return int(step_part)
        except ValueError:
            # Fallback to modification time for files that don't follow the pattern
            mtime = os.path.getmtime(os.path.join(folder, filename))
            return int(mtime)

    # Filter by exact step if specified
    if exact_step is not None:
        filtered_files = []
        for f in files:
            step_num = get_step_number(f)
            if isinstance(step_num, int) and step_num == exact_step:
                filtered_files.append(f)
        files = filtered_files

    if not files:
        return None

    # Sort by timestamp within the same step (latest first)
    def get_timestamp_from_filename_helper(filename):
        parts = filename.split("_")
        if len(parts) >= 3:
            # Join timestamp parts: "2025-01-01-12-00-00"
            timestamp_str = "_".join(parts[2:]).replace(".h5ad", "")
            return timestamp_str
        else:
            # Fallback to modification time
            mtime = os.path.getmtime(os.path.join(folder, filename))
            return str(int(mtime))

    files.sort(key=get_timestamp_from_filename_helper, reverse=True)
    latest_file = files[index_from_end]
    return os.path.join(folder, latest_file)


def get_latest_file(
    folder, prefix, index_from_end=0, dataset_name=None, exact_step=None
) -> Optional[str]:
    """
    Get the latest file with given prefix from folder.
    With new naming convention (e.g., 0_rna_2025-01-01-12-00-00.h5ad), finds files from exact step.

    Args:
        folder: Base folder to search in
        prefix: File prefix to match (e.g., "rna", "protein")
        index_from_end: Index from end of sorted files (0 = latest/highest number)
        dataset_name: Optional dataset name to search in specific subdirectory
        exact_step: Exact step number to load (e.g., if exact_step=2, will ONLY load step 2 files)

    Returns:
        Path to the latest matching file, or None if not found
    """
    base_folder = Path(folder)

    # If dataset_name is provided, search in dataset-specific subdirectory first
    if dataset_name:
        dataset_folder = base_folder / dataset_name
        if dataset_folder.exists():
            files = [f for f in os.listdir(dataset_folder) if f.endswith(".h5ad")]
            # Filter files that match the pattern: step_prefix_timestamp.h5ad or step_prefix.h5ad
            pattern_files = []
            for f in files:
                # Check if file matches NEW pattern: number_prefix_timestamp.h5ad or number_prefix.h5ad
                # Must start with a digit followed by underscore
                if f[0].isdigit() and "_" in f:
                    parts = f.split("_")
                    if len(parts) >= 2:  # At least step_prefix
                        step_part = parts[0]
                        prefix_part = parts[1]
                        if step_part.isdigit() and prefix_part == prefix:
                            pattern_files.append(f)
            if pattern_files:
                return _get_latest_from_numbered_files(
                    dataset_folder, pattern_files, index_from_end, exact_step
                )

    # Search in all subdirectories and main folder
    all_files = []

    # Check main folder
    if base_folder.exists():
        main_files = [f for f in os.listdir(base_folder) if f.endswith(".h5ad")]
        for f in main_files:
            # Check if file matches NEW pattern: number_prefix_timestamp.h5ad or number_prefix.h5ad
            # Must start with a digit followed by underscore
            if f[0].isdigit() and "_" in f:
                parts = f.split("_")
                if len(parts) >= 2:
                    step_part = parts[0]
                    prefix_part = parts[1]
                    if step_part.isdigit() and prefix_part == prefix:
                        all_files.append((base_folder, f))

    # Check subdirectories (dataset-specific folders)
    if base_folder.exists():
        for subdir in base_folder.iterdir():
            if subdir.is_dir():
                sub_files = [f for f in os.listdir(subdir) if f.endswith(".h5ad")]
                for f in sub_files:
                    # Check if file matches NEW pattern: number_prefix_timestamp.h5ad or number_prefix.h5ad
                    # Must start with a digit followed by underscore
                    if f[0].isdigit() and "_" in f:
                        parts = f.split("_")
                        if len(parts) >= 2:
                            step_part = parts[0]
                            prefix_part = parts[1]
                            if step_part.isdigit() and prefix_part == prefix:
                                all_files.append((subdir, f))

    if not all_files:
        return None

    # Filter by exact step if specified
    if exact_step is not None:
        filtered_files = []
        for folder_path, filename in all_files:
            step_part = filename.split("_")[0]
            try:
                step_num = int(step_part)
                if step_num == exact_step:
                    filtered_files.append((folder_path, filename))
            except ValueError:
                # Skip files that don't follow the pattern when exact_step is specified
                pass
        all_files = filtered_files

    if not all_files:
        return None

    # Sort all files by timestamp within the same step (latest first)
    def get_timestamp_from_filename(folder_file_tuple):
        folder_path, filename = folder_file_tuple
        # Extract timestamp from filename like "3_rna_2025-01-01-12-00-00.h5ad"
        parts = filename.split("_")
        if len(parts) >= 3:
            # Join timestamp parts: "2025-01-01-12-00-00"
            timestamp_str = "_".join(parts[2:]).replace(".h5ad", "")
            return timestamp_str
        else:
            # Fallback to modification time
            mtime = os.path.getmtime(folder_path / filename)
            return str(int(mtime))

    all_files.sort(key=get_timestamp_from_filename, reverse=True)

    # Get the requested file
    folder_path, latest_file = all_files[index_from_end]
    full_path = folder_path / latest_file

    # Print file information
    file_time = datetime.fromtimestamp(os.path.getmtime(full_path))
    time_diff = datetime.now() - file_time

    if time_diff.days > 0:
        print(
            f"{full_path} was created {time_diff.days} days, {time_diff.seconds//3600} hours, {(time_diff.seconds%3600)//60} minutes ago"
        )
    elif time_diff.seconds > 3600:
        print(
            f"{full_path} was created {time_diff.seconds//3600} hours, {(time_diff.seconds%3600)//60} minutes ago"
        )
    else:
        print(f"{full_path} was created {time_diff.seconds} seconds ago")

    return str(full_path)

