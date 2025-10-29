"""I/O utilities for consistent artifact writing."""

import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


def ensure_dir(path: str) -> None:
    """Ensure directory exists, creating parents if needed."""
    dir_path = Path(path).parent
    dir_path.mkdir(parents=True, exist_ok=True)


def write_json(path: str, data: Dict[str, Any]) -> None:
    """
    Write dictionary to JSON file with error handling.
    
    Args:
        path: Path to JSON file
        data: Dictionary to write
        
    Raises:
        IOError: If file cannot be written
    """
    try:
        ensure_dir(path)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to write JSON to {path}: {e}")


def append_csv_row(
    path: str, 
    row: Dict[str, Any], 
    header_order: Optional[List[str]] = None
) -> None:
    """
    Append row to CSV file, creating header if file doesn't exist.
    
    Args:
        path: Path to CSV file
        row: Dictionary with row data
        header_order: Optional list of column names in desired order.
                     If None, uses sorted keys of row dict.
    
    Raises:
        IOError: If file cannot be written
    """
    try:
        ensure_dir(path)
        file_exists = Path(path).exists()
        
        if header_order is None:
            header_order = sorted(row.keys())
        
        mode = 'a' if file_exists else 'w'
        with open(path, mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header_order)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
    except Exception as e:
        raise IOError(f"Failed to append CSV row to {path}: {e}")

