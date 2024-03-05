"""Hacky script to occupy and then release RAM in order to flush disk cache."""

import psutil
import gc
from tqdm import tqdm


def get_available_ram():
    """
    Get the total available RAM in bytes.
    """
    return psutil.virtual_memory().available


def occupy_ram(keep_unoccupied: int = 3 * 1024**3):
    """
    Occupy the specified amount of RAM.

    Args:
        keep_unoccupied: The amount of RAM to keep unoccupied in bytes (default: 3GB).
    """
    print(f"Occupying RAM...")
    chunk_size = 2 * (1024**3)  # Size of each chunk (2GB)
    occupied_memory = []
    occupied = 0
    with tqdm(total=get_available_ram() - keep_unoccupied, desc="Occupying RAM", unit="B", unit_scale=True) as pbar:
        while (free := get_available_ram() - keep_unoccupied) > 0:
            # Adjust the chunk size if the remaining amount is less than 2GB
            current_chunk_size = min(chunk_size, free)
            # Allocate memory and append it to the list
            occupied_memory.append(bytearray(current_chunk_size))
            occupied += current_chunk_size
            pbar.total = free + occupied
            pbar.update(current_chunk_size)


def release_ram():
    """
    Release the occupied RAM.
    """
    print("Releasing RAM...")
    gc.collect()  # Force garbage collection to free up memory


def ram_summary():
    vmem = psutil.virtual_memory()
    print("=== RAM Summary ===")
    print(f"Available RAM: {vmem.available / (1024**3):.2f} GB")
    print(f"Cached:        {vmem.cached / (1024**3):.2f} GB")
    print(f"Buffers:       {vmem.buffers / (1024**3):.2f} GB")
    print(f"Total RAM:     {vmem.total / (1024**3):.2f} GB")
    print("===================")


if __name__ == "__main__":
    try:
        ram_summary()
        occupy_ram()
    finally:
        release_ram()
        print("RAM released.")
        ram_summary()
