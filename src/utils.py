import os
import re
import sys
from typing import Union, Tuple
from pathlib import Path
from src.logger import logging
from src.exception import CustomException

def validate_file_path(
        file_path: Union[str, Path],
        allowed_extensions: tuple = None,
        min_size_bytes: int = None,
        max_size_bytes: int = None,
        create_dir: bool = False
) -> bool:
    """
    Validate a file path with multiple checks including existence, permissions,
    file extension, size limits, and path security.

    Args:
        file_path (Union[str, Path]): Path to validate
        allowed_extensions (tuple, optional): Tuple of allowed file extensions (e.g., ('.jpg', '.png'))
        min_size_bytes (int, optional): Minimum file size in bytes
        max_size_bytes (int, optional): Maximum file size in bytes
        create_dir (bool): Whether to create parent directories if they don't exist

    Returns:
        bool: True if path is valid, False otherwise
    """
    try:
        # Convert to Path object for better path handling
        logging.info("Starting the validation")
        path = Path(file_path)

        # Security check: Prevent directory traversal attacks
        try:
            path = path.resolve()
            if not str(path).startswith(str(Path.cwd().resolve())):
                logging.info("Path points outside the current working directory")
                return False
        except Exception as e:
            logging.error(f"Error in path resolution: {str(e)}")
            raise CustomException(e, sys)

        # Check for invalid characters in path
        invalid_chars = '<>:"|?*'
        if any(char in str(path) for char in invalid_chars):
            logging.info(f"Path contains invalid characters: {invalid_chars}")
            return False

        # Validate parent directory
        parent_dir = path.parent
        if not parent_dir.exists():
            if create_dir:
                try:
                    parent_dir.mkdir(parents=True, exist_ok=True)
                    logging.info(f"Created directory: {parent_dir}")
                except Exception as e:
                    logging.error(f"Failed to create directory: {str(e)}")
                    raise CustomException(e, sys)
            else:
                logging.info(f"Parent directory does not exist: {parent_dir}")
                return False

        # Check if parent directory is writable
        if not os.access(str(parent_dir), os.W_OK):
            logging.info(f"Parent directory is not writable: {parent_dir}")
            return False

        # If file exists, perform additional checks
        if path.exists():
            # Check if it's actually a file (not a directory)
            if not path.is_file():
                logging.info(f"Path exists but is not a file: {path}")
                return False

            # Check file permissions
            if not os.access(str(path), os.R_OK):
                logging.info("File exists but is not readable")
                return False

            # Check file size if limits are specified
            file_size = path.stat().st_size
            if min_size_bytes is not None and file_size < min_size_bytes:
                logging.info(f"File size ({file_size} bytes) is below minimum ({min_size_bytes} bytes)")
                return False
            if max_size_bytes is not None and file_size > max_size_bytes:
                logging.info(f"File size ({file_size} bytes) exceeds maximum ({max_size_bytes} bytes)")
                return False

        # Validate extension if specified
        if allowed_extensions:
            if not path.suffix.lower() in allowed_extensions:
                logging.info(f"File extension {path.suffix} not in allowed extensions: {allowed_extensions}")
                return False

        logging.info("File path validation successful")
        return True

    except Exception as e:
        logging.error(f"Validation error: {str(e)}")
        raise CustomException(e, sys)