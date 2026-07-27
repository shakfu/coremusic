#!/usr/bin/env python3
"""Sequential with Logging."""

# --8<-- [start:example]
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def batch_process_sequential(input_dir, output_dir, processor_func):
    """Process all audio files in directory sequentially."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_files = list(input_path.glob("*.wav"))
    total = len(audio_files)

    logger.info(f"Processing {total} files...")

    for i, input_file in enumerate(audio_files, 1):
        output_file = output_path / input_file.name

        try:
            processor_func(str(input_file), str(output_file))
            logger.info(f"[{i}/{total}] Processed: {input_file.name}")
        except Exception as e:
            logger.error(f"[{i}/{total}] Failed: {input_file.name} - {e}")

    logger.info("Batch processing complete")
# --8<-- [end:example]
