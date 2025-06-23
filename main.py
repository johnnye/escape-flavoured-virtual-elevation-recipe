import logging
import os
import sys

from PySide6.QtWidgets import QApplication

from ui.file_selector import FileSelector


def main():
    # Configure logging - INFO level for normal use, DEBUG available via environment variable
    log_level = (
        logging.DEBUG
        if "DEBUG" in sys.argv or "VE_DEBUG" in os.environ
        else logging.INFO
    )

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("virtual_elevation.log")]
        + ([logging.StreamHandler(sys.stdout)] if log_level == logging.DEBUG else []),
    )

    logger = logging.getLogger(__name__)
    if log_level == logging.DEBUG:
        logger.info("Starting Virtual Elevation application with debug logging enabled")
    else:
        logger.info("Starting Virtual Elevation application")

    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Start with file selector window
    window = FileSelector()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
