import sys

from PySide6.QtWidgets import QApplication

from ui.file_selector import FileSelector


def main():
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Start with file selector window
    window = FileSelector()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
