from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QSlider, QWidget


class SliderTextBox(QWidget):
    """
    A combined slider and text box widget that maintains synchronization between both controls.
    
    Features:
    - Slider updates text box when dragged
    - Text box updates slider when edited (with validation)
    - Supports both integer and float values
    - Enforces min/max boundaries
    - Emits valueChanged signal when value changes from either control
    """
    
    valueChanged = Signal(object)  # Emits the new value (int or float)
    
    def __init__(self, minimum=0, maximum=100, value=50, is_float=False, decimals=2, label=None):
        super().__init__()
        
        self.minimum = minimum
        self.maximum = maximum
        self.is_float = is_float
        self.decimals = decimals
        self._updating = False  # Prevent recursive updates
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Add label if provided
        if label:
            self.label = QLabel(label)
            layout.addWidget(self.label)
        else:
            self.label = None
        
        # Create slider
        self.slider = QSlider(Qt.Horizontal)
        if self.is_float:
            # For float values, use integer slider with scaled values
            self.scale_factor = 10 ** self.decimals
            self.slider.setMinimum(int(minimum * self.scale_factor))
            self.slider.setMaximum(int(maximum * self.scale_factor))
            self.slider.setValue(int(value * self.scale_factor))
        else:
            self.slider.setMinimum(int(minimum))
            self.slider.setMaximum(int(maximum))
            self.slider.setValue(int(value))
        
        # Create text box
        self.textbox = QLineEdit()
        self.textbox.setFixedWidth(80)
        
        # Set initial value
        self.set_value(value)
        
        # Connect signals
        self.slider.valueChanged.connect(self._on_slider_changed)
        self.textbox.editingFinished.connect(self._on_textbox_changed)
        
        # Add widgets to layout
        layout.addWidget(self.slider, 1)  # Slider takes most space
        layout.addWidget(self.textbox, 0)  # Text box fixed width
    
    def _on_slider_changed(self, slider_value):
        """Handle slider value change"""
        if self._updating:
            return
        
        self._updating = True
        try:
            if self.is_float:
                value = slider_value / self.scale_factor
                self.textbox.setText(f"{value:.{self.decimals}f}")
            else:
                value = slider_value
                self.textbox.setText(str(value))
            
            self.valueChanged.emit(value)
        finally:
            self._updating = False
    
    def _on_textbox_changed(self):
        """Handle text box value change"""
        if self._updating:
            return
        
        self._updating = True
        try:
            text = self.textbox.text().strip()
            if not text:
                return
            
            try:
                if self.is_float:
                    value = float(text)
                else:
                    value = int(text)
                
                # Clamp to boundaries
                value = max(self.minimum, min(self.maximum, value))
                
                # Update controls
                if self.is_float:
                    self.slider.setValue(int(value * self.scale_factor))
                    self.textbox.setText(f"{value:.{self.decimals}f}")
                else:
                    self.slider.setValue(int(value))
                    self.textbox.setText(str(value))
                
                self.valueChanged.emit(value)
                
            except ValueError:
                # Invalid input - revert to current slider value
                if self.is_float:
                    current_value = self.slider.value() / self.scale_factor
                    self.textbox.setText(f"{current_value:.{self.decimals}f}")
                else:
                    current_value = self.slider.value()
                    self.textbox.setText(str(current_value))
        finally:
            self._updating = False
    
    def set_value(self, value):
        """Set the value programmatically"""
        if self._updating:
            return
        
        self._updating = True
        try:
            # Clamp to boundaries
            value = max(self.minimum, min(self.maximum, value))
            
            if self.is_float:
                self.slider.setValue(int(value * self.scale_factor))
                self.textbox.setText(f"{value:.{self.decimals}f}")
            else:
                value = int(value)
                self.slider.setValue(value)
                self.textbox.setText(str(value))
        finally:
            self._updating = False
    
    def get_value(self):
        """Get the current value"""
        if self.is_float:
            return self.slider.value() / self.scale_factor
        else:
            return self.slider.value()
    
    def set_range(self, minimum, maximum):
        """Update the min/max range"""
        self.minimum = minimum
        self.maximum = maximum
        
        if self.is_float:
            self.slider.setMinimum(int(minimum * self.scale_factor))
            self.slider.setMaximum(int(maximum * self.scale_factor))
        else:
            self.slider.setMinimum(int(minimum))
            self.slider.setMaximum(int(maximum))
        
        # Clamp current value to new range
        current_value = self.get_value()
        if current_value < minimum or current_value > maximum:
            self.set_value(max(minimum, min(maximum, current_value)))
    
    def set_enabled(self, enabled):
        """Enable/disable the widget"""
        self.slider.setEnabled(enabled)
        self.textbox.setEnabled(enabled)
        if self.label:
            self.label.setEnabled(enabled)