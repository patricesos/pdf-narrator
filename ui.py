# ui.py

import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from PyQt6.QtWidgets import QApplication, QFileDialog # Keep for file dialogs if preferred
import ttkbootstrap as tb
from ttkbootstrap.constants import *
import os
import sys
import threading
import time
import json
import subprocess # For opening folders cross-platform

# Keep these imports - assuming they exist and work
from extract import extract_book
from generate_audiobook_kokoro import (
    generate_audiobooks_kokoro,
    generate_audio_for_all_voices_kokoro,
    test_single_voice_kokoro,
    available_voices # Assuming this function is now in kokoro module
)

# --- Constants ---
CONFIG_FILE = "config.json"
DEFAULT_THEME = "flatly"
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Helper Classes ---

class LogRedirector:
    """Redirects stdout/stderr to a Tkinter widget."""
    def __init__(self, write_callback):
        self.write_callback = write_callback
        self.is_logging = False

    def write(self, message):
        if self.is_logging: return
        self.is_logging = True
        try:
            if message.strip():
                self.write_callback(message)
        finally:
            self.is_logging = False

    def flush(self):
        pass

# --- GUI Frames ---

class SourceFrame(tb.Frame):
    """Frame for PDF/EPUB source selection and extraction options."""
    def __init__(self, master, app, **kwargs):
        super().__init__(master, padding=(15, 10), **kwargs)
        self.app = app # Reference to the main AudiobookApp instance

        # Variables
        self.pdf_path = tk.StringVar()
        self.pdf_folder = tk.StringVar()
        self.manual_extracted_dir = tk.StringVar()
        self.extracted_text_dir = tk.StringVar() # Auto-populated output
        self.use_toc = tk.BooleanVar(value=True)
        self.extract_mode = tk.StringVar(value="chapters")
        self.source_option = tk.StringVar(value="single")

        self.grid_columnconfigure(1, weight=1) # Make entry fields expand

        # --- Source Selection ---
        source_lf = tb.Labelframe(self, text="Input Source", padding=15, bootstyle=INFO)
        source_lf.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=(0, 10))
        source_lf.grid_columnconfigure(1, weight=1)

        # Single Book
        self.single_rb = tb.Radiobutton(
            source_lf, text="Single Book (PDF/EPUB)", variable=self.source_option,
            value="single", command=self._update_ui, width=22
        )
        self.single_rb.grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        self.single_entry = tb.Entry(source_lf, textvariable=self.pdf_path, width=50)
        self.single_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.single_browse_btn = tb.Button(source_lf, text="Browse...", command=self._browse_single_pdf, width=10)
        self.single_browse_btn.grid(row=0, column=2, sticky="e", padx=(5, 0), pady=5)

        # Batch Books
        self.batch_rb = tb.Radiobutton(
            source_lf, text="Batch Folder (PDF/EPUB)", variable=self.source_option,
            value="batch", command=self._update_ui, width=22
        )
        self.batch_rb.grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
        self.batch_entry = tb.Entry(source_lf, textvariable=self.pdf_folder, width=50)
        self.batch_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.batch_browse_btn = tb.Button(source_lf, text="Browse...", command=self._browse_pdf_folder, width=10)
        self.batch_browse_btn.grid(row=1, column=2, sticky="e", padx=(5, 0), pady=5)

        # Use Existing Text
        self.skip_rb = tb.Radiobutton(
            source_lf, text="Existing Text Folder", variable=self.source_option,
            value="skip", command=self._update_ui, width=22
        )
        self.skip_rb.grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        self.skip_entry = tb.Entry(source_lf, textvariable=self.manual_extracted_dir, width=50)
        self.skip_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=5)
        self.skip_browse_btn = tb.Button(source_lf, text="Browse...", command=self._browse_extracted_folder, width=10)
        self.skip_browse_btn.grid(row=2, column=2, sticky="e", padx=(5, 0), pady=5)

        # --- Extraction Options (Only enabled if not 'skip') ---
        self.options_lf = tb.Labelframe(self, text="Extraction Options", padding=15, bootstyle=INFO)
        self.options_lf.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=10)

        self.toc_check = tb.Checkbutton(self.options_lf, text="Use TOC/Metadata (if available)", variable=self.use_toc)
        self.toc_check.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 10))

        tb.Label(self.options_lf, text="Extract by:").grid(row=1, column=0, sticky="w")
        mode_frame = tb.Frame(self.options_lf) # Frame to hold radio buttons
        mode_frame.grid(row=1, column=1, sticky="w")
        self.chapters_rb = tb.Radiobutton(mode_frame, text="Chapters/Sections", variable=self.extract_mode, value="chapters")
        self.chapters_rb.pack(side=LEFT, padx=(0, 15))
        self.whole_rb = tb.Radiobutton(mode_frame, text="Whole Book (Single File)", variable=self.extract_mode, value="whole")
        self.whole_rb.pack(side=LEFT)

        # --- Output Directory Display ---
        out_lf = tb.Labelframe(self, text="Extracted Text Output (Preview)", padding=15, bootstyle=SECONDARY)
        out_lf.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(10, 0))
        out_lf.grid_columnconfigure(0, weight=1)
        self.extracted_entry = tb.Entry(out_lf, textvariable=self.extracted_text_dir, state="readonly")
        self.extracted_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.open_extracted_btn = tb.Button(out_lf, text="Open", command=self._open_extracted_folder, width=8)
        self.open_extracted_btn.grid(row=0, column=1, sticky="e")


    def _update_ui(self):
        """Enable/disable controls based on the selected source option."""
        option = self.source_option.get()

        # Enable/disable entry fields and browse buttons
        self.single_entry.config(state=NORMAL if option == "single" else DISABLED)
        self.single_browse_btn.config(state=NORMAL if option == "single" else DISABLED)
        self.batch_entry.config(state=NORMAL if option == "batch" else DISABLED)
        self.batch_browse_btn.config(state=NORMAL if option == "batch" else DISABLED)
        self.skip_entry.config(state=NORMAL if option == "skip" else DISABLED)
        self.skip_browse_btn.config(state=NORMAL if option == "skip" else DISABLED)

        # Enable/disable extraction options frame
        options_state = NORMAL if option != "skip" else DISABLED
        for child in self.options_lf.winfo_children():
             # Handle nested frames like mode_frame
            if isinstance(child, (tb.Checkbutton, tb.Label, tb.Radiobutton)):
                child.config(state=options_state)
            elif isinstance(child, tb.Frame):
                 for grandchild in child.winfo_children():
                      if isinstance(grandchild, tb.Radiobutton):
                           grandchild.config(state=options_state)

        # Update output preview based on current selection (if valid)
        self._update_output_paths()

    def _update_output_paths(self, source_path=None, source_type=None):
        """Update preview paths for extracted text and audiobook output."""
        print(f"\n--- _update_output_paths called ---") # Log entry point
        print(f"  Initial Args: source_path='{source_path}', source_type='{source_type}'")

        option = source_type or self.source_option.get()
        path = source_path
        print(f"  Determined Option: '{option}'")

        if not path: # If called without specific path, get from current selection
            print(f"  Path not provided, attempting to get from state...")
            if option == "single": path = self.pdf_path.get()
            elif option == "batch": path = self.pdf_folder.get()
            elif option == "skip": path = self.manual_extracted_dir.get()
            print(f"  Path from state: '{path}'")
        else:
             print(f"  Path provided: '{path}'")


        base_name = ""
        extracted_dir = ""
        audio_dir = ""

        # Check if the determined path is valid before proceeding
        path_exists = path and os.path.exists(path)
        print(f"  Path exists check ('{path}'): {path_exists}")

        if path_exists:
            is_file = os.path.isfile(path)
            is_dir = os.path.isdir(path)
            print(f"  Path is file: {is_file}, Path is dir: {is_dir}")

            if option == "single" and is_file:
                base_name = os.path.splitext(os.path.basename(path))[0]
                print(f"  Mode 'single', extracted base_name: '{base_name}'")
            elif (option == "batch" or option == "skip") and is_dir:
                base_name = os.path.basename(path.rstrip(os.sep))
                print(f"  Mode '{option}', extracted base_name: '{base_name}'")
            else:
                 print(f"  Warning: Path type doesn't match mode '{option}'. IsFile={is_file}, IsDir={is_dir}")

        else:
             print(f"  Path is invalid or does not exist, cannot determine base_name.")


        if base_name:
            extracted_dir = os.path.join(PROJECT_DIR, "extracted_books", base_name)
            audio_dir = os.path.join(PROJECT_DIR, "audiobooks", base_name)
            print(f"  Calculated extracted_dir: '{extracted_dir}'")
            print(f"  Calculated audio_dir   : '{audio_dir}'")
        else:
             print(f"  Base_name is empty, output directories will be empty.")


        # Set the extracted text preview
        extracted_display_path = extracted_dir if option != "skip" else path if path_exists else ""
        self.extracted_text_dir.set(extracted_display_path)
        print(f"  Setting extracted text preview to: '{extracted_display_path}'")

        # Log before updating audio display
        print(f"  Updating Audio Directory Display to: '{audio_dir}'") # Corrected Log Message
        # Update the shared audio output dir in the app
        self.app.update_audio_output_dir_display(audio_dir)
        print(f"--- _update_output_paths finished ---\n") # Log exit point

    def _browse_file_or_folder(self, mode):
        """Generalized browse function."""
        qt_app = QApplication.instance() or QApplication(sys.argv)
        path = ""
        if mode == "single":
            path, _ = QFileDialog.getOpenFileName(
                None, "Select Book File", PROJECT_DIR,
                "Book Files (*.pdf *.epub *.txt *.html *.htm);;All Files (*.*)"
            )
            if path: self.pdf_path.set(path)
        elif mode == "batch":
            path = QFileDialog.getExistingDirectory(
                None, "Select Folder Containing Books", PROJECT_DIR
            )
            if path: self.pdf_folder.set(path)
        elif mode == "skip":
            path = QFileDialog.getExistingDirectory(
                None, "Select Existing Text Folder", PROJECT_DIR
            )
            if path: self.manual_extracted_dir.set(path)

        if path:
            self._update_output_paths(path, mode) # Update paths immediately after selection

    def _browse_single_pdf(self):
        self._browse_file_or_folder("single")

    def _browse_pdf_folder(self):
        self._browse_file_or_folder("batch")

    def _browse_extracted_folder(self):
        self._browse_file_or_folder("skip")

    def _open_extracted_folder(self):
        """Opens the displayed extracted text folder."""
        folder = self.extracted_text_dir.get()
        self.app.open_folder(folder)

    # --- Getters ---
    def get_config(self):
        return {
            "source_option": self.source_option.get(),
            "pdf_path": self.pdf_path.get(),
            "pdf_folder": self.pdf_folder.get(),
            "manual_extracted_dir": self.manual_extracted_dir.get(),
            "use_toc": self.use_toc.get(),
            "extract_mode": self.extract_mode.get(),
            # extracted_text_dir is derived, no need to save explicitly
        }

    def set_config(self, config):
        """Sets the frame's state based on a loaded configuration dictionary."""
        self.source_option.set(config.get("source_option", "single"))
        self.pdf_path.set(config.get("pdf_path", ""))
        self.pdf_folder.set(config.get("pdf_folder", ""))
        self.manual_extracted_dir.set(config.get("manual_extracted_dir", ""))
        self.use_toc.set(config.get("use_toc", True))
        self.extract_mode.set(config.get("extract_mode", "chapters"))

        # Update the UI elements to reflect the loaded mode (enable/disable fields)
        self._update_ui()

        # Determine path/mode from just-loaded config to pass explicitly
        loaded_option = self.source_option.get()
        loaded_path = ""
        if loaded_option == "single": loaded_path = self.pdf_path.get()
        elif loaded_option == "batch": loaded_path = self.pdf_folder.get()
        elif loaded_option == "skip": loaded_path = self.manual_extracted_dir.get()

        # Pass the explicitly determined values
        self._update_output_paths(source_path=loaded_path, source_type=loaded_option)


class AudioFrame(tb.Frame):
    """Frame for Kokoro voice selection and audiobook generation settings."""
    def __init__(self, master, app, **kwargs):
        super().__init__(master, padding=(15, 10), **kwargs)
        self.app = app

        # Variables
        self.voicepack = tk.StringVar() # No default, set from list
        self.chunk_size = tk.IntVar(value=510)
        self.chunk_size_display = tk.StringVar(value="510 (Small)") # For combobox
        self.audio_format = tk.StringVar(value=".wav")
        self.audio_format_display = tk.StringVar(value=".wav (High Quality)") # For combobox
        # Detect CUDA availability at runtime; default to CPU on macOS/where CUDA isn't available
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False
        default_device = "cuda" if cuda_available else "cpu"
        self.device = tk.StringVar(value=default_device)
        self.audio_output_dir = tk.StringVar() # Display only, set by app

        self.grid_columnconfigure(1, weight=1)

        # --- Voice Selection ---
        voice_lf = tb.Labelframe(self, text="Voice Selection (Kokoro)", padding=15, bootstyle=INFO)
        voice_lf.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        voice_lf.grid_columnconfigure(1, weight=1)

        tb.Label(voice_lf, text="Voice:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        self.voice_list = available_voices()
        self.voice_combo = tb.Combobox(
            voice_lf, textvariable=self.voicepack, values=self.voice_list,
            state="readonly", width=35
        )
        self.voice_combo.grid(row=0, column=1, sticky="ew", pady=5)
        if self.voice_list: # Set default selection
            default_voice = "am_liam" # Preferred default
            if default_voice in self.voice_list:
                 self.voice_combo.set(default_voice)
            else:
                 self.voice_combo.current(0) # Fallback to first voice
        self.voice_combo.bind("<<ComboboxSelected>>", self._check_voice_selection)

        # --- Generation Settings ---
        settings_lf = tb.Labelframe(self, text="Generation Settings", padding=15, bootstyle=INFO)
        settings_lf.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        settings_lf.grid_columnconfigure(1, weight=1)

        # Chunk Size
        tb.Label(settings_lf, text="Chunk Size:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=5)
        chunk_options = ["510 (Small)", "1020 (Medium)", "2040 (Large)"]
        self.chunk_combo = tb.Combobox(
            settings_lf, textvariable=self.chunk_size_display, values=chunk_options,
            state="readonly", width=18
        )
        self.chunk_combo.grid(row=0, column=1, sticky="w", pady=5)
        self.chunk_combo.bind("<<ComboboxSelected>>", self._update_chunk_size)

        # Audio Format
        tb.Label(settings_lf, text="Output Format:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=5)
        formats = [".wav (High Quality)", ".mp3 (Smaller Size)"]
        self.format_combo = tb.Combobox(
            settings_lf, textvariable=self.audio_format_display, values=formats,
            state="readonly", width=18
        )
        self.format_combo.grid(row=1, column=1, sticky="w", pady=5)
        self.format_combo.bind("<<ComboboxSelected>>", self._update_audio_format)

        # Device Selection
        tb.Label(settings_lf, text="Device:").grid(row=2, column=0, sticky="w", padx=(0, 10), pady=5)
        device_frame = tb.Frame(settings_lf)
        device_frame.grid(row=2, column=1, sticky="w", pady=5)
        self.gpu_rb = tb.Radiobutton(device_frame, text="GPU (CUDA)", variable=self.device, value="cuda")
        self.gpu_rb.pack(side=LEFT, padx=(0, 15))
        self.cpu_rb = tb.Radiobutton(device_frame, text="CPU", variable=self.device, value="cpu")
        self.cpu_rb.pack(side=LEFT)
        # Disable GPU option if CUDA is not available
        if not cuda_available:
            try:
                self.gpu_rb.config(state=DISABLED)
            except Exception:
                pass

        # --- Output Directory Display ---
        out_lf = tb.Labelframe(self, text="Audiobook Output (Preview)", padding=15, bootstyle=SECONDARY)
        out_lf.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 0))
        out_lf.grid_columnconfigure(0, weight=1)
        self.output_entry = tb.Entry(out_lf, textvariable=self.audio_output_dir, state="readonly")
        self.output_entry.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.open_output_btn = tb.Button(out_lf, text="Open", command=self._open_audio_folder, width=8)
        self.open_output_btn.grid(row=0, column=1, sticky="e")

    def _check_voice_selection(self, event=None):
        # Placeholder if validation is needed in the future
        pass

    def _update_chunk_size(self, event):
        selection = self.chunk_size_display.get()
        if "Small" in selection: self.chunk_size.set(510)
        elif "Medium" in selection: self.chunk_size.set(1020)
        elif "Large" in selection: self.chunk_size.set(2040)

    def _update_audio_format(self, event):
        selection = self.audio_format_display.get()
        if "wav" in selection: self.audio_format.set(".wav")
        elif "mp3" in selection: self.audio_format.set(".mp3")

    def update_output_display(self, path):
        """Updates the read-only output directory field."""
        self.audio_output_dir.set(path)
        self.open_output_btn.config(state=NORMAL if path and os.path.isdir(path) else DISABLED)


    def _open_audio_folder(self):
        """Opens the displayed audiobook output folder."""
        folder = self.audio_output_dir.get()
        self.app.open_folder(folder)

    # --- Getters ---
    def get_config(self):
         # Find the actual display value matching the internal value
        chunk_map = {510: "510 (Small)", 1020: "1020 (Medium)", 2040: "2040 (Large)"}
        format_map = {".wav": ".wav (High Quality)", ".mp3": ".mp3 (Smaller Size)"}
        self.chunk_size_display.set(chunk_map.get(self.chunk_size.get(), "510 (Small)"))
        self.audio_format_display.set(format_map.get(self.audio_format.get(), ".wav (High Quality)"))

        return {
            "voicepack": self.voicepack.get(),
            "chunk_size": self.chunk_size.get(),
            "audio_format": self.audio_format.get(),
            "device": self.device.get(),
        }

    def set_config(self, config):
        selected_voice = config.get("voicepack", "")
        if selected_voice in self.voice_list:
             self.voicepack.set(selected_voice)
        elif self.voice_list:
             self.voicepack.set(self.voice_list[0]) # Fallback

        self.chunk_size.set(config.get("chunk_size", 510))
        self.audio_format.set(config.get("audio_format", ".wav"))
        self.device.set(config.get("device", "cuda"))

        # Update display variables based on loaded internal values
        chunk_map = {510: "510 (Small)", 1020: "1020 (Medium)", 2040: "2040 (Large)"}
        format_map = {".wav": ".wav (High Quality)", ".mp3": ".mp3 (Smaller Size)"}
        self.chunk_size_display.set(chunk_map.get(self.chunk_size.get(), "510 (Small)"))
        self.audio_format_display.set(format_map.get(self.audio_format.get(), ".wav (High Quality)"))
    def get_device(self):
        """Returns the currently selected device string ('cuda' or 'cpu')."""
        return self.device.get() # Retrieve the value from the tk.StringVar

class ControlFrame(tb.Frame):
    """Frame for process control buttons and primary status display."""
    def __init__(self, master, app, **kwargs):
        super().__init__(master, padding=(10, 5), **kwargs)
        self.app = app

        # Variables
        self.status_text = tk.StringVar(value="Ready")
        self.current_action_text = tk.StringVar(value="-") # e.g., "Extracting:", "Generating Audio:"
        self.current_file_text = tk.StringVar(value="") # e.g., "Chapter 1.txt" or "book.pdf"
        self.progress_count_text = tk.StringVar(value="") # e.g., "(1 of 10)"
        self.estimated_time_text = tk.StringVar(value="Est. Time: N/A")

        self.grid_columnconfigure(1, weight=1) # Make status expand

        # --- Buttons ---
        button_frame = tb.Frame(self)
        button_frame.grid(row=0, column=0, padx=(0, 20), pady=5, sticky="ns")

        self.start_button = tb.Button(
            button_frame, text="▶ Start", bootstyle=SUCCESS,
            command=self.app.start_process, width=10
        )
        self.start_button.pack(side=LEFT, padx=5)

        self.pause_button = tb.Button(
            button_frame, text="⏸ Pause", bootstyle=WARNING,
            command=self.app.pause_process, state=DISABLED, width=10
        )
        self.pause_button.pack(side=LEFT, padx=5)

        self.resume_button = tb.Button(
            button_frame, text="⏯ Resume", bootstyle=INFO,
            command=self.app.resume_process, state=DISABLED, width=10
        )
        self.resume_button.pack(side=LEFT, padx=5)

        self.cancel_button = tb.Button(
            button_frame, text="⏹ Cancel", bootstyle=DANGER,
            command=self.app.cancel_process, state=DISABLED, width=10
        )
        self.cancel_button.pack(side=LEFT, padx=5)

        # --- Status Display ---
        status_frame = tb.Frame(self)
        status_frame.grid(row=0, column=1, sticky="ew", pady=5)
        status_frame.grid_columnconfigure(1, weight=1) # Make file name expand

        # Row 1: Overall Status & Time
        tb.Label(status_frame, text="Status:", width=7).grid(row=0, column=0, sticky="w", padx=(0, 5))
        tb.Label(status_frame, textvariable=self.status_text, font="-weight bold").grid(row=0, column=1, sticky="w")
        tb.Label(status_frame, textvariable=self.estimated_time_text).grid(row=0, column=2, sticky="e", padx=10)

        # Row 2: Current Action & File
        tb.Label(status_frame, textvariable=self.current_action_text, width=15).grid(row=1, column=0, sticky="w", padx=(0,5))
        tb.Label(status_frame, textvariable=self.current_file_text).grid(row=1, column=1, sticky="w")
        tb.Label(status_frame, textvariable=self.progress_count_text).grid(row=1, column=2, sticky="e", padx=10)

    # --- Update Methods (called by app) ---
    def update_status(self, status=None, action=None, file=None, count_str=None, est_time_str=None):
        """Updates status display elements. Only updates fields that are provided."""
        if status is not None:
            self.status_text.set(status)
        if action is not None:
            self.current_action_text.set(action)
        # Handle file update carefully - allow setting it back to empty
        if file is not None:
             self.current_file_text.set(file)
        if count_str is not None:
            self.progress_count_text.set(count_str)
        if est_time_str is not None:
            self.estimated_time_text.set(est_time_str)

    def set_button_states(self, running=False, paused=False):
        self.start_button.config(state=DISABLED if running else NORMAL)
        self.pause_button.config(state=NORMAL if running and not paused else DISABLED)
        self.resume_button.config(state=NORMAL if running and paused else DISABLED)
        self.cancel_button.config(state=NORMAL if running else DISABLED)


class ProgressFrame(tb.Frame):
    """Frame for detailed progress bars and logs."""
    def __init__(self, master, app, **kwargs):
        super().__init__(master, padding=(15, 10), **kwargs)
        self.app = app

        # Variables
        self.extract_progress = tk.DoubleVar(value=0.0)
        self.audio_progress = tk.DoubleVar(value=0.0)
        self.extract_percent = tk.StringVar(value="0%")
        self.audio_percent = tk.StringVar(value="0%")

        self.grid_columnconfigure(0, weight=1) # Make progress bars expand
        self.grid_rowconfigure(1, weight=1)    # Make log area expand

        # --- Progress Bars ---
        progress_lf = tb.Labelframe(self, text="Detailed Progress", padding=15, bootstyle=INFO)
        progress_lf.grid(row=0, column=0, sticky="ew", padx=10, pady=(0, 10))
        progress_lf.grid_columnconfigure(1, weight=1) # Make bars expand inside labelframe

        # Extract Progress
        tb.Label(progress_lf, text="📄 Text Extraction:", width=18).grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.extract_bar = tb.Progressbar(progress_lf, variable=self.extract_progress, maximum=100, bootstyle=INFO)
        self.extract_bar.grid(row=0, column=1, sticky="ew", pady=5)
        tb.Label(progress_lf, textvariable=self.extract_percent, width=5).grid(row=0, column=2, sticky="e", padx=(10, 0))

        # Audio Progress
        tb.Label(progress_lf, text="🔊 Audio Generation:", width=18).grid(row=1, column=0, sticky="w", padx=(0, 10))
        self.audio_bar = tb.Progressbar(progress_lf, variable=self.audio_progress, maximum=100, bootstyle=SUCCESS)
        self.audio_bar.grid(row=1, column=1, sticky="ew", pady=5)
        tb.Label(progress_lf, textvariable=self.audio_percent, width=5).grid(row=1, column=2, sticky="e", padx=(10, 0))

        # --- Logs ---
        log_lf = tb.Labelframe(self, text="Process Logs", padding=(15, 10), bootstyle=SECONDARY)
        log_lf.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 0))
        log_lf.grid_columnconfigure(0, weight=1)
        log_lf.grid_rowconfigure(1, weight=1) # Make text widget expand

        clear_log_btn = tb.Button(log_lf, text="Clear Logs", bootstyle="secondary-outline", width=10,
                                  command=lambda: self.log_text.delete(1.0, tk.END))
        clear_log_btn.grid(row=0, column=0, sticky="e", pady=(0, 5))

        self.log_text = scrolledtext.ScrolledText(log_lf, height=10, wrap=tk.WORD, bd=0, relief="flat")
        self.log_text.grid(row=1, column=0, sticky="nsew")

        # Redirect stdout/stderr
        sys.stdout = LogRedirector(self._log_message_gui)
        sys.stderr = LogRedirector(self._log_message_gui)

    def _log_message_gui(self, msg):
        """Safely appends message to the log text widget from any thread."""
        try:
            # Use 'after' to ensure GUI update happens in the main thread
            self.app.after(0, self._insert_log, msg)
        except Exception as e:
            # Fallback to console if GUI is closing or unavailable
            print(f"(Log Error: {e}) {msg}", file=sys.__stderr__)

    def _insert_log(self, msg):
        """Inserts message and scrolls the log widget (called via 'after')."""
        if self.log_text.winfo_exists(): # Check if widget still exists
            self.log_text.insert(tk.END, msg.strip() + "\n")
            self.log_text.see(tk.END)

    # --- Update Methods (called by app) ---
    def update_progress(self, extract_val=None, audio_val=None):
        if extract_val is not None:
            self.extract_progress.set(extract_val)
            self.extract_percent.set(f"{int(extract_val)}%")
        if audio_val is not None:
            self.audio_progress.set(audio_val)
            self.audio_percent.set(f"{int(audio_val)}%")

    def reset_progress(self):
        self.update_progress(0, 0)


class VoiceTestFrame(tb.Frame):
    """Frame for testing TTS voices with sample text."""
    def __init__(self, master, app, **kwargs):
        super().__init__(master, padding=(15, 10), **kwargs)
        self.app = app
        self.test_thread = None
        self.cancellation_flag = False
        self.pause_event = threading.Event() # For potential future pause/resume in test
        self.pause_event.set()

        # Variables
        self.test_text = tk.StringVar(value="This is a sample text to test the selected voice.")
        self.test_mode = tk.StringVar(value="single")
        self.selected_voice = tk.StringVar() # Set from available voices
        self.test_output_dir = tk.StringVar(value=os.path.join(PROJECT_DIR, "voice_tests"))
        self.progress_var = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="Ready to test")
        self.progress_label_text = tk.StringVar(value="")

        os.makedirs(self.test_output_dir.get(), exist_ok=True)

        self.grid_columnconfigure(0, weight=1)

        # --- Test Mode ---
        mode_lf = tb.Labelframe(self, text="Test Mode", padding=15, bootstyle=INFO)
        mode_lf.grid(row=0, column=0, sticky="ew", padx=10, pady=(0, 10))
        tb.Radiobutton(
            mode_lf, text="Test Single Voice", variable=self.test_mode,
            value="single", command=self._update_ui
        ).pack(side=LEFT, padx=(0, 20))
        tb.Radiobutton(
            mode_lf, text="Test All Available Voices", variable=self.test_mode,
            value="all", command=self._update_ui
        ).pack(side=LEFT)

        # --- Voice Selection (Single Mode Only) ---
        self.voice_lf = tb.Labelframe(self, text="Voice Selection", padding=15, bootstyle=INFO)
        # self.voice_lf is gridded in _update_ui
        self.voice_lf.grid_columnconfigure(1, weight=1)
        tb.Label(self.voice_lf, text="Select Voice:").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.voice_list = available_voices()
        self.voice_combo = tb.Combobox(
            self.voice_lf, textvariable=self.selected_voice, values=self.voice_list,
            state="readonly", width=35
        )
        self.voice_combo.grid(row=0, column=1, sticky="ew")
        if self.voice_list: # Set default
             default_voice = "am_liam"
             if default_voice in self.voice_list: self.voice_combo.set(default_voice)
             else: self.voice_combo.current(0)

        # --- Sample Text ---
        text_lf = tb.Labelframe(self, text="Sample Text", padding=(15, 10), bootstyle=SECONDARY)
        text_lf.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        text_lf.grid_columnconfigure(0, weight=1)
        text_lf.grid_rowconfigure(0, weight=1) # Allow text box to grow slightly if needed
        self.text_input = scrolledtext.ScrolledText(text_lf, height=5, wrap=tk.WORD, bd=0)
        self.text_input.grid(row=0, column=0, sticky="ew")
        self.text_input.insert(tk.END, self.test_text.get())

        # --- Output & Controls ---
        control_lf = tb.Labelframe(self, text="Output & Controls", padding=15, bootstyle=SECONDARY)
        control_lf.grid(row=3, column=0, sticky="ew", padx=10, pady=10)
        control_lf.grid_columnconfigure(1, weight=1) # Make status expand

        # Output Dir
        tb.Label(control_lf, text="Test Output:").grid(row=0, column=0, sticky="w", padx=(0, 10), pady=(0, 10))
        out_frame = tb.Frame(control_lf) # Frame for entry and button
        out_frame.grid(row=0, column=1, columnspan=2, sticky="ew", pady=(0, 10))
        out_frame.grid_columnconfigure(0, weight=1)
        tb.Entry(out_frame, textvariable=self.test_output_dir, state=READONLY).grid(row=0, column=0, sticky="ew", padx=(0,10))
        tb.Button(out_frame, text="Open", command=self._open_output_folder, width=8).grid(row=0, column=1, sticky="e")

        # Buttons
        self.run_button = tb.Button(
            control_lf, text="▶ Run Test", bootstyle=SUCCESS, width=12, command=self._start_test
        )
        self.run_button.grid(row=1, column=0, sticky="w", pady=(5, 0))
        self.stop_button = tb.Button(
            control_lf, text="⏹ Stop Test", bootstyle=DANGER, width=12, command=self._stop_test, state=DISABLED
        )
        self.stop_button.grid(row=1, column=1, sticky="w", padx=10, pady=(5, 0))

        # Status
        self.status_label = tb.Label(control_lf, textvariable=self.status_text)
        self.status_label.grid(row=1, column=2, sticky="e", padx=(10, 0), pady=(5, 0))


        # --- Progress Bar ---
        progress_lf = tb.Labelframe(self, text="Test Progress", padding=(15,10), bootstyle=SECONDARY)
        progress_lf.grid(row=4, column=0, sticky="ew", padx=10, pady=(10, 0))
        progress_lf.grid_columnconfigure(0, weight=1)

        self.progressbar = tb.Progressbar(progress_lf, variable=self.progress_var, maximum=100)
        self.progressbar.grid(row=0, column=0, sticky="ew", pady=5)
        self.progress_label = tb.Label(progress_lf, textvariable=self.progress_label_text)
        self.progress_label.grid(row=1, column=0, sticky="ew", pady=(0, 5))

        self._update_ui() # Initial setup

    def _update_ui(self):
        """Show/hide single voice selection based on mode."""
        if self.test_mode.get() == "single":
            self.voice_lf.grid(row=1, column=0, sticky="ew", padx=10, pady=10)
        else:
            self.voice_lf.grid_forget()

    def _open_output_folder(self):
        folder = self.test_output_dir.get()
        self.app.open_folder(folder)

    def _update_status(self, status, progress_label=""):
        self.status_text.set(status)
        self.progress_label_text.set(progress_label)

    def _set_button_state(self, running):
        self.run_button.config(state=DISABLED if running else NORMAL)
        self.stop_button.config(state=NORMAL if running else DISABLED)
        # Disable mode/voice selection while running
        state = DISABLED if running else NORMAL
        for child in self.master.master.winfo_children(): # Access mode radio buttons via parent notebook tab
             if isinstance(child, tb.Labelframe) and "Mode" in child.cget('text'):
                  for rb in child.winfo_children():
                       if isinstance(rb, tb.Radiobutton): rb.config(state=state)
        self.voice_combo.config(state=DISABLED if running else 'readonly')
        self.text_input.config(state=DISABLED if running else NORMAL)


    def _progress_callback(self, progress, label_info="", index=0, total=0):
        """Callback for updating progress bar and labels during test."""
        self.app.after(0, self._update_progress_gui, progress, label_info, index, total)

    def _update_progress_gui(self, progress, label_info="", index=0, total=0):
         if not self.winfo_exists(): return
         # Handle None progress (e.g., on error/cancel)
         progress_val = progress if progress is not None else 0
         self.progress_var.set(progress_val)

         label = ""
         if total > 0: # Multi-voice test
             label = f"Processing voice {index}/{total}: {label_info}"
         elif label_info: # Single voice test or final status
              label = f"Status: {label_info}"

         self.progress_label_text.set(label)


    def _start_test(self):
        if self.test_thread and self.test_thread.is_alive():
            messagebox.showinfo("In Progress", "A voice test is already running.")
            return

        test_text = self.text_input.get("1.0", tk.END).strip()
        if not test_text:
            messagebox.showwarning("Input Required", "Please enter sample text to test.")
            return

        if self.test_mode.get() == "single" and not self.selected_voice.get():
             messagebox.showwarning("Input Required", "Please select a voice to test.")
             return

        self.cancellation_flag = False
        self.progress_var.set(0)
        self._update_status("Starting test...")
        self._set_button_state(running=True)

        self.test_thread = threading.Thread(target=self._run_test_thread, args=(test_text,), daemon=True)
        self.test_thread.start()

    def _stop_test(self):
        if self.test_thread and self.test_thread.is_alive():
            self._update_status("Stopping test...")
            self.cancellation_flag = True
            # No need to explicitly disable buttons here, _run_test_thread finally block handles it
        else:
             self._update_status("No test running")


    def _run_test_thread(self, test_text):
        """Background thread for running the voice tests."""
        try:
            mode = self.test_mode.get()
            output_dir = self.test_output_dir.get()
            device = self.app.audio_frame.get_device() # Get device from audio frame
            os.makedirs(output_dir, exist_ok=True)

            if mode == "single":
                voice = self.selected_voice.get()
                if not voice: raise ValueError("No voice selected for single test.")

                # --- CORRECT lang_code derivation for single test ---
                lang_code = voice[0] # Use first letter (e.g., 'a' from 'am_liam')
                # --- End Correction ---

                self._update_status(f"Testing: {voice}")
                output_file = os.path.join(output_dir, f"test_{voice}.wav")

                # Pass the corrected lang_code and device
                test_single_voice_kokoro(
                    input_text=test_text,
                    voice=voice,
                    output_path=output_file,
                    lang_code=lang_code, # Pass the single letter code
                    device=device,       # Pass the selected device
                    speed=1.0,
                    split_pattern=r'[.!?]+', # Simpler split for testing
                    cancellation_flag=lambda: self.cancellation_flag,
                    # Modify progress callback for single test context if needed
                    progress_callback=lambda p, fname, idx, total: self._progress_callback(p, voice), # Simplified for single file
                    pause_event=self.pause_event
                )
                final_progress = 100

            else: # mode == "all"
                temp_dir = os.path.join(PROJECT_DIR, "voice_test_temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, "test_text.txt")
                with open(temp_file, "w", encoding="utf-8") as f: f.write(test_text)

                voices_to_test = self.voice_list
                if not voices_to_test: raise ValueError("No voices available to test.")

                total_voices = len(voices_to_test)
                self._update_status(f"Testing {total_voices} voices...")

                # --- CORRECT lang_code derivation for 'all' test ---
                # Assume all voices tested belong to the same language family
                # Use the lang code from the *first* voice in the list
                lang_code = voices_to_test[0][0]
                # --- End Correction ---

                # Pass the corrected lang_code and device
                generate_audio_for_all_voices_kokoro(
                    input_path=temp_file,
                    lang_code=lang_code, # Pass the single letter code
                    voices=voices_to_test,
                    output_dir=output_dir,
                    device=device,       # Pass the selected device
                    speed=1.0,
                    split_pattern=r'[.!?]+', # Simpler split for testing
                    cancellation_flag=lambda: self.cancellation_flag,
                    # Pass the correct progress callback signature expected by the function
                    progress_callback=self._progress_callback, # UI method handles overall progress
                    pause_event=self.pause_event
                )
                final_progress = 100

                # Clean up temp file
                try:
                    if os.path.exists(temp_file): os.remove(temp_file)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir): os.rmdir(temp_dir)
                except OSError as e: print(f"Warning: Could not clean up temp test file: {e}")


            # --- GUI updates after completion (run via app.after) ---
            if not self.cancellation_flag:
                 self.app.after(0, lambda: self._progress_callback(final_progress, "Completed")) # Ensure 100%
                 self._update_status("Test completed!")
                 self.app.after(100, self._ask_open_folder)
            else:
                 self._update_status("Test cancelled.")
                 self.app.after(0, lambda: self._progress_callback(0, "Cancelled")) # Reset progress bar


        except Exception as e:
            error_msg = f"Test Error: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.app.after(0, lambda: messagebox.showerror("Test Error", error_msg))
            self._update_status("Test failed!")
            self.app.after(0, lambda: self._progress_callback(0, "Error")) # Reset progress bar
        finally:
            self.app.after(0, self._set_button_state, False)

    def _ask_open_folder(self):
         if messagebox.askyesno("Test Complete", "Voice test finished. Open the output folder?"):
                self._open_output_folder()


# --- Main Application Window ---

class AudiobookApp(tb.Window):
    """Main application window for PDF/EPUB to Audiobook conversion."""
    def __init__(self, *args, **kwargs):
        self.selected_theme = self._load_theme_from_config()
        super().__init__(themename=self.selected_theme, *args, **kwargs)

        self.title("PDF Narrator")
        self.geometry("950x750") # Adjusted size
        self.minsize(800, 650)

        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- Process State ---
        self.process_thread = None
        self.cancellation_flag = False
        self.pause_event = threading.Event()
        self.pause_event.set() # Start in the 'running' (not paused) state
        self.is_running = False
        self.is_paused = False

        # --- Main Layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1) # Notebook area expands

        # Header (Optional - simple title)
        # tb.Label(self, text="Narrator Studio", font="-size 16 -weight bold").grid(
        #     row=0, column=0, pady=(10, 5)
        # )

        # --- Notebook for Tabs ---
        self.notebook = tb.Notebook(self, bootstyle="primary")
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 5))

        # Create tab frames (simple frames, content added by specific classes)
        self.source_tab = tb.Frame(self.notebook)
        self.audio_tab = tb.Frame(self.notebook)
        self.progress_tab = tb.Frame(self.notebook)
        self.voice_test_tab = tb.Frame(self.notebook)

        self.notebook.add(self.source_tab, text=" 1. Source Setup ")
        self.notebook.add(self.audio_tab, text=" 2. Audio Setup ")
        self.notebook.add(self.progress_tab, text=" 3. Process & Logs ")
        self.notebook.add(self.voice_test_tab, text=" Voice Test ")

        # --- Populate Tabs ---
        # Pass 'self' (the app instance) to frames that need it
        self.source_frame = SourceFrame(self.source_tab, app=self)
        self.source_frame.pack(fill=BOTH, expand=True)

        self.audio_frame = AudioFrame(self.audio_tab, app=self)
        self.audio_frame.pack(fill=BOTH, expand=True)

        self.progress_frame = ProgressFrame(self.progress_tab, app=self) # Manages bars & logs
        self.progress_frame.pack(fill=BOTH, expand=True)

        self.voice_test_frame = VoiceTestFrame(self.voice_test_tab, app=self)
        self.voice_test_frame.pack(fill=BOTH, expand=True)

        # --- Control Area (Below Notebook) ---
        self.control_frame = ControlFrame(self, app=self) # Manages buttons & status
        self.control_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        tb.Separator(self, orient='horizontal').grid(row=3, column=0, sticky="ew", padx=10, pady=(0,5))

        # --- Footer ---
        footer_frame = tb.Frame(self, padding=(10, 5))
        footer_frame.grid(row=4, column=0, sticky="ew")

        reset_btn = tb.Button(footer_frame, text="Reset Settings", bootstyle="secondary", command=self._reset_config)
        reset_btn.pack(side=LEFT, padx=(10, 5))

        # Theme Selector
        theme_frame = tb.Frame(footer_frame)
        theme_frame.pack(side=RIGHT, padx=10)
        tb.Label(theme_frame, text="Theme:").pack(side=LEFT, padx=(0, 5))
        self.theme_var = tk.StringVar(value=self.selected_theme)
        themes = tb.Style().theme_names()
        theme_combo = tb.Combobox(
            theme_frame, textvariable=self.theme_var, values=themes,
            state="readonly", width=15
        )
        theme_combo.pack(side=LEFT)
        theme_combo.bind("<<ComboboxSelected>>", self._change_theme)

        exit_btn = tb.Button(footer_frame, text="Exit", command=self.on_close, width=8)
        exit_btn.pack(side=RIGHT, padx=5)


        # --- Load Config and Finalize UI ---
        self.load_config()
        # Ensure initial output paths are set based on loaded config
        self.source_frame._update_output_paths()
        self.control_frame.set_button_states(running=False, paused=False) # Initial button state

        self.source_frame._update_ui() # Initial UI update

    # --- Theme and Config Handling ---
    def _load_theme_from_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, 'r') as f:
                    config = json.load(f)
                    return config.get("theme", DEFAULT_THEME)
            except Exception: pass # Ignore errors, use default
        return DEFAULT_THEME

    def load_config(self):
        if not os.path.exists(CONFIG_FILE): return
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)

            self.source_frame.set_config(config.get("source_settings", {}))
            self.audio_frame.set_config(config.get("audio_settings", {}))

            # Load and apply theme last
            self.selected_theme = config.get("theme", DEFAULT_THEME)
            self.theme_var.set(self.selected_theme)
            self.style.theme_use(self.selected_theme)

        except Exception as e:
            print(f"Error loading config: {e}")
            messagebox.showwarning("Config Error", f"Could not load settings from {CONFIG_FILE}.\nError: {e}")

    def save_config(self):
        config = {
            "theme": self.selected_theme,
            "source_settings": self.source_frame.get_config(),
            "audio_settings": self.audio_frame.get_config(),
        }
        try:
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
            # Optionally show a warning to the user

    def _change_theme(self, event=None):
        new_theme = self.theme_var.get()
        self.style.theme_use(new_theme)
        self.selected_theme = new_theme
        # Force redraw/update of widgets if needed, though ttkbootstrap usually handles it

    def _reset_config(self):
        if messagebox.askyesno("Reset Settings", "Reset all source and audio settings to defaults?"):
            # Create default instances to get default values (avoids hardcoding defaults here)
            temp_source = SourceFrame(self.source_tab, self)
            temp_audio = AudioFrame(self.audio_tab, self)
            self.source_frame.set_config(temp_source.get_config())
            self.audio_frame.set_config(temp_audio.get_config())
            temp_source.destroy()
            temp_audio.destroy()
            self.source_frame._update_output_paths() # Update derived paths
            messagebox.showinfo("Settings Reset", "Settings have been reset to defaults.")


    # --- UI Update and Interaction ---
    def update_audio_output_dir_display(self, path):
        """Called by SourceFrame to update the display in AudioFrame."""
        if hasattr(self, 'audio_frame'):
            self.audio_frame.update_output_display(path)
        else:
             # This might happen during initialization before audio_frame exists
             self.after(100, lambda: self.update_audio_output_dir_display(path))


    def open_folder(self, folder_path):
        """Safely opens a folder in the default file explorer."""
        if not folder_path or not os.path.isdir(folder_path):
            messagebox.showwarning("Folder Not Found", f"The folder does not exist:\n{folder_path}")
            return
        try:
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin": # macOS
                subprocess.Popen(["open", folder_path])
            else: # Linux and other POSIX
                subprocess.Popen(["xdg-open", folder_path])
        except Exception as e:
            messagebox.showerror("Error Opening Folder", f"Could not open folder:\n{folder_path}\nError: {e}")


    # --- Process Control Methods ---
    def start_process(self):
        if self.is_running:
            print("Process already running.")
            return

        # --- Gather all configuration just before starting ---
        try:
            self.config_data = {
                "source": self.source_frame.get_config(),
                "audio": self.audio_frame.get_config(),
                "extracted_text_output": self.source_frame.extracted_text_dir.get(), # Get derived dir
                "audio_output": self.audio_frame.audio_output_dir.get() # Get derived dir
            }

            # --- Basic Validation ---
            source_opt = self.config_data["source"]["source_option"]
            if source_opt == "single" and not self.config_data["source"]["pdf_path"]:
                raise ValueError("Single Book: Please select a PDF/EPUB file.")
            if source_opt == "batch" and not self.config_data["source"]["pdf_folder"]:
                 raise ValueError("Batch Folder: Please select a source folder.")
            if source_opt == "skip" and not self.config_data["source"]["manual_extracted_dir"]:
                 raise ValueError("Existing Text: Please select the text folder.")
            if not self.config_data["audio"]["voicepack"]:
                 raise ValueError("Audio Setup: Please select a voice.")
            if source_opt != "skip" and not self.config_data["extracted_text_output"]:
                 raise ValueError("Source Setup: Could not determine extracted text output directory.")
            if not self.config_data["audio_output"]:
                 raise ValueError("Source/Audio Setup: Could not determine audiobook output directory.")

            # Ensure output directories exist (create if needed)
            if source_opt != "skip":
                 os.makedirs(self.config_data["extracted_text_output"], exist_ok=True)
            os.makedirs(self.config_data["audio_output"], exist_ok=True)


        except ValueError as ve:
             messagebox.showerror("Configuration Error", str(ve))
             return
        except Exception as e:
             messagebox.showerror("Error", f"An unexpected error occurred during setup: {e}")
             return

        # --- Start Process ---
        self.is_running = True
        self.is_paused = False
        self.cancellation_flag = False
        self.pause_event.set() # Ensure not paused
        self.control_frame.set_button_states(running=True, paused=False)
        self.progress_frame.reset_progress()
        self.control_frame.update_status("Starting...")
        print("-" * 20 + " Starting Process " + "-" * 20) # Log separator

        self.process_thread = threading.Thread(target=self._run_process_thread, daemon=True)
        self.process_thread.start()

    def pause_process(self):
        if self.is_running and not self.is_paused:
            self.pause_event.clear() # Signal thread to pause
            self.is_paused = True
            self.control_frame.set_button_states(running=True, paused=True)
            self.control_frame.update_status("Paused")
            print("Process paused.")

    def resume_process(self):
        if self.is_running and self.is_paused:
            self.pause_event.set() # Signal thread to resume
            self.is_paused = False
            self.control_frame.set_button_states(running=True, paused=False)
            # Status will be updated by the running thread
            print("Process resumed.")

    def cancel_process(self):
        if self.is_running:
            print("Cancellation requested...")
            self.cancellation_flag = True
            self.pause_event.set() # Ensure thread is not stuck waiting if paused
            self.control_frame.update_status("Cancelling...")
            self.control_frame.set_button_states(running=True, paused=False) # Keep cancel active, disable others
            # The thread's finally block will reset state fully

    def _update_gui_progress(self, extract_p=None, audio_p=None, status=None, action=None, file=None, count_str=None, est_time_str=None):
         """Helper to safely update GUI elements from the process thread via 'after'."""
         update_args = {}
         if status is not None: update_args["status"] = status
         if action is not None: update_args["action"] = action
         if file is not None: update_args["file"] = file
         if count_str is not None: update_args["count_str"] = count_str
         if est_time_str is not None: update_args["est_time_str"] = est_time_str

         self.after(0, lambda: (
              self.progress_frame.update_progress(extract_p, audio_p),
              self.control_frame.update_status(**update_args) if update_args else None
         ))

    def _format_time(self, seconds):
        """Formats seconds into Hh Mm Ss or Mm Ss or Ss string."""
        if seconds <= 0: return "N/A"
        m, s = divmod(int(seconds), 60)
        h, m = divmod(m, 60)
        if h > 0: return f"{h}h {m}m {s}s"
        if m > 0: return f"{m}m {s}s"
        return f"{s}s"

    # --- Background Process Thread ---
    def _run_process_thread(self):
        """The actual work happens here in a background thread."""
        start_time = time.time()
        try:
            # --- Get config gathered by start_process ---
            cfg = self.config_data
            source_opt = cfg["source"]["source_option"]
            extracted_base_output = cfg["extracted_text_output"]
            audio_base_output = cfg["audio_output"]

            all_task_folders = [] # List of tuples: (text_input_folder, audio_output_folder)

            # --- 1. Extraction Phase (if not 'skip') ---
            if source_opt != "skip":
                self._update_gui_progress(extract_p=0, status="Extracting Text...")

                use_toc = cfg["source"]["use_toc"]
                extract_mode = cfg["source"]["extract_mode"]

                if source_opt == "single":
                    pdf_path = cfg["source"]["pdf_path"]
                    filename = os.path.basename(pdf_path)
                    self._update_gui_progress(action="Extracting:", file=filename, count_str="(1 of 1)")

                    def single_extract_progress(p):
                        if self.cancellation_flag: raise InterruptedError("Extraction cancelled")
                        self._update_gui_progress(extract_p=p)
                        # Simple time estimate based on progress
                        elapsed = time.time() - start_time
                        est = (elapsed / (p / 100.0)) - elapsed if p > 0 else 0
                        self._update_gui_progress(est_time_str=f"Est. Time: {self._format_time(est)}")


                    # Output directly into the determined base folder for single files
                    extract_book(
                        pdf_path, use_toc=use_toc, extract_mode=extract_mode,
                        output_dir=extracted_base_output, # e.g., ./extracted_books/MyBook
                        progress_callback=single_extract_progress
                    )
                    all_task_folders.append((extracted_base_output, audio_base_output))
                    print(f"Successfully extracted: {filename}")

                elif source_opt == "batch":
                    source_folder = cfg["source"]["pdf_folder"]
                    book_files = []
                    for root, _, files in os.walk(source_folder):
                        for file in files:
                            if file.lower().endswith(('.pdf', '.epub', '.txt', '.html', '.htm')):
                                book_files.append(os.path.join(root, file))

                    if not book_files: raise FileNotFoundError("No PDF/EPUB/TXT/HTML files found in batch folder.")
                    total_files = len(book_files)
                    print(f"Found {total_files} book files for batch processing.")

                    for i, book_path in enumerate(book_files, start=1):
                        if self.cancellation_flag: raise InterruptedError("Batch extraction cancelled")

                        filename = os.path.basename(book_path)
                        self._update_gui_progress(action="Extracting:", file=filename, count_str=f"({i} of {total_files})")

                        # Determine relative path for output structure mirroring
                        rel_path = os.path.relpath(os.path.dirname(book_path), source_folder)
                        book_name_no_ext = os.path.splitext(filename)[0]

                        # Output folder for this specific book's text
                        current_extract_output = os.path.join(extracted_base_output, rel_path, book_name_no_ext)
                        # Corresponding audio output folder
                        current_audio_output = os.path.join(audio_base_output, rel_path, book_name_no_ext)

                        os.makedirs(current_extract_output, exist_ok=True)

                        def batch_extract_progress(p):
                            if self.cancellation_flag: raise InterruptedError("Extraction cancelled")
                            overall_p = ((i - 1 + p / 100.0) / total_files) * 100
                            self._update_gui_progress(extract_p=overall_p)
                            elapsed = time.time() - start_time
                            est = (elapsed / (overall_p / 100.0)) - elapsed if overall_p > 0 else 0
                            self._update_gui_progress(est_time_str=f"Est. Time: {self._format_time(est)}")

                        extract_book(
                            book_path, use_toc=use_toc, extract_mode=extract_mode,
                            output_dir=current_extract_output,
                            progress_callback=batch_extract_progress
                        )
                        all_task_folders.append((current_extract_output, current_audio_output))
                        print(f"({i}/{total_files}) Extracted: {filename}")

                self._update_gui_progress(extract_p=100) # Mark extraction as complete

            else: # source_opt == "skip"
                print("Skipping extraction phase.")
                self._update_gui_progress(extract_p=100, status="Using Existing Text")
                manual_dir = cfg["source"]["manual_extracted_dir"]

                # Need to find relevant text folders within the manual dir
                # Assumes structure might mirror batch output (or be flat)
                if not os.path.isdir(manual_dir):
                     raise FileNotFoundError("Existing text folder not found or is invalid.")

                # Heuristic: Find folders containing .txt files directly, or assume the top level is one task
                found_text = False
                for root, _, files in os.walk(manual_dir):
                     if any(f.lower().endswith(".txt") for f in files):
                          rel_path = os.path.relpath(root, manual_dir)
                          text_input_folder = root
                          audio_output_folder = os.path.join(audio_base_output, rel_path)
                          all_task_folders.append((text_input_folder, audio_output_folder))
                          found_text = True

                if not found_text and any(f.lower().endswith(".txt") for f in os.listdir(manual_dir)):
                     # If text files are directly in the root, treat the root as the task
                     all_task_folders.append((manual_dir, audio_base_output))
                     found_text = True

                if not all_task_folders:
                    raise FileNotFoundError(f"No .txt files found within the specified existing text folder: {manual_dir}")

                print(f"Found {len(all_task_folders)} text folder(s) to process for audio generation.")


            # --- 2. Audio Generation Phase ---
            if self.cancellation_flag: raise InterruptedError("Cancelled before audio generation")

            self._update_gui_progress(audio_p=0, status="Generating Audiobooks...")
            audio_cfg = cfg["audio"]
            voice = audio_cfg["voicepack"] # Already validated that it's selected
            if voice and len(voice) > 0:
                 lang_code = voice[0] # Use the first letter (e.g., 'a' from 'af_alloy')
            else:
                 # Handle case where voice might be empty, though validation should prevent this
                 raise ValueError("Invalid or empty voice selected.")
            audio_format = audio_cfg["audio_format"]
            chunk_size = audio_cfg["chunk_size"] # Not directly used by kokoro func? Check generate_audiobooks_kokoro
            device = audio_cfg["device"] # Not directly used by kokoro func? Check generate_audiobooks_kokoro

            total_tasks = len(all_task_folders)
            if total_tasks == 0:
                 print("Warning: No text folders found to generate audio from.")
                 # No error, just nothing to do

            task_start_time = time.time() # Reset timer for audio phase estimate

            for task_idx, (text_input_dir, audio_output_dir) in enumerate(all_task_folders, start=1):
                 if self.cancellation_flag: raise InterruptedError("Audio generation cancelled")

                 task_name = os.path.basename(text_input_dir)
                 self._update_gui_progress(action="Generating Audio:", file=task_name, count_str=f"({task_idx} of {total_tasks})")
                 print(f"--- Generating audio for: {task_name} ({task_idx}/{total_tasks}) ---")

                 os.makedirs(audio_output_dir, exist_ok=True)

                 def audio_progress_callback(progress, current_file="", file_idx=0, files_total=0):
                     if self.cancellation_flag: raise InterruptedError("Audio generation cancelled")

                     overall_progress = ((task_idx - 1 + progress / 100.0) / total_tasks) * 100
                     self._update_gui_progress(audio_p=overall_progress)

                     # Update current file within the task
                     file_count_str = f"({file_idx}/{files_total})" if files_total > 0 else ""
                     self._update_gui_progress(file=f"{task_name} / {current_file}", count_str=f"{file_count_str}") # Append file name

                     # Time estimate for audio phase
                     elapsed = time.time() - task_start_time
                     est = (elapsed / (overall_progress / 100.0)) - elapsed if overall_progress > 0 else 0
                     self._update_gui_progress(est_time_str=f"Est. Time: {self._format_time(est)}")

                 # Call the Kokoro generation function
                 # Pass necessary parameters - adjust based on actual function signature
                 generate_audiobooks_kokoro(
                     input_dir=text_input_dir,
                     output_dir=audio_output_dir,
                     voice=voice,
                     lang_code=lang_code, # Pass derived lang code
                     audio_format=audio_format,
                     # speed=1.0, # Assuming default speed, add if needed
                     # split_pattern=r'\n+', # Assuming default split, add if needed
                     # device=device # Pass device if kokoro func supports it
                     # chunk_size=chunk_size # Pass chunk_size if kokoro func supports it
                     progress_callback=audio_progress_callback, # Use the combined callback
                     cancellation_flag=lambda: self.cancellation_flag,
                     pause_event=self.pause_event, # Pass the pause event
                     # Add file_callback if generate_audiobooks_kokoro supports it for finer file updates
                     # file_callback=lambda filename, i, total: ...
                 )
                 print(f"Finished audio for: {task_name}")

            self._update_gui_progress(audio_p=100) # Mark audio as complete

            # --- Completion ---
            total_time = time.time() - start_time
            success_msg = f"Process completed successfully in {self._format_time(total_time)}."
            print(success_msg)
            self._update_gui_progress(status="Completed", action="-", file="", count_str="", est_time_str=f"Total Time: {self._format_time(total_time)}")
            self.after(100, lambda: messagebox.showinfo("Success", success_msg))


        except FileNotFoundError as e:
             error_msg = f"Error: {e}"
             print(error_msg)
             self._update_gui_progress(status="Failed", action="Error")
             self.after(0, lambda: messagebox.showerror("File Not Found", error_msg))
        except ValueError as e: # Configuration errors
             error_msg = f"Configuration Error: {e}"
             print(error_msg)
             self._update_gui_progress(status="Failed", action="Config Error")
             self.after(0, lambda: messagebox.showerror("Configuration Error", error_msg))
        except InterruptedError as e: # User cancellation
             error_msg = f"Process Cancelled: {e}"
             print(error_msg)
             self._update_gui_progress(status="Cancelled", action="-", file="", count_str="")
             self.after(0, lambda: messagebox.showwarning("Cancelled", "The process was cancelled by the user."))
        except Exception as e:
            error_msg = f"An unexpected error occurred: {e}"
            print(error_msg)
            import traceback
            traceback.print_exc() # Log full traceback to console
            self._update_gui_progress(status="Failed", action="Error")
            self.after(0, lambda: messagebox.showerror("Runtime Error", error_msg))
        finally:
            # --- Reset state regardless of outcome ---
            self.is_running = False
            self.is_paused = False
            self.cancellation_flag = False
            self.pause_event.set() # Ensure it's set for the next run
            # Update button states via 'after' to ensure it's done in main thread
            self.after(0, lambda: self.control_frame.set_button_states(running=False, paused=False))
            print("-" * 20 + " Process Finished " + "-" * 20)


    # --- Window Closing ---
    def on_close(self):
        """Handles window closing, saves config, stops threads."""
        print("Closing application...")
        if self.is_running:
            if messagebox.askyesno("Process Running", "A process is currently running. Do you want to stop it and exit?"):
                self.cancel_process()
                # Give the thread a moment to acknowledge cancellation
                if self.process_thread and self.process_thread.is_alive():
                     self.process_thread.join(timeout=1.5)
            else:
                return # Don't close if user cancels

        # Also check and stop voice test thread if running
        if self.voice_test_frame.test_thread and self.voice_test_frame.test_thread.is_alive():
             print("Stopping voice test thread...")
             self.voice_test_frame.cancellation_flag = True
             self.voice_test_frame.test_thread.join(timeout=1.0)


        self.save_config()
        self.destroy() # Close the Tkinter window


# --- Main Execution ---
if __name__ == "__main__":
    # Optional: Initialize QApplication early if using QFileDialog consistently
    # qt_app = QApplication.instance() or QApplication(sys.argv)

    app = AudiobookApp()
    app.mainloop()