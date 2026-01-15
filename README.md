# PDF Narrator

**Updated for Kokoro v1.0!**  
Now setting up is easier—simply install the required Python dependencies (including the updated Kokoro package) and run the app. No more manual downloads or moving model files into specific folders.

PDF Narrator (Kokoro Edition) transforms your **PDF and EPUB documents** into audiobooks effortlessly using **advanced text extraction** and **Kokoro TTS** technology. With Kokoro v1.0, the integration is seamless and the setup is as simple as installing the requirements and running the application.

---

## Demo

1. **Screenshot**  
   Check out the GUI in the screenshot below:  
   ![Demo Screenshot](assets/demo.png)

2. **Audio Sample**  
   Listen to a short sample of the generated audiobook:  
   - af_sky
   
https://github.com/user-attachments/assets/f1e0a88b-9d44-40d6-82c4-dcf070db6856

   - am_liam
   
https://github.com/user-attachments/assets/de611592-9404-416a-8118-1acca1651edf

   - af_heart
   
https://github.com/user-attachments/assets/b9d87ed4-3458-4116-accf-0816ec1448b2



---

## Features

- **Intelligent Text Extraction**
  - Supports both **PDF** and **EPUB** formats.
  - For PDFs: Skips headers, footers, and page numbers; optionally splits based on Table of Contents (TOC).
  - For EPUBs: Extracts chapters based on internal HTML structure.

- **Kokoro TTS Integration**
  - Generate natural-sounding audiobooks with the updated [Kokoro v1.0 model](https://huggingface.co/hexgrad/Kokoro-82M).
  - Easily select different voicepacks.

- **User-Friendly GUI**
  - Modern interface built with **ttkbootstrap** (theme selector, scrolled logs, progress bars).
  - Pause/resume and cancel your audiobook generation anytime.

- **Configurable for Low-VRAM Systems**
  - Choose the chunk size for text to accommodate limited GPU resources.
  - Switch to CPU if no GPU is available.

- **Voice Testing Made Simple**
  - Test a single voice or loop through all available voices directly from the GUI.
  - Use the dedicated Voice Test tab to generate and listen to sample audio files.

---

## Prerequisites

- **Python 3.8+**
- **FFmpeg** (for audio-related tasks on some systems)
- **Torch** (PyTorch for the Kokoro TTS model)
- Other dependencies as listed in `requirements.txt`

---

## Installation 
> ARM64 MacOS users see the section below

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mateogon/pdf-narrator.git
   cd pdf-narrator
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install Python Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Installation for ARM64 MacOS Users
1. **Clone the Repository**
   ```bash
   git clone https://github.com/mateogon/pdf-narrator.git
   cd pdf-narrator
   ```

2. **Run the Setup Script to Download Dependencies**
   ```bash
   chmod +x scripts/setup_macos_arm64.sh
   ./scripts/setup_macos_arm64.sh
   ```

3. **Create and Activate the Conda Virtual Environment**
   ```bash
   source "$HOME/miniforge3/bin/activate" && conda activate pdf-narrator
   ```

---
## Windows Additional Installation Notes

For Windows users, some libraries may require extra steps:

### 1. **Prerequisites**

- **Python 3.12.7**  
  Download and install [Python 3.12.7](https://www.python.org/downloads/). Ensure `python` and `pip` are added to your system's PATH.

- **CUDA 12.4** (for GPU acceleration)  
  Install the [CUDA 12.4 Toolkit](https://developer.nvidia.com/cuda-downloads) if you plan to use GPU acceleration.

### 2. **Installing eSpeak NG**

eSpeak NG is required for phoneme-based operations.

1. **Download the Installer**  
   [eSpeak NG X64 Installer](https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-X64.msi)

2. **Run the Installer**  
   Follow the on-screen instructions.

3. **Set Environment Variables**  
   Add the following environment variables:

   - `PHONEMIZER_ESPEAK_LIBRARY` → `C:\Program Files\eSpeak NG\libespeak-ng.dll`
   - `PHONEMIZER_ESPEAK_PATH` → `C:\Program Files (x86)\eSpeak\command_line\espeak.exe`

   (Right-click "This PC" → Properties → Advanced system settings → Environment Variables)

4. **Verify Installation**

   Open Command Prompt and run:

   ```cmd
   espeak-ng --version
   ```

### 3. **Using Precompiled Wheels for DeepSpeed and lxml**

1. **Download Wheels**

   - **DeepSpeed** (for Python 3.12.7, CUDA 12.4): [DeepSpeed Wheel](https://huggingface.co/NM156/deepspeed_wheel/tree/main)
   - **lxml** (for Python 3.12): [lxml Release](https://github.com/lxml/lxml/releases/tag/lxml-5.3.0)

2. **Install the Wheels**

   Activate your virtual environment and run:

   ```cmd
   pip install path\to\deepspeed-0.11.2+cuda124-cp312-cp312-win_amd64.whl
   pip install path\to\lxml-5.3.0-cp312-cp312-win_amd64.whl
   ```

3. **Verify Installation**

   ```cmd
   deepspeed --version
   pip show lxml
   espeak-ng --version
   ```

---
## Quick Start

1. **Launch the App**
   ```bash
   python main.py
   ```

2. **Select a Mode**
   - **Single Book:** Choose a PDF or EPUB file and extract its text.
   - **Batch Books:** Select a folder with multiple PDFs and/or EPUBs (processes all, preserving folder structure).
   - **Skip Extraction:** Use pre-extracted text files.

3. **Extract Text**
   - **PDFs:** Splits into chapters if TOC is available; otherwise, extracts entire document.
   - **EPUBs:** Extracts chapters based on internal structure.

4. **Configure Kokoro TTS Settings**
   - Select a voice.
   - Adjust chunk size and output format (`.wav` or `.mp3`).

5. **Generate Audiobook**
   - Click Start Process and monitor progress.
   - Find your audio files in the output folder.

---

## Testing Voices

With the latest update, you can now quickly test voices directly within the app:

1. **Navigate to the Voice Test Tab**  
   In the main window, click on the **Voice Test** tab to access voice testing features.

2. **Enter Sample Text**  
   Modify or use the default sample text in the provided text area.

3. **Select Test Mode**  
   - **Test Single Voice:** Pick one voice from the dropdown to generate a sample.
   - **Test All Available Voices:** Choose this option to automatically generate samples for every available voice.

4. **Run the Test**  
   Click **Run Voice Test** to start. The app creates a temporary file with your sample text, processes it through Kokoro TTS for each voice (if testing all voices), and saves the output audio files in the designated folder.

5. **Monitor Progress and Listen**  
   A progress bar and status labels keep you updated. Once the test is complete, you’ll be prompted to open the output folder where you can listen to the generated samples.

6. **Stop the Test if Needed**  
   If you need to cancel while testing, click **Stop Test** to interrupt the process.

---

## Technical Highlights

- **Text Extraction**
  - PDF: Built on PyMuPDF for efficient parsing, with TOC-based splitting.
  - EPUB: Extracts text from HTML content files within the EPUB structure.

- **Kokoro TTS**
  - Advanced text normalization and phonemization.
  - Splits text into chunks (<510 tokens) and joins audio outputs.

---

## Contributing

Fork the repository, create a branch, and submit a pull request.  
Report bugs or suggest features via Issues.

---

## License

This project is released under the MIT License (LICENSE.md).

---

Enjoy converting your PDFs and EPUBs into immersive audiobooks with Kokoro v1.0 TTS—and now, easily test voices to pick your perfect sound!
