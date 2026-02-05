# generate_audiobook_kokoro.py

import os
import time
import numpy as np
import torch
import soundfile as sf
import traceback  # For more detailed error logging
from kokoro import (
    KPipeline,
)  # Assuming KPipeline handles device internally or takes it as arg

# --- Constants ---
DEFAULT_SAMPLE_RATE = 24000
if os.name == "Darwin":
    DEVICE = "cpu"  # Default to CPU on macOS for compatibility
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Helper Functions ---


def available_voices():
    """Return the hard-coded list of available Kokoro voice identifiers."""
    return [
        "af_heart",
        "af_alloy",
        "af_aoede",
        "af_bella",
        "af_jessica",
        "af_kore",
        "af_nicole",
        "af_nova",
        "af_river",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_echo",
        "am_eric",
        "am_fenrir",
        "am_liam",
        "am_michael",
        "am_onyx",
        "am_puck",
        "am_santa",
        "bf_alice",
        "bf_emma",
        "bf_isabella",
        "bf_lily",
        "bm_daniel",
        "bm_fable",
        "bm_george",
        "bm_lewis",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "pf_dora",
        "pm_alex",
        "pm_santa",
    ]


# --- Core Audio Generation for a Single File ---


def generate_audio_for_file_kokoro(
    input_path,
    pipeline,
    voice,
    output_path,
    speed=1.0,
    split_pattern=r"\n+",
    cancellation_flag=None,
    chunk_progress_callback=None,
    pause_event=None,
):
    """
    Generates audio for a single text file using a pre-initialized Kokoro pipeline.

    Args:
        input_path (str): Path to the input text file.
        pipeline (KPipeline): An initialized Kokoro pipeline instance.
        voice (str): The voice identifier to use.
        output_path (str): Path to save the generated audio file.
        speed (float): Speech speed multiplier.
        split_pattern (str): Regex pattern for splitting text into chunks for TTS.
        cancellation_flag (callable): Function returning True to cancel.
        chunk_progress_callback (callable): Callback reporting (chars_in_chunk, chunk_duration).
        pause_event (threading.Event): Event to pause processing.

    Returns:
        bool: True if audio generation was successful and saved, False otherwise.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            print(
                f"Warning: Input file '{os.path.basename(input_path)}' is empty. Skipping."
            )
            return False
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return False
    except Exception as e:
        print(f"Error reading file '{os.path.basename(input_path)}': {e}")
        return False
    # print(f"Read file in {time.time() - start_file_read:.3f}s") # Optional debug log

    if cancellation_flag and cancellation_flag():
        print("      Cancellation detected before audio synthesis.")
        raise InterruptedError("Processing cancelled by user.")
    if pause_event:
        pause_event.wait()  # Wait if paused

    audio_chunks = []
    chars_processed_in_file = 0
    start_synth_time = time.time()
    last_callback_time = start_synth_time

    print("Synthesizing audio...")
    try:
        # Iterate through generated audio chunks from the pipeline
        for chunk_index, (gs, ps, audio) in enumerate(
            pipeline(text, voice=voice, speed=speed, split_pattern=split_pattern)
        ):
            if cancellation_flag and cancellation_flag():
                print("      Cancellation detected during audio synthesis.")
                raise InterruptedError("Processing cancelled by user.")
            if pause_event:
                pause_event.wait()  # Wait if paused

            # Process the audio chunk
            if isinstance(audio, torch.Tensor):
                audio = (
                    audio.cpu().numpy()
                )  # Move to CPU and convert to NumPy if needed
            audio_chunks.append(audio)

            # Update progress based on this chunk
            chars_in_chunk = len(gs) if gs else 0  # Length of graphemes in the chunk
            chars_processed_in_file += chars_in_chunk
            current_time = time.time()
            chunk_duration = current_time - last_callback_time
            last_callback_time = current_time

            if chunk_progress_callback and chars_in_chunk > 0:
                # Report characters processed in this chunk and its duration
                chunk_progress_callback(chars_in_chunk, chunk_duration)

    except Exception as e:
        print(
            f"Error during Kokoro pipeline processing for '{os.path.basename(input_path)}': {e}"
        )
        traceback.print_exc()  # Print detailed traceback for debugging
        return False  # Indicate failure for this file

    if not audio_chunks:
        print(
            f"Warning: No audio chunks generated for '{os.path.basename(input_path)}'."
        )
        return False

    # Concatenate, Normalize, and Save
    try:
        print(f"Concatenating {len(audio_chunks)} audio chunks...")
        combined_audio = np.concatenate(audio_chunks)

        # Normalize audio to prevent clipping and fit int16 range
        max_abs_val = np.max(np.abs(combined_audio))
        if max_abs_val > 0:  # Avoid division by zero for silent audio
            # Normalize to ~95% of max range to leave some headroom
            normalized_audio = (combined_audio / max_abs_val * 32767 * 0.95).astype(
                np.int16
            )
        else:
            normalized_audio = combined_audio.astype(np.int16)  # Already silent

        print(f"      Saving audio to '{os.path.basename(output_path)}'...")
        sf.write(output_path, normalized_audio, DEFAULT_SAMPLE_RATE)
        # Removed verbose "Audio saved to..." log from here

    except Exception as e:
        print(
            f"      Error concatenating or saving audio for '{os.path.basename(output_path)}': {e}"
        )
        return False

    return True  # Indicate success for this file


# --- Main Function for Processing a Directory ---


def generate_audiobooks_kokoro(
    input_dir,
    lang_code,  # Language code for the pipeline (e.g., 'a')
    voice,  # Voice identifier (e.g., "am_liam")
    device=DEVICE,  # Device for TTS computation ('cuda' or 'cpu')
    output_dir=None,  # Optional: Defaults to 'input_dir_audio' sibling folder
    audio_format=".wav",  # Output audio format (ensure soundfile supports it)
    speed=1.0,
    split_pattern=r"\n+",
    progress_callback=None,  # Callback for overall progress (percentage, current_file, index, total)
    cancellation_flag=None,
    pause_event=None,
    # Removed file_callback (merged into progress_callback)
    # Removed update_estimate_callback (handled internally if needed or by UI)
):
    """
    Generates audio files from all .txt files in a directory using Kokoro TTS.

    Args:
        input_dir (str): Path to the directory containing .txt files.
        lang_code (str): Kokoro language code (e.g., 'a', 'b', 'j').
        voice (str): Kokoro voice identifier (e.g., 'am_liam').
        device (str): Computation device ('cuda' or 'cpu').
        output_dir (str, optional): Directory to save audio files. Defaults to sibling directory.
        audio_format (str): File extension for audio output (e.g., '.wav', '.mp3').
        speed (float): Speech speed multiplier.
        split_pattern (str): Regex for splitting text for TTS processing.
        progress_callback (callable, optional): Reports overall progress.
            Receives: (overall_percentage, current_filename, current_index, total_files).
        cancellation_flag (callable, optional): Function returning True to cancel.
        pause_event (threading.Event, optional): Event to pause processing.

    Returns:
        list[str]: List of paths to successfully generated audio files.

    Raises:
        FileNotFoundError: If input_dir does not exist.
        ValueError: If lang_code or device is invalid.
        Exception: For errors during pipeline initialization or processing.
    """
    start_process_time = time.time()
    # Resolve device: if CUDA requested but not available, fall back to CPU
    try:
        if device == "cuda" and not torch.cuda.is_available():
            print("  Warning: CUDA requested but not available; falling back to CPU.")
            device = "cpu"
    except Exception:
        # If torch is not functional for device checks, default to CPU
        device = "cpu"
    print(f"\n--- Starting Audiobook Generation Task ---")
    print(f"  Input Directory : '{input_dir}'")
    print(f"  Language / Voice: {lang_code} / {voice}")
    print(f"  Device          : {device}")

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: '{input_dir}'")

    # --- Determine and Create Output Directory ---
    if output_dir is None:
        parent_dir = os.path.dirname(os.path.normpath(input_dir))
        book_name = os.path.basename(os.path.normpath(input_dir))
        output_dir = os.path.join(parent_dir, f"{book_name}_audio")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output Directory: '{output_dir}'")
    except OSError as e:
        print(f"  Error creating output directory '{output_dir}': {e}")
        raise

    # --- Gather and Sort Text Files ---
    try:
        files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(".txt")])
        total_files = len(files)
        if total_files == 0:
            print(
                "  Warning: No .txt files found in the input directory. Nothing to process."
            )
            return []
        print(f"  Files to process: {total_files}")
    except Exception as e:
        print(f"  Error listing files in '{input_dir}': {e}")
        raise

    # --- Initialize Kokoro Pipeline ---
    pipeline = None  # Define outside try block
    try:
        print(
            f"  Initializing Kokoro pipeline for lang='{lang_code}' on device='{device}'..."
        )
        init_start_time = time.time()
        # *** CRUCIAL: Assuming KPipeline accepts 'device' argument ***
        pipeline = KPipeline(
            lang_code=lang_code, device=device, repo_id="hexgrad/Kokoro-82M"
        )
        print(f"  Pipeline initialized in {time.time() - init_start_time:.2f}s.")
    except AssertionError as e:
        # Catch assertion errors specifically, often related to invalid lang_code
        print(f"  Error: Invalid language code '{lang_code}' provided for KPipeline.")
        print(f"  Details: {e}")
        raise ValueError(f"Invalid language code: {lang_code}") from e
    except Exception as e:
        print(f"  Error initializing Kokoro pipeline: {e}")
        traceback.print_exc()
        raise  # Re-raise other initialization errors

    # --- Prepare for Progress Tracking ---
    # Pre-calculate total characters for smoother progress estimation
    total_characters_all_files = 0
    print("  Calculating total text size for progress estimation...")
    for text_file in files:
        try:
            with open(os.path.join(input_dir, text_file), "r", encoding="utf-8") as f:
                total_characters_all_files += len(f.read())
        except Exception as e:
            print(
                f"    Warning: Could not read file '{text_file}' for size calculation: {e}"
            )
    print(f"  Total characters approx: {total_characters_all_files}")

    characters_processed_so_far = 0
    start_loop_time = time.time()  # For rate calculation within the loop
    generated_files = []
    files_processed_successfully = 0

    # --- Define Inner Callback for Chunk Progress ---
    def internal_chunk_progress_callback(
        chars_in_chunk, chunk_duration, current_filename, current_index, total_files
    ):
        nonlocal characters_processed_so_far
        characters_processed_so_far += chars_in_chunk

        # Calculate overall progress percentage based on characters
        overall_progress = 0
        if total_characters_all_files > 0:
            overall_progress = (
                characters_processed_so_far / total_characters_all_files
            ) * 100
            overall_progress = min(
                max(overall_progress, 0), 100
            )  # Clamp between 0 and 100

        # Report overall progress back to the main UI callback
        if progress_callback:
            # Pass overall %, current filename, current index, total files
            progress_callback(
                overall_progress, current_filename, current_index, total_files
            )

        # Note: Time estimation can be done here or in the UI based on progress and elapsed time
        # elapsed_loop_time = time.time() - start_loop_time
        # if overall_progress > 1 and update_estimate_callback: # Avoid early jumpy estimates
        #     total_estimated_time = (elapsed_loop_time / (overall_progress / 100.0))
        #     remaining_time = max(0, total_estimated_time - elapsed_loop_time)
        #     update_estimate_callback(int(remaining_time))

    # --- Process Each File ---
    print("\n--- Processing Files ---")
    try:
        for i, text_file in enumerate(files, start=1):
            if cancellation_flag and cancellation_flag():
                print(f"\nCancellation detected before processing '{text_file}'.")
                raise InterruptedError("Processing cancelled by user.")
            if pause_event:
                pause_event.wait()  # Check pause before each file

            print(f"\n[{i}/{total_files}] Processing: '{text_file}'")
            file_start_time = time.time()
            input_path = os.path.join(input_dir, text_file)
            base_name = os.path.splitext(text_file)[0]
            output_filename = f"{base_name}{audio_format}"
            output_path = os.path.join(output_dir, output_filename)

            # --- Call the file generation function ---
            # Pass a lambda that captures the current file context for the internal callback
            file_chunk_callback = lambda chars, duration: (
                internal_chunk_progress_callback(
                    chars, duration, text_file, i, total_files
                )
            )

            success = generate_audio_for_file_kokoro(
                input_path=input_path,
                pipeline=pipeline,
                voice=voice,
                output_path=output_path,
                speed=speed,
                split_pattern=split_pattern,
                cancellation_flag=cancellation_flag,
                chunk_progress_callback=file_chunk_callback,  # Use the context-aware lambda
                pause_event=pause_event,
            )

            file_elapsed_time = time.time() - file_start_time
            if success:
                print(
                    f"   Successfully processed '{text_file}' in {file_elapsed_time:.2f}s"
                )
                generated_files.append(output_path)
                files_processed_successfully += 1
            else:
                print(f"   Failed to process '{text_file}' (check logs above)")

    except InterruptedError as e:
        print("\n--- Audiobook Generation Cancelled ---")
        # Progress callback might not be 100%, handle in UI if needed
        if progress_callback:
            progress_callback(None, "Cancelled", i, total_files)  # Signal cancellation
        # Don't re-raise, allow finally block to run
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred During Processing ---")
        print(f"   Error: {e}")
        traceback.print_exc()
        if progress_callback:
            progress_callback(None, "Error", i, total_files)  # Signal error
        # Don't re-raise, allow finally block to run
    finally:
        # --- Cleanup / Final Report ---
        print("\n--- Audiobook Generation Finished ---")
        total_process_time = time.time() - start_process_time
        print(
            f"  Successfully generated: {files_processed_successfully} / {total_files} files"
        )
        print(f"  Total time elapsed  : {total_process_time:.2f} seconds")
        # Ensure progress reaches 100% only if fully completed without cancellation/error
        if files_processed_successfully == total_files and not (
            cancellation_flag and cancellation_flag()
        ):
            if progress_callback:
                progress_callback(100, "Completed", total_files, total_files)

    return generated_files


# --- Functions for Testing ---


def generate_audio_for_all_voices_kokoro(
    input_path,  # Path to the single .txt file for testing
    lang_code,  # Language code for the pipeline
    voices,  # List of voice identifiers to test
    output_dir,  # Directory to save test audio files
    device=DEVICE,  # Device ('cuda' or 'cpu')
    speed=1.0,
    split_pattern=r"\n+",
    cancellation_flag=None,  # Optional cancellation
    progress_callback=None,  # Callback(overall_perc, voice_name, index, total)
    pause_event=None,  # Optional pause event
):
    """
    Generates audio samples for multiple voices from a single text file.

    Args:
        input_path (str): Path to the source .txt file.
        lang_code (str): Kokoro language code.
        voices (list[str]): List of voice identifiers to test.
        output_dir (str): Directory to save output audio files.
        device (str): Computation device ('cuda' or 'cpu').
        speed (float): Speech speed multiplier.
        split_pattern (str): Regex for splitting text.
        cancellation_flag (callable, optional): Function returning True to cancel.
        progress_callback (callable, optional): Reports overall progress.
        pause_event (threading.Event, optional): Event to pause processing.
    """
    print(f"\n--- Starting Test Generation for All Voices ---")
    print(f"  Input File : '{input_path}'")
    print(f"  Language   : {lang_code}")
    print(f"  Device     : {device}")

    if not os.path.isfile(input_path):
        print(f"  Error: Input text file not found: '{input_path}'")
        return
    if not voices:
        print("  Warning: No voices provided for testing.")
        return

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output Dir : '{output_dir}'")
    except OSError as e:
        print(f"  Error creating output directory '{output_dir}': {e}")
        return

    # --- Initialize Pipeline Once ---
    pipeline = None
    try:
        print(
            f"  Initializing Kokoro pipeline for lang='{lang_code}' on device='{device}'..."
        )
        # Resolve device similarly as above
        try:
            if device == "cuda" and not torch.cuda.is_available():
                print(
                    "  Warning: CUDA requested but not available; falling back to CPU."
                )
                device = "cpu"
        except Exception:
            device = "cpu"
        pipeline = KPipeline(lang_code=lang_code, device=device)
    except Exception as e:
        print(f"  Error initializing Kokoro pipeline: {e}")
        traceback.print_exc()
        return

    total_voices = len(voices)
    print(f"  Voices to test: {total_voices}")

    characters_processed_so_far = 0
    total_chars_in_file = 0  # Calculate once
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            total_chars_in_file = len(f.read())
    except Exception:
        pass  # Ignore error here, will be caught later if file is bad

    # --- Define Inner Callback for Chunk Progress ---
    def internal_test_chunk_callback(
        chars_in_chunk, chunk_duration, current_voice, current_index, total_voices
    ):
        nonlocal characters_processed_so_far
        # Accumulate characters processed *within the current voice's generation*
        # For overall progress across voices, we use the voice index.
        # This callback might be less useful here unless you want intra-voice progress.

        # Calculate overall progress based on voice index
        overall_progress = (
            (current_index - 1) / total_voices
        ) * 100  # Progress before current voice finishes
        # We could try to estimate progress *within* the current voice based on chars,
        # but let's keep it simple and update based on completed voices.

        if progress_callback:
            # Pass overall % based on voice index, current voice name, index, total
            progress_callback(
                overall_progress, current_voice, current_index, total_voices
            )

    # --- Loop Through Voices ---
    print("\n--- Generating Voice Samples ---")
    try:
        for i, voice in enumerate(voices, start=1):
            if cancellation_flag and cancellation_flag():
                print(f"\nCancellation detected before processing voice '{voice}'.")
                raise InterruptedError("Processing cancelled by user.")
            if pause_event:
                pause_event.wait()

            print(f"\n[{i}/{total_voices}] Testing Voice: '{voice}'")
            file_start_time = time.time()
            output_filename = f"test_{voice}.wav"  # Use WAV for testing consistency
            output_path = os.path.join(output_dir, output_filename)

            # Pass a lambda that captures the current voice context
            test_chunk_callback = lambda chars, duration: internal_test_chunk_callback(
                chars, duration, voice, i, total_voices
            )

            success = generate_audio_for_file_kokoro(
                input_path=input_path,
                pipeline=pipeline,
                voice=voice,
                output_path=output_path,
                speed=speed,
                split_pattern=split_pattern,
                cancellation_flag=cancellation_flag,
                chunk_progress_callback=test_chunk_callback,  # Use context-aware lambda
                pause_event=pause_event,
            )

            file_elapsed_time = time.time() - file_start_time
            if success:
                print(
                    f"   Successfully generated sample for '{voice}' in {file_elapsed_time:.2f}s"
                )
                # Update progress after successful completion of a voice
                if progress_callback:
                    progress_callback((i / total_voices) * 100, voice, i, total_voices)
            else:
                print(f"   Failed to generate sample for '{voice}'")
                # Optionally break or continue on failure

    except InterruptedError:
        print("\n--- Voice Test Generation Cancelled ---")
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred During Voice Testing ---")
        print(f"   Error: {e}")
        traceback.print_exc()
    finally:
        print("\n--- Voice Test Generation Finished ---")
        # Ensure 100% is reported if fully completed
        if not (cancellation_flag and cancellation_flag()):
            if progress_callback:
                progress_callback(100, "Completed", total_voices, total_voices)


def test_single_voice_kokoro(
    input_text,  # Raw text string
    voice,  # Voice identifier
    output_path,  # Full path for the output audio file
    lang_code="a",  # Language code (must match voice)
    device=DEVICE,  # Device ('cuda' or 'cpu')
    speed=1.0,
    split_pattern=r"\n+",
    cancellation_flag=None,
    progress_callback=None,  # Callback(overall_perc, filename, 1, 1)
    pause_event=None,
):
    """
    Generates a test audio sample for a single voice from a text string.

    Creates a temporary file to hold the text.

    Args:
        input_text (str): The text to synthesize.
        voice (str): Kokoro voice identifier.
        output_path (str): Full path to save the output audio file.
        lang_code (str): Kokoro language code.
        device (str): Computation device ('cuda' or 'cpu').
        speed (float): Speech speed multiplier.
        split_pattern (str): Regex for splitting text.
        cancellation_flag (callable, optional): Function returning True to cancel.
        progress_callback (callable, optional): Reports overall progress (0-100).
        pause_event (threading.Event, optional): Event to pause processing.

    Returns:
        str or None: Path to the generated audio file on success, None on failure.
    """
    import tempfile  # Keep import local as it's only used here

    print(f"\n--- Starting Single Voice Test ---")
    print(f"  Voice      : {voice}")
    print(f"  Language   : {lang_code}")
    print(f"  Device     : {device}")
    print(f"  Output File: '{output_path}'")

    if not input_text.strip():
        print("  Error: Input text is empty.")
        return None

    temp_file_path = None  # Define outside try
    try:
        # --- Create Temporary File ---
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".txt", delete=False, encoding="utf-8"
        ) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write(input_text)
        print(f"  Created temp input file: '{temp_file_path}'")

        # --- Initialize Pipeline ---
        pipeline = None
        try:
            print(f"  Initializing Kokoro pipeline...")
            try:
                if device == "cuda" and not torch.cuda.is_available():
                    print(
                        "  Warning: CUDA requested but not available; falling back to CPU."
                    )
                    device = "cpu"
            except Exception:
                device = "cpu"
            pipeline = KPipeline(lang_code=lang_code, device=device)
        except Exception as e:
            print(f"  Error initializing Kokoro pipeline: {e}")
            traceback.print_exc()
            return None  # Cannot proceed without pipeline

        # --- Ensure Output Directory Exists ---
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        except OSError as e:
            print(f"  Error creating output directory for '{output_path}': {e}")
            return None

        # --- Define Progress Callback ---
        total_chars_in_file = len(input_text)
        chars_processed_so_far = 0

        def single_test_chunk_callback(chars_in_chunk, chunk_duration):
            nonlocal chars_processed_so_far
            chars_processed_so_far += chars_in_chunk
            overall_progress = 0
            if total_chars_in_file > 0:
                overall_progress = (chars_processed_so_far / total_chars_in_file) * 100
                overall_progress = min(max(overall_progress, 0), 100)
            if progress_callback:
                # Pass progress %, fixed filename context "Test Sample", index 1 of 1
                progress_callback(overall_progress, "Test Sample", 1, 1)

        # --- Generate Audio ---
        print(f"  Generating test audio...")
        start_time = time.time()
        success = generate_audio_for_file_kokoro(
            input_path=temp_file_path,
            pipeline=pipeline,
            voice=voice,
            output_path=output_path,
            speed=speed,
            split_pattern=split_pattern,
            cancellation_flag=cancellation_flag,
            chunk_progress_callback=single_test_chunk_callback,  # Use specific callback
            pause_event=pause_event,
        )
        elapsed_time = time.time() - start_time

        if success:
            print(f"  Successfully generated test sample in {elapsed_time:.2f}s")
            print(f"  Output saved to: '{output_path}'")
            if progress_callback:
                progress_callback(100, "Completed", 1, 1)  # Ensure 100% on success
            return output_path
        else:
            print(f"  Failed to generate test sample for voice '{voice}'")
            if progress_callback:
                progress_callback(None, "Failed", 1, 1)  # Signal failure
            return None

    except InterruptedError:
        print("\n--- Single Voice Test Cancelled ---")
        if progress_callback:
            progress_callback(None, "Cancelled", 1, 1)
        return None
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred During Single Voice Test ---")
        print(f"   Error: {e}")
        traceback.print_exc()
        if progress_callback:
            progress_callback(None, "Error", 1, 1)
        return None
    finally:
        # --- Clean up Temporary File ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                # print(f"  Cleaned up temp file: '{temp_file_path}'") # Optional debug log
            except OSError as e:
                print(
                    f"  Warning: Could not remove temporary file '{temp_file_path}': {e}"
                )
