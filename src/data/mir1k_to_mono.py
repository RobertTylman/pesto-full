
import os
import argparse
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

def process_mir1k(input_dir, output_dir):
    """
    Takes the MIR-1K dataset (audio in left side, music in right side)
    and combines it into a mono file.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    wav_files = list(input_path.glob("**/*.wav"))
    if not wav_files:
        print(f"No .wav files found in {input_dir}")
        return

    print(f"Found {len(wav_files)} files to process.")

    for wav_file in tqdm(wav_files):
        try:
            # Load the audio file using soundfile
            # soundfile returns (samples, channels) and sample_rate
            audio_data, sample_rate = sf.read(wav_file)
            
            # Convert to torch tensor and transpose to (channels, samples)
            waveform = torch.from_numpy(audio_data).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.t()

            # Check if it's stereo
            if waveform.shape[0] == 2:
                # Right channel is music (accompaniment)
                mono_waveform = waveform[1:2, :]
            else:
                # Already mono, just copy
                mono_waveform = waveform

            # Define output path (mirrors input structure if desired, or flat)
            relative_path = wav_file.relative_to(input_path)
            target_file = output_path / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)

            # Save the mono file
            # soundfile expects (samples, channels)
            waveform_np = mono_waveform.t().numpy()
            sf.write(target_file, waveform_np, sample_rate)

        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MIR-1K stereo files to mono (right channel only).")
    parser.add_argument("input_dir", type=str, help="Directory containing MIR-1K .wav files")
    parser.add_argument("output_dir", type=str, help="Directory to save converted mono files")
    
    args = parser.parse_args()
    process_mir1k(args.input_dir, args.output_dir)
