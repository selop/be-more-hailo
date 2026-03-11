import wave
import math
import struct
import os

SAMPLE_RATE = 22050

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def generate_tone(filepath, note_seq, duration=0.2, wave_type="square", volume=0.5):
    """Generates a raw 8-bit style chiptune wav file from a sequence of frequencies."""
    ensure_dir(os.path.dirname(filepath))
    
    with wave.open(filepath, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2) # 16-bit
        wav_file.setframerate(SAMPLE_RATE)
        
        for freq in note_seq:
            num_samples = int(SAMPLE_RATE * duration)
            for i in range(num_samples):
                # Calculate time
                t = float(i) / SAMPLE_RATE
                
                # ADSR style quick envelope
                env = 1.0
                if i < 0.1 * num_samples:
                    env = i / (0.1 * num_samples)
                elif i > 0.9 * num_samples:
                    env = (num_samples - i) / (0.1 * num_samples)
                    
                if wave_type == "square":
                    # Square wave
                    sample = 1.0 if math.sin(2.0 * math.pi * freq * t) > 0 else -1.0
                elif wave_type == "triangle":
                    # Triangle wave
                    sample = 2.0 * abs(2.0 * (t * freq - math.floor(t * freq + 0.5))) - 1.0
                elif wave_type == "noise":
                    # White noise for squeaks
                    import random
                    sample = random.uniform(-1.0, 1.0)
                else:
                    # Sine wave
                    sample = math.sin(2.0 * math.pi * freq * t)
                    
                sample_val = int(sample * 32767.0 * volume * env)
                wav_file.writeframes(struct.pack('<h', sample_val))

def generate_bee():
    # A fast, buzzing sound using low frequency square waves or triangle
    freqs = []
    # simulate a buzzing flight path
    for i in range(20):
        freqs.append(150 + 20 * math.sin(i * 0.5))
    generate_tone("sounds/personas/bee.wav", freqs, duration=0.05, wave_type="triangle", volume=0.3)

def generate_low_battery():
    # Descending warning beeps
    freqs = [800, 600, 400, 200]
    generate_tone("sounds/personas/low_battery.wav", freqs, duration=0.3, wave_type="square", volume=0.4)

def generate_sir_mano():
    # Polite, elegant classical arpeggio
    # C major arpeggio
    freqs = [523.25, 659.25, 783.99, 1046.50]
    generate_tone("sounds/personas/sir_mano.wav", freqs, duration=0.15, wave_type="square", volume=0.3)

def generate_detective():
    # Noir sting: slow minor interval
    freqs = [329.63, 261.63, 220.00] # E4, C4, A3
    generate_tone("sounds/personas/detective.wav", freqs, duration=0.4, wave_type="square", volume=0.4)

def generate_football():
    # Squeaky mirror wipe sound: high pitch noise sweeps
    freqs = [1500, 1800, 2100, 1500, 1800, 2100]
    generate_tone("sounds/personas/football.wav", freqs, duration=0.1, wave_type="triangle", volume=0.2)

if __name__ == "__main__":
    print("Synthesizing sounds...")
    generate_bee()
    generate_low_battery()
    generate_sir_mano()
    generate_detective()
    generate_football()
    print("Finished.")
