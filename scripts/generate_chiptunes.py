"""
Generate chiptune-style WAV music files for BMO using pure Python wave synthesis.
Each track uses square waves, triangle waves, and simple arpeggios to create
classic 8-bit sounding melodies. Output is 44100Hz 16-bit PCM mono WAV.
"""
import struct
import wave
import math
import random
import os

SAMPLE_RATE = 44100
AMPLITUDE = 0.3  # Keep volume moderate

def square_wave(freq, t, duty=0.5):
    """Generate a square wave sample."""
    if freq == 0:
        return 0.0
    phase = (t * freq) % 1.0
    return 1.0 if phase < duty else -1.0

def triangle_wave(freq, t):
    """Generate a triangle wave sample."""
    if freq == 0:
        return 0.0
    phase = (t * freq) % 1.0
    return 4.0 * abs(phase - 0.5) - 1.0

def noise(t):
    """Simple noise generator (seeded per-sample for consistency)."""
    return random.uniform(-1, 1) * 0.3

def note_freq(note_name):
    """Convert note name to frequency. e.g., 'C4' -> 261.63"""
    notes = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
             'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
    if note_name == 'R':  # Rest
        return 0
    # Parse note name and octave
    if len(note_name) == 3:  # e.g., 'C#4'
        n, octave = note_name[:2], int(note_name[2])
    else:  # e.g., 'C4'
        n, octave = note_name[0], int(note_name[1])
    semitone = notes[n] + (octave - 4) * 12
    return 440.0 * (2.0 ** ((semitone - 9) / 12.0))

def render_melody(notes_list, wave_func, bpm=140, volume=0.3):
    """Render a list of (note, beats) into samples."""
    samples = []
    beat_duration = 60.0 / bpm
    for note, beats in notes_list:
        freq = note_freq(note) if note != 'R' else 0
        duration = beat_duration * beats
        num_samples = int(duration * SAMPLE_RATE)
        for i in range(num_samples):
            t = i / SAMPLE_RATE
            # Apply envelope: quick attack, sustain, release at end
            env = 1.0
            attack = 0.005
            release = 0.02
            if t < attack:
                env = t / attack
            elif t > duration - release:
                env = max(0, (duration - t) / release)
            sample = wave_func(freq, t) * volume * env
            samples.append(sample)
    return samples

def mix_tracks(*tracks):
    """Mix multiple tracks together, padding shorter ones with silence."""
    max_len = max(len(t) for t in tracks)
    mixed = [0.0] * max_len
    for track in tracks:
        for i, s in enumerate(track):
            mixed[i] += s
    # Clamp
    return [max(-1.0, min(1.0, s)) for s in mixed]

def save_wav(filename, samples):
    """Save samples as 16-bit PCM WAV."""
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        for s in samples:
            val = int(s * 32767 * AMPLITUDE / max(AMPLITUDE, 0.01))
            val = max(-32768, min(32767, val))
            wf.writeframes(struct.pack('<h', val))
    print(f"  Saved: {filename} ({len(samples)/SAMPLE_RATE:.1f}s, {os.path.getsize(filename)/1024:.0f}KB)")


def generate_bmo_adventure():
    """A cheerful adventure melody — like BMO starting a quest!"""
    melody = [
        ('C5', 0.5), ('E5', 0.5), ('G5', 0.5), ('C6', 1),
        ('B5', 0.5), ('G5', 0.5), ('E5', 0.5), ('C5', 1),
        ('D5', 0.5), ('F5', 0.5), ('A5', 0.5), ('D6', 1),
        ('C6', 0.5), ('A5', 0.5), ('F5', 0.5), ('D5', 1),
        ('E5', 0.5), ('G5', 0.5), ('B5', 0.5), ('E6', 1),
        ('D6', 0.5), ('B5', 0.5), ('G5', 0.5), ('E5', 1),
        ('C5', 0.5), ('E5', 0.5), ('G5', 1), ('C6', 1.5),
        ('R', 0.5),
        # Repeat with variation
        ('C5', 0.5), ('D5', 0.5), ('E5', 0.5), ('G5', 1),
        ('A5', 0.5), ('G5', 0.5), ('E5', 0.5), ('D5', 1),
        ('C5', 0.5), ('E5', 0.5), ('G5', 0.5), ('C6', 1),
        ('E6', 0.5), ('D6', 0.5), ('C6', 1), ('G5', 1),
        ('C6', 2),
    ]
    bass = [
        ('C3', 2), ('G3', 2), ('C3', 2), ('G3', 2),
        ('D3', 2), ('A3', 2), ('D3', 2), ('A3', 2),
        ('E3', 2), ('B3', 2), ('E3', 2), ('B3', 2),
        ('C3', 2), ('G3', 2), ('C3', 4),
    ]
    mel = render_melody(melody, square_wave, bpm=160, volume=0.25)
    bas = render_melody(bass, triangle_wave, bpm=160, volume=0.2)
    return mix_tracks(mel, bas)


def generate_pixel_dance():
    """An upbeat dance tune with quick arpeggios."""
    melody = [
        ('E5', 0.25), ('G5', 0.25), ('B5', 0.25), ('E6', 0.25),
        ('D6', 0.5), ('B5', 0.5),
        ('E5', 0.25), ('G5', 0.25), ('B5', 0.25), ('E6', 0.25),
        ('C6', 0.5), ('A5', 0.5),
        ('D5', 0.25), ('F#5', 0.25), ('A5', 0.25), ('D6', 0.25),
        ('C6', 0.5), ('A5', 0.5),
        ('C5', 0.25), ('E5', 0.25), ('G5', 0.25), ('C6', 0.25),
        ('B5', 0.5), ('G5', 0.5),
        # Second phrase
        ('E5', 0.25), ('G5', 0.25), ('B5', 0.25), ('E6', 0.25),
        ('F#6', 0.5), ('E6', 0.5),
        ('D5', 0.25), ('F#5', 0.25), ('A5', 0.25), ('D6', 0.25),
        ('E6', 0.5), ('D6', 0.5),
        ('C5', 0.25), ('E5', 0.25), ('G5', 0.25), ('C6', 0.25),
        ('D6', 0.5), ('C6', 0.5),
        ('B5', 1), ('R', 0.5), ('E6', 1.5),
    ]
    bass = [
        ('E3', 1), ('E3', 1), ('E3', 1), ('E3', 1),
        ('D3', 1), ('D3', 1), ('C3', 1), ('C3', 1),
        ('E3', 1), ('E3', 1), ('D3', 1), ('D3', 1),
        ('C3', 1), ('C3', 1), ('B2', 1), ('E3', 1),
    ]
    mel = render_melody(melody, square_wave, bpm=180, volume=0.22)
    bas = render_melody(bass, triangle_wave, bpm=180, volume=0.18)
    return mix_tracks(mel, bas)


def generate_starry_night():
    """A dreamy, slower melody — BMO looking at the stars."""
    melody = [
        ('G4', 1), ('B4', 1), ('D5', 1.5), ('R', 0.5),
        ('E5', 1), ('D5', 0.5), ('B4', 0.5), ('G4', 1.5), ('R', 0.5),
        ('A4', 1), ('C5', 1), ('E5', 1.5), ('R', 0.5),
        ('D5', 1), ('C5', 0.5), ('A4', 0.5), ('G4', 2),
        ('R', 1),
        ('G4', 0.5), ('A4', 0.5), ('B4', 1), ('D5', 1.5), ('R', 0.5),
        ('E5', 1), ('G5', 1), ('F#5', 1), ('D5', 1),
        ('B4', 1), ('G4', 1), ('A4', 1), ('B4', 1),
        ('G4', 3),
    ]
    bass = [
        ('G2', 2), ('D3', 2), ('G2', 2), ('D3', 2),
        ('A2', 2), ('E3', 2), ('D3', 2), ('G2', 2),
        ('G2', 2), ('D3', 2), ('E3', 2), ('D3', 2),
        ('G2', 2), ('D3', 2), ('G2', 4),
    ]
    mel = render_melody(melody, triangle_wave, bpm=100, volume=0.25)
    bas = render_melody(bass, triangle_wave, bpm=100, volume=0.15)
    return mix_tracks(mel, bas)


def generate_robot_march():
    """A marching, rhythmic tune — BMO on a mission!"""
    melody = [
        ('C5', 0.5), ('C5', 0.5), ('E5', 0.5), ('G5', 0.5),
        ('G5', 0.5), ('E5', 0.5), ('C5', 1),
        ('D5', 0.5), ('D5', 0.5), ('F5', 0.5), ('A5', 0.5),
        ('A5', 0.5), ('F5', 0.5), ('D5', 1),
        ('E5', 0.5), ('E5', 0.5), ('G5', 0.5), ('B5', 0.5),
        ('C6', 1), ('G5', 1),
        ('E5', 0.5), ('D5', 0.5), ('C5', 2),
        ('R', 1),
        # Second part
        ('G5', 0.5), ('G5', 0.5), ('A5', 0.5), ('B5', 0.5),
        ('C6', 1), ('A5', 1),
        ('F5', 0.5), ('F5', 0.5), ('G5', 0.5), ('A5', 0.5),
        ('B5', 1), ('G5', 1),
        ('E5', 0.5), ('D5', 0.5), ('C5', 0.5), ('E5', 0.5),
        ('G5', 1), ('C6', 1),
        ('C6', 2),
    ]
    bass = [
        ('C3', 1), ('G3', 1), ('C3', 1), ('G3', 1),
        ('D3', 1), ('A3', 1), ('D3', 1), ('A3', 1),
        ('E3', 1), ('B3', 1), ('C3', 1), ('G3', 1),
        ('C3', 2), ('R', 1),
        ('G3', 1), ('C3', 1), ('A3', 1), ('F3', 1),
        ('G3', 1), ('D3', 1), ('E3', 1), ('B2', 1),
        ('C3', 1), ('G3', 1), ('C3', 2),
    ]
    mel = render_melody(melody, square_wave, bpm=140, volume=0.22)
    bas = render_melody(bass, square_wave, bpm=140, volume=0.15)
    return mix_tracks(mel, bas)


def generate_game_over_fanfare():
    """A victory/game-over fanfare — short and triumphant!"""
    melody = [
        ('C5', 0.25), ('E5', 0.25), ('G5', 0.25), ('C6', 0.75),
        ('R', 0.25),
        ('C5', 0.25), ('E5', 0.25), ('G5', 0.25), ('C6', 0.75),
        ('R', 0.25),
        ('D5', 0.25), ('F#5', 0.25), ('A5', 0.25), ('D6', 0.75),
        ('R', 0.25),
        ('E5', 0.25), ('G5', 0.25), ('B5', 0.5),
        ('C6', 0.5), ('E6', 0.5), ('G6', 1.5),
        ('R', 0.5),
        # Grand finale
        ('C6', 0.25), ('D6', 0.25), ('E6', 0.25), ('F6', 0.25),
        ('G6', 1), ('E6', 0.5), ('C6', 0.5),
        ('G6', 1), ('E6', 0.5),
        ('C7', 2),
    ]
    mel = render_melody(melody, square_wave, bpm=150, volume=0.3)
    return mel


def generate_lullaby():
    """A gentle lullaby — BMO singing you to sleep."""
    melody = [
        ('E5', 1), ('D5', 0.5), ('C5', 1.5),
        ('E5', 1), ('D5', 0.5), ('C5', 1.5),
        ('E5', 0.5), ('F5', 0.5), ('G5', 1.5), ('R', 0.5),
        ('A5', 1), ('G5', 0.5), ('F5', 0.5), ('E5', 1.5), ('R', 0.5),
        ('D5', 1), ('E5', 0.5), ('F5', 1.5),
        ('E5', 1), ('D5', 0.5), ('C5', 1.5),
        ('D5', 0.5), ('E5', 0.5), ('D5', 1), ('C5', 2),
        ('R', 1),
        # Repeat softer
        ('C5', 1), ('D5', 0.5), ('E5', 1.5),
        ('G5', 1), ('F5', 0.5), ('E5', 1.5),
        ('D5', 1), ('C5', 0.5), ('D5', 1.5),
        ('C5', 3),
    ]
    bass = [
        ('C3', 3), ('G3', 3), ('C3', 3), ('F3', 3),
        ('G3', 3), ('C3', 3), ('G3', 3), ('C3', 3),
        ('C3', 3), ('G3', 3), ('G3', 3), ('C3', 3),
    ]
    mel = render_melody(melody, triangle_wave, bpm=80, volume=0.2)
    bas = render_melody(bass, triangle_wave, bpm=80, volume=0.12)
    return mix_tracks(mel, bas)


if __name__ == '__main__':
    output_dir = r'c:\Users\clevercode\OneDrive - Moores\Documents\GitHub\be-more-agent\sounds\music'
    
    # Remove corrupt stub files
    stubs = ['adventure_time.wav', 'bmo_jams.wav', 'chiptune_boss.wav']
    for stub in stubs:
        path = os.path.join(output_dir, stub)
        if os.path.exists(path):
            size = os.path.getsize(path)
            os.remove(path)
            print(f"Removed corrupt stub: {stub} ({size} bytes)")
    
    print()
    print("Generating chiptune tracks...")
    
    tracks = [
        ("bmo_adventure.wav", generate_bmo_adventure, "Cheerful adventure melody"),
        ("pixel_dance.wav", generate_pixel_dance, "Upbeat dance with arpeggios"),
        ("starry_night.wav", generate_starry_night, "Dreamy stargazing melody"),
        ("robot_march.wav", generate_robot_march, "Marching mission tune"),
        ("victory_fanfare.wav", generate_game_over_fanfare, "Triumphant victory jingle"),
        ("bmo_lullaby.wav", generate_lullaby, "Gentle lullaby"),
    ]
    
    for filename, generator, desc in tracks:
        print(f"\n  Generating: {desc}...")
        samples = generator()
        filepath = os.path.join(output_dir, filename)
        save_wav(filepath, samples)
    
    print(f"\nDone! Generated {len(tracks)} new chiptune tracks.")
    print(f"\nFinal music directory contents:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f}: {size/1024:.0f}KB ({size/SAMPLE_RATE/2:.1f}s)")
