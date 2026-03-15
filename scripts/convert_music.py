import os
from pydub import AudioSegment

music_dir = 'sounds/music'
files = [f for f in os.listdir(music_dir) if f.endswith('.wav')]

for f in files:
    filepath = os.path.join(music_dir, f)
    print(f'Converting {f}...')
    try:
        # Load the MP3 disguised as a WAV
        audio = AudioSegment.from_file(filepath)
        # Export as true PCM WAV for aplay
        audio.export(filepath, format="wav", parameters=["-acodec", "pcm_s16le", "-ar", "44100"])
        print(f'Success: {f}')
    except Exception as e:
        print(f'Failed {f}: {e}')
