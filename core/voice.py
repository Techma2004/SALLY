import os
import tempfile
import sounddevice as sd
import soundfile as sf
from piper import PiperVoice
from pathlib import Path

# Get project root from this file's location
# core/voice.py -> SALLY/
PROJECT_ROOT = Path(__file__).parent.parent

# Import config for paths
from . import config

print(f"[SALLY Voice] Loading Piper TTS from {config.VOICE_MODEL_PATH}...")

# Ensure path exists, if not, give helpful error
if not Path(config.VOICE_MODEL_PATH).exists():
    raise FileNotFoundError(
        f"Voice model not found at {config.VOICE_MODEL_PATH}\n"
        f"Run: mkdir -p models/voices && cd models/voices && "
        f"wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx"
    )

tts_voice = PiperVoice.load(str(config.VOICE_MODEL_PATH))
print("[SALLY Voice] TTS ready.")

stt_model = None

def get_stt_model():
    global stt_model
    if stt_model is None:
        print(f"[SALLY Voice] Loading whisper {config.WHISPER_MODEL_SIZE}...")
        from faster_whisper import WhisperModel
        stt_model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
        print("[SALLY Voice] Whisper ready.")
    return stt_model

def speak(text: str):
    print(text)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    with sf.SoundFile(wav_path, 'w', samplerate=tts_voice.config.sample_rate, channels=1) as wav_file:
        for chunk in tts_voice.synthesize(text):
            wav_file.write(chunk.audio_float_array)

    try:
        data, samplerate = sf.read(wav_path)
        sd.play(data, samplerate)
        sd.wait()
    except Exception as e:
        print(f"[Audio Error] {e}")
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

def listen(duration=5):
    print(f"\n[Listening {duration}s...]")
    try:
        model = get_stt_model()
        samplerate = 16000
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
        sf.write(temp_wav, recording, samplerate)

        segments, _ = model.transcribe(temp_wav, beam_size=5, language="en")
        text = " ".join([s.text for s in segments]).strip()
        os.remove(temp_wav)
        print(f"You said: {text}")
        return text
    except Exception as e:
        print(f"[Listen Error] {e}")
        return ""
