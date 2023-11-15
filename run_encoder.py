import whisper

model = whisper.load_model("tiny")

options = dict(language="ja", task="translate", temperature=0, beam_size=1)
result = model.transcribe(audio="erwin_jp.wav", **options)
print(result)

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("erwin_jp.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)
audio_features = model._get_audio_features(mel)

