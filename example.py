import whisper

model = whisper.load_model("large")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("jfk.wav")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio)

# decode the audio
options = whisper.DecodingOptions(fp16=False, without_timestamps=False)
result = whisper.decode(model, mel, options)

# print the recognized text
print(result)
