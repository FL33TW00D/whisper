import whisper

model = whisper.load_model("large")
result = model.transcribe("mm0.wav")
tokens = [segment["tokens"] for segment in result["segments"]]
all_tokens = []
for t in tokens:
    all_tokens.extend(t)
print("All tokens:", all_tokens)
print("Number of tokens:", len(all_tokens))
