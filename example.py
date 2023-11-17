import whisper
from pprint import pprint

model = whisper.load_model("tiny")

options = dict(language="ja", task="translate", temperature=0)
result = model.transcribe(audio="erwin_jp.wav", **options)

print("Raw tokens: ", result["raw_tokens"])
flat_tokens = [token for segment in result["raw_tokens"] for token in segment]
print("Flat tokens: ", flat_tokens)

