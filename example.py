import whisper
from pprint import pprint

model = whisper.load_model("tiny")

options = dict(language="ja", task="translate", temperature=0, beam_size=1)
result = model.transcribe(audio="erwin_jp.wav", **options)

pprint(result)

