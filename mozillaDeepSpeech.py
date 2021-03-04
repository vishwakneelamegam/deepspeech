#neede modules deepspeech, wave, numpy
import deepspeech
import wave
import numpy as np
#alpha, beta and beam width values
lm_alpha = 0.75
lm_beta = 1.85
beam_width = 500
#audio file path
filename = 'audio/8455-210777-0068.wav'
#model path
modelPath = 'deepspeech-0.9.3-models.pbmm'
scorerPath = 'deepspeech-0.9.3-models.scorer'
model = deepspeech.Model(modelPath)
model.enableExternalScorer(scorerPath)
model.setScorerAlphaBeta(lm_alpha,lm_beta)
model.setBeamWidth(beam_width)
w = wave.open(filename,'r')
rate = w.getframerate()
frames = w.getnframes()
buffer = w.readframes(frames)
data16 = np.frombuffer(buffer,dtype=np.int16)
text = model.stt(data16)
#use it when metadata needed
#textWithMetaData = model.sttWithMetadata(data16)
print(text)
