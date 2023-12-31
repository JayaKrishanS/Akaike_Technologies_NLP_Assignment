"""
Speech-to-Text Model

This script provides a function for transcribing audio files into text using 
the Hugging Face Whisper-Large automatic speech recognition (ASR) model.

Importing the following packages:
- sys
- torch
- transformers.AutoModelForSpeechSeq2Seq
- transformers.AutoProcessor
- transformers.pipeline

Functions:
- get_transcription(filename: str) -> str`: Transcribes the input audio file 
(`filename`) into text using the Whisper-Large ASR model.
"""

# Importing the packages
import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

# Determine the device for computation
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# Define the pre-trained model
MODEL_ID = "openai/whisper-large-v3"

# Initializing the Speech-to-Text model with the specified model identifier
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    use_safetensors=True
)
model.to(DEVICE)

# Processor for the Whisper-Large model
processor = AutoProcessor.from_pretrained(MODEL_ID)

# Creating an ASR (Automatic Speech Recognition) pipeline using Hugging Face's transformers library
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=TORCH_DTYPE,
    device=DEVICE,
)

#Function to transcript the audio file
def get_transcription(filename: str):
    """
    This function is for Translating the audio to text.
    Args: filename (str): The path to the input audio file.
    Checking whether the filename is a string or not, else raise a Value error
    """
    if not isinstance(filename, str):
        raise ValueError("Input filename must be a string.")
    result = pipe(filename, generate_kwargs={"language": "marathi"})
    transcription = result["text"]
    return transcription

if __name__ == "__main__":
    if len(sys.argv) == 2:
        input_filename = sys.argv[1]
    else:
        input_filename = input()

    transcription_result = get_transcription(input_filename)
    print(transcription_result)
