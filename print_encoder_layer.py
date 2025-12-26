from finetune_codes.model import KimiAudioModel

model = KimiAudioModel.from_pretrained("output/pretrained_hf", 
                                        device_map=None)

# Apply low-rank decomposition to whisper encoder
whisper = model.whisper_model
print(whisper)
print(whisper.speech_encoder.config.num_hidden_layers)
print(whisper.speech_encoder)
# for key, value in whisper.named_modules():
#     print(f"{key}: {value}")