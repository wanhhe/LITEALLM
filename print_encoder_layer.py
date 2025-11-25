from finetune_codes.model import KimiAudioModel

model = KimiAudioModel.from_pretrained("output/pretrained_hf", 
                                        device_map=None)

# 针对 whisper encoder 应用低秩分解
whisper = model.whisper_model
print(whisper)
print(whisper.speech_encoder.config.num_hidden_layers)
print(whisper.speech_encoder)
# for key, value in whisper.named_modules():
#     print(f"{key}: {value}")