import torch
from audiocraft.utils.autocast import TorchAutocast
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout



def generate_long_seq(model, num_samples, total_gen_len, use_sampling, temp, top_k, top_p, cfg_coef):

    samples = []
    prev_generated = None
    descriptions = [None] 

    attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, None)
    remove_prompts = False

    with model.autocast:
        for i in range(num_samples):
            sample = model.lm.generate(prev_generated, attributes, 1, total_gen_len, use_sampling, temp, top_k, top_p, cfg_coef, remove_prompts=remove_prompts)
            remove_prompts = True
            
            if sample.shape[2] == total_gen_len:
                prev_generated = torch.clone(sample[..., total_gen_len//2:])
            else:
                prev_generated = torch.clone(sample)
            
            with torch.no_grad():
                gen_audio = model.compression_model.decode(sample, None)
            # free gpu 
            del sample

            gen_audio = gen_audio[0].detach().cpu().numpy()
            gen_audio = gen_audio.transpose(1, 0)
            samples.append(gen_audio)

    return samples

