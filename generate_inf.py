import torch
from audiocraft.utils.autocast import TorchAutocast
from audiocraft.modules.conditioners import ClassifierFreeGuidanceDropout


def generate_long_seq(
    model, num_samples, total_gen_len, use_sampling, temp, top_k, top_p, cfg_coef
):
    """Instead of using a text prompt, half the sample length of the previous generation is used as prompt input 

    Args:
        model (MusicGen): The pretrained MusicGen Model
        num_samples (integer): How many samples should be created, one sample is half the totgal_gen_len
        total_gen_len (integer): Maximum generation length.
        use_sampling (bool): Whether to use a sampling strategy or not.
        temp (float): Softmax temperature parameter. Defaults to 1.0.
        top_k (integer): top_k used for sampling. Defaults to 250.
        top_p (float): top_p used for sampling, when set to 0 top_k is used. Defaults to 0.0.
        cfg_coef (float): Coefficient used for classifier free guidance. Defaults to 3.0.

    Returns:
        List[torch.Tensor]: Output is a list of numpy vectors
    """
    samples = []
    prev_generated = None
    descriptions = [None]

    attributes, prompt_tokens = model._prepare_tokens_and_attributes(descriptions, None)
    remove_prompts = False

    with model.autocast:
        for i in range(num_samples):
            sample = model.lm.generate(
                prev_generated,
                attributes,
                1,
                total_gen_len,
                use_sampling,
                temp,
                top_k,
                top_p,
                cfg_coef,
                remove_prompts=remove_prompts,
            )
            remove_prompts = True

            if sample.shape[2] == total_gen_len:
                prev_generated = torch.clone(sample[..., total_gen_len // 2 :])
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
