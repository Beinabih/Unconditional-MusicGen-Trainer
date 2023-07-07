from audiocraft.models import MusicGen
import audiocraft
from audiocraft.models.encodec import EncodecModel
from audiocraft.models.lm import LMModel
from audiocraft import quantization as qt
from audiocraft.modules.codebooks_patterns import DelayedPatternProvider
from audiocraft.modules.conditioners import (
    ConditioningProvider,
    T5Conditioner,
    ConditionFuser,

)
import torch

def load_model(config, model_name):
    """load the pretained model or use the "small" model config for training from scratch

    Args:
        config (dict): train config
        model_name (str): model name

    Returns:
        MusicGen: MusicGen Model
    """
    if config["model"]:
        model = MusicGen.get_pretrained(config["model"])
    else:
        encoder_kwargs = {
            "dimension": 128,
            "channels": config["channels"],
            "causal": False,
            "n_filters": 64,
            "n_residual_layers": 1,
            "ratios": [8, 5, 4, 4],
            "activation": "ELU",
            "activation_params": {"alpha": 1.0},
            "norm": "weight_norm",
            "norm_params": {},
            "kernel_size": 7,
            "residual_kernel_size": 3,
            "last_kernel_size": 7,
            "dilation_base": 2,
            "pad_mode": "reflect",
            "true_skip": True,
            "compress": 2,
            "lstm": 2,
            "disable_norm_outer_blocks": 0,
        }
        decoder_kwargs = {
            "dimension": 128,
            "channels": config["channels"],
            "causal": False,
            "n_filters": 64,
            "n_residual_layers": 1,
            "ratios": [8, 5, 4, 4],
            "activation": "ELU",
            "activation_params": {"alpha": 1.0},
            "norm": "weight_norm",
            "norm_params": {},
            "kernel_size": 7,
            "residual_kernel_size": 3,
            "last_kernel_size": 7,
            "dilation_base": 2,
            "pad_mode": "reflect",
            "true_skip": True,
            "compress": 2,
            "lstm": 2,
            "disable_norm_outer_blocks": 0,
            "trim_right_ratio": 1.0,
            "final_activation": None,
            "final_activation_params": None,
        }

        quantizer_args = {
            "n_q": 4,
            "q_dropout": False,
            "bins": 2048,
            "decay": 0.99,
            "kmeans_init": True,
            "kmeans_iters": 50,
            "threshold_ema_dead_code": 2,
            "orthogonal_reg_weight": 0.0,
            "orthogonal_reg_active_codes_only": False,
            "dimension": 128,
        }

        encoder = audiocraft.modules.SEANetEncoder(**encoder_kwargs)
        decoder = audiocraft.modules.SEANetDecoder(**decoder_kwargs)

        quantizer = qt.ResidualVectorQuantizer(**quantizer_args)

        frame_rate = config["sample_rate"] // encoder.hop_length

        compression_model = EncodecModel(
            encoder,
            decoder,
            quantizer,
            frame_rate=frame_rate,
            channels=2,
            sample_rate=config["sample_rate"],
        ).to("cuda")

        pattern_provider = DelayedPatternProvider(
            n_q=4, **{"delays": [0, 1, 2, 3], "flatten_first": 0, "empty_initial": 0}
        )

        conditioners = {
            "description": T5Conditioner(
                output_dim=1024,
                device="cuda",
                **{
                    "name": "t5-base",
                    "finetune": False,
                    "word_dropout": 0.3,
                    "normalize_text": False,
                }
            )
        }

        condition_provider = ConditioningProvider(
            conditioners,
            device="cuda",
            **{"merge_text_conditions_p": 0.25, "drop_desc_p": 0.5}
        )

        fuser = ConditionFuser(
            fuse2cond={'sum': [], 'cross': ['description'], 'prepend': [], 'input_interpolate': []},
            **{"cross_attention_pos_emb": False, "cross_attention_pos_emb_scale": 1}
        )

        kwargs = {
            "dim": 1024,
            "num_heads": 4,
            "num_layers": 6,
            "hidden_scale": 2,
            "n_q": 4,
            "card": 2048,
            "dropout": 0.0,
            "emb_lr": None,
            "activation": "gelu",
            "norm_first": True,
            "bias_ff": False,
            "bias_attn": False,
            "bias_proj": False,
            "past_context": None,
            "causal": True,
            "custom": False,
            "memory_efficient": True,
            "attention_as_float32": False,
            "layer_scale": None,
            "positional_embedding": "sin",
            "xpos": False,
            "checkpointing": "none",
            "weight_init": "gaussian",
            "depthwise_init": "current",
            "zero_bias_init": True,
            "norm": "layer_norm",
            "cross_attention": False,
            "qk_layer_norm": False,
            "qk_layer_norm_cross": False,
            "attention_dropout": None,
            "kv_repeat": 1,
            "two_step_cfg": False,
        }
        lm = LMModel(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            cfg_dropout=0.3,
            cfg_coef=3.0,
            attribute_dropout={},
            dtype=torch.float16,
            device="cuda",
            **kwargs
        ).to("cuda")

        model = MusicGen(model_name, compression_model, lm)

    return model
