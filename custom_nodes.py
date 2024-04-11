from tsc_utils import *
from .efficiency_nodes import encode_prompts, TSC_Apply_ControlNet_Stack

from PIL import Image

import comfy.diffusers_load
import comfy.samplers
import comfy.sample
import comfy.sd
import comfy.utils
import comfy.controlnet

import comfy.clip_vision
import comfy.model_management
from nodes import MAX_RESOLUTION

import torch
import folder_paths
import numpy as np


########################################################################################################################
# Custom Number Operation
class NumberOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (
                    ["addition", "subtraction", "multiplication", "division"],
                ),
                "number1": (
                    "NUMBER",
                    {
                        "default": 0,
                    },
                ),
                "number2": (
                    "NUMBER",
                    {
                        "default": 0,
                    },
                ),
            },
        }

    RETURN_TYPES = ("NUMBER", "FLOAT", "INT")

    FUNCTION = "execute"

    CATEGORY = "custom"

    def execute(self, operation, number1, number2):
        if operation == "addition":
            ans = number1 + number2
        elif operation == "subtraction":
            ans = number1 - number2
        elif operation == "multiplication":
            ans = number1 * number2
        elif operation == "division":
            ans = number1 / number2

        return (ans, ans, int(ans))


########################################################################################################################
# Custom Float Operation
class FloatOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (
                    ["addition", "subtraction", "multiplication", "division"],
                ),
                "number1": (
                    "FLOAT",
                    {
                        "default": 0,
                    },
                ),
                "number2": (
                    "FLOAT",
                    {
                        "default": 0,
                    },
                ),
            },
        }

    RETURN_TYPES = ("NUMBER", "FLOAT", "INT")

    FUNCTION = "execute"

    CATEGORY = "custom"

    def execute(self, operation, number1, number2):
        if operation == "addition":
            ans = number1 + number2
        elif operation == "subtraction":
            ans = number1 - number2
        elif operation == "multiplication":
            ans = number1 * number2
        elif operation == "division":
            ans = number1 / number2

        return (ans, ans, int(ans))


########################################################################################################################
# Custom Int Operation
class IntOperation:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "operation": (
                    ["addition", "subtraction", "multiplication", "division"],
                ),
                "number1": (
                    "INT",
                    {
                        "default": 0,
                    },
                ),
                "number2": (
                    "INT",
                    {
                        "default": 0,
                    },
                ),
            },
        }

    RETURN_TYPES = ("NUMBER", "FLOAT", "INT")

    FUNCTION = "execute"

    CATEGORY = "custom"

    def execute(self, operation, number1, number2):
        if operation == "addition":
            ans = number1 + number2
        elif operation == "subtraction":
            ans = number1 - number2
        elif operation == "multiplication":
            ans = number1 * number2
        elif operation == "division":
            ans = number1 / number2

        return (ans, ans, int(ans))


########################################################################################################################
# Custom Loader
class CustomLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "image_input": (["true", "false"],),
                "weight_interpretation": (
                    ["comfy", "A1111", "compel", "comfy++", "down_weight"],
                ),
                "positive": (
                    "STRING",
                    {"multiline": True, "default": "positive prompt"},
                ),
                "negative": (
                    "STRING",
                    {"multiline": True, "default": "negative prompt"},
                ),
            },
            "optional": {
                "lora_stack": ("LORA_STACK",),
                "cnet_stack": ("CONTROL_NET_STACK",),
                "image": ("IMAGE",),
                "empty_latent_width": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64},
                ),
                "empty_latent_height": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64},
                ),
                "image_base_size": (
                    "INT",
                    {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64},
                ),
                "base_denoise": (
                    "FLOAT",
                    {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "LATENT",
        "VAE",
        "CLIP",
        "FLOAT",
        "DEPENDENCIES",
    )
    RETURN_NAMES = (
        "model",
        "positive",
        "negative",
        "latent",
        "vae",
        "clip",
        "initial_denoise",
        "dependencies",
    )
    FUNCTION = "execute"
    CATEGORY = "custom"

    def execute(
        self,
        model_name,
        vae_name,
        clip_skip,
        token_normalization,
        image_input,
        weight_interpretation,
        positive,
        negative,
        empty_latent_width=None,
        empty_latent_height=None,
        image_base_size=None,
        base_denoise=None,
        lora_stack=None,
        cnet_stack=None,
        refiner_name="None",
        image=None,
        ascore=None,
        prompt=None,
        my_unique_id=None,
        loader_type="regular",
    ):
        globals_cleanup(prompt)

        vae_cache, ckpt_cache, lora_cache, refn_cache = get_cache_numbers(
            "Custom Loader"
        )

        if image_input == "true" and image == None:
            raise ValueError("Image must be specified when using image input")
        if image_input == "true" and (not image_base_size or not base_denoise):
            raise ValueError("Image size must be specified when using image input")
        if image_input == "false" and (
            not empty_latent_height or not empty_latent_width
        ):
            raise ValueError(
                "Height and width must be specified when not using image input"
            )

        if lora_stack:
            lora_params = []
            lora_params.extend(lora_stack)

            model, clip = load_lora(
                lora_params,
                model_name,
                my_unique_id,
                cache=lora_cache,
                ckpt_cache=ckpt_cache,
                cache_overwrite=True,
            )

            if vae_name == "Baked VAE":
                vae = get_bvae_by_ckpt_name(model_name)
        else:
            model, clip, vae = load_checkpoint(
                model_name, my_unique_id, cache=ckpt_cache, cache_overwrite=True
            )
            lora_params = None

        if refiner_name != "None":
            refiner_model, refiner_clip, _ = load_checkpoint(
                refiner_name,
                my_unique_id,
                output_vae=False,
                cache=refn_cache,
                cache_overwrite=True,
                ckpt_type="refn",
            )
        else:
            refiner_model = refiner_clip = None

        refiner_clip_skip = clip_skip[1] if loader_type == "sdxl" else None
        clip_skip = clip_skip[0] if loader_type == "sdxl" else clip_skip

        (
            positive_encoded,
            negative_encoded,
            clip,
            refiner_positive_encoded,
            refiner_negative_encoded,
            refiner_clip,
        ) = encode_prompts(
            positive,
            negative,
            token_normalization,
            weight_interpretation,
            clip,
            clip_skip,
            refiner_clip,
            refiner_clip_skip,
            ascore,
            loader_type == "sdxl",
            empty_latent_width,
            empty_latent_height,
        )

        if cnet_stack:
            controlnet_conditioning = TSC_Apply_ControlNet_Stack().apply_cnet_stack(
                positive_encoded, negative_encoded, cnet_stack
            )
            positive_encoded, negative_encoded = (
                controlnet_conditioning[0],
                controlnet_conditioning[1],
            )

        if vae_name != "Baked VAE":
            vae = load_vae(
                vae_name, my_unique_id, cache=vae_cache, cache_overwrite=True
            )

        dependencies = (
            vae_name,
            model_name,
            clip,
            clip_skip,
            refiner_name,
            refiner_clip,
            refiner_clip_skip,
            positive,
            negative,
            token_normalization,
            weight_interpretation,
            ascore,
            empty_latent_width,
            empty_latent_height,
            lora_params,
            cnet_stack,
        )

        if image_input == "true":
            # Get image ratio
            w, h = tensor2pil(image).size
            ratio = w / h
            if w > h:
                height = image_base_size
                width = int(image_base_size * ratio)
            else:
                width = image_base_size
                height = int(image_base_size / ratio)

            print("Image width: ", width)
            print("Image height: ", height)
            samples = image.movedim(-1, 1)
            s = comfy.utils.common_upscale(
                samples, width, height, "bilinear", "disabled"
            )
            s = s.movedim(1, -1)
            latent = vae.encode(s[:, :, :, :3])
            denoise = base_denoise
        else:
            latent = torch.zeros(
                [1, 4, empty_latent_height // 8, empty_latent_width // 8]
            ).cpu()
            denoise = 1.0

        latent = {"samples": latent}
        if loader_type == "regular":
            return (
                model,
                positive_encoded,
                negative_encoded,
                latent,
                vae,
                clip,
                denoise,
                dependencies,
            )
        elif loader_type == "sdxl":
            return (
                (
                    model,
                    clip,
                    positive_encoded,
                    negative_encoded,
                    refiner_model,
                    refiner_clip,
                    refiner_positive_encoded,
                    refiner_negative_encoded,
                ),
                positive_encoded,
                negative_encoded,
                latent,
                vae,
                clip,
                denoise,
                dependencies,
            )


def register_nodes(c):
    c.update({"Custom Number Operation": NumberOperation})
    c.update({"Custom Float Operation": FloatOperation})
    c.update({"Custom Int Operation": IntOperation})
    c.update({"Custom Loader": CustomLoader})
