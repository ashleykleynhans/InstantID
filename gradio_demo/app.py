import sys
sys.path.append('./')

from typing import Tuple

import os
import cv2
import math
import torch
import random
import numpy as np
import argparse
import logging

import PIL
from PIL import Image

import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel
from diffusers import LCMScheduler

from huggingface_hub import hf_hub_download

from insightface.app import FaceAnalysis

from style_template import styles
from pipeline_stable_diffusion_xl_instantid_full import StableDiffusionXLInstantIDPipeline
from model_util import load_models_xl

import gradio as gr

# Global variables
MAX_SEED = np.iinfo(np.int32).max
LOG_LEVEL = logging.INFO
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"
DEFAULT_MODEL = "wangqixun/YamerMIX_v8"
MODEL_DIRECTORY = "./models"
CHECKPOINT_DIRECTORY = "./checkpoints"

# Set device and torch_dtype
torch_dtype = torch.float16

if torch.backends.mps.is_available():
    device = "mps"
    torch_dtype = torch.float32
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Download ControlNet checkpoint from Hugging Face Hub if the files do not already exist
if not os.path.exists(os.path.join(CHECKPOINT_DIRECTORY, "ControlNetModel/config.json")):
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/config.json",
        local_dir=CHECKPOINT_DIRECTORY,
        local_dir_use_symlinks=False
    )
if not os.path.exists(os.path.join(CHECKPOINT_DIRECTORY, "ControlNetModel/diffusion_pytorch_model.safetensors")):
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ControlNetModel/diffusion_pytorch_model.safetensors",
        local_dir=CHECKPOINT_DIRECTORY,
        local_dir_use_symlinks=False
    )

# Download IP-Adapter checkpoint from Hugging Face Hub if the files do not already exist
if not os.path.exists(os.path.join(CHECKPOINT_DIRECTORY, "ip-adapter.bin")):
    hf_hub_download(
        repo_id="InstantX/InstantID",
        filename="ip-adapter.bin",
        local_dir=CHECKPOINT_DIRECTORY,
        local_dir_use_symlinks=False
    )

# Load face encoder
app = FaceAnalysis(name="antelopev2", root="./", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f"./{CHECKPOINT_DIRECTORY}/ip-adapter.bin"
controlnet_path = f"./{CHECKPOINT_DIRECTORY}/ControlNetModel"

# Load ControlNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)


def get_pipeline(model_path):
    if model_path.endswith(
        ".ckpt"
    ) or model_path.endswith(".safetensors"):
        scheduler_kwargs = hf_hub_download(
            repo_id="wangqixun/YamerMIX_v8",
            subfolder="scheduler",
            filename="scheduler_config.json",
        )

        (tokenizers, text_encoders, unet, _, vae) = load_models_xl(
            pretrained_model_name_or_path=model_path,
            scheduler_name=None,
            weight_dtype=torch_dtype,
        )

        scheduler = diffusers.EulerDiscreteScheduler.from_config(scheduler_kwargs)
        pipe = StableDiffusionXLInstantIDPipeline(
            vae=vae,
            text_encoder=text_encoders[0],
            text_encoder_2=text_encoders[1],
            tokenizer=tokenizers[0],
            tokenizer_2=tokenizers[1],
            unet=unet,
            scheduler=scheduler,
            controlnet=controlnet,
        )

    else:
        pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            feature_extractor=None,
        )

        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    if device == "mps":
        pipe.to("mps", torch_dtype)
        pipe.enable_attention_slicing()
    elif device == "cuda":
        pipe.cuda()

    pipe.load_ip_adapter_instantid(face_adapter)

    if device == "mps" or device == "cuda":
        pipe.image_proj_model.to(device)
        pipe.unet.to(device)

    # Load and disable LCM
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdxl")
    pipe.disable_lora()

    return pipe


def toggle_lcm_ui(value):
    if value:
        return (
            gr.update(minimum=0, maximum=100, step=1, value=5),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=1.5)
        )
    else:
        return (
            gr.update(minimum=5, maximum=100, step=1, value=30),
            gr.update(minimum=0.1, maximum=20.0, step=0.1, value=5)
        )


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def remove_tips():
    return gr.update(visible=False)


def get_example():
    return [
        [
            "./examples/yann-lecun_resize.jpg",
            "a man",
            "Snow",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/musk_resize.jpeg",
            "a man",
            "Mars",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/sam_resize.png",
            "a man",
            "Jungle",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, gree",
        ],
        [
            "./examples/schmidhuber_resize.png",
            "a man",
            "Neon",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
        [
            "./examples/kaifu_resize.png",
            "a man",
            "Vibrant Color",
            "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, photo, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
        ],
    ]


def run_for_examples(face_file, prompt, style, negative_prompt):
    global model_path

    return generate_image(
        model_path,
        face_file,
        None,
        prompt,
        negative_prompt,
        style,
        30,
        0.8,
        0.8,
        5,
        42,
        False,
        True
    )


def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_kps(image_pil, kps, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]

        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil


def resize_img(input_image, max_side=1280, min_side=1024, size=None,
               pad_to_max_side=False, mode=PIL.Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def apply_style(style_name: str, positive: str, negative: str = "") -> Tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative


def generate_image(model_path, face_image_path, pose_image_path, prompt, negative_prompt,
                   style_name, num_steps, identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, seed,
                   enable_lcm, enhance_face_region, progress=gr.Progress(track_tqdm=True)):
    global min_side, max_side

    if model_path is None:
        model_path = DEFAULT_MODEL

    if face_image_path is None:
        raise gr.Error(f"Cannot find any input face image! Please upload the face image")

    if prompt is None:
        prompt = "a person"

    # apply the style template
    prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)

    face_image = load_image(face_image_path)
    face_image = resize_img(face_image, max_side, min_side)
    face_image_cv2 = convert_from_image_to_cv2(face_image)
    height, width, _ = face_image_cv2.shape

    # Extract face features
    face_info = app.get(face_image_cv2)

    if len(face_info) == 0:
        raise gr.Error(f"Cannot find any face in the image! Please upload another person image")

    face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[-1]  # only use the maximum face
    face_emb = face_info["embedding"]
    face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info["kps"])

    if pose_image_path is not None:
        pose_image = load_image(pose_image_path)
        pose_image = resize_img(pose_image, max_side, min_side)
        pose_image_cv2 = convert_from_image_to_cv2(pose_image)

        face_info = app.get(pose_image_cv2)

        if len(face_info) == 0:
            raise gr.Error(f"Cannot find any face in the reference image! Please upload another person image")

        face_info = face_info[-1]
        face_kps = draw_kps(pose_image, face_info['kps'])

        width, height = face_kps.size

    if enhance_face_region:
        control_mask = np.zeros([height, width, 3])
        x1, y1, x2, y2 = face_info["bbox"]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        control_mask[y1:y2, x1:x2] = 255
        control_mask = Image.fromarray(control_mask.astype(np.uint8))
    else:
        control_mask = None

    generator = torch.Generator(device=device).manual_seed(seed)

    logging.info("Start inference...")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Prompt: {prompt}")
    logging.info(f"Negative Prompt: {negative_prompt}")

    pipe = get_pipeline(model_path)
    pipe.set_ip_adapter_scale(adapter_strength_ratio)

    if enable_lcm:
        pipe.enable_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.disable_lora()
        pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    images = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image_embeds=face_emb,
        image=face_kps,
        control_mask=control_mask,
        controlnet_conditioning_scale=float(identitynet_strength_ratio),
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        height=height,
        width=width,
        generator=generator
    ).images

    return images[0], gr.update(visible=True)


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_available_models():
    files = []
    extensions = ['.safetensors', '.ckpt']

    for file in os.listdir(MODEL_DIRECTORY):
        if any(file.endswith(ext) for ext in extensions):
            files.append(os.path.join(MODEL_DIRECTORY, file))

    return files


def refresh_models(selected_model):
    global DEFAULT_MODEL
    models = [DEFAULT_MODEL] + get_available_models()

    if selected_model in models:
        default_model = selected_model
    else:
        default_model = DEFAULT_MODEL

    return gr.Dropdown(
        label="Model path",
        choices=models,
        value=default_model
    )


def launch_ui(launch_kwargs, model_path, enable_lcm_arg):
    title = r"""
    <h1 align="center">InstantID: Zero-shot Identity-Preserving Generation in Seconds</h1>
    """

    description = r"""
    <b>Official 🤗 Gradio demo</b> for <a href='https://github.com/InstantID/InstantID' target='_blank'><b>InstantID: Zero-shot Identity-Preserving Generation in Seconds</b></a>.<br>

    How to use:<br>
    1. Upload an image with a face. For images with multiple faces, we will only detect the largest face. Ensure the face is not too small and is clearly visible without significant obstructions or blurring.
    2. (Optional) You can upload another image as a reference for the face pose. If you don't, we will use the first detected face image to extract facial landmarks. If you use a cropped face at step 1, it is recommended to upload it to define a new face pose.
    3. Enter a text prompt, as done in normal text-to-image models.
    4. Click the <b>Submit</b> button to begin customization.
    5. Share your customized photo with your friends and enjoy! 😊
    """

    article = r"""
    ---
    📝 **Citation**
    <br>
    If our work is helpful for your research or applications, please cite us via:
    ```bibtex
    @article{wang2024instantid,
    title={InstantID: Zero-shot Identity-Preserving Generation in Seconds},
    author={Wang, Qixun and Bai, Xu and Wang, Haofan and Qin, Zekui and Chen, Anthony},
    journal={arXiv preprint arXiv:2401.07519},
    year={2024}
    }
    ```
    📧 **Contact**
    <br>
    If you have any questions, please feel free to open an issue or directly reach us out at <b>haofanwang.ai@gmail.com</b>.
    """

    tips = r"""
    ### Usage tips of InstantID
    1. If you're not satisfied with the similarity, try increasing the weight of "IdentityNet Strength" and "Adapter Strength."    
    2. If you feel that the saturation is too high, first decrease the Adapter strength. If it remains too high, then decrease the IdentityNet strength.
    3. If you find that text control is not as expected, decrease Adapter strength.
    4. If you find that realistic style is not good enough, go for our Github repo and use a more realistic base model.
    """

    css = '''
    .gradio-container {width: 85% !important}
    '''
    interface = gr.Blocks(
        css=css,
        title="InstantID: Zero-shot Identity-Preserving Generation in Seconds",
        theme=gr.themes.Default()
    )

    with interface:
        gr.Markdown(title)
        gr.Markdown(description)

        with gr.Row():
            with gr.Column():
                # upload face image
                face_file = gr.Image(label="Upload a photo of your face", type="filepath")

                # optional: upload a reference pose image
                pose_file = gr.Image(label="Upload a reference pose image (optional)", type="filepath")

                prompt = gr.Textbox(
                    label="Prompt",
                    info="Give simple prompt is enough to achieve good face fidelity",
                    placeholder="A photo of a person",
                    value=""
                )

                submit = gr.Button("Submit", variant="primary")

                # Allow a different model to be selected by loading models from disk
                # and displaying them in a dropdown
                model_choices = [model_path] + get_available_models()
                model = gr.Dropdown(
                    label="Model path",
                    choices=model_choices,
                    value=model_path
                )
                refresh_button = gr.Button("Refresh Models")
                refresh_button.click(fn=refresh_models, inputs=model, outputs=model)

                enable_lcm = gr.Checkbox(
                    label="Enable Fast Inference with LCM", value=enable_lcm_arg,
                    info="LCM speeds up the inference step, the trade-off is the quality of the generated image. It performs better with portrait face images rather than distant faces",
                )
                style = gr.Dropdown(label="Style template", choices=STYLE_NAMES, value=DEFAULT_STYLE_NAME)

                identitynet_strength_ratio = gr.Slider(
                    label="IdentityNet strength (for fidelity)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )
                adapter_strength_ratio = gr.Slider(
                    label="Image adapter strength (for detail)",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=0.80,
                )

                with gr.Accordion(open=False, label="Advanced Options"):
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt",
                        placeholder="low quality",
                        value="(lowres, low quality, worst quality:1.2), (text:1.2), watermark, (frame:1.2), deformed, ugly, deformed eyes, blur, out of focus, blurry, deformed cat, deformed, photo, anthropomorphic cat, monochrome, pet collar, gun, weapon, blue, 3d, drones, drone, buildings in background, green",
                    )
                    num_steps = gr.Slider(
                        label="Number of sample steps",
                        minimum=20,
                        maximum=100,
                        step=1,
                        value=5 if enable_lcm_arg else 30,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance scale",
                        minimum=0.1,
                        maximum=10.0,
                        step=0.1,
                        value=0 if enable_lcm_arg else 5,
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=42,
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                    enhance_face_region = gr.Checkbox(label="Enhance non-face region", value=True)

            with gr.Column():
                gallery = gr.Image(label="Generated Images")
                usage_tips = gr.Markdown(label="Usage tips of InstantID", value=tips ,visible=False)

            submit.click(
                fn=remove_tips,
                outputs=usage_tips,
            ).then(
                fn=randomize_seed_fn,
                inputs=[seed, randomize_seed],
                outputs=seed,
                queue=False,
                api_name=False,
            ).then(
                fn=generate_image,
                inputs=[
                    model,
                    face_file,
                    pose_file,
                    prompt,
                    negative_prompt,
                    style,
                    num_steps,
                    identitynet_strength_ratio,
                    adapter_strength_ratio,
                    guidance_scale,
                    seed,
                    enable_lcm,
                    enhance_face_region
                ],
                outputs=[gallery, usage_tips]
            ).then(
                fn=clear_cuda_cache
            )

            enable_lcm.input(fn=toggle_lcm_ui, inputs=[enable_lcm], outputs=[num_steps, guidance_scale], queue=False)

        gr.Examples(
            examples=get_example(),
            inputs=[face_file, prompt, style, negative_prompt],
            run_on_click=True,
            fn=run_for_examples,
            outputs=[gallery, usage_tips],
            cache_examples=True
        )

        gr.Markdown(article)

    interface.launch(**launch_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--enable_lcm", type=bool, default=os.environ.get("ENABLE_LCM", False))
    parser.add_argument("--inbrowser", action="store_true", help="Open in browser")
    parser.add_argument("--listen", type=str, default="0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1", help="IP to listen on for connections to Gradio")
    parser.add_argument("--server_port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Share the Gradio UI")
    parser.add_argument("--medvram", action="store_true", help="Medium VRAM settings")
    parser.add_argument("--lowvram", action="store_true", help="Low VRAM settings")
    parser.add_argument("--username", type=str, default="", help="Username for authentication")
    parser.add_argument("--password", type=str, default="", help="Password for authentication")
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s",
        level=LOG_LEVEL
    )

    logging.getLogger("httpx").setLevel(logging.WARNING)

    model_path = args.model_path

    if args.medvram:
        max_side, min_side = 1024, 832
    elif args.lowvram:
        max_side, min_side = 832, 640
    else:
        max_side, min_side = 1280, 1024

    launch_kwargs = {
        "server_name": args.listen,
        "server_port": args.server_port
    }

    if args.username and args.password:
        launch_kwargs["auth"] = (args.username, args.password)

    if args.inbrowser:
        launch_kwargs["inbrowser"] = args.inbrowser

    if args.share:
        launch_kwargs["share"] = args.share

    logging.info(f'max_side: {max_side}, min_side: {min_side}')
    launch_ui(
        launch_kwargs,
        model_path=model_path,
        enable_lcm_arg=args.enable_lcm,
    )
