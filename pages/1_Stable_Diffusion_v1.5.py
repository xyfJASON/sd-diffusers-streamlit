from PIL import Image

import streamlit as st

import torch
import diffusers
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


@st.cache_resource
def build_txt2img_pipeline():
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to('cuda')
    return pipeline


@st.cache_resource
def build_img2img_pipeline():
    pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to('cuda')
    return pipeline


def txt2img(
        st_comp: dict, prompt: str, neg_prompt: str, seed: int, batch_size: int, batch_count: int,
        height: int, width: int, scheduler: str, sample_steps: int, cfg_scale: float,
):
    # build pipeline
    pipeline = build_txt2img_pipeline()

    # set scheduler
    if scheduler == "PNDM":
        pipeline.scheduler = diffusers.PNDMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DDPM":
        pipeline.scheduler = diffusers.DDPMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DDIM":
        pipeline.scheduler = diffusers.DDIMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DPMSolverMultistep":
        pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DEISMultistep":
        pipeline.scheduler = diffusers.DEISMultistepScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "UniPCMultistep":
        pipeline.scheduler = diffusers.UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    # generate image
    rng = torch.Generator(device='cuda').manual_seed(seed)
    image_list = []
    for batch in range(batch_count):
        images = pipeline(
            prompt=[prompt] * batch_size,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            guidance_scale=cfg_scale,
            negative_prompt=[neg_prompt] * batch_size,
            generator=rng,
        ).images
        image_list.extend(images)
    st_comp['image_placeholder'].image(image_list)


def img2img(
        st_comp: dict, prompt: str, neg_prompt: str, image: Image, strength: float, seed: int, batch_size: int,
        batch_count: int, height: int, width: int, scheduler: str, sample_steps: int, cfg_scale: float,
):
    # build pipeline
    pipeline = build_img2img_pipeline()

    # set scheduler
    if scheduler == "PNDM":
        pipeline.scheduler = diffusers.PNDMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DDPM":
        pipeline.scheduler = diffusers.DDPMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DDIM":
        pipeline.scheduler = diffusers.DDIMScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DPMSolverMultistep":
        pipeline.scheduler = diffusers.DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "DEISMultistep":
        pipeline.scheduler = diffusers.DEISMultistepScheduler.from_config(pipeline.scheduler.config)
    elif scheduler == "UniPCMultistep":
        pipeline.scheduler = diffusers.UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler}")

    # resize image
    image = image.resize((width, height))

    # generate image
    rng = torch.Generator(device='cuda').manual_seed(seed)
    image_list = []
    for batch in range(batch_count):
        images = pipeline(
            prompt=[prompt] * batch_size,
            image=image,
            strength=strength,
            num_inference_steps=sample_steps,
            guidance_scale=cfg_scale,
            negative_prompt=[neg_prompt] * batch_size,
            generator=rng,
        ).images
        image_list.extend(images)
    st_comp['image_placeholder'].image(image_list)


def ui():
    # STREAMLIT SETUP
    st.set_page_config(page_title='Stable Diffusion v1.5', layout='wide')
    if st.session_state.get('page', None) != 'Stable Diffusion v1.5':
        st.cache_resource.clear()
        torch.cuda.empty_cache()
    st.session_state.page = 'Stable Diffusion v1.5'

    # PAGE TITLE
    st.title('Stable Diffusion v1.5')

    # IMAGE PLACEHOLDER
    with st.container(border=True, height=520):
        image_placeholder = st.empty()

    # SIDEBAR
    with st.sidebar:
        seed, batch_size, batch_count, scheduler, sample_steps, cfg_scale = ui_sidebar()

    # TABS
    tabs = st.tabs(["Text to Image", "Image to Image"])
    with tabs[0]:
        prompt, neg_prompt, height, width, bttn_txt2img = ui_tab_txt2img()
    with tabs[1]:
        prompt, neg_prompt, image, height, width, strength, bttn_img2img = ui_tab_img2img()

    # GENERATE IMAGE
    if bttn_txt2img:
        txt2img(
            st_comp=dict(image_placeholder=image_placeholder),
            prompt=prompt, neg_prompt=neg_prompt, seed=seed, batch_size=batch_size, batch_count=batch_count,
            height=height, width=width, scheduler=scheduler, sample_steps=sample_steps, cfg_scale=cfg_scale,
        )
    if bttn_img2img:
        if image is None:
            st.warning("Please upload an image.")
        else:
            img2img(
                st_comp=dict(image_placeholder=image_placeholder),
                prompt=prompt, neg_prompt=neg_prompt, image=image, strength=strength, seed=seed, batch_size=batch_size,
                batch_count=batch_count, height=height, width=width, scheduler=scheduler, sample_steps=sample_steps,
                cfg_scale=cfg_scale,
            )


def ui_sidebar():
    # BASIC OPTIONS
    with st.expander("Basic options", expanded=True):
        seed = st.number_input(f"Seed", min_value=-1, max_value=2**32-1, value=-1, step=1)
        if seed == -1:
            seed = torch.randint(0, 2**32-1, (1, )).item()

        cols = st.columns(2)
        with cols[0]:
            batch_size = st.slider("Batch size", min_value=1, max_value=8, value=1, step=1)
        with cols[1]:
            batch_count = st.slider("Batch count", min_value=1, max_value=8, value=1, step=1)

        cols = st.columns(2)
        with cols[0]:
            scheduler = st.selectbox("Scheduler", options=[
                "DDIM", "DDPM", "PNDM", "DPMSolverMultistep", "DEISMultistep", "UniPCMultistep",
            ])
        with cols[1]:
            sample_steps = st.number_input("Sample steps", min_value=1, max_value=1000, value=20)

        cfg_scale = st.slider("CFG scale", min_value=1.0, max_value=20.0, value=7.5, step=0.1)

    return seed, batch_size, batch_count, scheduler, sample_steps, cfg_scale


def ui_tab_txt2img():
    # PROMPT INPUT BOX
    cols = st.columns(2)
    with cols[0]:
        prompt = st.text_area('Prompt', value='A photo of a cat', height=200)
    with cols[1]:
        neg_prompt = st.text_area(
            'Negative Prompt', height=200,
            value='lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality',
        )

    # IMAGE SIZE
    cols = st.columns(2)
    with cols[0]:
        height = st.slider("Height", min_value=128, max_value=1024, value=512, step=128)
    with cols[1]:
        width = st.slider("Width", min_value=128, max_value=1024, value=512, step=128)

    # BUTTON
    bttn_txt2img = st.button('Generate', key='txt2img', type='primary', use_container_width=True)

    return prompt, neg_prompt, height, width, bttn_txt2img


def ui_tab_img2img():
    cols = st.columns([0.5, 0.3, 0.2])

    with cols[0]:
        # PROMPT INPUT BOX
        prompt = st.text_area('Prompt', value='A photo of a cat', height=100)
        neg_prompt = st.text_area(
            'Negative Prompt', height=100,
            value='lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality',
        )

    with cols[1]:
        # UPLOAD IMAGE
        image_file = st.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
        image = None
        if image_file is not None:
            image = Image.open(image_file)

        # RESIZE IMAGE
        cols2 = st.columns(2)
        with cols2[0]:
            height = st.slider("Resize to (height)", min_value=128, max_value=1024, value=512, step=128)
        with cols2[1]:
            width = st.slider("Resize to (width)", min_value=128, max_value=1024, value=512, step=128)

        # EDIT STRENGTH
        strength = st.slider("Edit strength", min_value=0.0, max_value=1.0, value=0.8, step=0.05)

    with cols[2]:
        # DISPLAY IMAGE
        if image is not None:
            st.image(image)
        else:
            st.empty()

    # BUTTON
    bttn_img2img = st.button('Generate', key='img2img', type='primary', use_container_width=True)

    return prompt, neg_prompt, image, height, width, strength, bttn_img2img


if __name__ == '__main__':
    ui()
