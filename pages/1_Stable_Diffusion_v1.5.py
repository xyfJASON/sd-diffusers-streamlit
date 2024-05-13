import torch
import diffusers
from diffusers import StableDiffusionPipeline

import streamlit as st


@st.cache_resource
def build_pipeline():
    pipeline = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float16,
        use_safetensors=True,
    ).to('cuda')
    return pipeline


def generate(
        st_comp: dict,
        prompt: str,
        neg_prompt: str,
        seed: int,
        batch_size: int,
        batch_count: int,
        height: int,
        width: int,
        scheduler: str,
        sample_steps: int,
        cfg_scale: float,
):
    # build pipeline
    pipeline = build_pipeline()

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

    st.divider()

    # PROMPT INPUT BOX
    cols = st.columns(2)
    with cols[0]:
        prompt = st.text_area('Prompt', value='A photo of a cat', height=200)
    with cols[1]:
        neg_prompt = st.text_area(
            'Negative Prompt', height=200,
            value='lowres, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality',
        )

    # BUTTON
    bttn_generate = st.button('Generate', type='primary', use_container_width=True)

    # SIDEBAR
    with st.sidebar:
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
                height = st.slider("Height", min_value=128, max_value=1024, value=512, step=128)
            with cols[1]:
                width = st.slider("Width", min_value=128, max_value=1024, value=512, step=128)

            cols = st.columns(2)
            with cols[0]:
                scheduler = st.selectbox("Scheduler", options=[
                    "DDIM", "DDPM", "PNDM", "DPMSolverMultistep", "DEISMultistep", "UniPCMultistep",
                ])
            with cols[1]:
                sample_steps = st.number_input("Sample steps", min_value=1, max_value=1000, value=20)

            cfg_scale = st.slider("CFG scale", min_value=1.0, max_value=20.0, value=7.5, step=0.1)

    # GENERATE IMAGE
    if bttn_generate:
        generate(
            st_comp=dict(image_placeholder=image_placeholder),
            prompt=prompt,
            neg_prompt=neg_prompt,
            seed=seed,
            batch_size=batch_size,
            batch_count=batch_count,
            height=height,
            width=width,
            scheduler=scheduler,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
        )


if __name__ == '__main__':
    ui()
