"""
References:
    - Diffusion Forcing: https://github.com/buoyancy99/diffusion-forcing
"""
import torch
from dit import DiT_models
from vae import VAE_models
from torchvision import transforms
from torchvision.io import read_video, write_video, write_png
from utils import one_hot_actions, sigmoid_beta_schedule, ACTION_KEYS
from tqdm import tqdm
from einops import rearrange
from torch import autocast
import devicetorch
import gradio as gr
import os

device = devicetorch.get(torch)
#assert torch.cuda.is_available()
#device = "cuda:0"

# load DiT checkpoint
ckpt = torch.load("oasis500m.pt", map_location=torch.device(device))
model = DiT_models["DiT-S/2"]()
model.load_state_dict(ckpt, strict=False)
model = model.to(device).eval()

# load VAE checkpoint
vae_ckpt = torch.load("vit-l-20.pt", map_location=torch.device(device))
vae = VAE_models["vit-l-20-shallow-encoder"]()
vae.load_state_dict(vae_ckpt)
vae = vae.to(device).eval()

# sampling params
B = 1
total_frames = 32
max_noise_level = 1000
ddim_noise_steps = 100
noise_range = torch.linspace(-1, max_noise_level - 1, ddim_noise_steps + 1)
noise_abs_max = 20
ctx_max_noise_idx = ddim_noise_steps // 10 * 3

# get input video 
#video_id = "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001"
def get_next_filename(directory, extension="png"):
    # List all files with the specified extension
    files = [f for f in os.listdir(directory) if f.endswith(f".{extension}")]

    # Find the highest numbered file
    max_num = 0
    for file in files:
        try:
            num = int(file.split('.')[0])  # Get the number before the extension
            if num > max_num:
                max_num = num
        except ValueError:
            continue  # Skip files that don't start with a number

    # Return the next filename in sequence
    return os.path.join(directory, f"{max_num + 1}.{extension}")


def generate(video_id, total_frames, offset, action):
    print(f"generate {video_id}, total_frames={total_frames}, offset={offset}")
    #mp4_path = f"sample_data/{video_id}.mp4"
    #actions_path = f"sample_data/{video_id}.actions.pt"
    #video = read_video(mp4_path, pts_unit="sec")[0].float() / 255
    #actions = one_hot_actions(torch.load(actions_path, map_location=torch.device(device)))
    video = read_video(video_id, pts_unit="sec")[0].float() / 255

    #arr2 = torch.load(actions_path, map_location=torch.device(device))
    #arr = []
    #for i in range(total_frames + offset):
    #    arr.append({ "forward": 1, "attack": 1, "jump": 1 })
    #for i, item in enumerate(arr):
    #    if len(arr2) > i:
    #        arr[i]["camera"] = arr2[i]["camera"]
    #        last_camera = arr[i]["camera"]
    #    else:
    #        arr[i]["camera"] = last_camera
    #    for j, action_key in enumerate(ACTION_KEYS):
    #        if action_key not in ["forward", "cameraX", "cameraY", "attack", "jump"]:
    #            arr[i][action_key] = 0
    arr = []
    for i in range(total_frames + offset):
        a = { "camera": [0,0] }
        for j, action_key in enumerate(ACTION_KEYS):
            if action_key in ["cameraX", "cameraY"]:
                print("ignore")
            else:
                a[action_key] = 0

        if action in ["cameraX", "cameraY"]:
            print("ignore")
        else:
            a[action] = 1
        arr.append(a)
    print(f"arr={arr}")
    actions = one_hot_actions(arr)
    video = video[offset:offset+total_frames].unsqueeze(0)
    actions = actions[offset:offset+total_frames].unsqueeze(0)

    # sampling inputs
    n_prompt_frames = 1
    x = video[:, :n_prompt_frames]
    x = x.to(device)
    actions = actions.to(device)

    # vae encoding
    scaling_factor = 0.07843137255
    x = rearrange(x, "b t h w c -> (b t) c h w")
    H, W = x.shape[-2:]
    with torch.no_grad():
        x = vae.encode(x * 2 - 1).mean * scaling_factor
    x = rearrange(x, "(b t) (h w) c -> b t c h w", t=n_prompt_frames, h=H//vae.patch_size, w=W//vae.patch_size)

    # get alphas
    betas = sigmoid_beta_schedule(max_noise_level).to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod = rearrange(alphas_cumprod, "T -> T 1 1 1")

    # sampling loop
    for i in tqdm(range(n_prompt_frames, total_frames)):
        chunk = torch.randn((B, 1, *x.shape[-3:]), device=device)
        chunk = torch.clamp(chunk, -noise_abs_max, +noise_abs_max)
        x = torch.cat([x, chunk], dim=1)
        start_frame = max(0, i + 1 - model.max_frames)

        for noise_idx in reversed(range(1, ddim_noise_steps + 1)):
            # set up noise values
            ctx_noise_idx = min(noise_idx, ctx_max_noise_idx)
            t_ctx  = torch.full((B, i), noise_range[ctx_noise_idx], dtype=torch.long, device=device)
            t      = torch.full((B, 1), noise_range[noise_idx],     dtype=torch.long, device=device)
            t_next = torch.full((B, 1), noise_range[noise_idx - 1], dtype=torch.long, device=device)
            t_next = torch.where(t_next < 0, t, t_next)
            t = torch.cat([t_ctx, t], dim=1)
            t_next = torch.cat([t_ctx, t_next], dim=1)

            # sliding window
            x_curr = x.clone()
            x_curr = x_curr[:, start_frame:]
            t = t[:, start_frame:]
            t_next = t_next[:, start_frame:]

            # add some noise to the context
            ctx_noise = torch.randn_like(x_curr[:, :-1])
            ctx_noise = torch.clamp(ctx_noise, -noise_abs_max, +noise_abs_max)
            x_curr[:, :-1] = alphas_cumprod[t[:, :-1]].sqrt() * x_curr[:, :-1] + (1 - alphas_cumprod[t[:, :-1]]).sqrt() * ctx_noise

            # get model predictions
            with torch.no_grad():
                if device == "cuda":
                    with autocast("cuda", dtype=torch.half):
                        v = model(x_curr, t, actions[:, start_frame : i + 1])
                else:
                    v = model(x_curr, t, actions[:, start_frame : i + 1])

            x_start = alphas_cumprod[t].sqrt() * x_curr - (1 - alphas_cumprod[t]).sqrt() * v
            x_noise = ((1 / alphas_cumprod[t]).sqrt() * x_curr - x_start) \
                    / (1 / alphas_cumprod[t] - 1).sqrt()

            # get frame prediction
            x_pred = alphas_cumprod[t_next].sqrt() * x_start + x_noise * (1 - alphas_cumprod[t_next]).sqrt()
            x[:, -1:] = x_pred[:, -1:]

    # vae decoding
    x = rearrange(x, "b t c h w -> (b t) (h w) c")
    with torch.no_grad():
        x = (vae.decode(x / scaling_factor) + 1) / 2
    x = rearrange(x, "(b t) c h w -> b t h w c", t=total_frames)

    # save video
    x = torch.clamp(x, 0, 1)
    x = (x * 255).byte()
    os.makedirs("tmp", exist_ok=True)
    write_video("tmp/video.mp4", x[0], fps=20)
    last_filename = None
    for i in range(total_frames):
    #for i, frame in enumerate(x[0]):
        frame = x[0, i]
        frame = frame.permute(2, 0, 1)
        print(f"shape={frame.shape}")
        filename = get_next_filename("tmp")
        print(f"filename={filename}")
        frame_cpu = frame.cpu()
        print(f"frame_cpu={frame_cpu}")
        write_png(frame_cpu, filename)
        last_filename = filename
    print("generation saved to video.mp4.")
    return [last_filename, "tmp/video.mp4"]
    #return "video.mp4"


video_paths = [
    "Player729-f153ac423f61-20210806-224813.chunk_000",
    "snippy-chartreuse-mastiff-f79998db196d-20220401-224517.chunk_001",
    "treechop-f153ac423f61-20210916-183423.chunk_000"
]

def set(name):
    return gr.update(value=f"sample_data/{name}.mp4")

with gr.Blocks() as demo:
    # Display video options for selection
    with gr.Row():
        with gr.Column():
            video_selector = gr.Video(label="Source")
            #video_selector = gr.Radio(
            #    choices=video_paths,
            #    label="Source"
            #)
            total_frames = gr.Number(label="Number of Frames", value=2, step=16, interactive=True)
            #total_frames = gr.Number(label="Number of Frames", value=32, step=16, interactive=True)
            offset = gr.Number(label="Start Frame", value=2, step=20, interactive=True)
#            button = gr.Button("generate")
        with gr.Column():
            vid = gr.Video(label="Source", elem_id="source", interactive=False)
            #output_video = gr.Video(label="Generated", autoplay=True)
            output_img = gr.Image(label="Generated")
    with gr.Row():
        for key in ACTION_KEYS:
            button = gr.Button(key)
            button.click(
              fn=generate,
              inputs=[video_selector, total_frames, offset, button],
              #outputs=[output_video]
              outputs=[output_img, vid]
            )
    offset.change(
        None,
        inputs=[offset],
        js="(x) => { console.log(x); document.querySelector('#source video').currentTime=Math.ceil(x/20) }"
    )
#    button.click(
#        fn=generate,
#        inputs=[video_selector, total_frames, offset],
#        outputs=output_video
#    )
    video_selector.change(
        fn=set,
        inputs=[video_selector],
        outputs=vid
    )

demo.launch()
