import glob
import torch
import os

from PIL import Image
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from models.gan import Generator, Discriminator, CustomDataset, compute_gradient_penalty, merge_state_dicts
stop_training_flag = None

import gradio as gr


def train_gan(
        resume,
        epochs,
        batch_size,
        learning_rate_generator,
        learning_rate_discriminator,
        lambda_gp,
        n_critic,
        dim,
        channels,
        blend_alpha,
        image_folder,
        text_embedding_folder,
        latent_folder,
        default_caption,
        base_sd,
        generator_resume_path,
        discriminator_resume_path,
        save_step=10,
        model_save_step=1000,
        update_image_callback=None  # Callback function to update the image in Gradio
):
    global stop_training_flag
    stop_training_flag = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = dim // 8  # Ensure it's divisible by 8
    z_dim = 100
    b1, b2 = 0.5, 0.999

    os.makedirs(text_embedding_folder, exist_ok=True)
    os.makedirs(latent_folder, exist_ok=True)
    os.makedirs("output", exist_ok=True)

    generator = Generator(z_dim, 1280, img_size, channels).to(device)
    discriminator = Discriminator(img_size, channels).to(device)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate_generator, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate_discriminator, betas=(b1, b2))

    if resume:
        generator_checkpoint_path = generator_resume_path
        discriminator_checkpoint_path = discriminator_resume_path
    else:
        generator_checkpoint_path = "output/x.pth"
        discriminator_checkpoint_path = "output/x.pth"

    if os.path.isfile(generator_checkpoint_path) and os.path.isfile(discriminator_checkpoint_path):
        generator.load_state_dict(torch.load(generator_checkpoint_path))
        discriminator.load_state_dict(torch.load(discriminator_checkpoint_path))
        print("Resumed training from the last checkpoint.")
    else:
        print("No checkpoint found. Starting training from scratch.")

    criterion = torch.nn.BCELoss()

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    image_processor = VaeImageProcessor(vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1))
    clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k").to(device)
    clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-bigG-14-laion2B-39B-b160k")

    image_paths = glob.glob(os.path.join(image_folder, '*.png'))

    if not resume:
        unet = UNet2DConditionModel.from_pretrained(
            base_sd, torch_dtype=torch.float16, subfolder="unet", variant="fp16", use_safetensors=True
        ).to(device)

        generator.load_state_dict(merge_state_dicts(unet.state_dict(), generator.state_dict(), alpha=blend_alpha))

        unet.to("cpu")
        del unet

    for image_path in image_paths:
        base_name = os.path.basename(image_path).replace('.png', '')

        latent_path = os.path.join(latent_folder, base_name + '.pt')
        text_path = os.path.join(image_folder, f"{base_name}.txt")
        text_embedding_path = os.path.join(text_embedding_folder, f"{base_name}.pt")

        if not os.path.exists(latent_path):
            image = Image.open(image_path).resize((dim, dim), resample=Image.Resampling.LANCZOS)
            latent = image_processor.preprocess(image).to(device)
            with torch.no_grad():
                encoded = vae.encode(latent).latent_dist.sample().detach().cpu()
                encoded = vae.config.scaling_factor * encoded
                torch.save(encoded, latent_path)

        if os.path.exists(text_path) and not os.path.exists(text_embedding_path):
            with open(text_path, 'r') as file:
                text = file.read().strip()
            with torch.no_grad():
                inputs = clip_processor(text=text[:150], return_tensors="pt", padding=True)
                text_embeddings = clip_model.get_text_features(**inputs.to(device))
                torch.save(text_embeddings, text_embedding_path)
        else:
            with torch.no_grad():
                inputs = clip_processor(text=default_caption[:150], return_tensors="pt", padding=True)
                text_embeddings = clip_model.get_text_features(**inputs.to(device))
                torch.save(text_embeddings, text_embedding_path)
    # Example usage during training:
    text = "cyberpunk city"
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    infer_embeddings = clip_model.get_text_features(**inputs.to(device))

    i = 0
    seed_generator = torch.manual_seed(42)

    # Initialize dataset and dataloader
    dataset = CustomDataset(image_paths, latent_folder, text_embedding_folder)
    from torch.utils.data import WeightedRandomSampler

    # Calculate total size needed (round up to nearest multiple of batch_size)
    total_size_needed = ((len(dataset) - 1) // batch_size + 1) * batch_size
    extra_samples_needed = total_size_needed - len(dataset)

    # Create a list of indices that wraps around the dataset
    indices = list(range(len(dataset))) * (1 + extra_samples_needed // len(dataset))
    indices += indices[:extra_samples_needed % len(dataset)]

    # Create a sampler that samples according to these indices
    sampler = WeightedRandomSampler(indices, num_samples=total_size_needed, replacement=True)

    # Use this sampler in the DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    infer_noise = torch.randn(1, z_dim, device=device, generator=torch.Generator("cuda").manual_seed(666))
    # Training loop
    for epoch in range(epochs):
        if stop_training_flag:
            print("Training stopped by user.")
            torch.save(generator.state_dict(), f"output/generator_interrupt.pth")
            torch.save(discriminator.state_dict(), f"output/discriminator_interrupt.pth")
            generator.to('cpu')
            discriminator.to('cpu')
            del generator, discriminator
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            break
        for i, (real_data, text_embeddings) in enumerate(dataloader):
            if stop_training_flag:
                break
            real_data = real_data.to(device).squeeze(1)
            text_embeddings = text_embeddings.to(device).squeeze(1)


            optimizer_D.zero_grad()
            noise = torch.randn(batch_size, z_dim, generator=seed_generator).to(device)
            fake_data = generator(noise, text_embeddings)
            real_validity = discriminator(real_data)
            fake_validity = discriminator(fake_data)

            # Calculate average outputs from the discriminator
            avg_real_validity = torch.mean(real_validity).item()
            avg_fake_validity = torch.mean(fake_validity).item()

            gradient_penalty = compute_gradient_penalty(discriminator, real_data.data, fake_data.data)
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            if i % n_critic == 0:
                fake_data = generator(noise, text_embeddings)
                fake_validity = discriminator(fake_data)
                g_loss = -torch.mean(fake_validity)
                g_loss.backward()
                optimizer_G.step()
            # if (i + 1) % 100 == 0:  # Print every 100th batch of an epoch
            #     print(f"Batch {i+1}: Discriminator avg validity for real data: {avg_real_validity}, for fake data: {avg_fake_validity}")

        if (i) % save_step == 0:
            # print(f"[Epoch {i+1}/{i}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")
            # print(
            #     f"Batch {i + 1}: Discriminator avg validity for real data: {avg_real_validity}, for fake data: {avg_fake_validity}")

            with torch.no_grad():
                fake = generator(infer_noise, infer_embeddings)
                fake = fake / vae.config.scaling_factor
                image = vae.decode(fake, return_dict=False)[0].detach()
                image = image_processor.postprocess(image, output_type="pil")[0]
                # image.save(f"output/epoch_{epoch + 1}.png", "PNG")
                image.save(f"current.png", "PNG")
                if update_image_callback is not None:
                    update_image_callback(image)  # Send the current image to Gradio
                yield image, f"d_loss: {d_loss.item()}, g_loss: {g_loss.item()}, R/F: {avg_real_validity}, {avg_fake_validity}"

        if (epoch + 1) % model_save_step == 0:
            torch.save(generator.state_dict(), f"output/generator_epoch_{epoch + 1}.pth")
            torch.save(discriminator.state_dict(), f"output/discriminator_epoch_{epoch + 1}.pth")


    torch.save(generator.state_dict(), f"output/generator_last.pth")
    torch.save(discriminator.state_dict(), f"output/discriminator_last.pth")
    print("Training complete.")
    return image

def setup_gradio_interface():
    with gr.Blocks() as demo:
        with gr.Row():
            resume_chk = gr.Checkbox(label="Resume Training", value=False)
            epochs_input = gr.Number(label="Epochs", value=10000)
            batch_size_input = gr.Number(label="Batch Size", value=6)
            lr_gen_input = gr.Number(label="Learning Rate (Generator)", value=0.0000005)
            lr_disc_input = gr.Number(label="Learning Rate (Discriminator)", value=0.0000013)
            lambda_gp_input = gr.Number(label="Lambda GP", value=10)
            n_critic_input = gr.Number(label="N Critic", value=10)
            dim_input = gr.Number(label="Dimension", value=512)
            channels_input = gr.Number(label="Channels", value=4)
            blend_alpha_input = gr.Number(label="Blend Alpha", value=0.75)
        with gr.Row():
            image_folder_input = gr.Textbox(label="Image Folder", value="images")
            text_embedding_folder_input = gr.Textbox(label="Text Embedding Folder", value="text_embeddings")
            latent_folder_input = gr.Textbox(label="Latent Folder", value="latents")
            default_caption_input = gr.Textbox(label="Default Caption", value="abstract")
            base_sd_input = gr.Textbox(label="Base Stability Diffusion Model", value="stabilityai/sdxl-turbo")
            generator_resume_path_input = gr.Textbox(label="Generator Resume Path", placeholder="Path to generator checkpoint")
            discriminator_resume_path_input = gr.Textbox(label="Discriminator Resume Path", placeholder="Path to discriminator checkpoint")
        status = gr.Textbox(label="status")
        current_image = gr.Image(label="")

        train_btn = gr.Button("Start Training")
        stop_training_btn = gr.Button("Stop Training")

        def stop_training():
            global stop_training_flag
            stop_training_flag = True

        train_btn.click(
            train_gan,
            inputs=[
                resume_chk, epochs_input, batch_size_input, lr_gen_input, lr_disc_input, lambda_gp_input,
                n_critic_input, dim_input, channels_input, blend_alpha_input, image_folder_input,
                text_embedding_folder_input, latent_folder_input, default_caption_input, base_sd_input,
                generator_resume_path_input, discriminator_resume_path_input
            ],
            outputs=[current_image, status]
        )
        stop_training_btn.click(
            stop_training,
            inputs=[],
            outputs=[]
        )

    return demo

if __name__ == "__main__":
    demo = setup_gradio_interface()
    demo.launch()


