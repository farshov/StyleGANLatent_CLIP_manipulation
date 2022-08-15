import os
import argparse

import torch
import torchvision
import clip
from PIL import Image
from tqdm import tqdm

from stylegan_models import g_synthesis
from style

torch.manual_seed(20)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("USING ", device)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--output_dir',
    type=str,
    required=True,
    help='',
)
parser.add_argument(
    '--batch_size',
    type=int,
    default=1,
    help='Batch Size',
)
parser.add_argument(
    '--prompts_path',
    type=str,
    required=True,
    help='',
)
parser.add_argument(
    '--num_iters',
    type=int,
    default=100,
)
parser.add_argument(
    '--lr',
    type=float,
    default=1e-2,
    help='',
)


def truncation(x, threshold=0.7, max_layer=8, batch_size=1):
    avg_latent = torch.zeros(batch_size, x.size(1), 512).to(device)
    interp = torch.lerp(avg_latent, x, threshold)
    do_trunc = (torch.arange(x.size(1)) < max_layer).view(1, -1, 1).to(device)
    return torch.where(do_trunc, interp, x)


def tensor_to_pil_img(img):
    img = (img.clamp(-1, 1) + 1) / 2.0
    img = img[0].permute(1, 2, 0).detach().cpu().numpy() * 255
    img = Image.fromarray(img.astype('uint8'))
    return img


clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)


def compute_clip_loss(img, text):
    img = torch.nn.functional.upsample_bilinear(img, (224, 224))
    tokenized_text = clip.tokenize([text]).to(device)

    img_logits, _text_logits = clip_model(img, tokenized_text)

    return 1 / img_logits * 100


vgg16 = torchvision.models.vgg16(pretrained=True).to(device)
vgg_layers = vgg16.features
vgg_layer_name_mapping = {
    '1': "relu1_1",
    '3': "relu1_2",
    '6': "relu2_1",
    '8': "relu2_2",
    # '15': "relu3_3",
    # '22': "relu4_3"
}


def main():
    args = parser.parse_args()

    batch_size = args.batch_size
    lr = args.lr

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    g_synthesis.eval()
    g_synthesis.to(device)
    for param in g_synthesis.parameters():
        param.requires_grad = False

    latent_shape = (batch_size, 1, 512)

    normal_generator = torch.distributions.normal.Normal(
        torch.tensor([0.0]),
        torch.tensor([1.]),
    )

    clip_normalize = torchvision.transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    )

    print('Start of optimization')
    prompts_batch, prompts_batch_idxs = [], []
    with open(args.prompts_path, 'r') as f:
        for prompt_num, prompt in enumerate(f):
            if prompt[-1] == '\n':
                prompt = prompt[:-1]
            prompts_batch.append(prompt)
            prompts_batch_idxs.append(prompt_num)
            if len(prompts_batch) < batch_size:
                continue
            # init_latents = normal_generator.sample(latent_shape).squeeze(-1).to(device)
            latents_init = torch.zeros(latent_shape).squeeze(-1).to(device)
            latents = torch.nn.Parameter(latents_init, requires_grad=True)
            optimizer = torch.optim.Adam(
                params=[latents],
                lr=lr,
                betas=(0.9, 0.999),
            )
            print(prompt)
            for i in range(args.num_iters):
                dlatents = latents.repeat(1, 18, 1)
                img = g_synthesis(dlatents)

                # NOTE: clip normalization did not seem to have much effect
                # img = clip_normalize(img)

                # TODO: Поправить аргументы
                loss = compute_clip_loss(img, prompt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(loss.detach().cpu())

            # TODO: Добавить сохранение батча
            img = tensor_to_pil_img(img)
            img.save(os.path.join(output_dir, f'{prompt_num}_{prompt}.png'))

            if len(prompts_batch) == batch_size:
                prompts_batch, prompts_batch_idxs = [], []


if __name__ == '__main__':
    main()
