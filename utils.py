from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageNet, ImageFolder
import torch.nn as nn
from models.clip import Clip
from models.dino import Dino
from models.siglip import Siglip
import os
import io
import glob
from PIL import Image
from transformers import AutoTokenizer, CLIPTextModelWithProjection


class ImageNetHFDataset(Dataset):
    """Loads ILSVRC/imagenet-1k from HuggingFace parquet files without conversion."""
    def __init__(self, root, split, transform=None):
        import pyarrow.parquet as pq
        split_name = "train" if split == "train" else "validation"
        pattern = os.path.join(root, "data", f"{split_name}-*.parquet")
        files = sorted(glob.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No parquet files found at {pattern}")
        tables = [pq.read_table(f, columns=["image", "label"]) for f in files]
        import pyarrow as pa
        self.table = pa.concat_tables(tables)
        self.transform = transform

    def __len__(self):
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.slice(idx, 1)
        img_field = row.column("image")[0].as_py()
        label = row.column("label")[0].as_py()
        if isinstance(img_field, dict):
            img = Image.open(io.BytesIO(img_field["bytes"])).convert("RGB")
        else:
            img = Image.open(io.BytesIO(img_field)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

def get_collate_fn(processor):
    def collate_fn(batch):
        images = [img[0] for img in batch]
        return processor(images=images, return_tensors="pt", padding=True)
    return collate_fn

def get_dataset(args, preprocess, processor, split, subset=1.0):
    if args.dataset_name == 'cc3m':
        # if subset < 1.0:
        #     raise NotImplementedError
        # return get_cc3m(args, preprocess, split)
        raise NotImplementedError
    elif args.dataset_name == 'inat_birds':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'inat':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)
    elif args.dataset_name == 'imagenet':
        ds = ImageNet(root=args.data_path, split=split, transform=preprocess)
    elif args.dataset_name == 'imagenet_hf':
        ds = ImageNetHFDataset(root=args.data_path, split=split, transform=preprocess)
    elif args.dataset_name == 'cub':
        ds = ImageFolder(root=os.path.join(args.data_path, split), transform=preprocess)

    keep_every = int(1.0 / subset)
    if keep_every > 1:
        ds = Subset(ds, list(range(0, len(ds), keep_every)))
    if processor is not None:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=get_collate_fn(processor))
    else:
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return ds, dl

def get_model(args):
    if args.model_name.startswith('clip'):
        clip = Clip(args.model_name, args.device)
        return clip, clip.processor
    elif args.model_name.startswith('dino'):
        dino = Dino(args.model_name, args.device)
        return dino, dino.processor
    elif args.model_name.startswith('siglip'):
        siglip = Siglip(args.model_name, args.device)
        return siglip, siglip.processor

def get_text_model(args):
    if args.model_name.startswith('clip'):
        model = CLIPTextModelWithProjection.from_pretrained(f"openai/{args.model_name}").to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(f"openai/{args.model_name}")
        return model, tokenizer

class IdentitySAE(nn.Module):
    def encode(self, x):
        return x
    def decode(self, x):
        return x
