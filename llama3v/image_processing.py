from torchvision import transforms
from transformers import AutoTokenizer
import torch
import math
from PIL import Image

def transform(image):
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform_pipeline(image)

def reshape(patch_size, image_tensor):
    patches = torch.nn.functional.unfold(image_tensor, (patch_size, patch_size), stride=patch_size)
    return patches.reshape(image_tensor.size(0), patch_size, -1)

def image_placeholder(image, tokenizer, query_num, max_slice_nums=9, scale_resolution=448, patch_size=14):
    placeholder = tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    slice_images = []

    source_image, patches, best_grid = slice_image(image, max_slice_nums, scale_resolution, patch_size)
    slice_images.append(source_image)

    if patches:
        for row in patches:
            slice_images.extend(row)
        placeholder = get_grid_placeholder(tokenizer, best_grid, query_num)
    
    return slice_images, placeholder

def get_grid_placeholder(tokenizer, grid, query_num):
    placeholder = tokenizer.im_start + tokenizer.unk_token * query_num + tokenizer.im_end
    cols, rows = grid

    slices = ["".join([placeholder] * cols) for _ in range(rows)]
    return tokenizer.slice_start + "\n".join(slices) + tokenizer.slice_end

def find_best_resize(orig_size, scale_res, patch_size, allow_upscale=False):
    w, h = orig_size
    if (w * h > scale_res * scale_res) or allow_upscale:
        r = w / h
        h = int(scale_res / math.sqrt(r))
        w = int(h * r)
    w = max(round(w / patch_size) * patch_size, patch_size)
    h = max(round(h / patch_size) * patch_size, patch_size)
    return w, h

def get_size(orig_size, grid, scale_res, patch_size, allow_upscale=False):
    w, h = orig_size
    gx, gy = grid

    rw = max(round(w / gx) * gx, gx)
    rh = max(round(h / gy) * gy, gy)

    grid_w = rw // gx
    grid_h = rh // gy

    best_grid = find_best_resize((grid_w, grid_h), scale_res, patch_size, allow_upscale)

    return (best_grid[0] * gx, best_grid[1] * gy)

def split_patches(image, grid):
    patches = []
    w, h = image.size
    gx, gy = w // grid[0], h // grid[1]

    for i in range(0, h, gy):
        row = []
        for j in range(0, w, gx):
            row.append(image.crop((j, i, j + gx, i + gy)))
        patches.append(row)
    
    return patches

def slice_image(image, max_slice_nums=9, scale_resolution=448, patch_size=14, never_split=False):
    original_size = image.size
    original_width, original_height = original_size
    log_ratio = math.log(original_width / original_height)
    ratio = original_width * original_height / (scale_resolution * scale_resolution)
    multiple = min(math.ceil(ratio), max_slice_nums)

    if multiple <= 1 or never_split:
        best_size = find_best_resize(original_size, scale_resolution, patch_size, allow_upscale=True)
        source_image = image.resize(best_size, Image.Resampling.BICUBIC)
        return source_image, [], [1, 1]

    candidate_split_grids_nums = [num for num in [multiple - 1, multiple, multiple + 1] if 1 < num <= max_slice_nums]

    best_resize = find_best_resize(original_size, scale_resolution, patch_size)
    source_image = image.copy().resize(best_resize, Image.Resampling.BICUBIC)

    candidate_grids = []
    for split_grids_nums in candidate_split_grids_nums:
        candidate_grids.extend([[m, split_grids_nums // m] for m in range(1, split_grids_nums + 1) if split_grids_nums % m == 0])

    best_grid = min(candidate_grids, key=lambda grid: abs(log_ratio - math.log(grid[0] / grid[1])))

    refine_size = get_size(original_size, best_grid, scale_resolution, patch_size, allow_upscale=True)
    refine_image = image.resize(refine_size, Image.Resampling.BICUBIC)
    patches = split_patches(refine_image, best_grid)

    return source_image, patches, best_grid

def process_image(image, tokenizer):
    images = []
    patch_sizes = []
    slice_images, _ = image_placeholder(
        image, tokenizer, 64, max_slice_nums=9, scale_resolution=448, patch_size=14
    )

    for slice_img in slice_images:
        slice_img = transform(slice_img)
        H, W = slice_img.shape[1:]
        images.append(reshape(14, slice_img))
        patch_sizes.append(torch.Tensor([H // 14, W // 14]).type(torch.int32))

    all_pixel_values = [i.flatten(end_dim=1).permute(1, 0) for i in images]
    img_cnt = len(images)

    return images, patch_sizes, all_pixel_values, img_cnt