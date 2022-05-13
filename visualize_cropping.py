import torch
import torch.nn as nn
from torchvision import transforms as pth_transforms
from absl import logging, app, flags
from PIL import Image
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS
flags.DEFINE_string("image", "", help="Image path to check cropping")
flags.DEFINE_integer("image_size", 480, help="input image size")

flags.DEFINE_string("model", "dino_vits", help="model name")
flags.DEFINE_integer("patch_size", 16, help="dino patch size")

flags.DEFINE_float("threshold", 0.3, help="threhsold (pass 0 to disable)")
flags.DEFINE_string("output", "output.png", help="Image output path")
flags.DEFINE_integer("sum_span", 30, "sum span")
flags.DEFINE_integer("output_width", 480, "output image size")
flags.DEFINE_integer("output_height", 360, "output image size")


def main(argv):
    if FLAGS.image == "":
        raise ValueError("You should pass --image=IMAGE_PATH argument")

    model_name = f"{FLAGS.model}{FLAGS.patch_size}"
    logging.info(f"model: {model_name}")
    logging.info(f"patch size: {FLAGS.patch_size}")
    logging.info(f"image size: ({FLAGS.image_size})")

    logging.info("Load dino model")
    model = torch.hub.load("facebookresearch/dino:main", model_name)
    preprocessor = _get_preprocessor()

    logging.info("Load image")
    with open(FLAGS.image, "rb") as f:
        img = Image.open(f)
        img = img.convert("RGB")

    logging.info("forward image")
    with torch.no_grad():
        (img, resized_img), (w_featmap, h_featmap) = preprocessor(img)
        attentions = model.get_last_selfattention(img)
    nh = attentions.shape[1] # number of head

    logging.info("modify attention for plot")
    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

    if FLAGS.threshold != 0:
        # we keep only a certain percentage of the mass
        val, idx = torch.sort(attentions)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - FLAGS.threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        # interpolate
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=FLAGS.patch_size, mode="nearest")[0]
        attentions = th_attn.sum(0)
    else:
        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=FLAGS.patch_size, mode="nearest")[0]
        attentions = attentions.sum(0)

    logging.info("Crop")
    crop_transform = pth_transforms.CenterCrop((FLAGS.output_height, FLAGS.output_width))
    h, w, _ = resized_img.size()

    conv_weight = torch.ones((1, 1, FLAGS.sum_span, FLAGS.sum_span), dtype=torch.float32)
    pad_size = FLAGS.sum_span // 2
    padded_attention = nn.functional.pad(attentions, (pad_size, pad_size, pad_size, pad_size), value=0)
    scores = nn.functional.conv2d(padded_attention.unsqueeze(0).unsqueeze(0), conv_weight)
    scores = scores[0,0]

    max_index = (scores==torch.max(scores)).nonzero()[0]
    logging.info(f"Center point: {max_index}")

    max_h_start = h - FLAGS.output_height
    max_w_start = w - FLAGS.output_width

    h_start = min(max(max_index[0] + (FLAGS.sum_span // 2) - (FLAGS.output_height // 2), 0), max_h_start)
    w_start = min(max(max_index[1] + (FLAGS.sum_span // 2) - (FLAGS.output_width // 2), 0), max_w_start)

    score_cropped = resized_img[h_start:h_start+FLAGS.output_height, w_start:w_start+FLAGS.output_width,:]
    center_cropped = crop_transform(resized_img.permute(2, 0, 1)).permute(1, 2, 0)

    logging.info("Save plot")
    _plot_and_save(resized_img, attentions, scores, center_cropped, score_cropped)


def _get_preprocessor():
    resize = pth_transforms.Compose([
        pth_transforms.Resize(FLAGS.image_size),
        pth_transforms.ToTensor(),
    ])
    normalize = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def _preprocess(img):
        resized = resize(img)
        img = normalize(resized)

        # make the image divisible by the patch size
        w = img.shape[1] - img.shape[1] % FLAGS.patch_size
        h = img.shape[2] - img.shape[2] % FLAGS.patch_size

        img = img[:, :w, :h].unsqueeze(0)
        resized = resized[:, :w, :h].permute(1, 2, 0)

        w_featmap = img.shape[-2] // FLAGS.patch_size
        h_featmap = img.shape[-1] // FLAGS.patch_size

        return ((img, resized), (w_featmap, h_featmap))

    return _preprocess


def _plot_and_save(img, attention, scores, center_cropped, score_cropped):
    fig = plt.figure(figsize=[25, 10], frameon=False)

    ax = fig.add_subplot(1, 5, 1)
    ax.imshow(img)
    ax.set_title("original Image")

    ax = fig.add_subplot(1, 5, 2)
    ax.imshow(attention)
    ax.set_title("attention")

    ax = fig.add_subplot(1, 5, 3)
    ax.imshow(scores)
    ax.set_title("scores for croppping")

    ax = fig.add_subplot(1, 5, 4)
    ax.imshow(center_cropped)
    ax.set_title("center cropped")

    ax = fig.add_subplot(1, 5, 5)
    ax.imshow(score_cropped)
    ax.set_title("cropped using attention")

    fig.savefig(f"{FLAGS.output}")


if __name__ == "__main__":
    app.run(main)
