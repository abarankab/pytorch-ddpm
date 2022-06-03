import copy
import json
import os
import warnings
import datetime

from absl import app, flags

import torch
import wandb
from torchvision.utils import make_grid, save_image

from diffusion import CombinedGaussianDiffusionSampler
from model import UNet
from sngan import Generator32
from score.both import get_inception_and_fid_score


FLAGS = flags.FLAGS
flags.DEFINE_bool('train', False, help='train from scratch')
flags.DEFINE_bool('eval', False, help='load model.pt and evaluate FID and IS')
# UNet
flags.DEFINE_integer('ch', 128, help='base channel of UNet')
flags.DEFINE_multi_integer('ch_mult', [1, 2, 2, 2], help='channel multiplier')
flags.DEFINE_multi_integer('attn', [1], help='add attention to these levels')
flags.DEFINE_integer('num_res_blocks', 2, help='# resblock in each level')
flags.DEFINE_float('dropout', 0.1, help='dropout rate of resblock')
flags.DEFINE_string('conv_block_name', None, help='conv block name (residual, mbconv)')
# Gaussian Diffusion
flags.DEFINE_float('beta_1', 1e-4, help='start beta value')
flags.DEFINE_float('beta_T', 0.02, help='end beta value')
flags.DEFINE_integer('T', 1000, help='total diffusion steps')
flags.DEFINE_enum('mean_type', 'epsilon', ['xprev', 'xstart', 'epsilon'], help='predict variable')
flags.DEFINE_enum('var_type', 'fixedlarge', ['fixedlarge', 'fixedsmall'], help='variance type')
# Training
flags.DEFINE_float('lr', 2e-4, help='target learning rate')
flags.DEFINE_float('grad_clip', 1., help="gradient norm clipping")
flags.DEFINE_integer('total_steps', 800000, help='total training steps')
flags.DEFINE_integer('img_size', 32, help='image size')
flags.DEFINE_integer('warmup', 5000, help='learning rate warmup')
flags.DEFINE_integer('batch_size', 128, help='batch size')
flags.DEFINE_integer('num_workers', 4, help='workers of Dataloader')
flags.DEFINE_float('ema_decay', 0.9999, help="ema decay rate")
flags.DEFINE_bool('parallel', False, help='multi gpu training')
# Logging & Sampling
flags.DEFINE_string('logdir', './logs/DDPM_CIFAR10_EPS', help='log directory')
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_integer('sample_step', 1000, help='frequency of sampling')
# Evaluation
flags.DEFINE_integer('save_step', 5000, help='frequency of saving checkpoints, 0 to disable during training')
flags.DEFINE_integer('eval_step', 0, help='frequency of evaluating model, 0 to disable during training')
flags.DEFINE_integer('num_images', 50000, help='the number of generated images for evaluation')
flags.DEFINE_bool('fid_use_torch', False, help='calculate IS and FID on gpu')
flags.DEFINE_string('fid_cache', './stats/cifar10.train.npz', help='FID cache')
flags.DEFINE_bool('sample_net', False, help='if True also draws samples from net model')
# Wandb
flags.DEFINE_bool("log_to_wandb", True, help="if True logs to wandb")
flags.DEFINE_string("project_name", "ddpm-cifar-2", help="wandb project name")
flags.DEFINE_string("run_name", datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M"), help="wandb run name")
flags.DEFINE_string("ckpt_filename_0", None, help="checkpoint filename")
flags.DEFINE_string("ckpt_filename_1", None, help="checkpoint filename")
flags.DEFINE_float("midpoint_ratio", 0.5, help="when to switch to different spatial dimension")

device = torch.device('cuda')


def evaluate(sampler, model, gan):
    model.eval()
    gan.eval()
    with torch.no_grad():
        images = []
        for i in range(0, FLAGS.num_images, FLAGS.batch_size):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            
            sample_z = torch.randn(batch_size, 128).to(device)
            x_T = gan(sample_z)

            batch_images, midpoint_images, mn_images = sampler(x_T.to(device))

            batch_images = batch_images.cpu()
            midpoint_images = midpoint_images.cpu()
            mn_images = mn_images.cpu()

            images.append((batch_images + 1) / 2)
            wandb.log({
                "step": i,
                "samples": [wandb.Image((make_grid(batch_images) + 1) / 2)],
                "midpoint_samples": [wandb.Image((make_grid(midpoint_images) + 1) / 2)],
                "mn_samples": [wandb.Image((make_grid(mn_images) + 1) / 2)],
            })

        images = torch.cat(images, dim=0).numpy()
    model.train()
    gan.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def eval():
    run = wandb.init(
        project="ddpm-spatial-eval",
        entity='treaptofun',
        config=FLAGS.flag_values_dict(),
        name=FLAGS.run_name,
    )

    # model setup
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1, conv_block_name="residual")
    gan = Generator32(128)

    print(int(FLAGS.T * FLAGS.midpoint_ratio))
    sampler = CombinedGaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, midpoint=int(FLAGS.T * FLAGS.midpoint_ratio), img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt_0 = torch.load(FLAGS.ckpt_filename_0)
    ckpt_1 = torch.load(FLAGS.ckpt_filename_1)

    model.load_state_dict(ckpt_0['ema_model'])
    gan.load_state_dict(ckpt_1)

    (IS, IS_std), FID, samples = evaluate(sampler, model, gan)
    wandb.log({"IS(EMA)": IS, "IS_STD(EMA)": IS_std, "FID(EMA)": FID})
    print("Model(EMA): IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
    save_image(
        torch.tensor(samples[:256]),
        os.path.join(FLAGS.logdir, 'samples_ema.png'),
        nrow=16)
    wandb.log({"samples(EMA)": [wandb.Image(make_grid(torch.tensor(samples[:256])))]})


def main(argv):
    # suppress annoying inception_v3 initialization warning
    warnings.simplefilter(action='ignore', category=FutureWarning)
    if FLAGS.train:
        raise NotImplementedError()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
