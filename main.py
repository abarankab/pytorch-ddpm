import copy
import json
import os
import warnings
import datetime

from absl import app, flags

import torch
import wandb
from tensorboardX import SummaryWriter
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid, save_image
from torchvision import transforms

from diffusion import GaussianDiffusionTrainer, GaussianDiffusionSampler
from model import UNet
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
flags.DEFINE_string('sampler_name', None, help='t sampler name')
flags.DEFINE_bool('reweight_loss', None, help='reweights loss if True')
flags.DEFINE_integer('update_loss_steps', 0, help='additional loss updates')
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
flags.DEFINE_string("ckpt_filename", None, help="checkpoint filename")

device = torch.device('cuda')


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))


def infiniteloop(dataloader):
    while True:
        for x, y in iter(dataloader):
            yield x


def warmup_lr(step):
    return min(step, FLAGS.warmup) / FLAGS.warmup


def evaluate(sampler, model):
    model.eval()
    with torch.no_grad():
        images = []
        desc = "generating images"
        for i in range(0, FLAGS.num_images, FLAGS.batch_size):
            wandb.log({"step": i, "images": len(images) * FLAGS.batch_size})
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - i)
            x_T = torch.randn((batch_size, 3, FLAGS.img_size, FLAGS.img_size))
            batch_images = sampler(x_T.to(device)).cpu()
            images.append((batch_images + 1) / 2)
        images = torch.cat(images, dim=0).numpy()
    model.train()
    (IS, IS_std), FID = get_inception_and_fid_score(
        images, FLAGS.fid_cache, num_images=FLAGS.num_images,
        use_torch=FLAGS.fid_use_torch, verbose=True)
    return (IS, IS_std), FID, images


def train():
    # dataset
    dataset = CIFAR10(
        root='./data', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True,
        num_workers=FLAGS.num_workers, drop_last=True)
    datalooper = infiniteloop(dataloader)

    test_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=False,
        num_workers=FLAGS.num_workers, drop_last=True)

    # model setup
    net_model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, conv_block_name=FLAGS.conv_block_name)
    ema_model = copy.deepcopy(net_model)
    optim = torch.optim.Adam(net_model.parameters(), lr=FLAGS.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=warmup_lr)
    trainer = GaussianDiffusionTrainer(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.sampler_name, FLAGS.reweight_loss, FLAGS.update_loss_steps).to(device)
    net_sampler = GaussianDiffusionSampler(
        net_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    ema_sampler = GaussianDiffusionSampler(
        ema_model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, FLAGS.img_size,
        FLAGS.mean_type, FLAGS.var_type).to(device)
    if FLAGS.parallel:
        trainer = torch.nn.DataParallel(trainer)
        net_sampler = torch.nn.DataParallel(net_sampler)
        ema_sampler = torch.nn.DataParallel(ema_sampler)


    # log setup
    os.makedirs(os.path.join(FLAGS.logdir, 'sample'))
    x_T = torch.randn(FLAGS.sample_size, 3, FLAGS.img_size, FLAGS.img_size)
    x_T = x_T.to(device)
    grid = (make_grid(next(iter(dataloader))[0][:FLAGS.sample_size]) + 1) / 2

    run = wandb.init(
        project=FLAGS.project_name,
        entity='treaptofun',
        config=FLAGS.flag_values_dict(),
        name=FLAGS.run_name,
    )
    wandb.watch(net_model)

    # backup all arguments
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # show model size
    model_size = 0
    for param in net_model.parameters():
        model_size += param.data.nelement()
    print('Model params: %.2f M' % (model_size / 1024 / 1024))

    loss_wandb = 0
    n_loss_wandb = 0
    loss_compare = 0
    n_loss_compare = 0

    if FLAGS.ckpt_filename is not None:
        ckpt = torch.load(FLAGS.ckpt_filename)
        net_model.load_state_dict(ckpt['net_model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        sched.load_state_dict(ckpt['sched'])
        optim.load_state_dict(ckpt['optim'])
        x_T = ckpt['x_T']

    # start training
    for step in range(FLAGS.total_steps):
        # train
        optim.zero_grad()
        x_0 = next(datalooper).to(device)
        loss = trainer(x_0).mean()
        loss.backward()

        loss_compare += trainer.get_true_loss(x_0).mean().item()
        n_loss_compare += 1

        loss_wandb += loss.item()
        n_loss_wandb += 1

        torch.nn.utils.clip_grad_norm_(
            net_model.parameters(), FLAGS.grad_clip)
        optim.step()
        sched.step()
        ema(net_model, ema_model, FLAGS.ema_decay)

        # sample
        if FLAGS.sample_step > 0 and step % FLAGS.sample_step == 0:
            net_model.eval()
            with torch.no_grad():
                x_0 = ema_sampler(x_T)
                grid = (make_grid(x_0) + 1) / 2
                path = os.path.join(
                    FLAGS.logdir, 'sample', '%d.png' % step)
                save_image(grid, path)

                test_loss = 0
                n_test_loss = 0
                for x, y in test_dataloader:
                    x = x.to(device)
                    test_loss += trainer.get_true_loss(x).mean().item()
                    n_test_loss += 1

                wandb.log({
                    "compare_loss": loss_compare / n_loss_compare,
                    "test_loss": test_loss / n_test_loss,
                    "samples": [wandb.Image(grid)],
                    "train_loss": loss_wandb / n_loss_wandb,
                })

                loss_wandb = 0
                n_loss_wandb = 0
                loss_compare = 0
                n_loss_compare = 0

            net_model.train()

        # save
        if FLAGS.save_step > 0 and (step + 1) % FLAGS.save_step == 0:
            ckpt = {
                'net_model': net_model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'sched': sched.state_dict(),
                'optim': optim.state_dict(),
                'step': step,
                'x_T': x_T,
            }
            torch.save(ckpt, os.path.join(FLAGS.logdir, f'ckpt{step}.pt'))

        # evaluate
        if FLAGS.eval_step > 0 and (step + 1) % FLAGS.eval_step == 0:
            net_IS, net_FID, _ = evaluate(net_sampler, net_model)
            ema_IS, ema_FID, _ = evaluate(ema_sampler, ema_model)
            metrics = {
                'IS': net_IS[0],
                'IS_std': net_IS[1],
                'FID': net_FID,
                'IS_EMA': ema_IS[0],
                'IS_std_EMA': ema_IS[1],
                'FID_EMA': ema_FID
            }
            with open(os.path.join(FLAGS.logdir, 'eval.txt'), 'a') as f:
                metrics['step'] = step
                f.write(json.dumps(metrics) + "\n")


def eval():
    run = wandb.init(
        project="ddpm-eval",
        entity='treaptofun',
        config=FLAGS.flag_values_dict(),
        name=FLAGS.run_name,
    )

    # model setup
    model = UNet(
        T=FLAGS.T, ch=FLAGS.ch, ch_mult=FLAGS.ch_mult, attn=FLAGS.attn,
        num_res_blocks=FLAGS.num_res_blocks, dropout=FLAGS.dropout, conv_block_name=FLAGS.conv_block_name)
    sampler = GaussianDiffusionSampler(
        model, FLAGS.beta_1, FLAGS.beta_T, FLAGS.T, img_size=FLAGS.img_size,
        mean_type=FLAGS.mean_type, var_type=FLAGS.var_type).to(device)
    if FLAGS.parallel:
        sampler = torch.nn.DataParallel(sampler)

    # load model and evaluate
    ckpt = torch.load(FLAGS.ckpt_filename)

    if FLAGS.sample_net:
        model.load_state_dict(ckpt['net_model'])
        (IS, IS_std), FID, samples = evaluate(sampler, model)
        wandb.log({"IS": IS, "IS_STD": IS_std, "FID": FID})
        print("Model     : IS:%6.3f(%.3f), FID:%7.3f" % (IS, IS_std, FID))
        save_image(
            torch.tensor(samples[:256]),
            os.path.join(FLAGS.logdir, 'samples.png'),
            nrow=16)
        wandb.log({"samples": [wandb.Image(make_grid(torch.tensor(samples[:256])))]})

    model.load_state_dict(ckpt['ema_model'])
    (IS, IS_std), FID, samples = evaluate(sampler, model)
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
        train()
    if FLAGS.eval:
        eval()
    if not FLAGS.train and not FLAGS.eval:
        print('Add --train and/or --eval to execute corresponding tasks')


if __name__ == '__main__':
    app.run(main)
