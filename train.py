import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint
from f0_package import crepe_pytorch, sampling, spectral_feats
import glob


torch.backends.cudnn.benchmark = True

from numba import cuda
from GPUtil import showUtilization as gpu_usage

def train(rank, a, h):
    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = Generator(h).to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist, pitch_path, sine_pitch_path = get_dataset_filelist(a)

    trainset = MelDataset(sine_pitch_path, pitch_path, training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=False,
                              drop_last=True)

    if rank == 0:
        validset = MelDataset(sine_pitch_path, pitch_path, validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)

        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    original_dirpath = os.getcwd() #should run from git_repo
    small_crepe = crepe_pytorch.load_crepe(os.path.join(original_dirpath, 'f0_package/small.pth'), device, 'small')




    generator.train()
    mpd.train()
    msd.train()
    total_loss_vec = []
    epoch_vec = range(max(0, last_epoch), a.training_epochs)

    minimal_loss_gen_total = 1000

    for epoch in range(max(0, last_epoch), a.training_epochs):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):
            #print(batch.shape)
            # if (i % 2) == 1:
            #     continue
            if rank == 0:
                start_b = time.time()
            #print('x type ', type(x))

            #10.9.21
            x, y, _, y_mel, sampling_rate_read, sampling_rate_sing, y_pitch_feat, y_sine_pitch_mel = batch
            # print(type(x), x.shape)
            #

            #if len(x) == 0:
            #    continue
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)
            y_sine_pitch_mel = torch.autograd.Variable(y_sine_pitch_mel.to(device, non_blocking=True))

            # print(sum([p.numel() for p in generator.parameters() if p.requires_grad]))

            y_g_hat = generator(x, y_sine_pitch_mel)

            # temporary padding for size matching
            pad_size = y.shape[2] - y_g_hat.shape[2]
            pad_dim = (0, pad_size)
            y_g_hat = F.pad(y_g_hat, pad_dim, "constant", 0)

            # del x
            # torch.cuda.empty_cache()
            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)

            # loss_disc_f.backward()
            # del loss_disc_f
            # torch.cuda.empty_cache()

            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)

            # loss_disc_s.backward()
            # del loss_disc_s
            # torch.cuda.empty_cache()

            loss_disc_all = loss_disc_s + loss_disc_f

            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            #del y_g_hat_mel
            #torch.cuda.empty_cache()

            _, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            # del fmap_s_r, fmap_s_g
            # torch.cuda.empty_cache()
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            # del y_ds_hat_g
            # torch.cuda.empty_cache()

            _, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            # del fmap_f_r, fmap_f_g
            # torch.cuda.empty_cache()
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            # del y_df_hat_g
            # torch.cuda.empty_cache()


            # f0 loss calculation:
            # 10.9.21
            f0_loss = 0
            if h.pitch_loss == True and steps % h.calc_pitch_loss_steps_denom == 0:
                for idx in range(h.batch_size):
                    # sampler_16k_read = sampling.Sampler(orig_freq=sampling_rate_read[idx], new_freq=16000, device=device)
                    # sampler_16k_sing = sampling.Sampler(orig_freq=sampling_rate_sing[idx], new_freq=16000, device=device)
                    # print(f'input size is:{y.squeeze(0).shape}')
                    # in_features = spectral_feats.py_get_activation(y[idx].squeeze(1), sampling_rate_read[idx], small_crepe,
                    #                                 layer=18, grad=False, sampler = None)
                    # in_features = in_features.detach()
                    in_features = torch.load(y_pitch_feat[idx])
                    out_features = spectral_feats.py_get_activation(y_g_hat[idx].squeeze(1), sampling_rate_sing[idx], small_crepe,
                                                 layer=18, grad=True, sampler=None)
                    f0_loss += F.l1_loss(out_features, in_features)
                f0_loss = f0_loss/h.batch_size


            # loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)
            # 10.9.21

            loss_gen_all = h.loss_gen_s_w*loss_gen_s + h.loss_gen_f_w*loss_gen_f + h.loss_fm_s_w*loss_fm_s + h.loss_fm_f_w*loss_fm_f + h.mel_loss_w*loss_mel + h.f0_hp*f0_loss
            # loss_gen_all = h.mel_loss_w*loss_mel + h.f0_hp*f0_loss/4

            #

            loss_gen_all.backward()
            optim_g.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()
                        ######
                        print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                        format(steps, loss_gen_all, mel_error, time.time() - start_b))

                        print(f'loss_fm_f = {loss_fm_f},loss_fm_s = {loss_fm_s}, loss_gen_f = {loss_gen_f},loss_gen_s = {loss_gen_s}, loss_mel = {loss_mel}, f0_loss = {h.f0_hp * f0_loss}')
                        # print(f'loss_mel = {loss_mel}, f0_loss = {h.f0_hp * f0_loss}')
                        ######
                # checkpointing
                save_best_gen = False
                if minimal_loss_gen_total > loss_gen_all and steps > h.minimal_steps_for_best:
                    minimal_loss_gen_total = loss_gen_all
                    save_best_gen = True
                    path_best = os.path.join(a.checkpoint_path, "best")
                    if not os.path.exists(path_best):
                        os.mkdir(path_best)

                if (steps % a.checkpoint_interval == 0 and steps != 0) or save_best_gen:
                    if save_best_gen:

                        checkpoint_path = glob.glob("{}/best/g_*".format(a.checkpoint_path))
                        if len(checkpoint_path) > 0:
                            os.remove(checkpoint_path[0])
                        checkpoint_path = "{}/best/g_{:08d}".format(a.checkpoint_path, steps)
                    else:
                        checkpoint_path = glob.glob("{}/g_*".format(a.checkpoint_path))
                        if len(checkpoint_path) > 0:
                            os.remove(checkpoint_path[0])
                        checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    if save_best_gen:
                        checkpoint_path = glob.glob("{}/best/do_*".format(a.checkpoint_path))
                        if len(checkpoint_path) > 0:
                            os.remove(checkpoint_path[0])
                        checkpoint_path = "{}/best/do_{:08d}".format(a.checkpoint_path, steps)
                    else:
                        checkpoint_path = glob.glob("{}/do_*".format(a.checkpoint_path))
                        if len(checkpoint_path) > 0:
                            os.remove(checkpoint_path[0])
                        checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})

                # Tensorboard summary logging
                if steps % a.summary_interval == 0:

                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", loss_mel, steps)
                    sw.add_scalar("training/f0_loss", h.f0_hp*f0_loss, steps)
                    sw.add_scalar("training/loss_gen_s", loss_gen_s, steps)
                    sw.add_scalar("training/loss_gen_f", loss_gen_f, steps)
                    sw.add_scalar("training/loss_fm_s", loss_fm_s, steps)
                    sw.add_scalar("training/loss_fm_f", loss_fm_f, steps)

                # Validation
                if steps % a.validation_interval == 0:# and steps != 0:
                    generator.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            #print(type(batch))
                            #print(j)
                            # if j % 2 == 1:
                            #     continue
                            x, y, _, y_mel, _, _, _ = batch
                            #if len(x) == 0:
                            #    continue
                            y_g_hat = generator(x.to(device))
                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            pad_size = y_mel.shape[2] - y_g_hat_mel.shape[2]
                            pad_dim = (0, pad_size)
                            y_g_hat_mel = F.pad(y_g_hat_mel, pad_dim, "constant", 0)

                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                    generator.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

    # added by us:
    # torch.save(generator.state_dict(), 'generated')
        total_loss_vec.append(loss_gen_all)
    fig = plt.figure()  # create a figure, just like in matlab
    ax = fig.add_subplot(1, 1, 1)  # create a subplot of certain size
    ax.plot(epoch_vec, total_loss_vec)
    ax.set_xlabel('epoch')
    ax.set_ylabel('total loss')
    ax.set_title("generator total loss")
    ax.grid()
    ax.legend()
    plt.savefig('./total_loss.png')


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_pitch_dir', required=True)
    parser.add_argument('--input_sine_pitch_dir', required=True)
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=100000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=1000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
