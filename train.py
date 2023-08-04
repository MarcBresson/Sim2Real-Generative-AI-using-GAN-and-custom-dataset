from pathlib import Path
import logging
logging.getLogger().setLevel(logging.INFO)

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.nn import L1Loss
from ignite.metrics import SSIM, Loss, Average

from src.logger import Checkpointer, MetricLogger
from src.models import GAN, init_weights
from src.data import CustomImageDataset, dataset_split
from src.data import transformation
from src.data.visualisation import batch_to_numpy, plot_streetview_with_discrimination, plot_sim, multichannels_to_individuals

data_path = Path("data")

annotations_file = data_path / Path(r"annotations.arrow")
streetview_dir = data_path / Path(r"mapillary")
simulated_dir = data_path / Path(r"blender")
checkpoints_dir = data_path / Path(r"checkpoints")
viz_dir = data_path / Path(r"visualisation")

NUMBER_OF_EPOCHS = 1000
BATCH_SIZE = 16

dataset = CustomImageDataset(
    annotations_file,
    streetview_dir,
    simulated_dir
)


dataset_train, dataset_val, dataset_viz = dataset_split(dataset, [10000, 2000, 5])

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True, num_workers=6, pin_memory=True)
dataloader_viz = DataLoader(dataset_viz, batch_size=1, num_workers=1)

transform = Compose([
    transformation.RandomPerspective(yaw=(0, 360), pitch=(-20, 70), w_fov=(60, 120)),
    transformation.Resize((256, 256), antialias=True),
    transformation.NormalizeChannels()
])


viz_transform = Compose([
    transformation.RandomPerspective(yaw=0, pitch=30, w_fov=100),
    transformation.Resize((256, 256), antialias=True),
    transformation.NormalizeChannels()
])

for i, _ in enumerate(dataset_viz):
    (viz_dir / f"sample_{i}").mkdir(parents=True, exist_ok=True)

for i_batch, batch in enumerate(dataloader_viz):
    transformed_batch = viz_transform(batch)
    sim_imgs = batch_to_numpy(transformed_batch["simulated"])
    for i_sample, sim_img in enumerate(sim_imgs):
        channels = multichannels_to_individuals(sim_img, dataset.passes_channel_nbr)
        save_path = viz_dir / f"sample_{i_batch * dataloader_viz.batch_size + i_sample}" / "real_simulated"
        fig = plot_sim(channels, list(dataset.render_passes.keys()))
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
        plt.close()


model = GAN()
init_weights(model, init_gain=1)
checkpointer = Checkpointer(model, checkpoints_dir, 50)


metrics = {
    "train_generator_sim2strtview_ssim": SSIM(1.0, device="cuda:0"),
    "train_generator_sim2strtview_l1loss": Loss(L1Loss(), device="cuda:0"),
    "train_loss_generator": Average(device="cuda:0"),
    "train_loss_generator_fooling": Average(device="cuda:0"),
    "train_loss_generator_loss_gen_strtview": Average(device="cuda:0"),
    "train_loss_discrim_strtview": Average(device="cuda:0"),

    "test_generator_sim2strtview_ssim": SSIM(1.0, device="cuda:0"),
    "test_generator_sim2strtview_l1loss": Loss(L1Loss(), device="cuda:0"),
    "test_loss_generator": Average(device="cuda:0"),
    "test_loss_generator_fooling": Average(device="cuda:0"),
    "test_loss_generator_loss_gen_strtview": Average(device="cuda:0"),
    "test_loss_discrim_strtview": Average(device="cuda:0"),
}
metric_logger = MetricLogger(data_path / "metrics.log", metrics)


for epoch in tqdm(range(NUMBER_OF_EPOCHS)):
    for batch in dataloader_train:
        transform(batch)
        model.fit_sample(batch)
        metric_logger.get("train_loss_generator").update(model.generator_loss_value)
        metric_logger.get("train_loss_generator_fooling").update(model.loss_fooling_discrim_strtview)
        metric_logger.get("train_loss_generator_loss_gen_strtview").update(model.loss_gen_strtview)
        metric_logger.get("train_loss_discrim_strtview").update(model.discriminator_Strtview.loss_value)
        metric_logger.get("train_generator_sim2strtview_ssim").update((model.gen_streetviews, model.real_streetviews))
        metric_logger.get("train_generator_sim2strtview_l1loss").update((model.gen_streetviews, model.real_streetviews))

    for batch in dataloader_val:
        transform(batch)
        model.test(batch)
        metric_logger.get("test_loss_generator").update(model.generator_loss_value)
        metric_logger.get("test_loss_generator_fooling").update(model.loss_fooling_discrim_strtview)
        metric_logger.get("test_loss_generator_loss_gen_strtview").update(model.loss_gen_strtview)
        metric_logger.get("test_loss_discrim_strtview").update(model.discriminator_Strtview.loss_value)
        metric_logger.get("test_generator_sim2strtview_ssim").update((model.gen_streetviews, model.real_streetviews))
        metric_logger.get("test_generator_sim2strtview_l1loss").update((model.gen_streetviews, model.real_streetviews))

    for i_batch, batch in enumerate(dataloader_viz):
        viz_transform(batch)
        model.test(batch)

        target_imgs = batch_to_numpy(model.real_streetviews)
        streetview_imgs = batch_to_numpy(model.gen_streetviews)
        discrimination_imgs = batch_to_numpy(model.discriminator_Strtview.gene_discrim)

        for i_sample, (strtview_img, discrim_img, target_img) in enumerate(zip(streetview_imgs, discrimination_imgs, target_imgs)):
            save_path = viz_dir / f"sample_{i_batch * dataloader_viz.batch_size + i_sample}" / f"strtviw_discrim&epoch_{epoch}"
            fig = plot_streetview_with_discrimination(strtview_img, discrim_img, target_img)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
            plt.close()

    checkpointer.step(epoch, "epoch")
    metric_logger.step()
