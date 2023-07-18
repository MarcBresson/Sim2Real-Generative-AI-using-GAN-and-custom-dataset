from pathlib import Path

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.nn import L1Loss
from ignite.metrics import SSIM, Loss, Average

from src.logger import Checkpointer, MetricLogger
from src.models.cycle_gan import CycleGAN, init_weights
from src.data import CustomImageDataset, dataset_split
from src.data.transformation import RandomPerspective, Resize
from src.data.visualisation import batch_to_numpy, plot_streetview_with_discrimination, plot_sim, multichannels_to_individuals


annotations_file = Path(r"./data/annotations.arrow")
streetview_dir = Path(r"./data/mapillary")
simulated_dir = Path(r"./data/blender")
checkpoints_dir = Path(r"./data/checkpoints")
viz_dir = Path(r"./data/visualisation")

NUMBER_OF_EPOCHS = 5
BATCH_SIZE = 1

dataset = CustomImageDataset(
    annotations_file,
    streetview_dir,
    simulated_dir
)


dataset_train, dataset_val, dataset_viz, _ = dataset_split(dataset, [50, 50, 5, 1.])

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=True)
dataloader_viz = DataLoader(dataset_viz, batch_size=1)


transform = Compose([
    RandomPerspective(yaw=(0, 360), pitch=(-20, 70), w_fov=(60, 120)),
    Resize((512, 512), antialias=True)
])


viz_transform = Compose([
    RandomPerspective(yaw=0, pitch=30, w_fov=100),
    Resize((512, 512), antialias=True)
])

for i, _ in enumerate(dataset_viz):
    (viz_dir / f"sample_{i}").mkdir(parents=True, exist_ok=True)

for i_batch, batch in enumerate(dataloader_viz):
    transformed_batch = viz_transform(batch)
    sim_imgs = batch_to_numpy(transformed_batch["simulated"])
    for i_sample, sim_img in enumerate(sim_imgs):
        channels = multichannels_to_individuals(sim_img, dataset.passes_channel_nbr)
        save_path = viz_dir / f"sample_{i_batch * BATCH_SIZE + i_sample}" / "real_simulated"
        fig = plot_sim(channels, list(dataset.render_passes.keys()), horizontal=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)


model = CycleGAN()
init_weights(model)
checkpointer = Checkpointer(model, checkpoints_dir, 2)


metrics = {
    "training_generator_sim2strtview_ssim": SSIM(1.0),
    "training_generator_sim2strtview_l1loss": Loss(L1Loss()),
    "training_loss_generator": Average(),
    "training_loss_discrim_strtview": Average(),
    "training_loss_discrim_sim": Average(),

    "test_generator_sim2strtview_ssim": SSIM(1.0),
    "test_generator_sim2strtview_l1loss": Loss(L1Loss()),
    "test_loss_generator": Average(),
    "test_loss_discrim_strtview": Average(),
    "test_loss_discrim_sim": Average(),
}
metric_logger = MetricLogger(Path("./data/metrics.log"), metrics)


for epoch in tqdm(range(NUMBER_OF_EPOCHS)):
    for batch in dataloader_train:
        transformed_batch = transform(batch)
        model.fit_sample(batch)
        metric_logger.get("training_loss_generator").update(model.loss_value)
        metric_logger.get("training_loss_discrim_strtview").update(model.discriminator_Strtview.loss_value)
        metric_logger.get("training_loss_discrim_sim").update(model.discriminator_Sim.loss_value)
        metric_logger.get("training_generator_sim2strtview_ssim").update((model.gen_streetviews, model.real_streetviews))
        metric_logger.get("training_generator_sim2strtview_l1loss").update((model.gen_streetviews, model.real_streetviews))

    for batch in dataloader_val:
        transformed_batch = transform(batch)
        model.test(batch)
        metric_logger.get("test_loss_generator").update(model.loss_value)
        metric_logger.get("test_loss_discrim_strtview").update(model.discriminator_Strtview.loss_value)
        metric_logger.get("test_loss_discrim_sim").update(model.discriminator_Sim.loss_value)
        metric_logger.get("test_generator_sim2strtview_ssim").update((model.gen_streetviews, model.real_streetviews))
        metric_logger.get("test_generator_sim2strtview_l1loss").update((model.gen_streetviews, model.real_streetviews))

    for i_batch, batch in enumerate(dataloader_viz):
        transformed_batch = viz_transform(batch)
        model.test(batch)

        target_imgs = batch_to_numpy(model.real_streetviews)
        streetview_imgs = batch_to_numpy(model.gen_streetviews)
        discrimination_imgs = batch_to_numpy(model.discriminator_Strtview.gene_discrim)

        for i_sample, (strtview_img, discrim_img, target_img) in enumerate(zip(streetview_imgs, discrimination_imgs, target_imgs)):
            save_path = viz_dir / f"sample_{i_batch * BATCH_SIZE + i_sample}" / f"strtviw_discrim&epoch_{epoch}"
            fig = plot_streetview_with_discrimination(strtview_img, discrim_img, target_img, horizontal=True)
            fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    checkpointer.step(epoch, "epoch")
    metric_logger.step()
