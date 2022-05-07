import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader


class FullyConnectedModel(pl.LightningModule):
    def __init__(
        self,
        in_features=2,
        intermediate_layer_size=None,
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        if intermediate_layer_size is None:
             intermediate_layer_size = []

        for num_in, num_out in zip(
            [in_features, *intermediate_layer_size],
            [*intermediate_layer_size, 1]
        ):
            self.layers.append(torch.nn.Linear(num_in, num_out))

    def forward(self, x):
        z = x.view(x.size(0), -1)
        for layer in self.layers:
            z = layer(z)
            z = torch.relu(z)
        return z

    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y.view(-1, 1))
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


class LambdaDataset(Dataset):
    def __init__(self, in_features, max_iter, f):
        super().__init__()
        self.in_features = in_features
        self.f = f
        self.max_iter = max_iter

    def __len__(self):
        return self.max_iter

    def __getitem__(self, idx):
        x = torch.rand(self.in_features)
        y = self.f(x)
        return x, y


class LambdaDataModule(pl.LightningDataModule):
    def __init__(self, in_features, max_iter, f, batch_size):
        super().__init__()
        self.in_features = in_features
        self.max_iter = max_iter
        self.f = f
        self.batch_size = batch_size
        self.num_workers = 8

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.train_dataset = LambdaDataset(self.in_features, self.max_iter, self.f)
        self.valid_dataset = LambdaDataset(self.in_features, 100, self.f)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def teardown(self, stage=None):
        pass


if __name__ == "__main__":
    in_features = 2
    f = torch.sum
    model = FullyConnectedModel(in_features=2, intermediate_layer_size=[100, 100, 100])
    print(model.layers)

    datamodule = LambdaDataModule(in_features, max_iter=4000, f=f, batch_size=8)
    trainer = pl.Trainer(max_epochs=1)  # why there's some delay between at the end of the each epochs?
    trainer.fit(model, datamodule=datamodule)

    for _ in range(100):
        z = torch.rand((1, in_features))
        print(z, f(z), model(z))

