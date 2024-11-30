import optax
from flax import nnx
from model import CNN
from dataset import get_loaders
from tqdm.auto import tqdm

config = {
    'lr':0.005,
    'momentum':0.9
}
class Module:
    def __init__(self):
        self.valid_loader = None
        self.train_loader = None
        self.model = None
        self.optimizer = None
        self.metrics = None
        self.configure_model()
        self.configure_optimizers()
        self.configure_metrics()

    def configure_model(self):
        self.model = CNN(rngs = nnx.Rngs(0))

    def retrieve_loaders(self):
        train_loader, valid_loader = get_loaders(8)
        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def configure_metrics(self):
        metrics = nnx.MultiMetric(
            accuracy = nnx.metrics.Accuracy(),
            loss = nnx.metrics.Average('loss')
        )
        self.metrics = metrics

    def configure_optimizers(self):
        learning_rate = config['lr']
        momentum = config['momentum']
        optimizer = nnx.Optimizer(self.model, optax.adamw(learning_rate, momentum))
        self.optimizer = optimizer

    def loss_fn(self, model:CNN, batch):
        logits = model(batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits = logits,
            labels = batch['label']
        ).mean()
        return loss, logits

    @nnx.jit
    def train_step(self, model:CNN, batch, optimizer:nnx.Optimizer, metrics: nnx.MultiMetric):
        grad_fn = nnx.value_and_grad(self.loss_fn, has_aux = True)
        (loss, logits), grads =  grad_fn(model, batch)
        metrics.update(loss = loss, logits = logits, labels = batch['label'])
        optimizer.update(grads)

    @nnx.jit
    def eval_step(self, model:CNN, batch, metrics: nnx.MultiMetric):
        loss, logits = self.loss_fn(model, batch)
        metrics.update(loss = loss, logits = logits, labels = batch['label'])

    def run_single_epoch(self):
        eval_every = 10
        metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }
        for step, batch in enumerate(tqdm(self.train_loader)):
            self.train_step(
                model = self.model,
                batch = batch,
                optimizer = self.optimizer,
                metrics = self.metrics
            )


        for step, batch in enumerate(tqdm(self.valid_loader)):
            self.eval_step(
                model = self.model,
                batch = batch,
                metrics = self.metrics
            )


