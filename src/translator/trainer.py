import functools
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from translator import LOG_SEPARATOR
from translator.datasets import ParallelCorpus, ParallelDataLoader, VectorizedDataPointBatch, VectorizedParallelDataset
from translator.models import Translator

logger: logging.Logger = logging.getLogger("translator")


class ModelTrainer:
    def __init__(
        self,
        model: Translator,
        train_split: ParallelCorpus,
        dev_split: ParallelCorpus,
        test_split: ParallelCorpus,
    ) -> None:
        self.model = model

        self.train_split = train_split
        self.dev_split = dev_split
        self.test_split = test_split

    def _create_vectorized_datasets(
        self,
    ) -> tuple[VectorizedParallelDataset, VectorizedParallelDataset, VectorizedParallelDataset]:
        create: Callable[[Sequence[str], Sequence[str]], VectorizedParallelDataset] = functools.partial(
            VectorizedParallelDataset,
            source_tokenizer=self.model.source_tokenizer,
            target_tokenizer=self.model.target_tokenizer,
            source_language=self.model.source_language,
            target_language=self.model.target_language,
        )

        vectorized_train_dataset: VectorizedParallelDataset = create(
            self.train_split.source_sentences,
            self.train_split.target_sentences,
        )
        vectorized_dev_dataset: VectorizedParallelDataset = create(
            self.dev_split.source_sentences,
            self.dev_split.target_sentences,
        )
        vectorized_test_dataset: VectorizedParallelDataset = create(
            self.test_split.source_sentences,
            self.test_split.target_sentences,
        )

        return vectorized_train_dataset, vectorized_dev_dataset, vectorized_test_dataset

    @staticmethod
    def _create_data_loaders(
        train_dataset: VectorizedParallelDataset,
        dev_dataset: VectorizedParallelDataset,
        test_dataset: VectorizedParallelDataset,
        batch_size: int,
        num_workers: int,
    ) -> tuple[ParallelDataLoader, ParallelDataLoader, ParallelDataLoader]:
        train_data_loader = ParallelDataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle the training data to reduce overfitting to the data point order.
            num_workers=num_workers,
        )
        dev_data_loader = ParallelDataLoader(dev_dataset, batch_size=batch_size, num_workers=num_workers)
        test_data_loader = ParallelDataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
        return train_data_loader, dev_data_loader, test_data_loader

    def _log_train_parameters(
        self,
        max_epochs: int,
        learning_rate: float,
        batch_size: int,
        patience: int,
        scheduler_patience: int,
        teacher_forcing_ratio: float,
        num_workers: int,
    ) -> None:
        # Log model
        logger.info(LOG_SEPARATOR)
        logger.info("Training Model")
        logger.info(LOG_SEPARATOR)
        logger.info(self.model)

        # Log (hyper)parameters
        logger.info(LOG_SEPARATOR)
        logger.info("Training Hyperparameters:")
        logger.info(" - max_epochs: %r", max_epochs)
        logger.info(" - learning_rate: %r", learning_rate)
        logger.info(" - batch_size: %r", batch_size)
        logger.info(" - patience: %r", patience)
        logger.info(" - scheduler_patience: %r", scheduler_patience)
        logger.info(" - teacher_forcing_ratio: %r", teacher_forcing_ratio)

        logger.info(LOG_SEPARATOR)
        logger.info("Computational Parameters:")
        logger.info(" - num_workers: %r", num_workers)
        logger.info(" - device: %r", self.model.device)

        logger.info(LOG_SEPARATOR)
        logger.info("Dataset Splits:")
        logger.info(" - train: %d data points", len(self.train_split))
        logger.info(" - dev: %d data points", len(self.dev_split))
        logger.info(" - test: %d data points", len(self.test_split))

    def train_epoch(
        self,
        data_loader: ParallelDataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: Callable[[Tensor, Tensor], Tensor],
        teacher_forcing_ratio: float = 0.5,
    ) -> float:
        """Trains the model in-place for one epoch over the given dataset.

        Args:
            data_loader: The data loader used for training.
            optimizer: The optimizer used for training.
            criterion: The defined loss function
                (accepting the model's output log probabilities and the target ground truth).
            teacher_forcing_ratio: The probability that teacher forcing will be used.

        Returns:
            The loss of the model on the training dataset after this epoch.
        """
        self.model.train()  # Bring the model into training mode.

        train_loss: float = 0.0
        start_time: float = time.time()

        data_point_batch: VectorizedDataPointBatch
        for batch_index, data_point_batch in enumerate(data_loader, start=1):
            # Move data to the model's device.
            sources: Tensor = data_point_batch.sources.to(self.model.device)
            targets: Tensor = data_point_batch.targets.to(self.model.device)

            # Zero parameter gradients.
            optimizer.zero_grad()

            # Using the model, compute the log probabilities for each token in the target sequences.
            predicted_log_probabilities: Tensor = self.model(sources, targets, teacher_forcing_ratio)
            ground_truth_targets: Tensor = self.model.decoder.make_ground_truth_sequences(targets)

            # Compute the loss and update the model parameters via backpropagation.
            loss: Tensor = criterion(predicted_log_probabilities.flatten(end_dim=1), ground_truth_targets.flatten())
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            # Log intermediate results.
            if batch_index % max(1, len(data_loader) // 10) == 0:
                msg: str = (
                    f"batch {batch_index}/{len(data_loader)}"
                    f" - loss {train_loss / batch_index:.8f}"
                    f" - lr {optimizer.param_groups[0]['lr']:.4f}"
                    f" - time {time.time() - start_time:.2f}s"
                )
                logger.info(msg)

        return train_loss / len(data_loader)

    def train(
        self,
        *,
        max_epochs: int = 20,
        learning_rate: float = 0.5,
        batch_size: int = 32,
        patience: int = 5,
        scheduler_patience: int = 3,
        teacher_forcing_ratio: float = 0.5,
        evaluate_on_train: bool = False,
        num_workers: int = 0,
    ) -> float:
        # Create torch datasets and data loaders for each dataset split.
        train_dataset, dev_dataset, test_dataset = self._create_vectorized_datasets()
        train_data_loader, dev_data_loader, test_data_loader = self._create_data_loaders(
            train_dataset,
            dev_dataset,
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        # Use negative log-likelihood (equivalent to the cross entropy after a log softmax operation)
        # as the loss function and initialize an optimizer and scheduler.
        padding_index: int = -100 if self.model.decoder.padding_index is None else self.model.decoder.padding_index
        criterion = torch.nn.NLLLoss(ignore_index=padding_index)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience)

        self._log_train_parameters(
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            patience=patience,
            scheduler_patience=scheduler_patience,
            teacher_forcing_ratio=teacher_forcing_ratio,
            num_workers=num_workers,
        )

        # Train the model
        best_dev_perplexity: float = float("inf")
        best_model_state_dict: dict[str, Any] = self.model.state_dict()
        early_stopping_counter: int = 0

        try:
            # Loop over the dataset multiple times; one time is one epoch.
            for epoch in range(1, max_epochs + 1):
                logger.info(LOG_SEPARATOR)
                logger.info("EPOCH %d", epoch)

                train_loss: float = self.train_epoch(
                    train_data_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                )

                logger.info(LOG_SEPARATOR)
                logger.info("EPOCH %d DONE", epoch)

                # Validation phase
                dev_loss, dev_perplexity = self.model.evaluate(dev_data_loader, criterion=criterion)
                logger.info("TRAIN Loss:       %.4f", train_loss)
                logger.info("DEV   Loss:       %.4f", dev_loss)
                if evaluate_on_train:
                    _, train_perplexity = self.model.evaluate(train_data_loader, criterion=criterion)
                    logger.info("TRAIN Perplexity: %.4f", train_perplexity)
                logger.info("DEV   Perplexity: %.4f", dev_perplexity)

                # After validation phase, perform learning rate scheduler update step.
                scheduler.step(dev_loss)

                # Implementation of early stopping: If the model's perplexity on the dev set does not improve
                # in `patience` epochs, the training will be terminated.
                if dev_perplexity >= best_dev_perplexity:
                    early_stopping_counter += 1
                    logger.info("No improvement for %d epoch(s)", early_stopping_counter)
                else:
                    logger.info("New best score!")
                    best_dev_perplexity = dev_perplexity
                    best_model_state_dict = self.model.state_dict()
                    early_stopping_counter = 0

                if early_stopping_counter == patience:
                    logger.info("Patience reached: Terminating model training due to early stopping")
                    break

        except KeyboardInterrupt:
            logger.info(LOG_SEPARATOR)
            logger.warning("Manually interrupted training!")

        # The model has been trained as many epochs as were specified (subject to possible early stopping).
        logger.info(LOG_SEPARATOR)
        logger.info("Finished Training")

        # Now, evaluate the best model on test set.
        self.model.load_state_dict(best_model_state_dict)
        _, test_perplexity = self.model.evaluate(test_data_loader, criterion=criterion)
        logger.info("TEST  Perplexity: %.4f", test_perplexity)

        return test_perplexity
