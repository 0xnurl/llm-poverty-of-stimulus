# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import math
import pathlib
import shutil
import time

import torch
import torch.nn as nn
from lib import RETRAINING_DATA_PATH
from loguru import logger
from modified_external_sources.colorlessgreenRNNs.src.language_models.dictionary_corpus import (
    Corpus,
)
from modified_external_sources.lm_povstim_with_childes.utils_lm_povstim import (
    batchify,
    batchify_finetuning,
    get_batch,
    repackage_hidden,
)
from utils import get_free_gpu, kwargs_to_id

import model
from lm_argparser import lm_parser

parser = argparse.ArgumentParser(
    parents=[lm_parser], description="Basic training and evaluation for RNN LM"
)

parser.add_argument(
    "--experiment-id",
    dest="experiment_id",
    required=True,
    help="ID for experiment, corresponds to 'train_<id>.txt', 'valid_<id>.txt'.",
)
parser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="'wikipedia', 'childes', 'none'",
)
parser.add_argument(
    "--gpu",
    dest="gpu_id",
    type=int,
    default=None,
    help="Force GPU device ID. Defaults to search for GPU with largest free RAM.",
)

args = parser.parse_args()


# Set the random seed manually for reproducibility.
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.empty_cache()
        if args.gpu_id is None:
            gpu_id = get_free_gpu()
        else:
            gpu_id = args.gpu_id
        logger.info(f"Using GPU ID {gpu_id}")
        torch.cuda.set_device(gpu_id)
###############################################################################
# Load data
###############################################################################

logger.info("Loading data")
start = time.time()
corpus = Corpus(
    base_path=pathlib.Path(args.data),
    dataset_name=args.dataset_name,
    experiment_id=args.experiment_id,
)
logger.info("( %.2f )" % (time.time() - start))
ntokens = len(corpus.dictionary)
logger.info("Vocab size %d", ntokens)
eval_batch_size = 10

logger.info("Batchifying..")
if args.finetune:
    train_data = batchify_finetuning(
        corpus.train,
        args.batch_size,
        corpus.dictionary.word2idx["?"],
        args.cuda,
        padding_id=0,
    )
    val_data = batchify_finetuning(
        corpus.valid,
        eval_batch_size,
        corpus.dictionary.word2idx["?"],
        args.cuda,
        padding_id=0,
    )
    test_data = batchify_finetuning(
        corpus.test,
        eval_batch_size,
        corpus.dictionary.word2idx["?"],
        args.cuda,
        padding_id=0,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=0)
else:
    train_data = batchify(corpus.train, args.batch_size, args.cuda)
    val_data = batchify(corpus.valid, eval_batch_size, args.cuda)
    test_data = batchify(corpus.test, eval_batch_size, args.cuda)
    criterion = nn.CrossEntropyLoss()


###############################################################################
# Build the model
###############################################################################

logger.info("Building the model")

if args.load == None:
    if args.model == "Transformer":
        model = model.TransformerModel(
            ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout
        )
    else:
        model = model.RNNModel(
            args.model,
            ntokens,
            args.emsize,
            args.nhid,
            args.nlayers,
            args.dropout,
            args.tied,
        )
else:
    with open(args.load, "rb") as f:
        if args.cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location=lambda storage, loc: storage)

if args.cuda:
    model.cuda()


###############################################################################
# Training code
###############################################################################


def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    if args.model != "Transformer":
        hidden = model.init_hidden(eval_batch_size)

    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i, args.bptt)
            if args.model == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                output = output.view(-1, ntokens)
                hidden = repackage_hidden(hidden)

            total_loss += len(data) * nn.CrossEntropyLoss()(output, targets).item()

    return total_loss / (len(data_source) - 1)


def evaluate_finetuned(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for batch, b in enumerate(data_source):
            data, targets = b[:, :-1].T, b[:, 1:].T.flatten()

            if args.model == "Transformer":
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                hidden = model.init_hidden(len(b))
                hidden = repackage_hidden(hidden)
                output, hidden = model(data, hidden)
                output = output.view(-1, ntokens)

            total_loss += criterion(output, targets).item()

    return total_loss / (len(data_source) - 1)


def finetune():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    # Loop over training set in chunks of decl quest pairs
    for batch, b in enumerate(train_data):
        data, targets = b[:, :-1].T, b[:, 1:].T.flatten()

        model.zero_grad()

        if args.model == "Transformer":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            # truncated BPP
            hidden = model.init_hidden(len(b))
            hidden = repackage_hidden(hidden)  # is this necessary?
            output, hidden = model(data, hidden)
            output = output.view(-1, ntokens)

        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # TODO(Nur): needed for transformers?
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data),
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


def train():
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()

    if args.model != "Transformer":
        hidden = model.init_hidden(args.batch_size)

    # Loop over training set in chunks of size args.bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i, args.bptt)

        model.zero_grad()

        if args.model == "Transformer":
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            # truncated BPP
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info(
                "| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | "
                "loss {:5.2f} | ppl {:8.2f}".format(
                    epoch,
                    batch,
                    len(train_data) // args.bptt,
                    lr,
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss),
                )
            )
            total_loss = 0
            start_time = time.time()


# Loop over epochs.
lr = args.lr
best_val_loss = None

logger.info(f"Running Transformer with args: {vars(args)}")
args_id = kwargs_to_id(vars(args))
logger.info(f"Transformer id: {args_id}")
save_path = (
    RETRAINING_DATA_PATH
    / "checkpoints"
    / f"{args.experiment_id}__transformer__{args_id}.pt"
)
save_path.parent.mkdir(exist_ok=True, parents=True)
save_path = str(save_path)

# At any point you can hit Ctrl + C to break out of training early.
try:
    patience_exhausted = False
    epochs_since_improved = 0
    epoch = 0

    while not patience_exhausted:
        epoch_start_time = time.time()

        if args.finetune:
            finetune()
            val_loss = evaluate_finetuned(val_data)
        else:
            train()
            val_loss = evaluate(val_data)

        logger.info("-" * 89)
        logger.info(
            "| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | "
            "valid ppl {:8.2f}".format(
                epoch, (time.time() - epoch_start_time), val_loss, math.exp(val_loss)
            )
        )
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(save_path, "wb") as f:
                torch.save(model, f, _use_new_zipfile_serialization=False)
                shutil.copy(save_path, save_path[:-3] + f"__epoch_{epoch}.pt")
            if epoch > 0:
                prev_epoch_filename = save_path[:-3] + f"__epoch_{epoch-1}.pt"
                pathlib.Path(prev_epoch_filename).unlink(missing_ok=True)
            best_val_loss = val_loss
            epochs_since_improved = 0
        else:
            epochs_since_improved += 1
            if epochs_since_improved >= args.patience:
                patience_exhausted = True

            # Anneal the learning rate if no improvement has been seen in the validation dataset.
            lr /= 4.0
            logger.info("| epochs since loss improved: " + str(epochs_since_improved))
            logger.info("| reducing learning rate to " + str(lr))

            # Return to the best saved model checkpoint
            with open(save_path, "rb") as f:
                model = torch.load(f)

        logger.info("-" * 89)
        epoch += 1

except KeyboardInterrupt:
    logger.info("-" * 89)
    logger.info("Exiting from training early")

# Load the best saved model.
with open(save_path, "rb") as f:
    model = torch.load(f)

# Run on test data.
if args.finetune:
    test_loss = evaluate_finetuned(test_data)
else:
    test_loss = evaluate(test_data)
logger.info("=" * 89)
logger.info(
    "| End of training | test loss {:5.2f} | test ppl {:8.2f}".format(
        test_loss, math.exp(test_loss)
    )
)
logger.info("=" * 89)
