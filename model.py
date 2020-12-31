#!/usr/bin/env python3
# Student name: Yiyang Hua
# Student number: 1003201475
# UTORid: huayiyan
"""Statistical modelling/parsing classes"""

from itertools import islice
from pathlib import Path
from sys import stdout

import click
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data import load_and_preprocess_data
from data import score_arcs
from parse import minibatch_parse


class Config(object):
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    n_word_ids = None  # inferred
    n_tag_ids = None  # inferred
    n_deprel_ids = None  # inferred
    n_word_features = None  # inferred
    n_tag_features = None  # inferred
    n_deprel_features = None  # inferred
    n_classes = None  # inferred
    dropout = 0.5
    embed_size = None  # inferred
    hidden_size = 200
    batch_size = 2048
    n_epochs = 10
    lr = 0.001


class ParserModel(nn.Module):
    """
    Implements a feedforward neural network with an embedding layer and single
    hidden layer. This network will predict which transition should be applied
    to a given partial parse configuration.
    """
    def create_embeddings(self, word_embeddings: torch.Tensor) -> None:
        """Create embeddings that map word, tag, and deprels to vectors

        Args:
            word_embeddings:
                torch.Tensor of shape (n_word_ids, embed_size) representing
                matrix of pre-trained word embeddings

        Embedding layers convert sparse ID representations to dense vector
        representations.
         - Create 3 embedding layers using nn.Embedding, one for each of
           the input types:
           - The word embedding layer must be initialized with the value of the
             argument word_embeddings, so you will want to create it using
             nn.Embedding.from_pretrained(...). Make sure not to freeze the
             embeddings!
           - You don't need to do anything special for initializing the other
             two embedding layers, so use nn.Embedding(...) for them.
         - The relevant values for the number of embeddings for each type can
           be found in {n_word_ids, n_tag_ids, n_deprel_ids}.
         - Assign the layers to self as attributes:
               self.word_embed
               self.tag_embed
               self.deprel_embed
           (Don't use different variable names!)
        """
        ##****BEGIN YOUR CODE****
        self.word_embed = nn.Embedding.from_pretrained(word_embeddings, freeze=False)
        self.tag_embed = nn.Embedding(self.config.n_tag_ids, self.config.embed_size)
        self.deprel_embed = nn.Embedding(self.config.n_deprel_ids, self.config.embed_size)
        ##****END YOUR CODE****

    def create_net_layers(self) -> None:
        """Create layer weights and biases for this neural network

        Our neural network computes predictions from the embedded input
        using a single hidden layer as well as an output layer. This method
        creates the hidden and output layers, including their weights and
        biases (but PyTorch will manage the weights and biases; you will not
        need to access them yourself). Note that the layers will only compute
        the result of the multiplication and addition (i.e., no activation
        function is applied, so the hidden layer will not apply the ReLu
        function).

         - Create the two layers mentioned above using nn.Linear. You will need
           to fill in the correct sizes for the nn.Linear(...) calls. Keep in mind
           the layer sizes:
               input layer (x): N * embed_size
               hidden layer (h): hidden_size
               output layer (pred): n_classes
           where N = n_word_features + n_tag_features + n_deprel_features
         - Assign the two layers to self as attributes:
               self.hidden_layer
               self.output_layer
           (Don't use different variable names!)

        nn.Linear will take care of randomly initializing the weight and bias
        tensors automatically, so that's all that is to be done here.
        """
        ##****BEGIN YOUR CODE****
        input_layer_size = (self.config.n_word_features + self.config.n_tag_features +
                    self.config.n_deprel_features) * self.config.embed_size
        self.hidden_layer = nn.Linear(input_layer_size, self.config.hidden_size)
        self.output_layer = nn.Linear(self.config.hidden_size, self.config.n_classes)
        ##****END YOUR CODE****

    def reshape_embedded(self, embedded_batch: torch.Tensor) -> torch.Tensor:
        """Reshape an embedded input to combine the various embedded features

        Remember that we use various features based on the parser's state for
        our classifier, such as word on the top of the stack, next word in the
        buffer, etc. Each feature (such as a word) has its own embedding. But
        we will not want to keep the features separate for the classifier, so
        we must merge them all together. This method takes a tensor with
        separated embeddings for each feature and reshapes it accordingly.

        Args:
            embedded_batch:
                torch.Tensor of dtype float and shape (B, N, embed_size)
                where B is the batch_size and N is one of {n_word_features,
                n_tag_features, n_deprel_features}.
        Returns:
            reshaped_batch:
                torch.Tensor of dtype float and shape (B, N * embed_size).

         - Reshape the embedded batch tensor into the specified shape using
           torch.reshape. You may find the value of -1 handy for one of the
           shape dimensions; see the docs for torch.reshape for what it does.
           You may alternatively use the embedded_batch.view(...) or
           embedded_batch.reshape(...) methods if you prefer.
        """
        ##****BEGIN YOUR CODE****
        b, n, embed_size = embedded_batch.shape
        reshaped_batch = embedded_batch.reshape(b, n * embed_size)
        ##****END YOUR CODE****
        return reshaped_batch

    def get_concat_embeddings(self, word_id_batch: torch.Tensor,
                              tag_id_batch: torch.Tensor,
                              deprel_id_batch: torch.Tensor) -> torch.Tensor:
        """Get, reshape, and concatenate word, tag, and deprel embeddings

        Recall that in our neural network, we concatenate the word, tag, and
        deprel embeddings to use as input for our hidden layer. This method
        retrieves all word, tag, and deprel embeddings and concatenates them
        together.

        Args:
            word_id_batch:
                torch.Tensor of dtype int64 and shape (B, n_word_features)
            tag_id_batch:
                torch.Tensor of dtype int64 and shape (B, n_tag_features)
            deprel_id_batch:
                torch.Tensor of dtype int64 and shape (B, n_deprel_features)
            where B is the batch size
        Returns:
            x:
                torch.Tensor of dtype float and shape (B, N * embed_size) where
                N = n_word_features + n_tag_features + n_deprel_features

         - Look up the embeddings for the IDs represented by the word_id_batch,
           tag_id_batch, and deprel_id_batch tensors using the embedding layers
           you defined in self.create_embeddings. (You do not need to call that
           method from this one; that is done automatically for you elsewhere.)
         - Use the self.reshape_embedded method you implemented on each of the
           resulting embedded batch tensors from the previous step.
         - Concatenate the reshaped embedded inputs together using torch.cat to
           get the necessary shape specified above and return the result.
        """
        ##****BEGIN YOUR CODE****
        word_id_batch_temp = self.word_embed(word_id_batch)
        reshape_word = self.reshape_embedded(word_id_batch_temp)
        tag_id_batch_temp = self.tag_embed(tag_id_batch)
        reshape_tag = self.reshape_embedded(tag_id_batch_temp)
        deprel_id_batch_temp = self.deprel_embed(deprel_id_batch)
        reshape_deprel = self.reshape_embedded(deprel_id_batch_temp)
        x = torch.cat([reshape_word, reshape_tag, reshape_deprel], -1)
        ##****END YOUR CODE****
        return x

    def forward(self,
                word_id_batch: numpy.ndarray,
                tag_id_batch: numpy.ndarray,
                deprel_id_batch: numpy.ndarray) -> torch.Tensor:
        """Compute the forward pass of the single-layer neural network

        In our single-hidden-layer neural network, our predictions are computed
        as follows from the concatenated embedded input x:
          1. x is passed through the linear hidden layer to produce h.
          2. Dropout is applied to h to produce h_drop.
          3. h_drop is passed through the output layer to produce pred.
        This method computes pred from the x with the help of the setup done by
        the other methods in this class. Note that, compared to the assignment
        handout, we've added dropout to the hidden layer and we will not be
        applying the softmax activation at all in this model code. See the
        get_loss method if you are curious as to why.

        Args:
            word_id_batch:
                numpy.ndarray of dtype int64 and shape (B, n_word_features)
            tag_id_batch:
                numpy.ndarray of dtype int64 and shape (B, n_tag_features)
            deprel_id_batch:
                numpy.ndarray of dtype int64 and shape (B, n_deprel_features)
        Returns:
            pred: torch.Tensor of shape (B, n_classes)

        - Use self.hidden_layer that you defined in self.create_net_layers to
          compute the pre-activation hidden layer values.
        - Use the torch.nn.functional.relu function to activate the result of
          the previous step and then use the torch.nn.functional.dropout
          function to apply dropout with the appropriate dropout rate. This
          file already imports torch.nn.functional as F, so the function calls
          you will use are F.relu(...) and F.dropout(...).
          - Remember that dropout behaves differently when training vs. when
          evaluating. The F.dropout function reflects this via its arguments.
          You can use self.training to indicate whether or not the model is
          currently being trained.
        - Finally, use self.output_layer to compute the model outputs from the
          result of the previous step.
        """
        x = self.get_concat_embeddings(torch.tensor(word_id_batch),
                                       torch.tensor(tag_id_batch),
                                       torch.tensor(deprel_id_batch))

        ##****BEGIN YOUR CODE****
        h = F.relu(self.hidden_layer(x))
        h_drop = F.dropout(h, self.config.dropout, self.training)
        pred = self.output_layer(h_drop)
        ##****END YOUR CODE****
        return pred

    def get_loss(self, prediction_batch: torch.Tensor,
                 class_batch: torch.Tensor) -> torch.Tensor:
        """Calculate the value of the loss function

        In this case we are using cross entropy loss. The loss will be averaged
        over all examples in the current minibatch. Use F.cross_entropy to
        compute the loss.
        Note that we are not applying softmax to prediction_batch, since
        F.cross_entropy handles that in a more efficient way. Excluding the
        softmax in predictions won't change the expected transition. (Convince
        yourself of this.)

        Args:
            prediction_batch:
                A torch.Tensor of shape (batch_size, n_classes) and dtype float
                containing the logits of the neural network, i.e., the output
                predictions of the neural network without the softmax
                activation.
            class_batch:
                A torch.Tensor of shape (batch_size,) and dtype int64
                containing the ground truth class labels.
        Returns:
            loss: A 0d tensor (scalar) of dtype float
        """
        ##****BEGIN YOUR CODE****
        loss = F.cross_entropy(prediction_batch, class_batch)
        ##****END YOUR CODE****
        return loss

    def add_optimizer(self):
        """Sets up the optimizer.

        Creates an instance of the Adam optimizer and sets it as an attribute
        for this class.
        """
        self.optimizer = torch.optim.Adam(self.parameters(), self.config.lr)

    def _fit_batch(self, word_id_batch, tag_id_batch, deprel_id_batch,
                   class_batch):
        self.optimizer.zero_grad()
        pred_batch = self(word_id_batch, tag_id_batch, deprel_id_batch)
        loss = self.get_loss(pred_batch, torch.tensor(class_batch).argmax(-1))
        loss.backward()

        self.optimizer.step()

        return loss

    def fit_epoch(self, train_data, epoch, trn_progbar, batch_size=None):
        """Fit on training data for an epoch"""
        self.train()
        desc = 'Epoch %d/%d' % (epoch + 1, self.config.n_epochs)
        total = len(train_data) * batch_size if batch_size else len(train_data)
        bar_fmt = '{l_bar}{bar}| [{elapsed}<{remaining}{postfix}]'
        with tqdm(desc=desc, total=total, leave=False, miniters=1, unit='ex',
                  unit_scale=True, bar_format=bar_fmt, position=1) as progbar:
            trn_loss = 0
            trn_done = 0
            for ((word_id_batch, tag_id_batch, deprel_id_batch),
                 class_batch) in train_data:

                loss = self._fit_batch(word_id_batch, tag_id_batch,
                                       deprel_id_batch, class_batch)
                trn_loss += loss.item() * word_id_batch.shape[0]
                trn_done += word_id_batch.shape[0]
                progbar.set_postfix({'loss': '%.3g' % (trn_loss / trn_done)})
                progbar.update(word_id_batch.shape[0])
                trn_progbar.update(word_id_batch.shape[0] / total)
        return trn_loss / trn_done

    def predict(self, partial_parses):
        """Use this model to predict the next transitions/deprels of pps"""
        self.eval()
        feats = self.transducer.pps2feats(partial_parses)
        td_vecs = self(*feats).cpu().detach().numpy()
        preds = [
            self.transducer.td_vec2trans_deprel(td_vec) for td_vec in td_vecs]
        return preds

    def evaluate(self, sentences, ex_arcs):
        """LAS on either training or test sets"""
        act_arcs = minibatch_parse(sentences, self, self.config.batch_size)
        ex_arcs = tuple([(a[0], a[1],
                          self.transducer.id2deprel[a[2]]) for a in pp]
                        for pp in ex_arcs)
        stdout.flush()
        return score_arcs(act_arcs, ex_arcs)

    def __init__(self, transducer, config, word_embeddings):
        self.transducer = transducer
        self.config = config

        super().__init__()

        self.create_embeddings(torch.from_numpy(word_embeddings))
        self.create_net_layers()

        self.add_optimizer()


@click.command()
@click.option('--debug', is_flag=True)
def main(debug):
    """Main function

    Args:
    debug :
        whether to use a fraction of the data. Make sure not to use this flag
        when you're ready to train your model for real!
    """
    print(80 * '=')
    print(f'INITIALIZING{" debug mode" if debug else ""}')
    print(80 * '=')
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        torch.cuda.manual_seed_all(1234)
        print('Running on GPU: {}.'.format(torch.cuda.get_device_name()))
    else:
        print('Running on CPU.')
    config = Config()
    data = load_and_preprocess_data(max_batch_size=config.batch_size,
                                    transition_cache=0 if debug else None)
    transducer, word_embeddings, train_data = data[:3]
    dev_sents, dev_arcs = data[3:5]
    test_sents, test_arcs = data[5:]
    config.n_word_ids = len(transducer.id2word) + 1  # plus null
    config.n_tag_ids = len(transducer.id2tag) + 1
    config.n_deprel_ids = len(transducer.id2deprel) + 1
    config.embed_size = word_embeddings.shape[1]
    for (word_batch, tag_batch, deprel_batch), td_batch in \
            train_data.get_iterator(shuffled=False):
        config.n_word_features = word_batch.shape[-1]
        config.n_tag_features = tag_batch.shape[-1]
        config.n_deprel_features = deprel_batch.shape[-1]
        config.n_classes = td_batch.shape[-1]
        break
    print('# word features: {}'.format(config.n_word_features))
    print('# tag features: {}'.format(config.n_tag_features))
    print('# deprel features: {}'.format(config.n_deprel_features))
    print('# classes: {}'.format(config.n_classes))
    if debug:
        dev_sents = dev_sents[:500]
        dev_arcs = dev_arcs[:500]
        test_sents = test_sents[:500]
        test_arcs = test_arcs[:500]

    print(80 * '=')
    print('TRAINING')
    print(80 * '=')
    weights_file = Path('weights.pt')
    print('Best weights will be saved to:', weights_file)
    model = ParserModel(transducer, config, word_embeddings)
    if torch.cuda.is_available():
        model = model.cuda()
    best_las = 0.
    trnbar_fmt = '{l_bar}{bar}| [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    with tqdm(desc='Training', total=config.n_epochs, leave=False,
              unit='epoch', position=0, bar_format=trnbar_fmt) as progbar:
        for epoch in range(config.n_epochs):
            if debug:
                trn_loss = model.fit_epoch(list(islice(train_data, 32)), epoch,
                                           progbar, config.batch_size)
            else:
                trn_loss = model.fit_epoch(train_data, epoch, progbar)
            tqdm.write('Epoch {:>2} training loss: {:.3g}'.format(epoch + 1,
                                                                  trn_loss))
            stdout.flush()
            dev_las, dev_uas = model.evaluate(dev_sents, dev_arcs)
            best = dev_las > best_las
            if best:
                best_las = dev_las
                if not debug:
                    torch.save(model.state_dict(), str(weights_file))
            tqdm.write('         validation LAS: {:.3f}{} UAS: {:.3f}'.format(
                dev_las, ' (BEST!)' if best else '        ', dev_uas))
    if not debug:
        print()
        print(80 * '=')
        print('TESTING')
        print(80 * '=')
        print('Restoring the best model weights found on the dev set.')
        model.load_state_dict(torch.load(str(weights_file)))
        stdout.flush()
        las, uas = model.evaluate(test_sents, test_arcs)
        if las:
            print('Test LAS: {:.3f}'.format(las), end='       ')
        print('UAS: {:.3f}'.format(uas))
        print('Done.')
    return 0


if __name__ == '__main__':
    main()
