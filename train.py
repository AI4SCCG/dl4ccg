import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
import os
import time
import threading
import matplotlib.pyplot as plt

import utils
import config
import data
import models
import eval

from utils import read_pickle, Datagen_tree


class Train(object):

    def __init__(self, vocab_file_path=None, model_file_path=None):

        # dataset
        self.train_dataset = data.CodePtrDataset(source_path=config.train_source_path,
                                                 code_path=config.train_code_path,
                                                 ast_path=config.train_sbt_path,
                                                 nl_path=config.train_nl_path)

        # CodePtrDataset  return
        self.train_dataset_size = len(self.train_dataset)

        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           collate_fn=lambda *args: utils.unsort_collate_fn(args,
                                                                                            source_vocab=self.source_vocab,
                                                                                            code_vocab=self.code_vocab,
                                                                                            ast_vocab=self.ast_vocab,
                                                                                            nl_vocab=self.nl_vocab))

        trn_data = read_pickle("dataset/nl/train.pkl")
        code_w2i = read_pickle("dataset/code_w2i.pkl")
        self.code_w2i = utils.read_pickle("dataset/code_w2i.pkl")
        self.nl_w2i = utils.read_pickle("dataset/nl_w2i.pkl")

        trn_x, _ = zip(*sorted(trn_data.items()))
        self.trn_gen = Datagen_tree(trn_x, config.batch_size, code_w2i, train=False)

        # vocab
        self.source_vocab: utils.Vocab
        self.code_vocab: utils.Vocab
        self.ast_vocab: utils.Vocab
        self.nl_vocab: utils.Vocab
        # load vocab from given path
        if vocab_file_path:
            source_vocab_path, code_vocab_path, ast_vocab_path, nl_vocab_path = vocab_file_path
            self.source_vocab = utils.load_vocab_pk(source_vocab_path)
            self.code_vocab = utils.load_vocab_pk(code_vocab_path)
            self.ast_vocab = utils.load_vocab_pk(ast_vocab_path)
            self.nl_vocab = utils.load_vocab_pk(nl_vocab_path)
        # new vocab
        else:
            self.source_vocab = utils.Vocab('source_vocab')
            self.code_vocab = utils.Vocab('code_vocab')
            self.ast_vocab = utils.Vocab('ast_vocab')
            self.nl_vocab = utils.Vocab('nl_vocab')
            sources, codes, asts, nls = self.train_dataset.get_dataset()
            for source, code, ast, nl in zip(sources, codes, asts, nls):
                self.source_vocab.add_sentence(source)
                self.code_vocab.add_sentence(code)
                self.ast_vocab.add_sentence(ast)
                self.nl_vocab.add_sentence(nl)
            self.origin_source_vocab_size = len(self.source_vocab)
            self.origin_code_vocab_size = len(self.code_vocab)
            self.origin_nl_vocab_size = len(self.nl_vocab)

            # trim vocabulary
            self.code_vocab.trim(config.code_vocab_size)
            self.nl_vocab.trim(config.nl_vocab_size)
            # save vocabulary
            self.source_vocab.save(config.source_vocab_path)
            self.code_vocab.save(config.code_vocab_path)
            self.ast_vocab.save(config.ast_vocab_path)
            self.nl_vocab.save(config.nl_vocab_path)
            self.source_vocab.save(config.source_vocab_txt_path)
            self.code_vocab.save_txt(config.code_vocab_txt_path)
            self.ast_vocab.save_txt(config.ast_vocab_txt_path)
            self.nl_vocab.save_txt(config.nl_vocab_txt_path)

        self.source_vocab_size = len(self.source_vocab)
        self.code_vocab_size = len(self.code_vocab)
        self.ast_vocab_size = len(self.ast_vocab)
        self.nl_vocab_size = len(self.nl_vocab)
        self.len_code_w2i = len(self.code_w2i)
        self.len_nl_w2i = len(self.nl_w2i)
        # model
        self.model = models.Model(source_vocab_size=self.source_vocab_size,
                                  code_vocab_size=self.code_vocab_size,
                                  ast_vocab_size=self.ast_vocab_size,
                                  nl_vocab_size=self.nl_vocab_size,
                                  len_code_w2i = self.len_code_w2i,
                                  len_nl_w2i = self.len_nl_w2i,
                                  model_file_path=model_file_path)

        self.params = list(self.model.source_encoder.parameters()) + \
            list(self.model.code_encoder.parameters()) + \
            list(self.model.ast_encoder.parameters()) + \
            list(self.model.tree_encoder.parameters()) + \
            list(self.model.reduce_hidden.parameters()) + \
            list(self.model.decoder.parameters())
             
        # optimizer
        self.optimizer = Adam([
            {'params': self.model.source_encoder.parameters(), 'lr': config.source_encoder_lr},
            {'params': self.model.code_encoder.parameters(), 'lr': config.code_encoder_lr},
            {'params': self.model.ast_encoder.parameters(), 'lr': config.ast_encoder_lr},
            {'params': self.model.tree_encoder.parameters(), 'lr':config.tree_encoder_lr},
            {'params': self.model.reduce_hidden.parameters(), 'lr': config.reduce_hidden_lr},
            {'params': self.model.decoder.parameters(), 'lr': config.decoder_lr},
            
        ], betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

        if config.use_lr_decay:
            self.lr_scheduler = lr_scheduler.StepLR(self.optimizer,
                                                    step_size=config.lr_decay_every,
                                                    gamma=config.lr_decay_rate)

        # best score and model(state dict)
        self.min_loss: float = 1000
        self.best_model: dict = {}
        self.best_epoch_batch: (int, int) = (None, None)

        # eval instance
        self.eval_instance = eval.Eval(self.get_cur_state_dict())

        # early stopping
        self.early_stopping = None
        if config.use_early_stopping:
            self.early_stopping = utils.EarlyStopping()

        config.model_dir = os.path.join(config.model_dir, utils.get_timestamp())
        if not os.path.exists(config.model_dir):
            os.makedirs(config.model_dir)




    def run_train(self):
        """
        start training
        :return:
        """
        self.train_iter()
        return self.best_model

    def train_one_batch(self, x, batch, batch_size, criterion):
        """
        train one batch
        :param batch: get from collate_fn of corresponding dataloader
        :param batch_size: batch size
        :param criterion: loss function
        :return: avg loss
        """
        nl_batch = batch[6]

        self.optimizer.zero_grad()

        decoder_outputs = self.model(x, batch, batch_size, self.nl_vocab)     # [T, B, nl_vocab_size]

        decoder_outputs = decoder_outputs.view(-1, config.nl_vocab_size)
        nl_batch = nl_batch.view(-1)

        loss = criterion(decoder_outputs, nl_batch)
        loss.backward()

        # address over fit
        torch.nn.utils.clip_grad_norm_(self.params, 5)

        self.optimizer.step()

        return loss

    def train_iter(self):
        start_time = time.time()

        plot_losses = []

        criterion = nn.NLLLoss(ignore_index=utils.get_pad_index(self.nl_vocab))

        for epoch in range(config.n_epochs):
            print_loss = 0
            plot_loss = 0
            last_print_index = 0
            last_plot_index = 0
            t = self.trn_gen(3047)
            for (index_batch, batch), (x) in zip(enumerate(self.train_dataloader), t):
                batch_size = len(batch[0][0])

                loss = self.train_one_batch(x, batch, batch_size, criterion)
                print_loss += loss.item()
                plot_loss += loss.item()

                # print train progress details  打印train过程细节
                if index_batch % config.print_every == 0:
                    cur_time = time.time()
                    utils.print_train_progress(start_time=start_time, cur_time=cur_time, epoch=epoch,
                                               n_epochs=config.n_epochs, index_batch=index_batch, batch_size=batch_size,
                                               dataset_size=self.train_dataset_size, loss=print_loss,
                                               last_print_index=last_print_index)
                    print_loss = 0
                    last_print_index = index_batch

                # plot train progress details  绘制train过程细节
                if index_batch % config.plot_every == 0:
                    batch_length = index_batch - last_plot_index
                    if batch_length != 0:
                        plot_loss = plot_loss / batch_length
                    plot_losses.append(plot_loss)
                    plot_loss = 0
                    last_plot_index = index_batch

                # save check point
                if config.use_check_point and index_batch % config.save_check_point_every == 0:
                    pass

                # validate on the valid dataset every config.valid_every batches
                if config.validate_during_train and index_batch % config.validate_every == 0 and index_batch != 0:
                    print('\nValidating the model at epoch {}, batch {} on valid dataset......'.format(
                        epoch, index_batch))
                    config.logger.info('Validating the model at epoch {}, batch {} on valid dataset.'.format(
                        epoch, index_batch))
                    self.valid_state_dict(state_dict=self.get_cur_state_dict(), epoch=epoch, batch=index_batch)

                    if config.use_early_stopping:
                        if self.early_stopping.early_stop:
                            break
            if config.use_early_stopping:
                if self.early_stopping.early_stop:
                    break

            # validate on the valid dataset every epoch
            if config.validate_during_train:
                print('\nValidating the model at the end of epoch {} on valid dataset......'.format(epoch))
                config.logger.info('Validating the model at the end of epoch {} on valid dataset.'.format(epoch))
                self.valid_state_dict(self.get_cur_state_dict(), epoch=epoch)

                if config.use_early_stopping:
                    if self.early_stopping.early_stop:
                        break

            if config.use_lr_decay:
                self.lr_scheduler.step()

        plt.xlabel('every {} batches'.format(config.plot_every))
        plt.ylabel('avg loss')
        plt.plot(plot_losses)
        plt.savefig(os.path.join(config.out_dir, 'train_loss_{}.svg'.format(utils.get_timestamp())),
                    dpi=600, format='svg')
        utils.save_pickle(plot_losses, os.path.join(config.out_dir, 'plot_losses_{}.pk'.format(utils.get_timestamp())))

        # save the best model
        if config.save_best_model:
            best_model_name = 'best_epoch-{}_batch-{}.pt'.format(
                self.best_epoch_batch[0], self.best_epoch_batch[1] if self.best_epoch_batch[1] != -1 else 'last')
            self.save_model(name=best_model_name, state_dict=self.best_model)

    def save_model(self, name=None, state_dict=None):
        """
        save current model
        :param name: if given, name the model file by given name, else by current time
        :param state_dict: if given, save the given state dict, else save current model
        :return:
        """
        if state_dict is None:
            state_dict = self.get_cur_state_dict()
        if name is None:
            model_save_path = os.path.join(config.model_dir, 'model_{}.pt'.format(utils.get_timestamp()))
        else:
            model_save_path = os.path.join(config.model_dir, name)
        torch.save(state_dict, model_save_path)

    def save_check_point(self):
        pass

    def get_cur_state_dict(self) -> dict:
        """
        get current state dict of model
        :return:
        """
        state_dict = {
                'source_encoder': self.model.source_encoder.state_dict(),
                'code_encoder': self.model.code_encoder.state_dict(),
                'ast_encoder': self.model.ast_encoder.state_dict(),
                'tree_encoder': self.model.tree_encoder.state_dict(),
                'reduce_hidden': self.model.reduce_hidden.state_dict(),
                'decoder': self.model.decoder.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }
        return state_dict

    def valid_state_dict(self, state_dict, epoch, batch=-1):
        self.eval_instance.set_state_dict(state_dict)
        loss = self.eval_instance.run_eval()

        if config.save_valid_model:
            model_name = 'model_valid-loss-{:.4f}_epoch-{}_batch-{}.pt'.format(loss, epoch, batch)
            save_thread = threading.Thread(target=self.save_model, args=(model_name, state_dict))
            save_thread.start()

        if loss < self.min_loss:
            self.min_loss = loss
            self.best_model = state_dict
            self.best_epoch_batch = (epoch, batch)

        if config.use_early_stopping:
            self.early_stopping(loss)
