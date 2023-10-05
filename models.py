import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
import random
import data
import config
import utils


def init_rnn_wt(rnn):
    for names in rnn._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(rnn, name)
                wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)


def init_linear_wt(linear):
    """
    initialize the weight and bias(if) of the given linear layer
    :param linear: linear layer
    :return:
    """
    linear.weight.data.normal_(std=config.init_normal_std)
    if linear.bias is not None:
        linear.bias.data.normal_(std=config.init_normal_std)


def init_wt_normal(wt):
    """
    initialize the given weight following the normal distribution
    :param wt: weight to be normal initialized
    :return:
    """
    wt.data.normal_(std=config.init_normal_std)


def init_wt_uniform(wt):
    """
    initialize the given weight following the uniform distribution
    :param wt: weight to be uniform initialized
    :return:
    """
    wt.data.uniform_(-config.init_uniform_mag, config.init_uniform_mag)


class Encoder(nn.Module):
    """
    Encoder for both code and ast
    """

    def __init__(self, vocab_size):
        super(Encoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.num_directions = 2

        # vocab_size: config.code_vocab_size for code encoder, size of sbt vocabulary for ast encoder
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.gru = nn.GRU(config.embedding_dim, self.hidden_size, bidirectional=True)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)

    def forward(self, inputs: torch.Tensor, seq_lens: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """

        :param inputs: sorted by length in descending order, [T, B]
        :param seq_lens: should be in descending order
        :return: outputs: [T, B, H]
                hidden: [2, B, H]
        """
        embedded = self.embedding(inputs)   # [T, B, embedding_dim]
        packed = pack_padded_sequence(embedded, seq_lens, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs)  # [T, B, 2*H]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]

        # outputs: [T, B, H]
        # hidden: [2, B, H]
        return outputs, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_directions, batch_size, self.hidden_size, device=config.device)

class BiLSTM(nn.Module):
    def __init__(self, dim, return_seq=False):
        super(BiLSTM, self).__init__()
        self.c_init_f = nn.Parameter(torch.randn(1, dim) * 0.01).to("cuda:0")
        self.h_init_f = nn.Parameter(torch.randn(1, dim) * 0.01).to("cuda:0")
        self.c_init_b = nn.Parameter(torch.randn(1, dim) * 0.01).to("cuda:0")
        self.h_init_b = nn.Parameter(torch.randn(1, dim) * 0.01).to("cuda:0")
        self.lay_f = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, bidirectional=False).to("cuda:0")
        self.lay_b = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, bidirectional=False).to("cuda:0")
        self.fc = nn.Linear(dim * 2, dim, bias=False).to("cuda:0")
        self.return_seq = return_seq

    def forward(self, x, length):
        '''x: [batch, length, dim]'''
        x = x.to("cuda:0")  # 将输入张量移动到 CUDA 设备上

        batch = x.size(0)

        x_back = torch.zeros_like(x).to("cuda:0")
        for i in range(batch):
            x_back[i, :length[i], :] = torch.flip(x[i, :length[i], :], dims=[0])

        init_state_f = (torch.tile(self.h_init_f, [batch, 1]).unsqueeze(0).to("cuda:0"),
                        torch.tile(self.c_init_f, [batch, 1]).unsqueeze(0).to("cuda:0"))
        init_state_b = (torch.tile(self.h_init_b, [batch, 1]).unsqueeze(0).to("cuda:0"),
                        torch.tile(self.c_init_b, [batch, 1]).unsqueeze(0).to("cuda:0"))

        y_f, (h_f, c_f) = self.lay_f(x, init_state_f)
        y_b, (h_b, c_b) = self.lay_b(x_back, init_state_b)

        y = torch.cat([y_f, y_b], -1)

        if self.return_seq:
            return self.fc(y)
        else:
            y_last = y[torch.arange(batch), (length - 1).long()]

            return self.fc(y_last)

class ShidoTreeLSTMLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ShidoTreeLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = BiLSTM(dim_out, return_seq=True).to("cuda:0")
        self.U_i = BiLSTM(dim_out).to("cuda:0")
        self.U_u = BiLSTM(dim_out).to("cuda:0")
        self.U_o = BiLSTM(dim_out).to("cuda:0")
        self.W = nn.Linear(dim_in, dim_out * 4).to("cuda:0")
        self.h_init = torch.zeros([1, dim_out], dtype=torch.float32, requires_grad=True).to("cuda:0")
        self.c_init = torch.zeros([1, dim_out], dtype=torch.float32, requires_grad=True).to("cuda:0")

    def forward(self, tensor, indices):
        for i in range(len(tensor)):
            tensor[i] = tensor[i].to("cuda:0")

        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []

        for indice, x in zip(indices, tensor):

            h_tensor, c_tensor = self.forward_step(x, h_tensor, c_tensor, indice)

            res_h.append(h_tensor[:, :])
            res_c.append(c_tensor[:, :])

            h_tensor = torch.cat([self.h_init, h_tensor], dim=0)
            c_tensor = torch.cat([self.c_init, c_tensor], dim=0)

        return res_h, res_c


    def forward_step(self, x, h_tensor, c_tensor, indice):
        mask_bool = torch.ne(indice, -1)
        mask = mask_bool.float()  # [nodes, child]
        length = mask.sum(dim=1).int()
        h_1 = torch.zeros_like(indice)
        h_2 = torch.where(mask_bool, indice, h_1)
        h = h_tensor[h_2.long()]
        c = c_tensor[h_2.long()]
        c = c.to("cuda:0")
        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = self.U_f(h, length)
        branch_f_k = torch.sigmoid(torch.unsqueeze(W_f_x, dim=1) + branch_f_k)
        branch_f_k = branch_f_k.to("cuda:0")

        branch_f = torch.sum(branch_f_k * c * torch.unsqueeze(mask, -1).to("cuda:0"), dim=1)


        branch_i = self.U_i(h, length)  # [nodes, dim_out]
        branch_i = torch.sigmoid(branch_i + W_i_x)  # [nodes, dim_out]
        branch_u = self.U_u(h, length)  # [nodes, dim_out]
        branch_u = torch.tanh(branch_u + W_u_x)
        branch_o = self.U_o(h, length)  # [nodes, dim_out]
        branch_o = torch.sigmoid(branch_o + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node, dim_out]

        return new_h, new_c

class TreeEncoder(nn.Module):

    def __init__(self, in_vocab, out_vocab, layer=1, dropout=0.3):
        super(TreeEncoder, self).__init__()
        self.dim_E = config.hidden_size   # dim_E 表示编码器输入（代码）的嵌入维度
        self.dim_rep = config.hidden_size  # dim_rep 表示编码器和解码器之间共享的表示层的维度
        self.in_vocab = in_vocab   # len(code_w2i)
        self.out_vocab = out_vocab  # len(nl_w2i)
        self.layer = layer
        self.dropout = dropout  # 表示在LSTM层之间应用的dropout率
        self.E = TreeEmbeddingLayer(self.dim_E, in_vocab).to("cuda:0")
        for i in range(layer):
            setattr(self, "layer{}".format(i), ShidoTreeLSTMLayer(self.dim_E, self.dim_rep).to("cuda:0"))

    def forward(self, x):
        tensor, indice, tree_num = x
        tensor = self.E(tensor)
        for i in range(self.layer):
            skip = tensor
            tensor, c = getattr(self, "layer{}".format(i))(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]
        hidden = tensor[-1]
        cell = c[-1]
        return (hidden, cell)

        # output:[T,B,H],[hidden,cell]  [B,H]


class TreeEmbeddingLayer(nn.Module):
    def __init__(self, dim_E, in_vocab):
        super(TreeEmbeddingLayer, self).__init__()
        self.E = nn.Parameter(torch.Tensor(in_vocab, dim_E)).to("cuda:0")
        nn.init.uniform_(self.E, -0.05, 0.05)

    def forward(self, x_tensor):
        '''x: list of [1,]'''
        x_len = [xx.shape[0] for xx in x_tensor]
        x_tensor = [torch.LongTensor(xx).to("cuda:0") for xx in x_tensor]
        ex = self.E[x_tensor[0].flatten()]
        for i in range(1, len(x_tensor)):
            ex_i = self.E[x_tensor[i].flatten()]
            ex = torch.cat((ex, ex_i))
        exs = list(torch.split(ex, x_len, dim=0))

        return exs


class ReduceHidden(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(ReduceHidden, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(3 * self.hidden_size, self.hidden_size)

        init_linear_wt(self.linear)

    def forward(self, code_hidden, ast_hidden, tree_hidden):
        """

        :param code_hidden: hidden state of code encoder, [1, B, H]
        :param ast_hidden: hidden state of ast encoder, [1, B, H]
        :return: [1, B, H]
        """
        hidden = torch.cat((code_hidden, ast_hidden, tree_hidden), dim=2)
        hidden = self.linear(hidden)
        hidden = F.relu(hidden)
        return hidden


class Attention(nn.Module):

    def __init__(self, hidden_size=config.hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(2 * self.hidden_size, self.hidden_size)
        self.v = nn.Parameter(torch.rand(self.hidden_size), requires_grad=True)   # [H]
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs):
        time_step, batch_size, _ = encoder_outputs.size()
        h = hidden.repeat(time_step, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], dim=2)))
        energy = energy.transpose(1, 2)     # [B, H, T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)      # [B, 1, H]
        energy = torch.bmm(v, energy)   # [B, 1, T]
        return energy.squeeze(1)


class Decoder(nn.Module):

    def __init__(self, vocab_size, hidden_size=config.hidden_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.decoder_dropout_rate)
        self.source_attention = Attention()
        self.code_attention = Attention()
        self.ast_attention = Attention()
        self.gru = nn.GRU(config.embedding_dim + self.hidden_size, self.hidden_size)
        self.out = nn.Linear(2 * self.hidden_size, config.nl_vocab_size)

        init_wt_normal(self.embedding.weight)
        init_rnn_wt(self.gru)
        init_linear_wt(self.out)

    def forward(self, inputs: torch.Tensor, last_hidden: torch.Tensor,
                source_outputs: torch.Tensor, code_outputs: torch.Tensor, ast_outputs: torch.Tensor) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        forward the net
        :param inputs: word input of current time step, [B]
        :param last_hidden: last decoder hidden state, [1, B, H]
        :param code_outputs: outputs of code encoder, [T, B, H]
        :param ast_outputs: outputs of ast encoder, [T, B, H]
        :return: output: [B, nl_vocab_size]
                hidden: [1, B, H]
                attn_weights: [B, 1, T]
        """
        embedded = self.embedding(inputs).unsqueeze(0)      # [1, B, embedding_dim]
        # embedded = self.dropout(embedded)
        source_attn_weights = self.source_attention(last_hidden, source_outputs)  # [B, 1, T]
        source_context = source_attn_weights.bmm(source_outputs.transpose(0, 1))  # [B, 1, H]
        source_context = source_context.transpose(0, 1)  # [1, B, H]

        code_attn_weights = self.code_attention(last_hidden, code_outputs)  # [B, 1, T]
        code_context = code_attn_weights.bmm(code_outputs.transpose(0, 1))  # [B, 1, H]
        code_context = code_context.transpose(0, 1)     # [1, B, H]

        ast_attn_weights = self.ast_attention(last_hidden, ast_outputs)  # [B, 1, T]
        ast_context = ast_attn_weights.bmm(ast_outputs.transpose(0, 1))     # [B, 1, H]
        ast_context = ast_context.transpose(0, 1)   # [1, B, H]

        context = 0.5 * source_context + 0.5 * code_context + ast_context  # [1, B, H]

        rnn_input = torch.cat([embedded, context], dim=2)   # [1, B, embedding_dim + H]
        outputs, hidden = self.gru(rnn_input, last_hidden)  # [1, B, H] for both
        outputs = outputs.squeeze(0)    # [B, H]
        context = context.squeeze(0)    # [B, H]
        outputs = self.out(torch.cat([outputs, context], 1))    # [B, nl_vocab_size]
        outputs = F.log_softmax(outputs, dim=1)     # [B, nl_vocab_size]
        return outputs, hidden, source_attn_weights, code_attn_weights, ast_attn_weights


class Model(nn.Module):

    def __init__(self, source_vocab_size, code_vocab_size, ast_vocab_size, nl_vocab_size, len_code_w2i, len_nl_w2i,
                 model_file_path=None, model_state_dict=None, is_eval=False):
        super(Model, self).__init__()

        # vocabulary size for encoders
        self.source_vocab_size = source_vocab_size
        self.code_vocab_size = code_vocab_size
        self.ast_vocab_size = ast_vocab_size
        self.is_eval = is_eval
        # code_w2i = utils.read_pickle("dataset/code_w2i.pkl")
        # nl_w2i = utils.read_pickle("dataset/nl_w2i.pkl")
        self.len_code_w2i = len_code_w2i
        self.len_nl_w2i = len_nl_w2i
        # init models
        self.source_encoder = Encoder(self.source_vocab_size)
        self.code_encoder = Encoder(self.code_vocab_size)
        self.ast_encoder = Encoder(self.ast_vocab_size)
        self.tree_encoder = TreeEncoder(self.len_code_w2i, self.len_nl_w2i)
        self.reduce_hidden = ReduceHidden()
        self.decoder = Decoder(nl_vocab_size)

        if config.use_cuda:
            self.source_encoder = self.source_encoder.cuda()
            self.code_encoder = self.code_encoder.cuda()
            self.ast_encoder = self.ast_encoder.cuda()
            self.tree_encoder = self.tree_encoder.cuda()
            self.reduce_hidden = self.reduce_hidden.cuda()
            self.decoder = self.decoder.cuda()

        if model_file_path:
            state = torch.load(model_file_path)
            self.set_state_dict(state)

        if model_state_dict:
            self.set_state_dict(model_state_dict)

        if is_eval:
            self.source_encoder.eval()
            self.code_encoder.eval()
            self.ast_encoder.eval()
            self.tree_encoder.eval()
            self.reduce_hidden.eval()
            self.decoder.eval()

    def forward(self, x, batch, batch_size, nl_vocab, is_test=False):
        """

        :param batch:
        :param batch_size:
        :param nl_vocab:
        :param is_test: if True, function will return before decoding
        :return: decoder_outputs: [T, B, nl_vocab_size]
        """
        # batch: [T, B]
        source_batch, source_seq_lens, code_batch, code_seq_lens, ast_batch, ast_seq_lens, nl_batch, nl_seq_lens = batch

        # encode
        # outputs: [T, B, H]
        # hidden: [2, B, H]
        source_outputs, source_hidden = self.source_encoder(source_batch, source_seq_lens)
        code_outputs, code_hidden = self.code_encoder(code_batch, code_seq_lens)
        ast_outputs, ast_hidden = self.ast_encoder(ast_batch, ast_seq_lens)

        # data for decoder
        code_hidden = code_hidden[:1]  # [1, B, H]
        ast_hidden = ast_hidden[:1]  # [1, B, H]

        (tree_hidden, tree_cell) = self.tree_encoder(x)
        tree_hidden = tree_hidden.unsqueeze(0)  
        tree_cell = tree_cell.unsqueeze(0)  

        zero_hidden = torch.zeros_like(ast_hidden).to("cuda:0")	
        tree_hidden = self.reduce_hidden(tree_hidden, tree_cell, zero_hidden)
        decoder_hidden = self.reduce_hidden(code_hidden, ast_hidden, tree_hidden)  # [1, B, H]

        if is_test:
            return source_outputs, code_outputs, ast_outputs, decoder_hidden

        if nl_seq_lens is None:
            max_decode_step = config.max_decode_steps
        else:
            max_decode_step = max(nl_seq_lens)

        decoder_inputs = utils.init_decoder_inputs(batch_size=batch_size, vocab=nl_vocab)  # [B]

        decoder_outputs = torch.zeros((max_decode_step, batch_size, config.nl_vocab_size), device=config.device)

        for step in range(max_decode_step):
            # decoder_outputs: [B, nl_vocab_size]
            # decoder_hidden: [1, B, H]
            # attn_weights: [B, 1, T]
            decoder_output, decoder_hidden, \
                source_attn_weights, code_attn_weights, ast_attn_weights = self.decoder(inputs=decoder_inputs,
                                                                   last_hidden=decoder_hidden,
                                                                   source_outputs=source_outputs,
                                                                   code_outputs=code_outputs,
                                                                   ast_outputs=ast_outputs)
            decoder_outputs[step] = decoder_output

            if config.use_teacher_forcing and random.random() < config.teacher_forcing_ratio and not self.is_eval:
                # use teacher forcing, ground truth to be the next input
                decoder_inputs = nl_batch[step]
            else:
                # output of last step to be the next input
                _, indices = decoder_output.topk(1)  # [B, 1]
                decoder_inputs = indices.squeeze(1).detach()  # [B]
                decoder_inputs = decoder_inputs.to(config.device)

        return decoder_outputs

    def set_state_dict(self, state_dict):
        self.source_encoder.load_state_dict(state_dict["source_encoder"])
        self.code_encoder.load_state_dict(state_dict["code_encoder"])
        self.ast_encoder.load_state_dict(state_dict["ast_encoder"])
        self.tree_encoder.load_state_dict(state_dict["tree_encoder"])
        self.reduce_hidden.load_state_dict(state_dict["reduce_hidden"])
        self.decoder.load_state_dict(state_dict["decoder"])
