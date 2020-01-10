import os
import torch
from torch import nn
from torch.autograd import Variable


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias, dtype):
        """
        Initialize the ConvLSTM cell
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.height, self.width = input_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2   #padding 是kernel的一半做卷积的时候尺寸就不会变化，满足GRU不同时间点输入输出尺度的一致性
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.dtype = dtype

        self.reset_conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,    #【ht-1,x】,用于对输入和隐含层的卷积
                                    out_channels=self.hidden_dim,  #
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.update_conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,    #【ht-1,x】,用于对输入和隐含层的卷积
                                    out_channels=self.hidden_dim,  #
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.reset_gate_bn=nn.BatchNorm2d(self.hidden_dim)
        self.update_gate_bn = nn.BatchNorm2d(self.hidden_dim)

        self.conv_can = nn.Conv2d(in_channels=input_dim+hidden_dim,          #【ht-1,x】,用于对输入和隐含层的卷积
                              out_channels=self.hidden_dim, # for candidate neural memory 卷积输出作为候选输出值
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=self.bias)
        self.cand_bn = nn.BatchNorm2d(self.hidden_dim)
        self.relu = torch.nn.LeakyReLU(negative_slope=0.1)

        ## gru初始化很很重要
        #init_cnn_weight= torch.nn.init.kaiming_normal_
        #init_cnn_weight = torch.nn.init.orthogonal_
        #init_cnn_weight= torch.nn.init.xavier_uniform_
        init_cnn_weight = torch.nn.init.orthogonal_
        init_cnn_bias =torch.nn.init.zeros_

        init_cnn_weight( self.reset_conv_gates.weight)
        init_cnn_bias(self.reset_conv_gates.bias)

        init_cnn_weight(self.update_conv_gates.weight)
        init_cnn_bias(self.update_conv_gates.bias)
        init_cnn_weight(self.conv_can.weight)
        init_cnn_bias(self.conv_can.bias)




    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).type(self.dtype))

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)          #输入和隐层组合，在通道维度上

        gamma =self.reset_conv_gates(combined)
        beta = self.update_conv_gates(combined)

        #使用bn归一化后再激活
        gamma = self.reset_gate_bn(gamma)
        beta  = self.update_gate_bn(beta)
        #激活
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)     #复位门作用在隐层之后，与输入在通道维度上concat
        cc_cnm = self.conv_can(combined)                                  #最后输出门卷积得到候选值
        cc_cnm =self.cand_bn(cc_cnm)
        # cnm = torch.tanh(cc_cnm)
        cnm =  self.relu(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm           #融合上一状态和候选值就得到新的状态
        return h_next


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM model
    Reference:
      Xingjian Shi et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting."
    """

    def __init__(self, shape, input_channel, filter_size, hidden_size):
        super().__init__()
        self._shape = shape
        self._input_channel = input_channel
        self._filter_size = filter_size
        self._hidden_size = hidden_size
        self._conv = nn.Conv2d(in_channels=self._input_channel + self._hidden_size,
                               ###hidden state has similar spational struture as inputs, we simply concatenate them on the feature dimension
                               out_channels=self._hidden_size * 4,  ##lstm has four gates
                               kernel_size=self._filter_size,
                               stride=1,
                               padding=1)

    def forward(self, x, state):  # x为【4,8,20,20】

        _hidden, _cell = state  # 【4,64,20,20】，【4,64,20,20】
        # print(x.shape,_hidden.shape)
        cat_x = torch.cat([x, _hidden], dim=1)  # 在通道维度上concat ,也就是Lstm中 【x,ht-1]的合并
        Conv_x = self._conv(cat_x)

        i, f, o, j = torch.chunk(Conv_x, 4, dim=1)  # 卷积之后在拆分开，在通道维度上拆开，分别对应输入门，遗忘门，输出门，更新门

        i = torch.sigmoid(i)  # 加上激活之后才叫门
        f = torch.sigmoid(f)
        m = nn.LeakyReLU(0.1)
        # m= torch.tanh
        cell = _cell * f + i * m(j)  # 使用leakrelu激活，防止梯度消失
        o = torch.sigmoid(o)
        hidden = o * m(cell)

        return hidden, cell



class ConvGRU(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 dtype, batch_first=False, bias=True, return_all_layers=False):
        """
        :param input_size: (int, int)
            Height and width of input tensor as (height, width).
        :param input_dim: int e.g. 256
            Number of channels of input tensor.
        :param hidden_dim: int e.g. 1024
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param num_layers: int
            Number of ConvLSTM layers
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        :param alexnet_path: str
            pretrained alexnet parameters
        :param batch_first: bool
            if the first position of array is batch or not
        :param bias: bool
            Whether or not to add the bias.
        :param return_all_layers: bool
            if return hidden and cell states for all layers
        """
        super(ConvGRU, self).__init__()

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)   #如果输入的参数，不是list类型，则扩展为list类型，扩展份数有num_layers指定
        hidden_dim  = self._extend_for_multilayer(hidden_dim, num_layers)    #隐藏层的卷积层层数
        if not len(kernel_size) == len(hidden_dim) == num_layers:             #判断list的长度相等
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.dtype = dtype
        self.num_layers = num_layers
        self.batch_first = batch_first                                        #batch_first适合dataloader操作
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim[i - 1]         #GRU输入的第一层为特征输入维度，否则级联的GRU的输入就算中间的隐含状态层
            cell_list.append(ConvGRUCell(input_size=(self.height, self.width),
                                         input_dim=cur_input_dim,
                                         hidden_dim=self.hidden_dim[i],
                                         kernel_size=self.kernel_size[i],
                                         bias=self.bias,
                                         dtype=self.dtype))

        # convert python list to pytorch module
        self.cell_list = nn.ModuleList(cell_list)            #将python中的list转换为pytorch种的模块，

    def forward(self, input_tensor, hidden_state=None):
        """
        :param input_tensor: (b, t, c, h, w) or (t,b,c,h,w) depends on if batch first or not
            extracted features from alexnet
        :param hidden_state:
        :return: layer_output_list, last_state_list
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4).contiguous()



        # print("input_shape",input_tensor.shape)
        # Implement stateful ConvLSTM
        # 初始状态一直为0，测试的时候可以考虑更改
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))   #取出batch的值，同时利用之前初始化好的hidden层的大小，定义隐含层变量



        layer_output_list = []
        last_state_list   = []

        seq_len = input_tensor.size(1)    # n s c h w 中的s,
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):    #遍历构建级联的RRU
            h = hidden_state[layer_idx]             #取出隐含层

            cur_layer_input_list=torch.chunk(cur_layer_input,seq_len,dim=1)     # [b,t,c,h,w]-->(b,1,c,h,w)的多个list元素
            output_inner = []
            for t in range(seq_len):
                # input current hidden and cell state then compute the next hidden and cell state through ConvLSTMCell forward function

                h = self.cell_list[layer_idx](input_tensor=cur_layer_input_list[t].squeeze(dim=1), # (b,1,c,h,w)-->(b,c,h,h)，的张量送入GRU模块
                                              h_cur=h)
                output_inner.append(h)                                                      #每次输入一个tensor会更新gru的状态，将gru的状态放入output_inner中缓存

            layer_output = torch.stack(output_inner, dim=1)    #【n c h w]的tensor在维度为1上stack 得到【n s c h w]作为当前这个gru模块的输出,隐含层结果包括每一步输入后的输出状态的结果，大小为【n,t,c,h,w】
            cur_layer_input = layer_output                     #当前这个gru模块的输出同时也是下一个gru模块的输入，所以赋值给cur_layer_input

            layer_output_list.append(layer_output)             #最后将每个gru模块中间隐层输出到list中，
            last_state_list.append(h)                        #输出最后一层的隐层，也就是最后一个状态，大小为【n c h w】,也就是layer_output的【n,-1,c,h,w】其中-1代表最后

        if not self.return_all_layers:                          #如果设置的不输出所有中间的GRU的隐含层的话，则只输出最后一层的隐含层（每一步对应的状态和最后一步对应的状态）
            layer_output_list = layer_output_list[-1:]          #只输出gru组的最后一个gru模块的状态的状态结果 ayer_output_list 是个list，包含有1个元素，这个元素大小为【n,t,c,h,w】
            last_state_list   = last_state_list[-1:]            ##last_state_list 也是个list，包含有1个元素 ，这个元素大小为【n,c,h,w】,也就是[n,t,c,h,w]中的最后一个tensor [n,-1,c,h,w]

        # 如果全部输出的话，这个信息还是蛮大的，layer_output_list 是个list，包含有self.num_layers个元素，每个元素大小为【n,t,c,h,w】
        #last_state_list 也是个list，包含有self.num_layers个元素，每个元素大小为【n,c,h,w】,也就是[n,t,c,h,w]中的最后一个tensor [n,-1,c,h,w]
        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                    (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):         #如果输入的参数，不是list类型，则扩展为list类型，扩展份数有num_layers指定
            param = [param] * num_layers
        return param


if __name__ == '__main__':
    # set CUDA device
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # detect if CUDA is available or not
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        dtype = torch.cuda.FloatTensor # computation in GPU
    else:
        dtype = torch.FloatTensor

    height = width = 26
    channels = 85

    num_layers =3             # number of stacked hidden layer GRU模块级联个数
    hidden_dim = [32,64,128]  #每个GRU模块对应的隐藏层的个数
    # kernel_size = (3,3) # kernel size for two stacked hidden layer
    kernel_size = [(3, 3),(5,5),(1,1)]  #也可以这样为3个gru模块指定不同的卷积核

    model = ConvGRU(input_size=(height, width),
                    input_dim=channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    dtype=dtype,
                    batch_first=True,
                    bias = True,
                    return_all_layers = False)

    model =model.to(device)
    print(model)
    for name,param in model.named_parameters():
        print(name)

    batch_size = 4
    time_steps = 7
    input_tensor = torch.rand(batch_size, time_steps, channels, height, width)  # (b,t,c,h,w)
    input_tensor = input_tensor.to(device)
    layer_output_list, last_state_list = model(input_tensor)

    print(len(layer_output_list),layer_output_list[0].shape)
    print(len(layer_output_list), len(layer_output_list[0]),layer_output_list[0][0].shape)