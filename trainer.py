import os
import numpy as np
import lightning as L
import onnxruntime
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from data.dataset_manager import DatasetManager
from model.monodert.lightning_model import LitMonoDeRT
from model.monodert.net import MonoDeRT
from model.unet.lightning_model import LitUNet
from model.unet.net import UNet
from utilities.plots import plot_predictions
from model.monodepth_rt.lightning_model import LitMonoDepthRT
from model.monodepth_rt.net import MonoDepthRT
from utilities.callbacks import get_callbacks
from utilities.logger import get_logger

class Trainer:
    def __init__(self, options):
        self.opt = options

        # misc
        self.device = self.opt.device
        self.size = (self.opt.channels, self.opt.height, self.opt.width)
        self.experiment_name = f'{self.opt.model_name}-d={self.opt.dataset}-lr={self.opt.learning_rate}-e={self.opt.num_epochs}'
        self.loaded = False

        # data
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = (None, None, None)

        # trainer settings
        self.checkpoint_path = None
        self.logger = get_logger(self.experiment_name)
        self.version = self.logger.version
        self.cb_list = get_callbacks()
        self.trainer = self.trainer = L.Trainer(max_epochs=self.opt.num_epochs, log_every_n_steps=1, logger=self.logger,
                                                callbacks=self.cb_list, accelerator='auto')

        # model loading
        self.plain_model = None
        self.lit_model = None
        self.select_model()

    def select_model(self):
        if self.opt.model_name == 'unet':
            self.plain_model = UNet(3, 1)
            self.lit_model = LitUNet(self.plain_model.to(self.device), self.opt.height, self.opt.learning_rate)
            return UNet, LitUNet

        elif self.opt.model_name == 'monodepthrt':
            self.plain_model = MonoDepthRT(3, 1, False)
            self.lit_model = LitMonoDepthRT(self.plain_model.to(self.device), self.opt.height, self.opt.learning_rate)
            return MonoDepthRT, LitMonoDepthRT

        elif self.opt.model_name == 'monodert':
            self.plain_model = MonoDeRT(3, 1, False)
            self.lit_model = LitMonoDeRT(self.plain_model.to(self.device), self.opt.height, self.opt.learning_rate)
            return MonoDepthRT, LitMonoDepthRT

        else:
            raise("Choose an existing model")

    def train(self):
        self.train_dataloader, self.valid_dataloader, self.test_dataloader = self.get_data()

        if self.checkpoint_path:
            self.trainer.fit(model=self.lit_model, ckpt_path=self.checkpoint_path,
                             train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)
        else:
            self.trainer.fit(model=self.lit_model, train_dataloaders=self.train_dataloader, val_dataloaders=self.valid_dataloader)

        self.loaded = True

    def eval(self):
        self.trainer.test(self.lit_model, dataloaders=self.test_dataloader)

    def save(self):
        self.trainer.save_checkpoint(f'{self.opt.checkpoint_dir}/{self.experiment_name}.ckpt')

    def load(self, checkpoint_name):
        if self.opt.in_root:
            self.checkpoint_path = f'{checkpoint_name}.ckpt'
        else:
            self.checkpoint_path = f'{self.opt.checkpoint_dir}/{checkpoint_name}.ckpt'

        _, lit_class = self.select_model()

        self.lit_model = lit_class.load_from_checkpoint(
            checkpoint_path=self.checkpoint_path,
            plain_model=self.plain_model,
            size=self.opt.width,
            lr=self.opt.learning_rate
        )

        self.loaded = True

    def set_eval(self):
        self.lit_model.eval()

    def set_train(self):
        self.lit_model.train()




    ###################################
    ###   Inference                 ###
    ###################################
    def predict(self, input):
        return self.lit_model(input.to(self.lit_model.device))





    ###################################
    ###   Data Management           ###
    ###################################
    def get_data(self):
        print('### INITIALIZING DATALOADERS')
        h = DatasetManager(self.opt.data_path, self.opt)
        if self.opt.dataset == 'nyu_v2':
            train_data, valid_data, test_data = h.load_nyu_v2()
        elif self.opt.dataset == 'diode_val':
            train_data, valid_data, test_data = h.load_diode()
        elif self.opt.dataset == 'nyu_v2_folder':
            train_data, valid_data, test_data = h.load_nyu_v2_folders()
        else:
            train_data, valid_data, test_data = (None, None, None)

        train_dataloader = DataLoader(train_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0) #self.opt.num_workers
        valid_dataloader = DataLoader(valid_data, batch_size=self.opt.batch_size, shuffle=False, num_workers=0)
        test_dataloader = DataLoader(test_data, batch_size=self.opt.batch_size, shuffle=True, num_workers=0)

        print('### DATALOADERS INITIALIZED')
        return (train_dataloader, valid_dataloader, test_dataloader)






    ###################################
    ###   Plotting                  ###
    ###################################
    def plot_metrics(self, path_to_metrics=None):
        if path_to_metrics is None:
            if not self.loaded:
                print('LOAD SOME CHECKPOINT')
                return
            path_to_metrics = self.logger.log_dir

        results = pd.read_csv(path_to_metrics)

        import matplotlib
        matplotlib.use('TkAgg')
        plt.style.use('default')
        plt.rcParams['text.usetex'] = False

        fig, axes = plt.subplots(2, 2, figsize=(15, 7))

        # fig, axes = plt.subplots(3, 2, figsize=(15, 7))
        # gs = axes[0, 0].get_gridspec()
        # for ax in axes[:2, 0]: ax.remove()
        # ax = fig.add_subplot(gs[:2, 0])
        ax = axes[0][0]
        ax.set_title('Losses per epoch')
        ax.plot(results['train_total_loss_epoch'].dropna(ignore_index=True), label='train total loss', color='dodgerblue')
        ax.plot(results['train_mse_loss_epoch'].dropna(ignore_index=True), label='train MSE loss', color='orangered')
        ax.plot(results['train_ssim_loss_epoch'].dropna(ignore_index=True), label='train MSE loss', color='mediumseagreen')
        ax.plot(results['train_berhu_loss_epoch'].dropna(ignore_index=True), label='train BerHu loss', color='goldenrod')
        ax.plot(results['train_eas_loss_epoch'].dropna(ignore_index=True), label='train EAS loss', color='darkgreen')
        ax.plot(results['train_silog_loss_epoch'].dropna(ignore_index=True), label='train SiLog loss', color='palegreen')
        ax.plot(results['train_sasinv_loss_epoch'].dropna(ignore_index=True), label='train SaSInv loss', color='coral')

        ax.plot(results['valid_total_loss'].dropna(ignore_index=True), label='validation total loss', color='dodgerblue', alpha=0.5)
        # ax.plot(results['valid_mse_loss'].dropna(ignore_index=True), label='validation MSE loss', color='orangered', alpha=0.5)
        ax.plot(results['valid_ssim_loss'].dropna(ignore_index=True), label='validation MSE loss', color='mediumseagreen', alpha=0.5)
        ax.plot(results['valid_berhu_loss'].dropna(ignore_index=True), label='validation BerHu loss', color='goldenrod', alpha=0.5)
        ax.plot(results['valid_eas_loss'].dropna(ignore_index=True), label='validation EAS loss', color='darkgreen', alpha=0.5)
        ax.plot(results['valid_silog_loss'].dropna(ignore_index=True), label='validation SiLog loss', color='palegreen', alpha=0.5)
        # ax.plot(results['valid_sasinv_loss'].dropna(ignore_index=True), label='validation SaSInv loss', color='coral', alpha=0.5)
        ax.legend(loc='upper center', prop = { "size": 7.5 }, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
        # ax.set_ylim(0, 2)
        ax.grid()



        ax = axes[1][1]
        ax.set_title('Deltas per epoch')
        # plt.rcParams['text.usetex'] = True
        ax.plot(results['train_delta1_epoch'].dropna(ignore_index=True), color='royalblue', label='δ_1 train')
        ax.plot(results['train_delta2_epoch'].dropna(ignore_index=True), color='cornflowerblue', label='δ_2 train')
        ax.plot(results['train_delta3_epoch'].dropna(ignore_index=True), color='lightsteelblue', label='δ_3 train')

        ax.plot(results['valid_delta1'].dropna(ignore_index=True), color='orangered', label='δ_1 validation')
        ax.plot(results['valid_delta2'].dropna(ignore_index=True), color='coral', label='δ_2 validation')
        ax.plot(results['valid_delta3'].dropna(ignore_index=True), color='darksalmon', label='δ_3 validation')
        ax.legend(loc='upper center', prop = { "size": 7.5 }, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
        ax.grid()
        # plt.rcParams['text.usetex'] = False



        ax = axes[1][0]
        ax.set_title('Absolute Relative Error per epoch')
        ax.plot(results['train_abs_rel_epoch'].dropna(ignore_index=True), color='crimson', label='(AbsRel) absolute relative error train')
        ax.plot(results['valid_abs_rel'].dropna(ignore_index=True), color='crimson', label='(AbsRel) absolute relative error validation', alpha=0.7)
        ax.legend(loc='upper center', prop = { "size": 7.5 }, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=1)
        ax.grid()



        # gs = axes[2, 0].get_gridspec()
        # for ax in axes[2, :2]: ax.remove()
        # ax = fig.add_subplot(gs[2, :2])
        ax = axes[0][1]
        ax.set_title('Errors per epoch')
        ax.plot(results['train_mae_epoch'].dropna(ignore_index=True), color='royalblue', label='(MAE) mean absolute error train')
        ax.plot(results['train_log_mae_epoch'].dropna(ignore_index=True), color='cornflowerblue', label='(MAElog) mean absolute log error train')
        ax.plot(results['train_rmse_epoch'].dropna(ignore_index=True), color='darkgreen', label='(RMSE) root mean squared error')
        ax.plot(results['train_log_rmse_epoch'].dropna(ignore_index=True), color='forestgreen', label='(RMSElog) root mean squared log error')

        ax.plot(results['valid_mae'].dropna(ignore_index=True), color='royalblue', label='(MAE) mean absolute error validation', alpha=0.7)
        ax.plot(results['valid_log_mae'].dropna(ignore_index=True), color='cornflowerblue', label='(MAElog) mean absolute log error validation', alpha=0.7)
        ax.plot(results['valid_rmse'].dropna(ignore_index=True), color='darkgreen', label='(RMSE) root mean squared error validation', alpha=0.7)
        ax.plot(results['valid_log_rmse'].dropna(ignore_index=True), color='forestgreen', label='(RMSElog) root mean squared log error validation', alpha=0.7)
        # ax.set_ylim(0, 2)
        ax.legend(loc='upper center', prop = { "size": 7.5 }, bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
        ax.grid()

        fig.tight_layout(pad=1.0)

        plt.show()

    def plot_batch_predictions(self, save=False, title=None):
        if not self.loaded:
            print('LOAD SOME CHECKPOINT')
            return
        self.set_train()

        if title is None: title = f'V{self.version}-{self.experiment_name}'

        if self.test_dataloader is None:
            _, _, self.test_dataloader = self.get_data()

        inputs, target = next(iter(self.test_dataloader))
        all_preds = self.predict(inputs)
        plot_predictions(inputs, target, all_preds, save=save, title=title)





    ###################################
    ###   Optimization              ###
    ###################################
    def quant(self):
        def print_model_size(mdl):
            torch.save(mdl.state_dict(), "tmp.pt")
            print("%.2f MB" % (os.path.getsize("tmp.pt") / 1e6))
            os.remove('tmp.pt')

        print_model_size(self.plain_model)

        # backend = "qnnpack"
        # backend = "x86"
        backend = "fbgemm"
        self.plain_model.qconfig = torch.quantization.get_default_qconfig(backend)
        torch.backends.quantized.engine = backend

        self.plain_model = torch.quantization.prepare(self.plain_model, inplace=False)

        # model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
        self.plain_model = torch.quantization.convert(self.plain_model, inplace=False)

        # print_model_size(model_static_quantized)
        print_model_size(self.plain_model)

    def prune(self):
        import torch.nn.utils.prune as prune

        for name, module in self.plain_model.named_modules():
            # prune 20% of connections in all 2D-conv layers
            if isinstance(module, torch.nn.Conv2d):
                prune.l1_unstructured(module, name='weight', amount=0.1)
                # prune.random_unstructured(module, name='weight', amount=0.2)
                # prune.ln_structured(module, name='weight', amount=0.2, n=2, dim=0)




    ###################################
    ###   TF Lite                   ###
    ###################################
    # def onnx_to_tf_lite(self, onnx_model_name="model.onnx"):
    #     import onnx
    #     from onnx_tf.backend import prepare
    #
    #     onnx_model = onnx.load(onnx_model_name)
    #     tf_rep = prepare(onnx_model)
    #     tf_rep.export_graph("model.pb")
    #
    #     # Convert TensorFlow to TensorFlow Lite
    #     converter = tf.lite.TFLiteConverter.from_saved_model("model.pb")
    #     tflite_model = converter.convert()
    #
    #     with open("model.tflite", "wb") as f:
    #         f.write(tflite_model)







    ###################################
    ###   TorchScript               ###
    ###################################
    def save_as_torch_script(self):
        self.plain_model.eval()
        traced_model = torch.jit.trace(self.plain_model, torch.randn(1, 3, 256, 256))
        traced_model.save("model.pt")







    ###################################
    ###   ONNX                      ###
    ###################################
    def save_as_onnx(self):
        self.plain_model.eval()
        torch.onnx.export(self.plain_model.cpu(), torch.randn(1, 3, 256, 256), "model.onnx", input_names=["input"], output_names=["output"])
                          #opset_version=11)

    def load_from_onnx(self):
        import onnx
        import onnxruntime as ort
        from onnx.tools.net_drawer import GetPydotGraph, GetOpNodeProducer

        # load ONNX model
        onnx_model = onnx.load("model.onnx")

        print_graph = False
        if print_graph:
            # onnx.helper.printable_graph(onnx_model.graph)
            print('Model :\n\n{}'.format(onnx.helper.printable_graph(onnx_model.graph)))

            pydot_graph = GetPydotGraph(
                onnx_model.graph,
                name=onnx_model.graph.name,
                rankdir="TB",
                node_producer=GetOpNodeProducer("docstring"),
            )
            pydot_graph.write_dot("graph.dot")
            os.system("dot -O -Tpng graph.dot")
            image = plt.imread("graph.dot.png")
            plt.imshow(image)
            plt.axis("off")

        onnx.checker.check_model(onnx_model)
        print("ONNX model is valid")

        self.ort_session = ort.InferenceSession("model.onnx")

    def load_from_onnx_optimized(self, model_name="model.onnx"):
        import onnxruntime as ort

        so = onnxruntime.SessionOptions()
        so.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        exec_providers = [
            ('CUDAExecutionProvider', {"cudnn_conv_use_max_workspace": '1'}),
            'CPUExecutionProvider'
        ]

        self.ort_session = ort.InferenceSession(model_name, so, providers=exec_providers)

        # options = self.ort_session.get_provider_options()
        # cuda_options = options['CUDAExecutionProvider']
        # cuda_options['cudnn_conv_use_max_workspace'] = '1'
        # self.ort_session.set_providers(['CUDAExecutionProvider'], [cuda_options])
        print("ONNX loaded")

    def load_from_onnx_quant_optimized(self):
        self.load_from_onnx_optimized("quantized_model.onnx")

    def onnx_quant(self):
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType

        # Create a calibration data reader
        class MyDataReader(CalibrationDataReader):
            def __init__(self, input_data):
                self.data = input_data
                self.enum_data = None

            def get_next(self):
                if self.enum_data is None:
                    self.enum_data = iter(self.data)
                return next(self.enum_data, None)

            def rewind(self):
                self.enum_data = None

        # Prepare your calibration dataset
        calibration_data = []

        # Create a data reader instance
        calibration_data_reader = MyDataReader(calibration_data)

        # Quantize the model
        quantize_static("model.onnx", "quantized_model.onnx", calibration_data_reader, quant_format=QuantType.QUInt8)

        print(f'ONNX Quantized')

    def onnx_predict(self, frame):
        return self.ort_session.run(None, {"input": frame})