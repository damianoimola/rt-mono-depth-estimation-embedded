import numpy as np
from thop import profile



class Compare:
    def __init__(self, analyzed_model, reference_model, verbose = False):
        self.m1 = analyzed_model
        self.m2 = reference_model

        self.verbose = verbose

        self.m1_input = self.m2_input = None


    def compare(self, m1_input, m2_input):
        self.m1_input = m1_input
        self.m2_input = m2_input

        p1, p2 =self.count_parameters()
        m1, m2 = self.count_macs()
        f1, f2 = self.count_flops(p1, p2)
        mem1, mem2 = self.get_model_memory_usage(p1, p2)

        if self.verbose:
            print(p1, p2)
            print(m1, m2)
            print(f1, f2)
            print(mem1, mem2)

        metrics_analyzed_model = [p1, m1, f1, mem1]
        metrics_reference_model = [p2, m2, f2, mem2]

        eer = self.compute_EER(metrics_analyzed_model, metrics_reference_model)
        return eer



    def compute_EER(self, metrics_analyzed_model, metrics_reference_model):
        return 1 / len(metrics_reference_model) * sum(np.divide(metrics_analyzed_model, metrics_reference_model))

    def count_parameters(self):
        m1_params = sum(p.numel() for p in self.m1.parameters() if p.requires_grad)
        m2_params = sum(p.numel() for p in self.m2.parameters() if p.requires_grad)

        return m1_params, m2_params

    def count_macs(self):
        m1_macs = profile(self.m1, inputs=[self.m1_input], verbose=False)
        m2_macs = profile(self.m2, inputs=[self.m2_input], verbose=False)
        return m1_macs[0], m2_macs[0]

    def count_flops(self, m1_macs, m2_macs):
        m1_flops = 2*m1_macs
        m2_flops = 2*m2_macs
        return m1_flops, m2_flops

    def get_model_memory_usage(self, m1_params, m2_params):
        # model 1
        activations1 = self.m1(self.m1_input)
        activations_memory1 = activations1.element_size() * activations1.nelement()
        total_memory1 = activations_memory1 + m1_params * 4  # assuming 4 bytes per parameter
        m1_mem = total_memory1 / (1024 ** 2)  # Megabytes

        # model 2
        activations2 = self.m2(self.m2_input)
        activations_memory2 = activations2.element_size() * activations2.nelement()
        total_memory2 = activations_memory2 + m2_params * 4  # assuming 4 bytes per parameter
        m2_mem = total_memory2 / (1024 ** 2)  # Megabytes

        return m1_mem, m2_mem
