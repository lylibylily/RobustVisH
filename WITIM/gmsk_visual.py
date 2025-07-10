#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: gmsk for visual data
# Author: Z
# GNU Radio version: 3.10.10.0

from gnuradio import analog
from gnuradio import blocks
import pmt
from gnuradio import digital
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from argparse import ArgumentParser
# 创建ArgumentParser对象
parser = ArgumentParser(description="gmsk for visual data")
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import time


class gmsk_visual(gr.top_block):

    def __init__(self, input_file, output_file):
        gr.top_block.__init__(self, "gmsk for visual data", catch_exceptions=True)

        ##################################################
        # Blocks
        ##################################################

        self.digital_gmsk_mod_0 = digital.gmsk_mod(
            samples_per_symbol=2,
            bt=0.35,
            verbose=False,
            log=False,
            do_unpack=True)
        self.digital_gmsk_demod_0 = digital.gmsk_demod(
            samples_per_symbol=2,
            gain_mu=0.175,
            mu=0.5,
            omega_relative_limit=0.005,
            freq_error=0.0,
            verbose=False,log=False)
        self.blocks_unpacked_to_packed_xx_0 = blocks.unpacked_to_packed_bb(1, gr.GR_MSB_FIRST)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_cc(1)
        self.blocks_float_to_complex_0 = blocks.float_to_complex(1)
        self.blocks_file_source_0 = blocks.file_source(gr.sizeof_char*1, input_file, False, 0, 0)
        self.blocks_file_source_0.set_begin_tag(pmt.PMT_NIL)
        self.blocks_file_sink_0 = blocks.file_sink(gr.sizeof_char*1, output_file, False)
        self.blocks_file_sink_0.set_unbuffered(False)
        self.blocks_add_xx_0 = blocks.add_vcc(1)
        self.analog_noise_source_x_1 = analog.noise_source_c(analog.GR_GAUSSIAN, 0.054, 42)
        self.analog_noise_source_x_0 = analog.noise_source_f(analog.GR_IMPULSE, 0.046, 42)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_noise_source_x_0, 0), (self.blocks_float_to_complex_0, 0))
        self.connect((self.analog_noise_source_x_1, 0), (self.blocks_add_xx_0, 2))
        self.connect((self.blocks_add_xx_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_file_source_0, 0), (self.digital_gmsk_mod_0, 0))
        self.connect((self.blocks_float_to_complex_0, 0), (self.blocks_add_xx_0, 1))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.digital_gmsk_demod_0, 0))
        self.connect((self.blocks_unpacked_to_packed_xx_0, 0), (self.blocks_file_sink_0, 0))
        self.connect((self.digital_gmsk_demod_0, 0), (self.blocks_unpacked_to_packed_xx_0, 0))
        self.connect((self.digital_gmsk_mod_0, 0), (self.blocks_add_xx_0, 0))





def main(top_block_cls=gmsk_visual, options=None):
    args = parser.parse_args()
    tb = top_block_cls(args.input, args.output)

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        sys.exit(0)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    tb.start()

    try:
        time.sleep(6)  # 暂停6秒
    except EOFError:
        pass
    tb.stop()
    tb.wait()


if __name__ == '__main__':
    parser.add_argument('-i', '--input', type=str, required=True, help='Input file path')
    parser.add_argument('-o', '--output', type=str, required=True, help='Output file path')
    main()


# D:/Software/radioconda/python.exe D:/GRC310/gmsk_visual.py -i D:/dataset/WTD-VTD/work/data/au/visual/object_01_01.jpg	-o D:/dataset/WTD-VTD/work/outputdata/au/visual/object_01_01.jpg
