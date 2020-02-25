# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys
import json
import time
import torch
from Core.utils.utils import sec_to_hm_str, normalize_image
from termcolor import colored 
class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        self._abbrev_name = kwargs.pop("abbrev_name", "")
        if len(self._abbrev_name):
            self._abbrev_name = self._abbrev_name + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        record.name = record.name.replace(self._root_name, self._abbrev_name)
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log



def setup_logger(ouput, save_dir, distributed_rank, color=True, name="Instance-wise-Depth", abbrev_name=None, filename="log.txt"):
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    
    if abbrev_name is None:
        abbrev_name = "IWD" if name == "Instance-wise-Depth" else name
    
    plain_formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M:%S"
    )    
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    if color:
            formatter = _ColorfulFormatter(
                colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
                datefmt="%m/%d %H:%M:%S",
                root_name=name,
                abbrev_name=str(abbrev_name),
            )
    else:
            formatter = plain_formatter
            
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_time(cfg, batch_idx, duration, loss, start_time, step, num_total_steps ):
    """Print a logging statement to the terminal
    """
    samples_per_sec = cfg.SOLVER.IMS_PER_BATCH / duration
    time_sofar = time.time() - start_time
    training_time_left = (
        num_total_steps / step - 1.0) * time_sofar if step > 0 else 0
    print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
        " | loss: {:.5f} | time elapsed: {} | time left: {}"
    print(print_string.format(cfg.MODEL.NUM_EPOCHS, batch_idx, samples_per_sec, loss,
                                sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

def log(cfg, writers, step, mode, inputs, outputs, losses):
    """Write an event to the tensorboard events file
    """
    writer = writers[mode]
    for l, v in losses.items():
        writer.add_scalar("{}".format(l), v, step)

    for j in range(min(4, cfg.SOLVER.IMS_PER_BATCH)):  # write a maxmimum of four images
        for s in cfg.MODEL.DEPTH.SCALES:
            for frame_id in cfg.MODEL.DEPTH.FRAMES_IDS:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, step)
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, step)

            writer.add_image(
                "disp_{}/{}".format(s, j),
                normalize_image(outputs[("disp", s)][j]), step)

                   
            writer.add_image(
                    "automask_{}/{}".format(s, j),
                    outputs["identity_selection/{}".format(s)][j][None, ...], step)


