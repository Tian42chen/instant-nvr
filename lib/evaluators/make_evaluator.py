import importlib
import os


def _evaluator_factory(cfg):
    from lib.evaluators.if_nerf import Evaluator
    module = cfg.evaluator_module
    evaluator:Evaluator = importlib.import_module(module).Evaluator()
    return evaluator


def make_evaluator(cfg):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg)
