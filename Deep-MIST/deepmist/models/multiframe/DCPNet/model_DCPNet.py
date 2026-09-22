"""
Model implementation adapted from DCPNet.
Only the network definition is integrated.
Training, testing, dataset loading, loss, and metrics are provided by this repository.
"""
import importlib.util
import os

_impl_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'deepmist', 'models', 'multiframe', 'DCPNet', 'model_DCPNet.py'
)
_spec = importlib.util.spec_from_file_location('_dcpnet_impl', _impl_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DCPNet = _mod.DCPNet
