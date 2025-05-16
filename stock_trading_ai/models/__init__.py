"""Model management for stock trading AI."""

from .model_manager import ModelManager
from .architectures import TimeSeriesCNN, create_model_architecture
from .forecast import get_cnn_signal
from .value_model import get_value_signal
from .sentiment import get_sentiment_signal
from .technical_patterns import get_technical_signal
from .insider_model import get_insider_signal_wrapped as get_insider_signal
from .macro_model import get_macro_signal
from .ensemble import weighted_vote

__all__ = [
    'ModelManager',
    'TimeSeriesCNN',
    'create_model_architecture',
    'get_cnn_signal',
    'get_value_signal',
    'get_sentiment_signal',
    'get_technical_signal',
    'get_insider_signal',
    'get_macro_signal',
    'weighted_vote'
]
