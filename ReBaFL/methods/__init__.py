
from .fedavg import *
from .bsmfedavg import *
from .fedrs import *
from .fedlc import *
from .moon import *
from .fedprox import *
from .fedrep import *
from .scaffold import *
from .feddyn import *
from .local import *
from .rebafl import *
from .fedbabu import *

def local_update(rule):
    # gradient aggregation rule
    LocalUpdate = {
                   'FedAvg':LocalUpdate_FedAvg,
                   'BSMFedAvg':LocalUpdate_BSMFedAvg,
                   'FedProx':LocalUpdate_FedProx,
                   'Scaffold':LocalUpdate_Scaffold,
                   'FedDyn':LocalUpdate_FedDyn,
                   'MOON':LocalUpdate_MOON,
                   'FedRS':LocalUpdate_FedRS,
                   'FedLC':LocalUpdate_FedLC,
                   'Local':LocalUpdate_StandAlone,
                   'ReBaFL':LocalUpdate_ReBaFL,
                   'FedBABU':LocalUpdate_FedBABU,
    }

    return LocalUpdate[rule]