"""Zero-truncated Poisson Regression"""

from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np

import pyttb as ttb
from pyttb.gcp.fg_setup import setup
from pyttb.gcp.handles import Objectives
from pyttb.gcp.optimizers import LBFGSB, StochasticSolver
from pyttb.gcp.samplers import GCPSampler


def ztp(  # noqa: PLR0913,
    data: Union[ttb.tensor, ttb.sptensor],
    rank: int,
    optimizer: Union[StochasticSolver, LBFGSB],
    init: Union[Literal["random"], ttb.ktensor, List[np.ndarray]] = "random",
    mask: Optional[Union[ttb.tensor, np.ndarray]] = None,
    sampler: Optional[GCPSampler] = None,
    printitn: int = 1,
) -> Tuple[ttb.ktensor, ttb.ktensor, Dict]:
    """Fits Zero-truncated Poisson Regression CP decomposition

    Parameters
    ----------
    TODO
    """
    if not isinstance(data, (ttb.tensor, ttb.sptensor)):
        raise ValueError("Input data must be tensor or sptensor.")

    # Set zero-truncated poisson function/gradient handles and lower bound
    objective = setup(Objectives.ZT_POISSON, data)

    # TODO handle user-provided mask?
    if mask is None:
        ind, _ = data.find()
        mask = ttb.tensor.from_function(np.zeros, data.shape)
        mask[ind] = 1.0

    # Pass to gcp_opt
    return ttb.gcp_opt(
        data=data,
        rank=rank,
        objective=objective,
        optimizer=optimizer,
        init=init,
        mask=mask,
        sampler=sampler,
        printitn=printitn,
    )
