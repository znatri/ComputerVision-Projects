"""
This class contains helper functions which will help get the optimizer
"""

from typing import Any, Dict

import torch


def get_optimizer(
    model: torch.nn.Module, config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Returns the optimizer initializer according to the config

    Note: config has a minimum of three entries.
    Feel free to add more entries if you want.
    But do not change the name of the three existing entries

    Args:
    - model: the model to optimize for
    - config: a dictionary containing parameters for the config
    Returns:
    - optimizer: the optimizer
    """

    optimizer = None

    optimizer_type = config.get("optimizer_type", "sgd")
    learning_rate = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 1e-5)

    ############################################################################
    # Student code begin
    ############################################################################

    if optimizer_type == "sdg":
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay, 
            momentum=0, 
            dampening=0, 
            nesterov=False
        )      
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay,
            betas=(0.9, 0.999), 
            eps=1e-08,
            amsgrad=False,
        )

    ############################################################################
    # Student code end
    ############################################################################

    return optimizer
