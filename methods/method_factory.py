from typing import Dict, Any
import sys
import os

# Add parent directory to path to import abstract method
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from methods.abstract_method import AbstractJailbreakMethod
from methods.baseline.vanilla_method import VanillaMethod
from methods.baseline.autodan_method import AutoDANMethod
from methods.baseline.pair import PairMethod
from methods.baseline.pap_method import PAPMethod
from methods.proposed.cka_agent import CKAAgentMethod
from methods.baseline.multi_agent_jailbreak import MultiAgentJailbreakMethod
from methods.baseline.agent_self_response import AgentSelfResponseMethod
from methods.baseline.actor_attack.actor_attack import ActorAttack
from methods.baseline.x_teaming_method import XTeamingMethod
from methods.baseline.parley_method import ParleyMethod


def create_method(method_name: str, config=None, model=None):
    """
    Factory function to create jailbreak methods.

    Args:
        method_name (str): Name of the method
        config (Dict[str, Any]): Method configuration
        model: Target model instance

    Returns:
        AbstractJailbreakMethod: Initialized method instance

    Raises:
        ValueError: If method_name is not supported
    """
    method_name = method_name.lower()
    if method_name == "vanilla":
        return VanillaMethod(name=method_name, config=config, model=model)
    if method_name == "autodan":
        return AutoDANMethod(name=method_name, config=config, model=model)
    if method_name == "pair":
        return PairMethod(name=method_name, config=config, model=model)
    if method_name == "cka-agent":
        return CKAAgentMethod(name=method_name, config=config, model=model)
    if method_name == "pap":
        return PAPMethod(name=method_name, config=config, model=model)
    if method_name == "multi_agent_jailbreak":
        return MultiAgentJailbreakMethod(name=method_name, config=config, model=model)
    if method_name == "agent_self_response":
        return AgentSelfResponseMethod(name=method_name, config=config, model=model)
    if method_name == "actor_attack":
        return ActorAttack(name=method_name, config=config, model=model)
    if method_name == "x_teaming":
        return XTeamingMethod(name=method_name, config=config, model=model)
    if method_name == "parley" or method_name == "tap":
        return ParleyMethod(name=method_name, config=config, model=model)
    raise ValueError(
        f"Unsupported method: {method_name}. "
        f"Supported methods: {supported_methods()}"
    )


def supported_methods():
    return [
        "vanilla",
        "autodan",
        "pair",
        "cka-agent",
        "pap",
        "multi_agent_jailbreak",
        "agent_self_response",
        "actor_attack",
        "x_teaming",
        "parley",
        "tap",
    ]  # Available methods


if __name__ == "__main__":
    vanilla_method = create_method("vanilla")
    print(f"Created method: {vanilla_method}")
