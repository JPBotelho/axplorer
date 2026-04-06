from src.envs.cage import CageEnvironment
from src.envs.cycle import SquareEnvironment
from src.envs.isosceles import IsoscelesEnvironment
from src.envs.sphere import SphereEnvironment

ENVS = {"square": SquareEnvironment, "isosceles": IsoscelesEnvironment, "sphere": SphereEnvironment, "cage": CageEnvironment}


def build_env(params):
    """
    Build environment.
    """
    env = ENVS[params.env_name](params)
    return env
