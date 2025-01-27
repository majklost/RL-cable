# Networks and policies for various algos
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomNeighbourNetwork(BaseException):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """


class LinearActor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(self, feature_dim: int, last_layer_dim_pi: int = 64, last_layer_dim_vf: int = 64):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        self.policy_net = nn.Linear(feature_dim, last_layer_dim_pi, bias=False)
        # Value network
        # self.value_net = nn.Sequential(
        #     nn.Linear(feature_dim, 64), nn.Tanh(), nn.Linear(
        #         64, last_layer_dim_vf), nn.Tanh()
        # )
        self.value_net = nn.Linear(feature_dim, last_layer_dim_vf, bias=False)

    def forward(self, features, deterministic=False):
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features):
        return self.policy_net(features)

    def forward_critic(self, features):
        return self.value_net(features)


class LinearActorCriticPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 *args,
                 **kwargs
                 ):
        kwargs['ortho_init'] = False
        super().__init__(observation_space, action_space, lr_schedule,
                         *args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = LinearActor(self.features_dim)


class UniversalPolicy(ActorCriticPolicy):
    def __init__(self,
                 observation_space,
                 action_space,
                 lr_schedule,
                 my_net,
                 *args,
                 **kwargs
                 ):
        kwargs['ortho_init'] = False
        self.my_net = my_net
        super().__init__(observation_space, action_space, lr_schedule,
                         *args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = self.my_net(self.features_dim)
