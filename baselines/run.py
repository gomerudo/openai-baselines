import sys
import multiprocessing
import os.path as osp
import os
import gym

try:
    import nasgym
except ImportError as ex:
    raise ex

from collections import defaultdict
import tensorflow as tf
import numpy as np
import pandas as pd
from baselines.common.vec_env import VecFrameStack, VecNormalize, VecEnv
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder
from baselines.common.cmd_util import common_arg_parser, parse_unknown_args, make_vec_env, make_env
from baselines.common.tf_util import get_session
from baselines import logger
from importlib import import_module

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

try:
    import pybullet_envs
except ImportError:
    pybullet_envs = None

try:
    import roboschool
except ImportError:
    roboschool = None

_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def train(args, extra_args):
    env_type, env_id = get_env_type(args)
    print('env_type: {}'.format(env_type))

    total_timesteps = int(args.num_timesteps)
    seed = args.seed

    learn = get_learn_function(args.alg)
    alg_kwargs = get_learn_function_defaults(args.alg, env_type)
    alg_kwargs.update(extra_args)

    env = build_env(args)
    if args.save_video_interval != 0:
        env = VecVideoRecorder(env, osp.join(logger.get_dir(), "videos"), record_video_trigger=lambda x: x % args.save_video_interval == 0, video_length=args.save_video_length)

    if args.network:
        alg_kwargs['network'] = args.network
    else:
        if alg_kwargs.get('network') is None:
            alg_kwargs['network'] = get_default_network(env_type)

    print('Training {} on {}:{} with arguments \n{}'.format(args.alg, env_type, env_id, alg_kwargs))

    model = learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        **alg_kwargs
    )

    return model, env


def build_env(args):
    ncpu = multiprocessing.cpu_count()
    if sys.platform == 'darwin': ncpu //= 2
    nenv = args.num_env or ncpu
    alg = args.alg
    seed = args.seed

    env_type, env_id = get_env_type(args)

    if env_type in {'atari', 'retro'}:
        if alg == 'deepq':
            env = make_env(env_id, env_type, seed=seed, wrapper_kwargs={'frame_stack': True})
        elif alg == 'trpo_mpi':
            env = make_env(env_id, env_type, seed=seed)
        else:
            frame_stack_size = 4
            env = make_vec_env(env_id, env_type, nenv, seed, gamestate=args.gamestate, reward_scale=args.reward_scale)
            env = VecFrameStack(env, frame_stack_size)

    else:
        config = tf.ConfigProto(allow_soft_placement=True,
                               intra_op_parallelism_threads=1,
                               inter_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        get_session(config=config)

        flatten_dict_observations = alg not in {'her'}
        env = make_vec_env(env_id, env_type, args.num_env or 1, seed, reward_scale=args.reward_scale, flatten_dict_observations=flatten_dict_observations)

        if env_type == 'mujoco':
            env = VecNormalize(env)

    return env


def get_env_type(args):
    env_id = args.env

    if args.env_type is not None:
        return args.env_type, env_id

    # Re-parse the gym registry, since we could have new envs since last time.
    for env in gym.envs.registry.all():
        env_type = env._entry_point.split(':')[0].split('.')[-1]
        _game_envs[env_type].add(env.id)  # This is a set so add is idempotent

    if env_id in _game_envs.keys():
        env_type = env_id
        env_id = [g for g in _game_envs[env_type]][0]
    else:
        env_type = None
        for g, e in _game_envs.items():
            if env_id in e:
                env_type = g
                break
        assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

    return env_type, env_id


def get_default_network(env_type):
    if env_type in {'atari', 'retro'}:
        return 'cnn'
    else:
        return 'mlp'

def get_alg_module(alg, submodule=None):
    submodule = submodule or alg
    try:
        # first try to import the alg module from baselines
        alg_module = import_module('.'.join(['baselines', alg, submodule]))
    except ImportError:
        # then from rl_algs
        alg_module = import_module('.'.join(['rl_' + 'algs', alg, submodule]))

    return alg_module


def get_learn_function(alg):
    return get_alg_module(alg).learn


def get_learn_function_defaults(alg, env_type):
    try:
        alg_defaults = get_alg_module(alg, 'defaults')
        kwargs = getattr(alg_defaults, env_type)()
    except (ImportError, AttributeError):
        kwargs = {}
    return kwargs



def parse_cmdline_kwargs(args):
    '''
    convert a list of '='-spaced command-line arguments to a dictionary, evaluating python objects when possible
    '''
    def parse(v):

        assert isinstance(v, str)
        try:
            return eval(v)
        except (NameError, SyntaxError):
            return v

    return {k: parse(v) for k,v in parse_unknown_args(args).items()}



def main(args):
    # configure logger, disable logging in child MPI processes (with rank > 0)

    arg_parser = common_arg_parser()
    args, unknown_args = arg_parser.parse_known_args(args)
    extra_args = parse_cmdline_kwargs(unknown_args)

    if args.extra_import is not None:
        import_module(args.extra_import)

    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure()
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    logger.log("Starting training")
    model, env = train(args, extra_args)
    logger.log("Training ended")

    if args.save_path is not None and rank == 0:
        save_path = osp.expanduser(args.save_path)
        logger.log("Saving trained model to", save_path)
        model.save(save_path)

    if args.play:
        # Make directory for episode logs
        episode_log_dir = "{dir}/play_logs".format(
            dir=logger.get_dir()
        )
        os.makedirs(episode_log_dir, exist_ok=True)
        episode_log_path = "{dir}/{name}.csv".format(
            dir=episode_log_dir,
            name="episode_results"
        )
        episode_df = None

        logger.log("Running trained model")
        obs = env.reset()

        # Temporal change for meta-rl
        p_actions = np.zeros((obs.shape[0]), dtype=np.int32)
        p_rewards = np.zeros((obs.shape[0], 1))
        timesteps = np.zeros((obs.shape[0], 1))

        state = model.initial_state if hasattr(model, 'initial_state') else None
        dones = np.zeros((1,))

        # Control the number of time-steps allowed for the playing
        play_count = 0
        episode_rew = 0
        while play_count < args.num_timesteps:
            if state is not None:
                actions, _, state, _ = model.step(
                    obs,
                    S=state,
                    M=dones,
                    p_action=p_actions,
                    p_reward=p_rewards,
                    timestep=timesteps
                )
            else:
                actions, _, _, _ = model.step(obs)

            obs, rew, done, info_dict = env.step(actions)
            episode_rew += rew[0] if isinstance(env, VecEnv) else rew
            p_actions = actions
            p_rewards = rew
            timesteps += 1
            dones = done

            env.render()

            if episode_df is None:
                headers = info_dict[0].keys()
                episode_df = pd.DataFrame(columns=headers)

            # TODO: Check if this works
            episode_df = episode_df.append(
                info_dict, ignore_index=True
            )

            done = done.any() if isinstance(done, np.ndarray) else done
            if done:
                # Reset the timestep (meta-rl)
                timesteps = np.zeros((obs.shape[0], 1))
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
                obs = env.reset()

                # Every time we are done, we will save the csv's
                if hasattr(env, 'save_db_experiments'):
                    logger.log("Saving database of experiments")
                    env.save_db_experiments()
                if episode_df is not None:
                    outfile = open(episode_log_path, 'w')
                    logger.log("Saving episode logs")
                    episode_df.to_csv(outfile)
                    outfile.close()
                    # episode_df = None

            play_count += 1

        # We also save when we exit
        if hasattr(env, 'save_db_experiments'):
            logger.log("Saving database of experiments")
            env.save_db_experiments()
        if episode_df is not None:
            outfile = open(episode_log_path, 'w')
            logger.log("Saving episode logs")
            episode_df.to_csv(outfile)
            outfile.close()
            # episode_df = None

    env.close()

    return model

if __name__ == '__main__':
    main(sys.argv)
