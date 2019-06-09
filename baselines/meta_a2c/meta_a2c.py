import time
import os
import glob
import functools
import tensorflow as tf
import pandas as pd

from baselines import logger

from baselines.common import set_global_seeds, explained_variance
from baselines.common import tf_util
from baselines.common.policies import build_policy


from baselines.meta_a2c.utils import Scheduler, find_trainable_variables
from baselines.meta_a2c.runner import Runner
import numpy as np

from tensorflow import losses

class Model(object):

    """
    We use this class to :
        __init__:
        - Creates the step_model
        - Creates the train_model

        train():
        - Make the training part (feedforward and retropropagation of gradients)

        save/load():
        - Save load the model
    """
    def __init__(self, policy, env, nsteps,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5, lr=7e-4,
            alpha=0.99, epsilon=1e-5, total_timesteps=int(80e6), lrschedule='linear'):

        sess = tf_util.get_session()
        nenvs = env.num_envs
        nbatch = nenvs*nsteps


        with tf.variable_scope('a2c_model', reuse=tf.AUTO_REUSE):
            # step_model is used for sampling
            step_model = policy(nenvs, 1, sess)

            # train_model is used to train our network
            train_model = policy(nbatch, nsteps, sess)

        A = tf.placeholder(train_model.action.dtype, train_model.action.shape)
        ADV = tf.placeholder(tf.float32, [nbatch])
        R = tf.placeholder(tf.float32, [nbatch])
        LR = tf.placeholder(tf.float32, [])

        # Calculate the loss
        # Total loss = Policy gradient loss - entropy * entropy coefficient + Value coefficient * value loss

        # Policy loss
        neglogpac = train_model.pd.neglogp(A)
        # L = A(s,a) * -logpi(a|s)
        pg_loss = tf.reduce_mean(ADV * neglogpac)

        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy.
        entropy = tf.reduce_mean(train_model.pd.entropy())

        # Value loss
        vf_loss = losses.mean_squared_error(tf.squeeze(train_model.vf), R)

        loss = pg_loss - entropy*ent_coef + vf_loss * vf_coef

        # Update parameters using loss
        # 1. Get the model parameters
        params = find_trainable_variables("a2c_model")

        # 2. Calculate the gradients
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # zip aggregate each gradient with parameters associated
        # For instance zip(ABCD, xyza) => Ax, By, Cz, Da

        # 3. Make op for one policy and value update step of A2C
        trainer = tf.train.RMSPropOptimizer(learning_rate=LR, decay=alpha, epsilon=epsilon)

        _train = trainer.apply_gradients(grads)

        lr = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        def train(obs, states, rewards, masks, actions, values, p_rewards, p_actions, timesteps):
            # Here we calculate advantage A(s,a) = R + yV(s') - V(s)
            # rewards = R + yV(s')
            advs = rewards - values
            for step in range(len(obs)):
                cur_lr = lr.value()

            td_map = {
                train_model.X: obs,
                train_model.p_action: p_actions,
                train_model.p_reward: p_rewards,
                train_model.timestep: timesteps,
                A: actions,
                ADV: advs,
                R: rewards,
                LR: cur_lr
            }
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            policy_loss, value_loss, policy_entropy, _ = sess.run(
                [pg_loss, vf_loss, entropy, _train],
                td_map
            )
            return policy_loss, value_loss, policy_entropy

        # def reset_state():

        self.train = train
        self.train_model = train_model
        self.step_model = step_model
        self.step = step_model.step
        self.value = step_model.value
        self.initial_state = step_model.initial_state
        # self.reset_internal_state = reset_state
        self.save = functools.partial(tf_util.save_variables, sess=sess)
        self.load = functools.partial(tf_util.load_variables, sess=sess)
        tf.global_variables_initializer().run(session=sess)


def learn(
    network,
    env,
    seed=None,
    nsteps=5,
    total_timesteps=int(80e6),
    vf_coef=0.5,
    ent_coef=0.01,
    max_grad_norm=0.5,
    lr=7e-4,
    lrschedule='linear',
    epsilon=1e-5,
    alpha=0.99,
    gamma=0.99,
    log_interval=100,
    load_path=None,
    n_tasks=5,  # For deep meta-rl: learning to reinforcement learn
    # tmp_save_path=None,
    **network_kwargs):

    '''
    Main entrypoint for A2C algorithm. Train a policy with given network architecture on a given environment using a2c algorithm.

    Parameters:
    -----------

    network:            policy network architecture. Either string (mlp, lstm, lnlstm, cnn_lstm, cnn, cnn_small, conv_only - see baselines.common/models.py for full list)
                        specifying the standard network architecture, or a function that takes tensorflow tensor as input and returns
                        tuple (output_tensor, extra_feed) where output tensor is the last network layer output, extra_feed is None for feed-forward
                        neural nets, and extra_feed is a dictionary describing how to feed state into the network for recurrent neural nets.
                        See baselines.common/policies.py/lstm for more details on using recurrent nets in policies


    env:                RL environment. Should implement interface similar to VecEnv (baselines.common/vec_env) or be wrapped with DummyVecEnv (baselines.common/vec_env/dummy_vec_env.py)


    seed:               seed to make random number sequence in the alorightm reproducible. By default is None which means seed from system noise generator (not reproducible)

    nsteps:             int, number of steps of the vectorized environment per update (i.e. batch size is nsteps * nenv where
                        nenv is number of environment copies simulated in parallel)

    total_timesteps:    int, total number of timesteps to train on (default: 80M)

    vf_coef:            float, coefficient in front of value function loss in the total loss function (default: 0.5)

    ent_coef:           float, coeffictiant in front of the policy entropy in the total loss function (default: 0.01)

    max_gradient_norm:  float, gradient is clipped to have global L2 norm no more than this value (default: 0.5)

    lr:                 float, learning rate for RMSProp (current implementation has RMSProp hardcoded in) (default: 7e-4)

    lrschedule:         schedule of learning rate. Can be 'linear', 'constant', or a function [0..1] -> [0..1] that takes fraction of the training progress as input and
                        returns fraction of the learning rate (specified as lr) as output

    epsilon:            float, RMSProp epsilon (stabilizes square root computation in denominator of RMSProp update) (default: 1e-5)

    alpha:              float, RMSProp decay parameter (default: 0.99)

    gamma:              float, reward discounting parameter (default: 0.99)

    log_interval:       int, specifies how frequently the logs are printed out (default: 100)

    **network_kwargs:   keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network
                        For instance, 'mlp' network architecture has arguments num_hidden and num_layers.

    '''

    # Do to avoid Too many files open exception.
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logger.log("RM Soft limix was:", soft)
    logger.log("RM will set to:", hard)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))

    set_global_seeds(seed)

    # Get the nb of env
    nenvs = env.num_envs
    policy = build_policy(env, network, **network_kwargs)

    logger.log("Number of environments (nenv):", nenvs)
    logger.log("Number of steps (nsteps):", nsteps)
    logger.log("Total timesteps (total_timesteps):", total_timesteps)

    # Instantiate the model object (that creates step_model and train_model)
    model = Model(
        policy=policy,
        env=env,
        nsteps=nsteps,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        lr=lr,
        alpha=alpha,
        epsilon=epsilon,
        total_timesteps=total_timesteps,
        lrschedule=lrschedule
    )

    if load_path is not None:
        logger.log("Loading model from path", load_path)
        model.load(load_path)

    # Save the graph
    file_writer = tf.summary.FileWriter(
        "{root}/graph/".format(root=logger.get_dir()),
        tf.get_default_graph()
    )
    file_writer.close()

    # Calculate the batch_size
    nbatch = nenvs*nsteps
    logger.log("Number of batches (nbatch):", nbatch)

    # Make directory for episode logs
    episode_log_dir = "{dir}/episode_logs".format(
        dir=logger.get_dir()
    )
    os.makedirs(episode_log_dir, exist_ok=True)

    n_stored_episodes = len(glob.glob("{dir}/*".format(dir=episode_log_dir)))

    models_save_dir = "{dir}/models".format(
        dir=logger.get_dir()
    )
    os.makedirs(models_save_dir, exist_ok=True)
    
    episode_df = None
    for task_i in range(1, n_tasks + 1):
        tstart = time.time()

        # Instantiate the runner object inside the for-loop, so we start from
        # the beginning.
        runner = Runner(env, model, nsteps=nsteps, gamma=gamma)
        logger.log("Starting task", task_i)
        # for update in range(1, 3):
        for update in range(1, total_timesteps//nbatch + 1):
            # Get mini batch of experiences
            obs, states, rewards, masks, actions, values, p_rewards, p_actions, p_timesteps, info_dicts = runner.run()

            policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values, p_rewards, p_actions, p_timesteps)
            nseconds = time.time() - tstart

            # Make the pandas dataframe for logging of the info_dict
            if episode_df is None:
                headers = info_dicts[0].keys()
                episode_df = pd.DataFrame(columns=headers)
            else:
                episode_df = episode_df.append(
                    list(info_dicts), ignore_index=True
                )

            # Calculate the fps (frame per second)
            fps = int((update*nbatch)/nseconds)
            if update % log_interval == 0 or update == 1:
                # Calculates if value function is a good predicator of the returns (ev > 1)
                # or if it's just worse than predicting nothing (ev =< 0)
                ev = explained_variance(values, rewards)
                logger.record_tabular("task", task_i)  # For Meta-A2C
                logger.record_tabular("total_time", nseconds)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update*nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("policy_loss", float(policy_loss))
                logger.record_tabular("explained_variance", float(ev))
                logger.dump_tabular()

            # if tmp_save_path is not None:
            # logger.log("Saving temporal training model")
            # tmp_save_path = "{dir}/meta_a2c_tmp-{n}.mdl".format(
            #     dir=models_save_dir,
            #     n=task_i
            # )
            # model.save(tmp_save_path)

        # Save path
        episode_log_path = "{dir}/{name}.csv".format(
            dir=episode_log_dir,
            name="episodes_results"
            # name="task-{t}_ep-{e}".format(
            #     t=task_i,
            #     e=update + n_stored_episodes
            # )
        )

        outfile = open(episode_log_path, 'w')
        episode_df.to_csv(outfile)
        outfile.close()

        # # 2. Reset the environment to start a new MDP (only if supported)
        # if hasattr(env, 'next_task'):
        #     env.next_task()

        if hasattr(env, 'save_db_experiments'):
            logger.log("Saving databse of experiments of the environment")
            env.save_db_experiments()

    return model

