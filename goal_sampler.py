import numpy as np
from utils import get_idxs_per_relation, get_idxs_per_object_permutations
from mpi4py import MPI


ALL_MASKS = True


class GoalSampler:
    def __init__(self, args):
        self.num_rollouts_per_mpi = args.num_rollouts_per_mpi
        self.rank = MPI.COMM_WORLD.Get_rank()

        self.goal_dim = args.env_params['goal']
        self.relation_ids = get_idxs_per_relation(n=args.n_blocks)

        self.n_blocks = args.n_blocks

        self.discovered_goals = []
        self.discovered_goals_str = []

        self.discovered_goals_dict = {}

        self.nb_classes = 0
        self.above_to_id_dict = {}
        self.id_to_above_dict = {}

        self.continuous = args.algo == 'continuous'

        self.init_stats()

    def sample_goal(self, n_goals, evaluation):
        """
        Sample n_goals goals to be targeted during rollouts
        evaluation controls whether or not to sample the goal uniformly or according to curriculum
        """
        if evaluation and len(self.discovered_goals) > 0:
            goals = np.random.choice(self.discovered_goals, size=self.num_rollouts_per_mpi)
        else:
            if len(self.discovered_goals) == 0:
                goals = np.random.choice([-1., 1.], size=(n_goals, self.goal_dim))
            else:
                # sample uniformly from discovered goals
                # goal_ids = np.random.choice(range(len(self.discovered_goals)), size=n_goals)
                # goals = np.array(self.discovered_goals)[goal_ids]
                # First select uniformly a class
                class_id = np.random.randint(self.nb_classes)
                goals_str = np.random.choice(list(self.discovered_goals_dict[self.id_to_above_dict[class_id]]), size=n_goals)
                goals = np.array([np.fromstring(g_s[1:-1], dtype=float, sep=' ') for g_s in goals_str])
        return goals

    def update(self, episodes, t):
        """
        Update discovered goals list from episodes
        Update list of successes and failures for LP curriculum
        Label each episode with the last ag (for buffer storage)
        """
        all_episodes = MPI.COMM_WORLD.gather(episodes, root=0)

        if self.rank == 0:
            all_episode_list = [e for eps in all_episodes for e in eps]

            for e in all_episode_list:
                # Add last achieved goal to memory if first time encountered
                if str(e['ag_binary'][-1]) not in self.discovered_goals_str:
                    above_predicates = e['ag_binary'][-1][-20:]
                    # get all permutations of achieved goal 
                    predicate_permutation_indexes = get_idxs_per_object_permutations(n=self.n_blocks)
                    res = [e['ag_binary'][-1]]
                    for pair in predicate_permutation_indexes:
                        current_vector = e['ag_binary'][-1].copy()
                        current_vector[pair[0]] = e['ag_binary'][-1][pair[1]].copy()
                        current_vector[pair[1]] = e['ag_binary'][-1][pair[0]].copy()
                        res.append(current_vector)
                    unique_permutations = np.unique(np.array(res), axis=0)
                    try: 
                        self.discovered_goals_dict[str(above_predicates)].union(set([str(e) for e in unique_permutations]))
                    except KeyError:
                        self.discovered_goals_dict[str(above_predicates)] = set([str(e) for e in unique_permutations])
                        self.above_to_id_dict[str(above_predicates)] = self.nb_classes
                        self.id_to_above_dict[self.nb_classes] = str(above_predicates)
                        self.nb_classes += 1
                    
                    self.discovered_goals += [e for e in unique_permutations if str(e) not in self.discovered_goals_str]
                    self.discovered_goals_str += [str(e) for e in unique_permutations if str(e) not in self.discovered_goals_str]
                    # self.discovered_goals.append(e['ag_binary'][-1].copy())
                    # self.discovered_goals_str.append(str(e['ag_binary'][-1]))
        self.sync()
        for e in episodes:
            last_above_ag = e['ag_binary'][-1][-20:]
            oracle_id = self.above_to_id_dict[str(last_above_ag)]
            e['last_ag_oracle_id'] = oracle_id

        return episodes

    def sync(self):
        self.discovered_goals = MPI.COMM_WORLD.bcast(self.discovered_goals, root=0)
        self.discovered_goals_str = MPI.COMM_WORLD.bcast(self.discovered_goals_str, root=0)
        self.discovered_goals_dict = MPI.COMM_WORLD.bcast(self.discovered_goals_dict, root=0)
        self.above_to_id_dict = MPI.COMM_WORLD.bcast(self.above_to_id_dict, root=0)
        self.id_to_above_dict = MPI.COMM_WORLD.bcast(self.id_to_above_dict, root=0)
        self.nb_classes = MPI.COMM_WORLD.bcast(self.nb_classes, root=0)

    def build_batch(self, batch_size):
        # goal_ids = np.random.choice(np.arange(len(self.discovered_goals)), size=batch_size)
        goal_ids = np.random.choice(np.arange(self.nb_classes), size=batch_size)
        return goal_ids

    def init_stats(self):
        self.stats = dict()
        # Number of classes of eval
        if self.goal_dim == 30:
            n = 11
        else:
            n = 6
        if not self.continuous:
            for i in np.arange(1, n+1):
                self.stats['Eval_SR_{}'.format(i)] = []
                self.stats['Av_Rew_{}'.format(i)] = []
        self.stats['epoch'] = []
        self.stats['episodes'] = []
        self.stats['global_sr'] = []
        self.stats['nb_discovered'] = []
        self.stats['nb_classes'] = []
        keys = ['goal_sampler', 'rollout', 'gs_update', 'store', 'norm_update',
                'policy_train', 'eval', 'epoch', 'total']
        for k in keys:
            self.stats['t_{}'.format(k)] = []

    def save(self, epoch, episode_count, av_res, av_rew, global_sr, time_dict):
        self.stats['epoch'].append(epoch)
        self.stats['episodes'].append(episode_count)
        self.stats['global_sr'].append(global_sr)
        for k in time_dict.keys():
            self.stats['t_{}'.format(k)].append(time_dict[k])
        self.stats['nb_discovered'].append(len(self.discovered_goals))
        self.stats['nb_classes'].append(len(self.discovered_goals_dict.keys()))
        if not self.continuous:
            for g_id in np.arange(1, len(av_res) + 1):
                self.stats['Eval_SR_{}'.format(g_id)].append(av_res[g_id-1])
                self.stats['Av_Rew_{}'.format(g_id)].append(av_rew[g_id-1])
