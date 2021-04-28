"""Performs computations and file manipulations for train statistics logging purposes"""
import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import pandas as pd
from dsr.program import Program, from_tokens
from dsr.utils import is_pareto_efficient
from itertools import compress


class StatsLogger():
    """ Class responsible for dealing with output files of training statistics. It encapsulates all outputs to files."""
    # Variables used as "Configurations". They will affect which files are saved
    sess = None
    logdir = None
    summary = None
    output_file = None
    save_all_r = None
    save_positional_entropy = None
    save_cache = None
    save_cache_r_min = None
    hof = None
    pareto_front = None

    # Variables to be initiated and used across this class, most of them are pointers to files.
    summary_writer = None
    all_r_output_file = None
    hof_output_file = None
    pf_output_file = None
    positional_entropy_output_file = None
    cache_output_file = None

    def __init__(self, sess, logdir="./log", summary=True, output_file=None, save_all_r=False, hof=10,
                 pareto_front=False,save_positional_entropy=False, save_cache=False, save_cache_r_min=0.9):
        self.sess = sess
        self.logdir = logdir
        self.summary = summary
        self.output_file = output_file
        self.save_all_r = save_all_r
        self.hof = hof
        self.pareto_front = pareto_front
        self.save_positional_entropy = save_positional_entropy
        self.save_cache = save_cache
        self.save_cache_r_min = save_cache_r_min

        self.setup_output_files()

    def setup_output_files(self):
        """
        Opens and prepares all output log files controlled by this class.
        """
        # Creates the summary writer
        if self.summary:
            timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
            summary_dir = os.path.join("summary", timestamp)
            self.summary_writer = tf.summary.FileWriter(summary_dir, self.sess.graph)

        if self.output_file is not None:
            os.makedirs(self.logdir, exist_ok=True)
            self.output_file = os.path.join(self.logdir, self.output_file)
            prefix, _ = os.path.splitext(self.output_file)
            self.all_r_output_file = "{}_all_r.npy".format(prefix)
            self.hof_output_file = "{}_hof.csv".format(prefix)
            self.pf_output_file = "{}_pf.csv".format(prefix)
            self.positional_entropy_output_file = "{}_positional_entropy.npy".format(prefix)
            self.cache_output_file = "{}_cache.csv".format(prefix)
            with open(self.output_file, 'w') as f:
                # r_best : Maximum across all iterations so far
                # r_max : Maximum across this iteration's batch
                # r_avg_full : Average across this iteration's full batch (before taking epsilon subset)
                # r_avg_sub : Average across this iteration's epsilon-subset batch
                # n_unique_* : Number of unique Programs in batch
                # n_novel_* : Number of never-before-seen Programs per batch
                # a_ent_* : Empirical positional entropy across sequences averaged over positions
                # invalid_avg_* : Fraction of invalid Programs per batch
                headers = ["base_r_best",
                           "base_r_max",
                           "base_r_avg_full",
                           "base_r_avg_sub",
                           "r_best",
                           "r_max",
                           "r_avg_full",
                           "r_avg_sub",
                           "l_avg_full",
                           "l_avg_sub",
                           "ewma",
                           "n_unique_full",
                           "n_unique_sub",
                           "n_novel_full",
                           "n_novel_sub",
                           "a_ent_full",
                           "a_ent_sub",
                           "invalid_avg_full",
                           "invalid_avg_sub"]
                f.write("{}\n".format(",".join(headers)))
        else:
            self.all_r_output_file = self.hof_output_file = self.pf_output_file = self.positional_entropy_output_file = \
                self.cache_output_file = None

    def save_stats(self, base_r, r, r_full, base_r_full, l, l_full, empirical_entropy, actions, actions_full, s, s_full, invalid, invalid_full, base_r_best, base_r_max, r_best,
                    r_max,ewma, summaries, epoch, s_history):
        """Computes and saves all statistics"""
        base_r_avg_full = np.mean(base_r_full)
        r_avg_full = np.mean(r_full)

        l_avg_full = np.mean(l_full)
        a_ent_full = np.mean(np.apply_along_axis(empirical_entropy, 0, actions_full))
        n_unique_full = len(set(s_full))
        n_novel_full = len(set(s_full).difference(s_history))
        invalid_avg_full = np.mean(invalid_full)

        if self.output_file is not None:
            base_r_avg_sub = np.mean(base_r)
            r_avg_sub = np.mean(r)
            l_avg_sub = np.mean(l)
            a_ent_sub = np.mean(np.apply_along_axis(empirical_entropy, 0, actions))
            n_unique_sub = len(set(s))
            n_novel_sub = len(set(s).difference(s_history))
            invalid_avg_sub = np.mean(invalid)
            stats = np.array([[
                base_r_best,
                base_r_max,
                base_r_avg_full,
                base_r_avg_sub,
                r_best,
                r_max,
                r_avg_full,
                r_avg_sub,
                l_avg_full,
                l_avg_sub,
                ewma,
                n_unique_full,
                n_unique_sub,
                n_novel_full,
                n_novel_sub,
                a_ent_full,
                a_ent_sub,
                invalid_avg_full,
                invalid_avg_sub
            ]], dtype=np.float32)
            with open(self.output_file, 'ab') as f:
                np.savetxt(f, stats, delimiter=',')
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, epoch)
            self.summary_writer.flush()

    def save_results(self, all_r, positional_entropy, base_r_history, pool):
        """ Saves stats that are available only after all epochs are finished"""
        if self.save_all_r:
            with open(self.all_r_output_file, 'ab') as f:
                np.save(f, all_r)

        if self.save_positional_entropy:
            with open(self.positional_entropy_output_file, 'ab') as f:
                np.save(f, positional_entropy)

        # Save the hall of fame
        if self.hof is not None and self.hof > 0:
            # For stochastic Tasks, average each unique Program's base_r_history,
            if Program.task.stochastic:

                # Define a helper function to generate a Program from its tostring() value
                def from_token_string(str_tokens, optimize):
                    tokens = np.fromstring(str_tokens, dtype=np.int32)
                    return from_tokens(tokens, optimize=optimize)

                # Generate each unique Program and manually set its base_r to the average of its base_r_history
                keys = base_r_history.keys()  # str_tokens for each unique Program
                vals = base_r_history.values()  # base_r histories for each unique Program
                programs = [from_token_string(str_tokens, optimize=False) for str_tokens in keys]
                for p, base_r in zip(programs, vals):
                    p.base_r = np.mean(base_r)
                    p.count = len(base_r)  # HACK
                    _ = p.r  # HACK: Need to cache reward here (serially) because pool doesn't know the complexity_function

            # For deterministic Programs, just use the cache
            else:
                programs = list(Program.cache.values())  # All unique Programs found during training

                """
                NOTE: Equivalence class computation is too expensive.
                Refactor later based on reward and/or semantics.
                """
                # # Filter out symbolically unique Programs. Assume N/A expressions are unique.
                # if pool is not None:
                #     results = pool.map(sympy_work, programs)
                #     for p, result in zip(programs, results):
                #         p.sympy_expr = result[0]
                # else:
                #     results = list(map(sympy_work, programs))
                # str_sympy_exprs = [result[1] for result in results]
                # unique_ids = np.unique(str_sympy_exprs, return_index=True)[1].tolist()
                # na_ids = [i for i in range(len(str_sympy_exprs)) if str_sympy_exprs[i] == "N/A"]
                # programs = list(map(programs.__getitem__, unique_ids + na_ids))

            def hof_work(p):
                return [p.r, p.base_r, p.count, repr(p.sympy_expr), repr(p), p.evaluate]

            base_r = [p.base_r for p in programs]
            i_hof = np.argsort(base_r)[-self.hof:][::-1]  # Indices of top hof Programs
            hof = [programs[i] for i in i_hof]


            if pool is not None:
                results = pool.map(hof_work, hof)
            else:
                results = list(map(hof_work, hof))

            eval_keys = list(results[0][-1].keys())
            columns = ["r", "base_r", "count", "expression", "traversal"] + eval_keys
            hof_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
            df = pd.DataFrame(hof_results, columns=columns)
            if self.hof_output_file is not None:
                print("Saving Hall of Fame to {}".format(self.hof_output_file))
                df.to_csv(self.hof_output_file, header=True, index=False)

            #save cache
            if self.save_cache and Program.cache:
                print("Saving cache to {}".format(self.cache_output_file))
                cache_data = [(repr(p), p.count, p.r) for p in Program.cache.values()]
                df_cache = pd.DataFrame(cache_data)
                df_cache.columns = ["str", "count", "r"]
                if self.save_cache_r_min is not None:
                    df_cache = df_cache[df_cache["r"] >= self.save_cache_r_min]
                df_cache.to_csv(self.cache_output_file, header=True, index=False)

            # Compute the pareto front
            if self.pareto_front:
                def pf_work(p):
                    return [p.complexity_eureqa, p.r, p.base_r, p.count, repr(p.sympy_expr), repr(p), p.evaluate]
                #if verbose:
                #    print("Evaluating the pareto front...")
                all_programs = list(Program.cache.values())
                costs = np.array([(p.complexity_eureqa, -p.r) for p in all_programs])
                pareto_efficient_mask = is_pareto_efficient(costs)  # List of bool
                pf = list(compress(all_programs, pareto_efficient_mask))
                pf.sort(key=lambda p: p.complexity_eureqa)  # Sort by complexity

                if pool is not None:
                    results = pool.map(pf_work, pf)
                else:
                    results = list(map(pf_work, pf))

                eval_keys = list(results[0][-1].keys())
                columns = ["complexity", "r", "base_r", "count", "expression", "traversal"] + eval_keys
                pf_results = [result[:-1] + [result[-1][k] for k in eval_keys] for result in results]
                df = pd.DataFrame(pf_results, columns=columns)
                if self.pf_output_file is not None:
                    print("Saving Pareto Front to {}".format(self.pf_output_file))
                    df.to_csv(self.pf_output_file, header=True, index=False)

                # Look for a success=True case within the Pareto front
                for p in pf:
                    if p.evaluate.get("success"):
                        p_final = p
                        break