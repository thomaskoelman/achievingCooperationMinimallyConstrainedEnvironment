from MatrixGenerator import MatrixGenerator
import numpy as np
import math
from nashpy import Game
import random

def cooperative_agent_algorithm(nb_moves):
   matrix_maker = MatrixGenerator(nb_moves)
   n = 200
   r = 0.1
   f_ab = 0.2
   f_nash = 0.1
   err_lvls = [.0, .001, .002, .004, .008, .016, .032, .064, .128, .256, .512, 1.0]
   err_distr = dict()
   for lvl in err_lvls:
      err_distr[str(lvl)] = 1/len(err_lvls)
   mean, var = 0, 1
   std = math.sqrt(var)
   particles = []
   for i in range(n):
      p_att = np.random.normal(mean, std)
      p_bel = np.random.normal(mean, std)
      nb_labels = matrix_maker.nb_lemke_howson_labels()
      labels = range(nb_labels)
      label = random.choice(labels)
      def choose_nash(game: Game):
         return game.lemke_howson(initial_dropped_label=label)
      particles.append((p_att, p_bel, choose_nash))
   weights = [1/n] * n

   A, B = matrix_maker.random_matrix()
   game = Game(A, B)

   att_opp = estimate_opponents_attitude(particles)
   bel_opp = estimate_opponents_belief(particles)
   nash_opp = estimate_opponents_method(weights, particles)

   att_agent = get_agent_attitude(att_opp, r)

   modded_game = modify_game(game, att_agent, att_opp)

   ne_agent, ne_opp = nash_opp(modded_game)

   move = pick_move(ne_agent)
   opp_move = pick_move(ne_opp)

   #-----update model-----
   #error estimate
   att_agent = bel_opp
   modded_game = modify_game(game, att_agent, att_opp)
   ne_agent, ne_opp = nash_opp(modded_game)
   j = ne_opp[opp_move]
   k = coop(att_opp, bel_opp)
   for l, lvl in enumerate(err_lvls):
      err_distr[str(lvl)] = err_distr[str(lvl)] # * t(j, k, l)
   normalize_dictionary(err_distr)
   err = est_error(err_lvls, err_distr)

   #---resample particles---
   for i, particle in enumerate(particles):
       p_att = particle[0]
       p_bel = particle[1]
       p_nash = particle[2]
       modded_game = modify_game(game, p_bel, p_att)
       _, ne_opp = p_nash(modded_game)
       weights[i] = ne_opp[opp_move]

   normalize_list(weights)
   particles = draw_particles(particles, weights)





   # gen = MatrixGenerator(nb_moves)
   # A, B = gen.random_matrix()
   # game = modify_game(Game(A, B), 0.5)
   # get_nash = particles[0][2]
   # print(get_nash)
   # print(get_nash(game))


def coop(att, bel):
   return (att + bel) / (math.sqrt(att ** 2 + 1) * math.sqrt(bel ** 2 + 1))

def est_error(err_lvls, err_distr):
   error = 0
   for lvl in err_lvls:
       error += lvl * err_distr[str(lvl)]
   return error

def modify_game(game: Game, att_agent_1, att_agent_2):
    A, B = game.payoff_matrices

    def agent_payoff(matrix, r_action, c_action):
        return matrix[r_action, c_action]

    def opp_payoff(matrix, r_action, c_action):
        return matrix.T[r_action, c_action]

    new_A = []
    for i, row in enumerate(A):
       new_row = []
       for j, val in enumerate(row):
          g_agent = agent_payoff(A, i, j)
          g_opp = opp_payoff(B, i, j)
          new_row.append(g_agent + att_agent_1 * g_opp)
       new_A.append(new_row)
    new_A = np.array(new_A)

    new_B = []
    for i, row in enumerate(B):
       new_row = []
       for j, val in enumerate(row):
          g_agent = agent_payoff(B, i, j)
          g_opp = opp_payoff(A, i, j)
          new_row.append(g_agent + att_agent_2 * g_opp)
       new_B.append(new_row)
    new_B = np.array(new_B)
    return Game(new_A, new_B)

def estimate_opponents_attitude(particles: list) -> float:
    n = len(particles)
    p_atts = [particle[0] for particle in particles]
    return sum(p_atts) / n

def estimate_opponents_belief(particles: list) -> float:
    n = len(particles)
    p_atts = [particle[1] for particle in particles]
    return sum(p_atts) / n

def estimate_opponents_method(weights: list, particles: list):
    index = np.argmax(np.array(weights))
    particle = particles[index]
    nash_opp = particle[2]
    return nash_opp

def get_agent_attitude(att_opp: float, r: float):
    return att_opp + r

def pick_from_distribution(distribution: list) -> int:
    rand = random.random()
    prob = 0
    for action, p in enumerate(distribution):
        prob += p
        if rand < prob:
            return action
    return -1

pick_particle = pick_from_distribution
pick_move = pick_from_distribution

def normalize_dictionary(d):
    factor = 1.0 / sum(d.values())
    for k in d:
        d[k] = d[k] * factor

def normalize_list(l):
    factor = 1.0 / sum(l)
    for i, e in enumerate(l):
        l[i] = e * factor

def draw_particles(particles: list, weights: list):
    n = len(particles)
    new_particle_ids = [pick_particle(weights) for _ in range(n)]
    new_particles = [particles[i] for i in new_particle_ids]
    return new_particles

cooperative_agent_algorithm(nb_moves=16)