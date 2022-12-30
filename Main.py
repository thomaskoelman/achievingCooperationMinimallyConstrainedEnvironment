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

   gen = MatrixGenerator(nb_moves)
   A, B = gen.random_matrix()
   game = modify_game(Game(A, B), 0.5)
   get_nash = particles[0][2]
   print(get_nash)
   print(get_nash(game))


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

cooperative_agent_algorithm(nb_moves=16)