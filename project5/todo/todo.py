import numpy as np
from util.util import softmax_prob, Message, discount, fmt_row
from util.frozen_lake import rollout
import pdb

def value_iteration(env, gamma, nIt):
  """
  Inputs:
      env: Environment description
      gamma: discount factor
      nIt: number of iterations
  Outputs:
      (value_functions, policies)
      
  len(value_functions) == nIt+1 and len(policies) == nIt+1
  """
  Vs = [np.zeros(env.nS)]
  pis = [np.zeros(env.nS,dtype='int')]  
  for it in range(nIt):
    V, pi = vstar_backup(Vs[-1], env, gamma)
    Vs.append(V)
    pis.append(pi)
  return Vs, pis

def policy_iteration(env, gamma, nIt):
  """
  Inputs:
      env: Environment description
      gamma: discount factor
      nIt: number of iterations
  Outputs:
      (value_functions, policies)
      
  len(value_functions) == nIt+1 and len(policies) == nIt+1
  """
  Vs = [np.zeros(env.nS)]
  pis = [np.zeros(env.nS,dtype='int')] 
  for it in range(nIt):
    vpi = policy_evaluation_v(pis[-1], env, gamma)
    qpi = policy_evaluation_q(vpi, env, gamma)
    pi = qpi.argmax(axis=1)
    Vs.append(vpi)
    pis.append(pi)
  return Vs, pis


def policy_gradient_optimize(env, policy, gamma,
      max_pathlength, timesteps_per_batch, n_iter, stepsize):
  from collections import defaultdict
  stat2timeseries = defaultdict(list)
  widths = (17,10,10,10,10)
  print fmt_row(widths, ["EpRewMean","EpLenMean","Perplexity","KLOldNew"])
  for i in xrange(n_iter):
      # collect rollouts
      total_ts = 0
      paths = []
      while True:
          path = rollout(env, policy, max_pathlength)                
          paths.append(path)
          total_ts += path["rewards"].shape[0] # Number of timesteps in the path
          #pathlength(path)
          if total_ts > timesteps_per_batch: 
              break
      print(len(paths))
      print(path['rewards'].shape[0])

      # get observations:
      obs_no = np.concatenate([path["observations"] for path in paths])
      # Update policy
      policy_gradient_step(policy, paths, gamma, stepsize)

      # Compute performance statistics
      pdists = np.concatenate([path["pdists"] for path in paths])
      kl = policy.compute_kl(pdists, policy.compute_pdists(obs_no)).mean()
      perplexity = np.exp(policy.compute_entropy(pdists).mean())

      stats = {  "EpRewMean" : np.mean([path["rewards"].sum() for path in paths]),
                 "EpRewSEM" : np.std([path["rewards"].sum() for path in paths])/np.sqrt(len(paths)),
                 "EpLenMean" : np.mean([path["rewards"].shape[0] for path in paths]), #pathlength(path) 
                 "Perplexity" : perplexity,
                 "KLOldNew" : kl }
      print fmt_row(widths, ['%.3f+-%.3f'%(stats["EpRewMean"], stats['EpRewSEM']), stats['EpLenMean'], stats['Perplexity'], stats['KLOldNew']])
      
      for (name,val) in stats.items():
          stat2timeseries[name].append(val)
  return stat2timeseries


#####################################################
## TODO: You need to implement all functions below ##
#####################################################
def vstar_backup(v_n, env, gamma):
  """
  Apply Bellman backup operator V -> T[V], i.e., perform one step of value iteration

  :param v_n: the state-value function (1D array) for the previous iteration
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: a pair (v_p, a_p), where 
  :  v_p is the updated state-value function and should be a 1D array (S -> R),
  :  a_p is the updated (deterministic) policy, which should also be a 1D array (S -> A)
  """
#  YOUR_CODE_HERE # TODO
  v_p = np.zeros(np.size(v_n))
  a_p = np.zeros(np.size(v_n))
  
  # store state value function for each control (at a given state)
  v_temp = np.zeros(4)
  
  for x in xrange(env.nS):
      for u in xrange(env.nA):
          data = env.P[x][u]
          
          prob_x_prime = np.asarray([d[0] for d in data])
          x_prime = np.asarray([d[1] for d in data])
          
          r = np.asarray([d[2] for d in data])
          r = np.sum( r * prob_x_prime )
          
          # for each state x, saving a state-value function for each control u
          v_temp[u] = r + gamma * np.sum(prob_x_prime * v_n[x_prime])
      
      # for each state, choosing the best value function and the control that leads to best value function
      v_p[x] = np.max(v_temp)
      a_p[x] = np.argmax(v_temp)
    
  
  # just to make sure v_p and a_p ar ethe right sizes (assert gives error if condition is false)
  assert v_p.shape == (env.nS,)
  assert a_p.shape == (env.nS,)
  return (v_p, a_p)

def policy_evaluation_v(pi, env, gamma):
  """
  :param pi: a deterministic policy (1D array: S -> A)
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: vpi, the state-value function for the policy pi
  
  Hint: use np.linalg.solve
  """
#  YOUR_CODE_HERE # TODO

  # transition matrix p(x'|x,pi(x))
  P_matrix = np.zeros((env.nS, env.nS))
  # reward vector r(i,pi(i))
  R_vector = np.zeros(env.nS)
  
  for x in xrange(env.nS):
      data = env.P[x][pi[x]]
      prob_x_prime = np.asarray([d[0] for d in data])
      x_prime = np.asarray([d[1] for d in data])
      r = np.asarray([d[2] for d in data])
      r = np.sum( r * prob_x_prime )
    
      P_matrix[x,x_prime] = np.copy(prob_x_prime)
      R_vector[x] = np.copy(r)

  vpi = np.linalg.solve(np.eye(env.nS)-gamma*P_matrix, R_vector)
  
  assert vpi.shape == (env.nS,)
  return vpi

def policy_evaluation_q(vpi, env, gamma):
  """
  :param vpi: the state-value function for the policy pi
  :param env: environment description providing the transition and reward functions
  :param gamma: the discount factor (scalar)
  :return: qpi, the state-action-value function for the policy pi
  """
#  YOUR_CODE_HERE # TODO

  qpi = np.zeros((env.nS,4))
  
  for x in xrange(env.nS):
      for u in xrange(env.nA):
          data = env.P[x][u]
          
          prob_x_prime = np.asarray([d[0] for d in data])
          x_prime = np.asarray([d[1] for d in data])
          
          r = np.asarray([d[2] for d in data])
          r = np.sum( r * prob_x_prime )
          
          qpi[x,u] = r + gamma * np.sum(prob_x_prime * vpi[x_prime])

  assert qpi.shape == (env.nS, env.nA)
  return qpi

def softmax_policy_gradient(f_sa, s_n, a_n, adv_n):
  """
  Compute policy gradient of policy for discrete MDP, where probabilities
  are obtained by exponentiating f_sa and normalizing.
  
  See softmax_prob and softmax_policy_checkfunc functions in util. This function
  should compute the gradient of softmax_policy_checkfunc.
  
  INPUT:
    f_sa : a matrix representing the policy parameters, whose first dimension s 
           indexes over states, and whose second dimension a indexes over actions
    s_n : states (vector of int)
    a_n : actions (vector of int)
    adv_n : discounted long-term returns (vector of float)
  """
#  YOUR_CODE_HERE # TODO
  # h(x,u,theta) = f_sa[x,u] = theta_x,u

  # probability of control given state and parameters
  pi_u_x_theta = np.zeros(f_sa.shape)
  for x in xrange(f_sa.shape[0]):
      pi_u_x_theta[x,:] = np.exp(f_sa[x,:]) / np.sum(np.exp(f_sa[x,:]))
#  pi_u_x_theta = softmax_prob(f_sa) # equivalent as above
  
  sum_gradient = np.zeros(f_sa.shape)
  
  for idx in xrange(s_n.shape[0]):
      x = s_n[idx]
      u = a_n[idx]
      
      gradient_along_x = np.zeros(f_sa.shape[1])      
      for a in xrange(f_sa.shape[1]):
          if a != u:
              gradient_along_x[a] = -pi_u_x_theta[x,a]
          else:
              gradient_along_x[a] = 1 - pi_u_x_theta[x,a]
              
      sum_gradient[x,:] += gradient_along_x * adv_n[idx]
    
#  grad_sa = adv_n  * sum_gradient
  grad_sa = sum_gradient / s_n.shape[0]

  assert grad_sa.shape == f_sa.shape
  return grad_sa


def policy_gradient_step(policy, paths, gamma,stepsize):
  """
  Compute the discounted returns, compute the policy gradient (using softmax_policy_gradient above),
  and update the policy parameters policy.f_sa
  """
#  YOUR_CODE_HERE # TODO
  theta = policy.f_sa
  grad = 0
  
  for item in paths:
      # discount_mult from gamma**num_timSteps (lowest) to 1 (highest)
      discount_mult = np.cumprod(gamma*np.ones(item['rewards'].shape[0]-1))
      discount_mult = np.insert(discount_mult,0,1)
      
      # item['rewards'] from time 0 to T
      gamma_times_r = item['rewards'] * discount_mult
      discounted_rewards = np.cumsum(gamma_times_r[::-1])[::-1]
      
      grad += softmax_policy_gradient(theta, item['observations'], item['actions'], discounted_rewards)
  
  grad = grad / len(paths)
  policy.f_sa += stepsize * grad


