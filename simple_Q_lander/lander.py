import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque
import random , argparse , time , uuid , pickle
import basic_Q_agent as Q

argparser = argparse.ArgumentParser()
argparser.add_argument( "--algo" , type=str , default="simple" , help="simple or double or crowd" )
argparser.add_argument( "--crowd" , action="store_true" , default=False , help="Enable crowd" )
argparser.add_argument( "--num_agents" , type=int , default=10 , help="Number of agents in crowd" )
argparser.add_argument( "--num_episodes" , type=int , default=1000 , help="Number of episodes" )
argparser.add_argument( "--batch_size" , type=int , default=64 , help="Batch size for learning" )
argparser.add_argument( "--epsilon_start" , type=float , default=1.0 , help="Starting epsilon value" )
argparser.add_argument( "--epsilon_end" , type=float , default=0.01 , help="Ending epsilon value" )
argparser.add_argument( "--epsilon_decay" , type=float , default=0.995 , help="Epsilon decay value" )
argparser.add_argument( "--gamma" , type=float , default=0.99 , help="Discount factor" )
argparser.add_argument( "--lr" , type=float , default=0.001 , help="Learning rate" )
argparser.add_argument( "--seed" , type=int , default=0 , help="Seed for random number generation" )
argparser.add_argument( "--gravity" , type=float , default=-10.0 , help="Gravity value" )
argparser.add_argument( "--enable_wind" , type=bool , default=False , help="Enable wind" )
argparser.add_argument( "--wind_power" , type=float , default=15.0 , help="Wind power" )
argparser.add_argument( "--turbulence_power" , type=float , default=1.5 , help="Turbulence power" )
argparser.add_argument( "--results_dir" , type=str , default=None , help="Directory to store results" )
argparser.add_argument( "--training_epsisode_length" , type=int , default=500 , help="Training episode length" )
argparser.add_argument( "--eval_interval" , type=int , default=0 , help="Evaluation interval" )
argparser.add_argument( "--eval_episodes" , type=int , default=32 , help="Number of episodes for evaluation" )
argparser.add_argument( "--eval_episode_length" , type=int , default=500 , help="Evaluation episode length" )
args = argparser.parse_args()

############################################################################################################
# random/set seed stuff
if args.seed :
    seed = args.seed
else :
    # create a random seed
    seed = int( time.time() )
print("Using seed: ", seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

############################################################################################################
# Environment setup
env = gym.make( "LunarLander-v3" , continuous=False , gravity=args.gravity ,
            enable_wind=args.enable_wind , wind_power=args.wind_power, 
            turbulence_power=args.turbulence_power )

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
memory = deque( maxlen=10000 )

############################################################################################################
# Agent setup

if args.crowd :
    if args.algo == "simple" :
        agents = [ Q.QAgent( state_size=state_size , action_size=action_size ) for i in range( args.num_agents ) ]
    elif args.algo == "double" :
        agents = [ Q.QAgent_DblQ( state_size=state_size , action_size=action_size ) for i in range( args.num_agents ) ]
    agent = Q.Crowd( agents , batch_size=args.batch_size , epsilon_start=args.epsilon_start , epsilon_end=args.epsilon_end , epsilon_decay=args.epsilon_decay )
else :
    if args.algo == "double" :
        agent = Q.QAgent_DblQ( state_size=state_size, action_size=action_size )

    if args.algo == "simple" :
        agent = Q.QAgent( state_size=state_size, action_size=action_size )

t = 0 ; training_evals = [] ; eval_evals = []
for e in range(args.num_episodes):

    state , info = env.reset()
    state = np.reshape(state, [1, state_size])
    total_reward = 0

    for time in range(500) :

        action = agent.act(state)
        next_state , reward , done , truncated , info = env.step(action)
        memory.append( ( state , action , reward , next_state , done ) )
        next_state = np.reshape(next_state, [1, state_size])
        if t % 4 == 0 and len(memory) > agent.batch_size :
            agent.learn( memory)
        state = next_state
        total_reward += reward
        t += 1

        if done or truncated or time == 499 :
            print(f"Episode: {e+1}/{args.num_episodes} , Score: {total_reward} , Done: {done} , Truncated: {truncated} , Time: {time}")
            break

    training_evals.append( { "episode" : e ,
                             "score" : total_reward ,
                             "done" : done ,
                             "truncated" : truncated ,
                             "time" : time ,
                             "cumulative time" : t } )

    if args.eval_interval > 0 and ( (e+1) % args.eval_interval == 0 or e == 0 ) :
        eval_scores = []
        for _ in range( args.eval_episodes ) :
            state , info = env.reset()
            state = np.reshape(state, [1, state_size])
            total_reward = 0
            for time in range( args.training_epsisode_length ) :
                action = agent.act(state , deterministic=True)
                next_state , reward , done , truncated , info = env.step(action)
                next_state = np.reshape(next_state, [1, state_size])
                state = next_state
                total_reward += reward
                if done or truncated or time == ( args.training_epsisode_length - 1 ) :
                    break
            eval_scores.append( total_reward )
        print( f"Evaluation: {t} , Mean score: {np.mean(eval_scores)} , Std score: {np.std(eval_scores)} , Max score: {np.max(eval_scores)} , Min score: {np.min(eval_scores)}" )
        eval_evals.append( { "episode" : e , 
                            "mean_score" : np.mean(eval_scores) , 
                            "std_score" : np.std(eval_scores) ,
                            "scores" : eval_scores } )

############################################################################################################
# Save results
if args.results_dir :
    filename = args.results_dir + "/" + str(uuid.uuid4()) + ".pkl"
    print("Saving results to ", filename)
    with open( filename , "wb" ) as fh :
        pickle.dump( args , fh )
        pickle.dump( training_evals , fh )
        pickle.dump( eval_evals , fh )

############################################################################################################
# print the means eval scores
print("Mean eval scores: ", [ eval["mean_score"] for eval in eval_evals ] )