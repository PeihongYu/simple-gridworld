RUNS=10
for ((i=0;i<${RUNS};i++));
do
#### Run PPO
#    python main.py --env centerSquare6x6_2a --algo PPO --run ${i} --seed ${i}
#    python main.py --env centerSquare6x6_3a --algo PPO --run ${i} --seed ${i}
#    python main.py --env centerSquare6x6_4a --algo PPO --run ${i} --seed ${i}
#### Run PPO with occupancy measure
    python main.py --env centerSquare6x6_2a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_3a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_4a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
#### generate suboptimal policies for each agent
#    python main.py --env centerSquare6x6_1a_0 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_1 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_2 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_3 --algo PPO --dense_reward --frames 400000
#### Run POfD
    python main.py --env centerSquare6x6_2a --algo POfD --pweight 0.02 --pdecay 1 --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_3a --algo POfD --pweight 0.02 --pdecay 1 --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_4a --algo POfD --pweight 0.02 --pdecay 1 --run ${i} --seed ${i}
done

#python main.py --env centerSquare6x6_2a --algo POfD --ppo_epoch 4 --num_mini_batch 4 --pweight 0.02 --pdecay 1 --frames 3000000 --use_value_norm
#python main.py --env centerSquare6x6_2a --algo POfD --ppo_epoch 4 --num_mini_batch 1 --pweight 0.02 --pdecay 1 --frames 3000000 --use_value_norm
#python main.py --env centerSquare6x6_2a --algo POfD --ppo_epoch 10 --num_mini_batch 1 --pweight 0.02 --pdecay 1 --frames 3000000 --use_value_norm
#python main.py --env centerSquare6x6_2a --algo POfD --ppo_epoch 10 --num_mini_batch 4 --pweight 0.02 --pdecay 1 --frames 3000000 --use_value_norm
#python main.py --env centerSquare6x6_2a --algo POfD --ppo_epoch 15 --num_mini_batch 1 --pweight 0.02 --pdecay 1 --frames 3000000 --use_value_norm

#python main.py --env mpe --frames 10000000
#python main.py --env mpe --frames 10000000 --algo POfD
##python main.py --env mpe --frames 4000000 --use_state_norm
##python main.py --env mpe --frames 4000000 --use_value_norm
#python main.py --env mpe --frames 10000000 --use_gae
#python main.py --env mpe --frames 10000000 --ppo_epoch 10 --num_mini_batch 1