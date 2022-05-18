RUNS=5
for ((i=0;i<${RUNS};i++));
do
  python main.py --env appleDoor_b --algo PPO --use_prior --N 100 --pweight 1 --pdecay 0.995 --add_noise --frames 5000000 --seed ${i} --run ${i}
  python main.py --env appleDoor_b --algo POfD --frames 5000000 --seed ${i} --run ${i}
  python main.py --env appleDoor_b --algo PPO --frames 5000000 --seed $((i+1)) --run $((i+1))

#    python main.py --env centerSquare_2a --N 1000 --add_noise --run ${i}
#    python main.py --env centerSquare6x6_2a --N 1000 --add_noise --run ${i}
#    python main.py --env centerSquare6x6_3a --N 1000 --add_noise --run ${i}
#    python main.py --env centerSquare6x6_4a --N 1000 --add_noise --run ${i}
#    python main.py --env centerSquare6x6_2a --run ${i}

# generate suboptimal policies for each agent
#    python main.py --env centerSquare6x6_1a_0 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_1 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_2 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_3 --algo PPO --dense_reward --frames 400000

#    python main.py --env centerSquare6x6_2a --algo POfD --pweight 0.02 --pdecay 1 --frames 2500000 --run 1
#    python main.py --env centerSquare6x6_3a --algo POfD --pweight 0.02 --pdecay 1 --run ${i}
#    python main.py --env centerSquare6x6_4a --algo POfD --pweight 0.02 --pdecay 1 --run ${i}
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

python main.py --env appleDoor_b_1 --frames 4000000 --save_interval
#python main.py --env appleDoor_b_2 --frames 1000000 --save_interval
#python main.py --env appleDoor_a --algo POfD --frames 5000000