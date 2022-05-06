RUNS=10
for ((i=0;i<${RUNS};i++));
do
#### Run PPO
    python main.py --env centerSquare6x6_2a --algo PPO --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_3a --algo PPO --run ${i} --seed ${i}
    python main.py --env centerSquare6x6_4a --algo PPO --run ${i} --seed ${i}
#### Run PPO with occupancy measure
#    python main.py --env centerSquare6x6_2a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
#    python main.py --env centerSquare6x6_3a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
#    python main.py --env centerSquare6x6_4a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
#### generate suboptimal policies for each agent
#    python main.py --env centerSquare6x6_1a_0 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_1 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_2 --algo PPO --dense_reward --frames 400000
#    python main.py --env centerSquare6x6_1a_3 --algo PPO --dense_reward --frames 400000
#### Run POfD
#    python main.py --env centerSquare6x6_2a --algo POfD --pweight 0.02 --pdecay 1 --run ${i} --seed ${i}
#    python main.py --env centerSquare6x6_3a --algo POfD --pweight 0.02 --pdecay 1 --run ${i} --seed ${i}
#    python main.py --env centerSquare6x6_4a --algo POfD --pweight 0.02 --pdecay 1 --run ${i} --seed ${i}
done