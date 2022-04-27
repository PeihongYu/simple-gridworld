RUNS=1
for ((i=0;i<${RUNS};i++));
do
#    python main.py --env centerSquare_2a --N 1000 --add-noise --run ${i}
#    python main.py --env centerSquare6x6_2a --N 1000 --add-noise --run ${i}
#    python main.py --env centerSquare6x6_3a --N 1000 --add-noise --run ${i}
#    python main.py --env centerSquare6x6_4a --N 1000 --add-noise --run ${i}
#    python main.py --env centerSquare6x6_2a --run ${i}

# generate suboptimal policies for each agent
    python main.py --env centerSquare6x6_1a_0 --algo PPO --dense-reward
    python main.py --env centerSquare6x6_1a_1 --algo PPO --dense-reward
    python main.py --env centerSquare6x6_1a_2 --algo PPO --dense-reward
    python main.py --env centerSquare6x6_1a_3 --algo PPO --dense-reward

#    python main.py --env centerSquare6x6_2a --algo POfD --pweight 0.02 --pdecay 1 --run ${i}
#    python main.py --env centerSquare6x6_3a --algo POfD --pweight 0.02 --pdecay 1 --run ${i}
#    python main.py --env centerSquare6x6_4a --algo POfD --pweight 0.02 --pdecay 1 --run ${i}
done
