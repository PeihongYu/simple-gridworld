RUNS=5
for ((i=0;i<${RUNS};i++));
do
    python main.py --env centerSquare6x6_2a --algo PPO --use-prior --N 1000 --add-noise --run ${i}
    python main.py --env centerSquare6x6_3a --algo PPO --use-prior --N 1000 --add-noise --run ${i}
    python main.py --env centerSquare6x6_4a --algo PPO --use-prior --N 1000 --add-noise --run ${i}
done
