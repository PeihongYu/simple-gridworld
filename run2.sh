RUNS=5
for ((i=0;i<${RUNS};i++));
do
  python main.py --env centerSquare6x6_2a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
  python main.py --env centerSquare6x6_3a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}
  python main.py --env centerSquare6x6_4a --algo PPO --use_prior --N 1000 --pweight 1 --pdecay 0.995 --add_noise --run ${i} --seed ${i}

#    python main.py --env centerSquare6x6_2a --algo PPO --use_prior --N 1000 --add_noise --run ${i}
#    python main.py --env centerSquare6x6_3a --algo PPO --use_prior --N 1000 --add_noise --run ${i}
#    python main.py --env centerSquare6x6_4a --algo PPO --use_prior --N 1000 --add_noise --run ${i}
done

#python main.py --env mpe --mpe_fixed_map --mpe_sparse_reward --frames 20000000 --algo PPO
#python main.py --env mpe --mpe_fixed_map --mpe_sparse_reward --frames 20000000 --algo POfD