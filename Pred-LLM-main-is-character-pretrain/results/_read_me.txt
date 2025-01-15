need to create 2 folders "_classification\pred_llm" and "_regression\pred_llm"

python -W ignore pred_llm.py --dataset 'classification' --method original --trainsize 1.0 --testsize 0.2 --gensize 1.0 --runs 3

python -W ignore pred_llm.py --dataset 'iris' --method original --trainsize 1.0 --testsize 0.2 --gensize 1.0 --runs 1