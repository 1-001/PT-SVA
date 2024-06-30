#!/bin/bash


sed  's/,/\t/g'  evaluate.csv | while read a b c d; do
    # echo $d
    # if [ "${b}" = "random_context" ]; then
	#     echo "${a}" "${b}" "${c}" "${d}"
    #     # python3 -u evaluate_models.py "${a}" "${b}" "${c}" "${d}"
    # fi
    echo "${a}" "${b}" "${c}" "${d}"
    python3 -u evaluate_models.py "${a}" "${b}" "${c}" "${d}"
    # break
done 