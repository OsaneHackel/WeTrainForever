#!/bin/bash
export COMPRL_SERVER_URL=comprl.cs.uni-tuebingen.de
export COMPRL_SERVER_PORT=65335
export COMPRL_ACCESS_TOKEN=9876bb30-5be0-45e0-b1f5-aa241c66b22f

cd ~/Hockey-TD3/tournament
# [
# current OU: 1986284 (good! but doesnt beat "strong")
    # this trained against strong, OU: 1989089
# pool, OU, PER: 1987372 (best option I think, wins against maximum dosage)
# pool, G, PER: 1987367 (okay) - looses against sac, Matooo
# pool, P, PER: 1988388  (rather weak, but ok maybe?)
# pretrained P PER 1986921 verliert die ganze zeit
# strong, OU, PER: 1989089 (okay, sometimes looses against SAC I think)
# pool with SAC, OU: 1989129 (okay, wins against RewardWasALie-SAC, wins against maximum_dosage, looses against team23-sac)
# strong, OU, PER: 1989088 (always lost)
# pool with SAC, OU: 1989351 (rather weak, better against MetaCritic-SACastic)
# pool with SAC, P, PER: 1989131 (is good-okay)

singularity exec ~/singularity_build/hockey_td3.simg \
    bash autorestart.sh \
    --args --agent=TD3_memo --saved_agent_path=/home/stud435/outputs/1987372/saved/td3_final.pt