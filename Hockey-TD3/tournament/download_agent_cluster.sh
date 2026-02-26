JOB_ID=1989740
mkdir -p "./$JOB_ID/saved"
mkdir -p "./$JOB_ID/logs"
scp -r "stud435@login1.tcml.uni-tuebingen.de:~/outputs/$JOB_ID/logs/" ./$JOB_ID/
scp -r "stud435@login1.tcml.uni-tuebingen.de:~/outputs/$JOB_ID/saved/td3_final.pt" ./$JOB_ID/saved/
#scp -r "stud435@login1.tcml.uni-tuebingen.de:~/outputs/$JOB_ID/saved/" ./$JOB_ID/