#sizes=(2 5 10 20 50 75 100 125 150 175 200)
sizes=(2)


path="/home/ejimenez-ruiz/Documents/ATI_AIDA/DivisionMatchingTask/neural_embedding_models/"

task[0]="ama_ncia"
task[1]="fma_nci"
task[2]="fma_snomed"
task[3]="snomed_nci"
task[4]="hp_mp"
task[5]="doid_ordo"


folder[0]="mouse"
folder[1]="fma2nci"
folder[2]="fma2snomed"
folder[3]="snomed2nci"
folder[4]="hp2mp"
folder[5]="doid2ordo"




for t in {0..5}
do
	model=$path"if_"${task[$t]}"_combined/model.tsv"
	#echo $model
	
	for size in ${sizes[*]}
	do
		clusterFile="/home/ejimenez-ruiz/Documents/ATI_AIDA/DivisionMatchingTask/experiments-ijcai/clusters-concepts/"${folder[$t]}"/cluster-"$size
		
		python3 /home/ejimenez-ruiz/Documents/ATI_AIDA/DivisionMatchingTask/neuro-onto-clustering/compute_clusters4concepts.py $model --n-clusters $size --output $clusterFile

		#echo $clusterFile
		#echo $size ${task[$t]}
	done
done




#python3 /home/ejimenez-ruiz/Documents/ATI_AIDA/DivisionMatchingTask/neuro-onto-clustering/compute_clusters4concepts.py 

#second-experiment/last_computed/converted/if_ama_ncia_combined/model.tsv --n-clusters $i --output 200_clusters_fma.txt

