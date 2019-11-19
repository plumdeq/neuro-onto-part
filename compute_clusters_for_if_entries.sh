sizes=(2 5 10 20 50 75 100 125 150 175 200)
#sizes=(2)


path="/home/ejimenez-ruiz/Documents/ATI_AIDA/DivisionMatchingTask/neural_embedding_models/"
pathIF="/home/ejimenez-ruiz/Documents/ATI_AIDA/DivisionMatchingTask/inverted_files/"

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

	if=$pathIF"if_"${task[$t]}"_combined"
	
	for size in ${sizes[*]}
	do
		clusterFile="/home/ejimenez-ruiz/Documents/ATI_AIDA/DivisionMatchingTask/experiments-ijcai/clusters-if/"${folder[$t]}"/cluster-"$size
		
		python3 /home/ejimenez-ruiz/Documents/ATI_AIDA/DivisionMatchingTask/neuro-onto-clustering/compute_clusters4if_entries.py $if $model --n-clusters $size --output $clusterFile

		#echo $clusterFile
		#echo $size ${task[$t]}
	done
done



