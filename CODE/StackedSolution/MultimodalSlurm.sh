#!/bin/bash 

# --gpus=2
# --mem-per-cpu=16G #good for 1 word

#SBATCH --job-name=k_vrh_K
#SBATCH --nodelist=gpu01
#SBATCH --mail-user=dario.massa@ideas-ncbr.pl
#SBATCH --mail-type=ALL

#SBATCH --gpus=ampere:1

#source activate density_cnns
eval "$(conda shell.bash hook)"
conda activate ~/anaconda3/envs/tfgpu1/
cd /raid/NFS_SHARE/home/dario.massa/PROJECT_3_corrected

#TEXT_MOD='albert-base-v2'
#text_mod='albert-base-v2'

TEXT_MOD='RobertaBase'
text_mod='roberta-base'

#TEXT_MOD='gpt2-medium'
#text_mod='gpt2-medium'

#TEXT_MOD='gpt2-large'
#text_mod='gpt2-large'

#text_info='1_word'
text_info='KEY'
#text_info='most'

#max_text_length=11 #10 #11 #9
max_text_length=100
#max_text_length=512

# Define the parameters
Splitting_test_and_hold=0.3
Drop=0.2 
batch=32 #16 #32
#LR_values=("0.0002" "0.0005" "0.0007" "0.001" "0.002" "0.005" "0.007" "0.01" "0.02")
LR_values=("0.0007")
model='IncV3'
numberfeat=1
number_ft_layers=5
number_ft_layers_2=100
LR_FT=0.00001

seed=420
epochs1=1
epochs2=700
epochs3=700

#feature='efermi'
#feature='g_vrh'
feature='k_vrh'
#feature='formation_energy_per_atom'


expname="${TEXT_MOD}_${feature}_${text_info}"

if [ ! -d "./results/${expname}" ]; then
    mkdir -p "./results/${expname}"
fi


for LR in "${LR_values[@]}"; do

    echo "Processing LR=$LR"
    # Create the name of the text file using the parameters
    file_name="${feature}_Parameters_${IncV3}_${TEXT_MOD}_${text_info}_${max_text_length}_Splitting_${Splitting_test_and_hold}_Drop_${Drop}_batch_${batch}_LR_{$LR}_NF_${numberfeat}_NFTL_${number_ft_layers}_NFTL2_${number_ft_layers_2}_LRFT_${LR_FT}_${epochs1}_${epochs2}_${epochs3}_s${seed}.txt"
    out_file_name="${feature}_OUT_${IncV3}_${TEXT_MOD}_${text_info}_${max_text_length}_Splitting_${Splitting_test_and_hold}_Drop_${Drop}_batch_${batch}_LR_${LR}_NF_${numberfeat}_NFTL_${number_ft_layers}_NFTL2_${number_ft_layers_2}_LRFT_${LR_FT}_${epochs1}_${epochs2}_${epochs3}_s${seed}.txt"

    # After defining the parameters, add this to print the parameter values.
    echo "Splitting_test_and_hold=${Splitting_test_and_hold}"
    echo "Drop=${Drop}"
    echo "batch=${batch}"
    echo "LR=${LR}"
    echo "numberfeat=${numberfeat}"
    echo "number_ft_layers=${number_ft_layers}"
    echo "number_ft_layers_2=${number_ft_layers_2}"
    echo "LR_FT=${LR_FT}"
    echo "feature=${feature}"
    echo "text_mod=${text_mod}"
    echo "text_info=${text_info}"
    echo "max_text_length=${max_text_length}"
    echo "epochs1=${epochs1}"
    echo "epochs2=${epochs2}"
    echo "epochs3=${epochs3}"
    echo "seed=${seed}"
    echo "expname=${expname}"

    # Write the parameters to the text file
    echo "Splitting_test_and_hold=${Splitting_test_and_hold}" > "${file_name}"
    echo "Drop=${Drop}" >> "${file_name}"
    echo "batch=${batch}" >> "${file_name}"
    echo "LR=${LR}" >> "${file_name}"
    echo "numberfeat=${numberfeat}" >> "${file_name}"
    echo "number_ft_layers=${number_ft_layers}" >> "${file_name}"
    echo "number_ft_layers_2=${number_ft_layers_2}" >> "${file_name}"
    echo "LR_FT=${LR_FT}" >> "${file_name}"
    echo "feature=${feature}" >> "${file_name}"
    echo "text_mod=${text_mod}" >> "${file_name}"
    echo "textinfo=${text_info}" >> "${file_name}"
    echo "max_text_length=${max_text_length}" >> "${file_name}"
    echo "epochs1=${epochs1}">> "${file_name}"
    echo "epochs2=${epochs2}">> "${file_name}"
    echo "epochs3=${epochs3}">> "${file_name}"
    echo "seed=${seed}">> "${file_name}"
    echo "expname=${expname}" >> "${file_name}"

    # Create a new Python file with the parameter values in its name
    new_python_file="${feature}_MultimodalFinal_${IncV3}_${TEXT_MOD}_${text_info}_${max_text_length}_Splitting_${Splitting_test_and_hold}_Drop_${Drop}_batch_${batch}_${LR}_NF_${numberfeat}_NFTL1_${number_ft_layers}_NFTL2_${number_ft_layers_2}_LRFT_${LR_FT}_${epochs1}_${epochs2}_${epochs3}_s${seed}.py"
    cp MultimodalFinal.py "${new_python_file}"
    # Run the copied Python code with the specific parameter values
    python -u "${new_python_file}" "${file_name}" > "./results/${expname}/${out_file_name}"

    mv "${new_python_file}" "./results/${expname}"
    mv "${file_name}" "./results/${expname}"

    sleep 5
done
``

