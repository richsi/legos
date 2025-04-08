# source env.sh

cp ../EXPLORA_gatech/output/aquarat_mistral7b_16_selected_exemplar.csv data/aquarat/aquarat_train.csv
cp ../EXPLORA_gatech/output/finqa_mistral7b_16_selected_exemplar.csv data/finqa/finqa_train.csv
cp ../EXPLORA_gatech/output/gsm8k_mistral7b_16_selected_exemplar.csv data/gsm8k/gsm8k_train.csv
cp ../EXPLORA_gatech/output/strategyqa_mistral7b_16_selected_exemplar.csv data/strategyqa/strategyqa_train.csv
cp ../EXPLORA_gatech/output/tabmwp_mistral7b_16_selected_exemplar.csv data/tabmwp/tabmwp_train.csv

python -m src.main -p insight_extraction -m mistral7b -d aquarat -n run1 && python -m src.main -p eval -e insight -m mistral7b -d aquarat -n run1 && python -m src.main -p eval -e exemplar -m mistral7b -d aquarat -n run1 && python -m src.main -p eval -e insight_exemplar -m mistral7b -d aquarat -n run1
python -m src.main -p insight_extraction -m mistral7b -d finqa -n run1 && python -m src.main -p eval -e insight -m mistral7b -d finqa -n run1 && python -m src.main -p eval -e exemplar -m mistral7b -d finqa -n run1 && python -m src.main -p eval -e insight_exemplar -m mistral7b -d finqa -n run1
python -m src.main -p insight_extraction -m mistral7b -d gsm8k -n run1 && python -m src.main -p eval -e insight -m mistral7b -d gsm8k -n run1 && python -m src.main -p eval -e exemplar -m mistral7b -d gsm8k -n run1 && python -m src.main -p eval -e insight_exemplar -m mistral7b -d gsm8k -n run1
python -m src.main -p insight_extraction -m mistral7b -d strategyqa -n run1 && python -m src.main -p eval -e insight -m mistral7b -d strategyqa -n run1 && python -m src.main -p eval -e exemplar -m mistral7b -d strategyqa -n run1 && python -m src.main -p eval -e insight_exemplar -m mistral7b -d strategyqa -n run1
python -m src.main -p insight_extraction -m mistral7b -d tabmwp -n run1 && python -m src.main -p eval -e insight -m mistral7b -d tabmwp -n run1 && python -m src.main -p eval -e exemplar -m mistral7b -d tabmwp -n run1 && python -m src.main -p eval -e insight_exemplar -m mistral7b -d tabmwp -n run1

cp ../EXPLORA_gatech/output/aquarat_llama3b_16_selected_exemplar.csv data/aquarat/aquarat_train.csv
cp ../EXPLORA_gatech/output/finqa_llama3b_16_selected_exemplar.csv data/finqa/finqa_train.csv
cp ../EXPLORA_gatech/output/gsm8k_llama3b_16_selected_exemplar.csv data/gsm8k/gsm8k_train.csv
cp ../EXPLORA_gatech/output/strategyqa_llama3b_16_selected_exemplar.csv data/strategyqa/strategyqa_train.csv
cp ../EXPLORA_gatech/output/tabmwp_llama3b_16_selected_exemplar.csv data/tabmwp/tabmwp_train.csv

python -m src.main -p insight_extraction -m llama3b -d aquarat -n run1 && python -m src.main -p eval -e insight -m llama3b -d aquarat -n run1 && python -m src.main -p eval -e exemplar -m llama3b -d aquarat -n run1 && python -m src.main -p eval -e insight_exemplar -m llama3b -d aquarat -n run1
python -m src.main -p insight_extraction -m llama3b -d finqa -n run1 && python -m src.main -p eval -e insight -m llama3b -d finqa -n run1 && python -m src.main -p eval -e exemplar -m llama3b -d finqa -n run1 && python -m src.main -p eval -e insight_exemplar -m llama3b -d finqa -n run1
python -m src.main -p insight_extraction -m llama3b -d gsm8k -n run1 && python -m src.main -p eval -e insight -m llama3b -d gsm8k -n run1 && python -m src.main -p eval -e exemplar -m llama3b -d gsm8k -n run1 && python -m src.main -p eval -e insight_exemplar -m llama3b -d gsm8k -n run1
python -m src.main -p insight_extraction -m llama3b -d strategyqa -n run1 && python -m src.main -p eval -e insight -m llama3b -d strategyqa -n run1 && python -m src.main -p eval -e exemplar -m llama3b -d strategyqa -n run1 && python -m src.main -p eval -e insight_exemplar -m llama3b -d strategyqa -n run1
python -m src.main -p insight_extraction -m llama3b -d tabmwp -n run1 && python -m src.main -p eval -e insight -m llama3b -d tabmwp -n run1 && python -m src.main -p eval -e exemplar -m llama3b -d tabmwp -n run1 && python -m src.main -p eval -e insight_exemplar -m llama3b -d tabmwp -n run1

cp ../EXPLORA_gatech/output/aquarat_llama1b_16_selected_exemplar.csv data/aquarat/aquarat_train.csv
cp ../EXPLORA_gatech/output/finqa_llama1b_16_selected_exemplar.csv data/finqa/finqa_train.csv
cp ../EXPLORA_gatech/output/gsm8k_llama1b_16_selected_exemplar.csv data/gsm8k/gsm8k_train.csv
cp ../EXPLORA_gatech/output/strategyqa_llama1b_16_selected_exemplar.csv data/strategyqa/strategyqa_train.csv
cp ../EXPLORA_gatech/output/tabmwp_llama1b_16_selected_exemplar.csv data/tabmwp/tabmwp_train.csv

python -m src.main -p insight_extraction -m llama1b -d aquarat -n run1 && python -m src.main -p eval -e insight -m llama1b -d aquarat -n run1 && python -m src.main -p eval -e exemplar -m llama1b -d aquarat -n run1 && python -m src.main -p eval -e insight_exemplar -m llama1b -d aquarat -n run1
python -m src.main -p insight_extraction -m llama1b -d finqa -n run1 && python -m src.main -p eval -e insight -m llama1b -d finqa -n run1 && python -m src.main -p eval -e exemplar -m llama1b -d finqa -n run1 && python -m src.main -p eval -e insight_exemplar -m llama1b -d finqa -n run1
python -m src.main -p insight_extraction -m llama1b -d gsm8k -n run1 && python -m src.main -p eval -e insight -m llama1b -d gsm8k -n run1 && python -m src.main -p eval -e exemplar -m llama1b -d gsm8k -n run1 && python -m src.main -p eval -e insight_exemplar -m llama1b -d gsm8k -n run1
python -m src.main -p insight_extraction -m llama1b -d strategyqa -n run1 && python -m src.main -p eval -e insight -m llama1b -d strategyqa -n run1 && python -m src.main -p eval -e exemplar -m llama1b -d strategyqa -n run1 && python -m src.main -p eval -e insight_exemplar -m llama1b -d strategyqa -n run1
python -m src.main -p insight_extraction -m llama1b -d tabmwp -n run1 && python -m src.main -p eval -e insight -m llama1b -d tabmwp -n run1 && python -m src.main -p eval -e exemplar -m llama1b -d tabmwp -n run1 && python -m src.main -p eval -e insight_exemplar -m llama1b -d tabmwp -n run1

