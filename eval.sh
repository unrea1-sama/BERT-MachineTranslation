export PYTHONPATH=${PYTHONPATH}:`pwd`
python3 input_pipeline.py --input_path ${2} --output_path ${3}
python3 thumt/bin/translator.py \
--input ${3} \
--output ${4} \
--checkpoints ${1} \
--tokenizer tokenizer.json \
--models transformer