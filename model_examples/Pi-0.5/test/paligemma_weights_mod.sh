paligemma_weights=$1

sed -i "102s|self.input_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)|self.input_tokenizer = AutoTokenizer.from_pretrained(\"${paligemma_weights}\")|" src/lerobot/processor/tokenizer_processor.py
